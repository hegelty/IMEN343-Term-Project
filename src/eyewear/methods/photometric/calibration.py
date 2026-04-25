from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eyewear.common.io.input_check import InputInfo
from eyewear.common.schema.models import LandmarkPoint
from eyewear.methods.mediapipe.runner import _try_mediapipe_face_mesh
from eyewear.methods.template import landmark_source, template_landmarks


ANCHOR_NAMES = [
    "left_iris_center",
    "right_iris_center",
    "left_inner_canthus",
    "right_inner_canthus",
    "left_outer_canthus",
    "right_outer_canthus",
    "pronasale",
    "left_alare",
    "right_alare",
]


@dataclass
class PosthocCalibration:
    status: str
    metric_ready: bool
    scale_source: str
    landmarks: dict[str, LandmarkPoint] | None = None
    diagnostics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")
    return np.asarray(vertices, dtype=float)


def mesh_bbox_proxy_landmarks(vertices: np.ndarray, method: str) -> dict[str, np.ndarray]:
    template = template_landmarks(method)
    template_xyz = np.asarray([point.xyz for point in template.values()], dtype=float)
    template_min = template_xyz.min(axis=0)
    template_range = np.maximum(template_xyz.max(axis=0) - template_min, 1e-6)

    mesh_min = vertices.min(axis=0)
    mesh_range = np.maximum(vertices.max(axis=0) - mesh_min, 1e-6)

    proxies: dict[str, np.ndarray] = {}
    for name, point in template.items():
        normalized = (np.asarray(point.xyz, dtype=float) - template_min) / template_range
        proxies[name] = mesh_min + normalized * mesh_range
    return proxies


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be shaped [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("at least three anchors are required for similarity fitting")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    covariance = src_centered.T @ tgt_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    variance = np.sum(src_centered ** 2)
    if variance <= 1e-9:
        raise ValueError("source anchors are degenerate")
    scale = float(np.sum(singular_values) / variance)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(point: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (rotation @ point) + translation


def calibrate_mesh_with_mediapipe_iris(
    input_info: InputInfo,
    mesh_path: Path | None,
    residual_threshold_mm: float = 20.0,
) -> PosthocCalibration:
    if mesh_path is None or not mesh_path.exists():
        return PosthocCalibration(
            status="mesh_missing",
            metric_ready=False,
            scale_source="iris_posthoc_calibration_missing_mesh",
            notes=["Post-hoc calibration skipped because no fitted mesh OBJ was available."],
        )

    extracted = _try_mediapipe_face_mesh(input_info)
    if extracted is None:
        return PosthocCalibration(
            status="mediapipe_iris_unavailable",
            metric_ready=False,
            scale_source="iris_posthoc_calibration_unavailable",
            notes=["Post-hoc calibration skipped because MediaPipe Iris landmarks were unavailable in this environment or image."],
        )

    mediapipe_landmarks, _, mediapipe_status, mediapipe_notes = extracted
    try:
        vertices = load_obj_vertices(mesh_path)
        proxy_points = mesh_bbox_proxy_landmarks(vertices, "photometric_mesh_bbox_proxy")
        anchors = [name for name in ANCHOR_NAMES if name in proxy_points and name in mediapipe_landmarks]
        source = np.asarray([proxy_points[name] for name in anchors], dtype=float)
        target = np.asarray([mediapipe_landmarks[name].xyz for name in anchors], dtype=float)
        scale, rotation, translation = fit_similarity_transform(source, target)
    except Exception as exc:
        return PosthocCalibration(
            status="calibration_failed",
            metric_ready=False,
            scale_source="iris_posthoc_calibration_failed",
            notes=[f"Post-hoc calibration failed: {exc}"],
        )

    transformed_anchors = np.asarray([apply_similarity(point, scale, rotation, translation) for point in source])
    residuals = np.linalg.norm(transformed_anchors - target, axis=1)
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    residual_max = float(np.max(residuals))
    metric_ready = residual_rms <= residual_threshold_mm and scale > 0

    calibrated_landmarks: dict[str, LandmarkPoint] = {}
    for name, proxy in proxy_points.items():
        xyz = apply_similarity(proxy, scale, rotation, translation)
        source_status = landmark_source(name)
        calibrated_landmarks[name] = LandmarkPoint(
            xyz=[round(float(v), 6) for v in xyz.tolist()],
            source=source_status,
            method="photometric_mesh_bbox_proxy_iris_calibrated",
            confidence=0.72 if source_status == "observed" else 0.55 if source_status == "derived" else 0.35,
        )

    transform_matrix = np.eye(4, dtype=float)
    transform_matrix[:3, :3] = scale * rotation
    transform_matrix[:3, 3] = translation

    status = "validated" if metric_ready else "residual_too_high"
    notes = [
        *mediapipe_notes,
        f"Post-hoc Method B scale used MediaPipe Iris anchors from backend '{mediapipe_status}'.",
        "Fitted OBJ vertices were converted to semantic proxy points via mesh bounding-box mapping, then aligned to iris-scaled eye/nose anchors.",
        "Temple, forehead, and ear/back-of-ear geometry remain proxy or estimated; they are intentionally excluded from metric scale validation.",
        "Dense mesh geometry is fitted by photometric_optimization, but semantic landmark mapping remains proxy-level until a validated FLAME vertex map is added.",
    ]
    if not metric_ready:
        notes.append(f"Calibration residual RMS {residual_rms:.2f} mm exceeded threshold {residual_threshold_mm:.2f} mm.")

    return PosthocCalibration(
        status=status,
        metric_ready=metric_ready,
        scale_source="iris_posthoc_calibration" if metric_ready else "iris_posthoc_calibration_residual_too_high",
        landmarks=calibrated_landmarks,
        diagnostics={
            "mesh_path": str(mesh_path),
            "mesh_vertex_count": int(vertices.shape[0]),
            "anchor_names": anchors,
            "anchor_count": len(anchors),
            "similarity_scale_mm_per_mesh_unit": scale,
            "similarity_rotation": rotation.tolist(),
            "similarity_translation_mm": translation.tolist(),
            "similarity_matrix": transform_matrix.tolist(),
            "residual_rms_mm": residual_rms,
            "residual_max_mm": residual_max,
            "residual_threshold_mm": residual_threshold_mm,
            "anchor_residuals_mm": {name: float(value) for name, value in zip(anchors, residuals)},
            "mediapipe_backend_status": mediapipe_status,
        },
        notes=notes,
    )

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eyewear.common.canonicalization.transform import canonicalize
from eyewear.common.geometry.math3d import midpoint
from eyewear.common.io.input_check import InputInfo, inspect_input_path
from eyewear.common.io.writers import write_common_outputs, write_raw_landmarks_csv
from eyewear.common.measurements.compute import compute_measurements
from eyewear.common.schema.models import CanonicalFace, LandmarkPoint
from eyewear.common.schema.validation import validate_required_fields
from eyewear.methods.template import ESTIMATED_LANDMARKS, landmark_source, template_landmarks


IRIS_DIAMETER_MM = 11.7


def _curve_template() -> dict:
    return {
        "nose_bridge_curve": {"type": "polyline", "unit": "mm", "points": [[0, 14, 6.5], [0, 10, 7], [0, 7, 7.5]]},
        "pad_line_cross_section": {"type": "polyline", "unit": "mm", "points": [[7, 8, 8], [0, 8, 8.3], [-7, 8, 8]]},
        "front_wrap_curve_at_hinge_height": {"type": "polyline", "unit": "mm", "points": [[66, 7, -6], [0, 8, 3], [-66, 7, -6]]},
    }


def _template_estimates_at_origin(method: str, origin: list[float]) -> dict[str, LandmarkPoint]:
    template_face = CanonicalFace(subject_id="template", method_name=method, landmarks=template_landmarks(method))
    template_face = canonicalize(template_face)
    estimates: dict[str, LandmarkPoint] = {}
    for name in ESTIMATED_LANDMARKS:
        p = template_face.landmarks[name]
        estimates[name] = LandmarkPoint(
            xyz=[origin[0] + p.xyz[0], origin[1] + p.xyz[1], origin[2] + p.xyz[2]],
            source="estimated",
            method=method,
            confidence=p.confidence,
        )
    return estimates


def _iris_scale_px(landmarks: list, width: int, height: int) -> float | None:
    if len(landmarks) < 478:
        return None

    def px(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=float)

    diameters = [
        np.linalg.norm(px(469) - px(471)),
        np.linalg.norm(px(470) - px(472)),
        np.linalg.norm(px(474) - px(476)),
        np.linalg.norm(px(475) - px(477)),
    ]
    valid = [d for d in diameters if d > 0.5]
    if not valid:
        return None
    return IRIS_DIAMETER_MM / float(np.mean(valid))


def _try_mediapipe_face_mesh(input_info: InputInfo) -> tuple[dict[str, LandmarkPoint], dict[int, dict[str, float]], str, list[str]] | None:
    if input_info.input_mode == "video":
        return None
    image_path = input_info.files[0]

    try:
        import mediapipe as mp  # type: ignore[import-not-found]
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency path
        return None

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    mm_per_px = _iris_scale_px(landmarks, width, height)
    if mm_per_px is None:
        return None

    def xyz(idx: int) -> list[float]:
        lm = landmarks[idx]
        x_px = lm.x * width
        y_px = lm.y * height
        return [
            -(x_px - width / 2.0) * mm_per_px,
            -(y_px - height / 2.0) * mm_per_px,
            -lm.z * width * mm_per_px,
        ]

    semantic_indices = {
        "left_iris_center": 473,
        "right_iris_center": 468,
        "left_inner_canthus": 362,
        "left_outer_canthus": 263,
        "right_inner_canthus": 133,
        "right_outer_canthus": 33,
        "sellion_or_nasion": 168,
        "bridge_top": 6,
        "bridge_mid": 197,
        "bridge_low": 195,
        "bridge_left_contact": 279,
        "bridge_right_contact": 49,
        "pronasale": 1,
        "left_alare": 358,
        "right_alare": 129,
        "forehead_center": 10,
        "left_forehead_wrap": 338,
        "right_forehead_wrap": 109,
        "left_front_width_point": 454,
        "right_front_width_point": 234,
        "left_temple_start": 356,
        "right_temple_start": 127,
    }

    points: dict[str, LandmarkPoint] = {}
    for name, idx in semantic_indices.items():
        points[name] = LandmarkPoint(
            xyz=xyz(idx),
            source=landmark_source(name),
            method="mediapipe_face_mesh_refined_landmarks",
            confidence=0.85,
        )

    origin = midpoint(points["left_iris_center"].xyz, points["right_iris_center"].xyz)
    points.update(_template_estimates_at_origin("visible_ear_proxy", origin))

    raw_rows = {}
    for i, lm in enumerate(landmarks):
        converted = xyz(i)
        raw_rows[i] = {
            "x_norm": lm.x,
            "y_norm": lm.y,
            "z_rel": lm.z,
            "x_mm": converted[0],
            "y_mm": converted[1],
            "z_mm": converted[2],
            "confidence": 0.85,
        }

    notes = [
        "MediaPipe Face Mesh refined landmarks were used as the practical Face Landmarker/Iris backend.",
        f"Iris scaling assumes average human iris diameter of {IRIS_DIAMETER_MM} mm.",
    ]
    return points, raw_rows, "mediapipe_face_mesh_refine_landmarks", notes


def _template_fallback_rows(face: CanonicalFace) -> dict[int, dict[str, float]]:
    return {
        i: {
            "x_norm": 0.5,
            "y_norm": 0.5,
            "z_rel": 0.0,
            "x_mm": p.xyz[0],
            "y_mm": p.xyz[1],
            "z_mm": p.xyz[2],
            "confidence": p.confidence,
        }
        for i, p in enumerate(face.landmarks.values())
    }


def _refresh_estimated_fields(face: CanonicalFace) -> None:
    face.estimated_fields = [k for k, v in face.landmarks.items() if v.source == "estimated"]
    face.estimated_fields.extend(k for k, v in face.measurements.items() if v.status == "estimated")


def run_mediapipe(subject_id: str, input_path: str, output_root: Path, input_mode: str = "single_image") -> dict:
    t0 = time.time()
    input_info = inspect_input_path(input_path, input_mode, allowed_modes={"single_image", "photo_set", "video"})

    raw_rows: dict[int, dict[str, float]]
    extracted = _try_mediapipe_face_mesh(input_info)
    if extracted is None:
        landmarks = template_landmarks("mediapipe_iris_template_proxy")
        backend_status = "template_proxy"
        raw_rows = {}
        quality_notes = [
            "MediaPipe runtime was unavailable, input was video, or no face was detected; deterministic template proxy was emitted for pipeline smoke testing.",
            "Do not treat template-proxy output as subject-specific measurement evidence.",
        ]
        scale_source = "iris_depth_template_proxy"
    else:
        landmarks, raw_rows, backend_status, quality_notes = extracted
        scale_source = "iris_depth"

    quality_notes.extend(input_info.notes)

    face = CanonicalFace(
        subject_id=subject_id,
        method_name="mediapipe_iris",
        landmarks=landmarks,
        scale_source=scale_source,
        metric_ready=extracted is not None,
        backend_name="MediaPipe Face Mesh + Iris",
        backend_status=backend_status,
        quality_notes=quality_notes,
    )
    face.curves = _curve_template()
    face = compute_measurements(canonicalize(face))
    _refresh_estimated_fields(face)
    missing_fields = validate_required_fields(face)

    out_dir = output_root / subject_id / "mediapipe"
    write_common_outputs(
        out_dir,
        face,
        runtime_sec=time.time() - t0,
        input_mode=input_mode,
        input_path=str(input_info.path),
        dependency_burden="low",
        gpu_required="optional",
        image_count=input_info.image_count,
        user_input_burden=input_info.user_input_burden,
        missing_fields=missing_fields,
    )

    write_raw_landmarks_csv(out_dir / "raw_landmarks.csv", raw_rows or _template_fallback_rows(face))

    return {
        "method": "mediapipe",
        "output_dir": str(out_dir),
        "missing_fields": missing_fields,
        "backend_status": backend_status,
        "runtime_sec": round(time.time() - t0, 3),
        "success": len(missing_fields) == 0,
    }

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from eyewear.common.schema.models import CanonicalFace, REQUIRED_LANDMARKS, REQUIRED_MEASUREMENTS
from eyewear.common.schema.validation import split_missing_fields
from eyewear.common.visualization.preview import write_preview


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_common_outputs(
    out_dir: Path,
    face: CanonicalFace,
    runtime_sec: float,
    input_mode: str,
    input_path: str,
    dependency_burden: str,
    gpu_required: str,
    image_count: int = 1,
    user_input_burden: str | None = None,
    missing_fields: list[str] | None = None,
) -> None:
    missing_fields = missing_fields or []
    missing_split = split_missing_fields(missing_fields)
    quality_notes = face.quality_notes or ["RGB-only limitations apply"]
    metadata = {
        "unit": "mm",
        "coordinate_system": {
            "origin": "midpoint_between_iris_centers",
            "x": "subject_right",
            "y": "up",
            "z": "forward",
            "handedness": "right-handed",
        },
        "capture": {
            "method": face.method_name,
            "image_count": image_count,
            "glasses_worn": False,
            "occlusion_notes": "unknown",
            "input_mode": input_mode,
            "input_path": input_path,
            "user_input_burden": user_input_burden or input_mode,
        },
        "scale_source": face.scale_source,
        "metric_ready": face.metric_ready,
        "pose_normalized": face.pose_normalized,
        "quality": {
            "overall_confidence": 0.8,
            "nose_region_confidence": 0.8,
            "ear_region_confidence": 0.5,
            "notes": " ".join(quality_notes),
        },
        "run": {
            "success": len(missing_fields) == 0,
            "backend_name": face.backend_name,
            "backend_status": face.backend_status,
            "landmark_count": len(face.landmarks),
            "required_landmark_count": len(REQUIRED_LANDMARKS),
            "missing_landmarks": missing_split["landmarks"],
            "required_measurement_count": len(REQUIRED_MEASUREMENTS),
            "missing_measurements": missing_split["measurements"],
            "estimated_fields": face.estimated_fields,
            "quality_notes": quality_notes,
        },
        "runtime_sec": runtime_sec,
        "dependency_burden": dependency_burden,
        "gpu_required": gpu_required,
        "comparison_factors": {
            "time_sec": runtime_sec,
            "dependency_burden": dependency_burden,
            "gpu_required": gpu_required,
            "input_burden": user_input_burden or input_mode,
            "output_completeness": "complete" if not missing_fields else "partial",
        },
        "estimated_fields": face.estimated_fields,
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "eyewear_landmarks.json", face.to_landmarks_json())
    write_json(out_dir / "measurements.json", face.to_measurements_json())
    write_json(out_dir / "design_curves.json", {"unit": "mm", "curves": face.curves})

    transform = {
        "facial_transformation_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "scale_source": face.scale_source,
        "metric_ready": face.metric_ready,
        "pose_deg": face.pose_deg,
        "backend_status": face.backend_status,
    }
    write_json(out_dir / "face_transform.json", transform)

    write_preview(out_dir / "preview_front.png", f"{face.subject_id} | {face.method_name} | front", face=face, view="front")
    write_preview(out_dir / "preview_side.png", f"{face.subject_id} | {face.method_name} | side", face=face, view="side")


def write_raw_landmarks_csv(path: Path, points: Dict[int, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x_norm", "y_norm", "z_rel", "x_mm", "y_mm", "z_mm", "confidence"])
        writer.writeheader()
        for pid, row in points.items():
            writer.writerow({"id": pid, **row})


def write_raw_mesh_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Proxy mesh placeholder. Replace with HavenFeng/photometric_optimization output when configured.\n"
        "v 0 0 0\nv 10 0 0\nv 0 10 0\nf 1 2 3\n",
        encoding="utf-8",
    )


def write_flame_params(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, shape=np.zeros(10), expression=np.zeros(10), pose=np.zeros(6), camera=np.zeros(3), texture_or_albedo=np.zeros(8), lighting=np.zeros(9))

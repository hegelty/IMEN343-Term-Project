from __future__ import annotations

import time
from pathlib import Path

from eyewear.common.canonicalization.transform import canonicalize
from eyewear.common.io.input_check import inspect_input_path
from eyewear.common.io.writers import write_common_outputs, write_flame_params, write_json, write_raw_mesh_obj
from eyewear.common.measurements.compute import compute_measurements
from eyewear.common.schema.models import CanonicalFace
from eyewear.common.schema.validation import validate_required_fields
from eyewear.methods.template import template_landmarks


def _curve_template() -> dict:
    return {
        "nose_bridge_curve": {"type": "polyline", "unit": "mm", "points": [[0, 14, 6.5], [0, 10, 7], [0, 7, 7.5]]},
        "pad_line_cross_section": {"type": "polyline", "unit": "mm", "points": [[7, 8, 8], [0, 8, 8.3], [-7, 8, 8]]},
        "front_wrap_curve_at_hinge_height": {"type": "polyline", "unit": "mm", "points": [[66, 7, -6], [0, 8, 3], [-66, 7, -6]]},
    }


def _photometric_backend_status() -> tuple[str, list[str]]:
    repo_root = Path(__file__).resolve().parents[4]
    upstream_dir = repo_root / "third_party" / "photometric_optimization"
    if not upstream_dir.exists():
        return "not_present", ["third_party/photometric_optimization is missing; using proxy mesh output only."]

    non_readme_files = [
        p for p in upstream_dir.rglob("*")
        if p.is_file() and p.name.lower() not in {"readme.md", "upstream.md"}
    ]
    if not non_readme_files:
        return "not_vendored_placeholder", [
            "HavenFeng/photometric_optimization is documented but not vendored or submodule-linked yet.",
            "raw_mesh.obj and flame_params.npz are proxy artifacts until the upstream backend/assets are installed.",
        ]
    return "vendored_or_submodule_present", ["Upstream photometric directory contains files; wrapper still requires environment-specific backend wiring."]


def _refresh_estimated_fields(face: CanonicalFace) -> None:
    face.estimated_fields = [k for k, v in face.landmarks.items() if v.source == "estimated"]
    face.estimated_fields.extend(k for k, v in face.measurements.items() if v.status == "estimated")


def run_photometric(subject_id: str, input_path: str, output_root: Path, input_mode: str = "single_image") -> dict:
    t0 = time.time()
    input_info = inspect_input_path(input_path, input_mode, allowed_modes={"single_image", "photo_set"})
    backend_status, backend_notes = _photometric_backend_status()

    face = CanonicalFace(
        subject_id=subject_id,
        method_name="photometric_optimization",
        landmarks=template_landmarks("mesh_fit_proxy"),
        scale_source="iris_posthoc_calibration_template_proxy",
        metric_ready=False,
        backend_name="HavenFeng/photometric_optimization",
        backend_status=backend_status,
        quality_notes=[
            *backend_notes,
            *input_info.notes,
            "Dense RGB fitting has inherent absolute-scale ambiguity; post-hoc iris calibration is required before using subject-specific mm claims.",
        ],
    )
    face.curves = _curve_template()
    face = compute_measurements(canonicalize(face))
    _refresh_estimated_fields(face)
    missing_fields = validate_required_fields(face)

    out_dir = output_root / subject_id / "photometric"
    write_common_outputs(
        out_dir,
        face,
        runtime_sec=time.time() - t0,
        input_mode=input_mode,
        input_path=str(input_info.path),
        dependency_burden="high",
        gpu_required="recommended",
        image_count=input_info.image_count,
        user_input_burden=input_info.user_input_burden,
        missing_fields=missing_fields,
    )
    write_raw_mesh_obj(out_dir / "raw_mesh.obj")
    write_flame_params(out_dir / "flame_params.npz")
    write_json(
        out_dir / "calibration.json",
        {
            "scale_source": face.scale_source,
            "status": "approximate",
            "backend_status": backend_status,
            "metric_ready": face.metric_ready,
            "assumed_iris_diameter_mm": 11.7,
            "notes": "Absolute scale may vary; iris-based post-hoc calibration is the intended strategy. Current proxy output is not a validated dense fit.",
        },
    )

    return {
        "method": "photometric",
        "output_dir": str(out_dir),
        "missing_fields": missing_fields,
        "backend_status": backend_status,
        "runtime_sec": round(time.time() - t0, 3),
        "success": len(missing_fields) == 0,
    }

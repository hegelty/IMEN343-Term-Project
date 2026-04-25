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
from eyewear.methods.photometric.calibration import calibrate_mesh_with_mediapipe_iris
from eyewear.methods.photometric.upstream import (
    UPSTREAM_HEAD,
    UPSTREAM_REPO,
    copy_upstream_outputs,
    inspect_upstream,
    run_upstream_fitting,
)


def _curve_template() -> dict:
    return {
        "nose_bridge_curve": {"type": "polyline", "unit": "mm", "points": [[0, 14, 6.5], [0, 10, 7], [0, 7, 7.5]]},
        "pad_line_cross_section": {"type": "polyline", "unit": "mm", "points": [[7, 8, 8], [0, 8, 8.3], [-7, 8, 8]]},
        "front_wrap_curve_at_hinge_height": {"type": "polyline", "unit": "mm", "points": [[66, 7, -6], [0, 8, 3], [-66, 7, -6]]},
    }


def _refresh_estimated_fields(face: CanonicalFace) -> None:
    face.estimated_fields = [k for k, v in face.landmarks.items() if v.source == "estimated"]
    face.estimated_fields.extend(k for k, v in face.measurements.items() if v.status == "estimated")


def run_photometric(
    subject_id: str,
    input_path: str,
    output_root: Path,
    input_mode: str = "single_image",
    device: str = "cpu",
    timeout_sec: int = 1800,
    run_upstream: bool = True,
) -> dict:
    t0 = time.time()
    input_info = inspect_input_path(input_path, input_mode, allowed_modes={"single_image", "photo_set"})
    upstream_check = inspect_upstream()
    upstream_result = run_upstream_fitting(input_info, subject_id, device=device, timeout_sec=timeout_sec) if run_upstream else None
    backend_status = upstream_result.status if upstream_result is not None else upstream_check.status
    backend_notes = upstream_result.notes if upstream_result is not None else upstream_check.notes
    backend_ran = bool(upstream_result and upstream_result.success)
    out_dir = output_root / subject_id / "photometric"
    imported = copy_upstream_outputs(upstream_result, out_dir) if upstream_result is not None else {}
    if not imported:
        write_raw_mesh_obj(out_dir / "raw_mesh.obj")
        write_flame_params(out_dir / "flame_params.npz")

    calibration = calibrate_mesh_with_mediapipe_iris(
        input_info,
        upstream_result.mesh_path if backend_ran and upstream_result is not None else None,
    )
    calibrated_landmarks = calibration.landmarks if calibration.metric_ready and calibration.landmarks is not None else None

    face = CanonicalFace(
        subject_id=subject_id,
        method_name="photometric_optimization",
        landmarks=calibrated_landmarks or template_landmarks("mesh_fit_semantic_proxy" if backend_ran else "mesh_fit_proxy"),
        scale_source=calibration.scale_source if backend_ran else "iris_posthoc_calibration_template_proxy",
        metric_ready=calibration.metric_ready if backend_ran else False,
        backend_name="HavenFeng/photometric_optimization",
        backend_status=backend_status,
        quality_notes=[
            *backend_notes,
            *calibration.notes,
            *input_info.notes,
            f"Upstream target: {UPSTREAM_REPO}@{UPSTREAM_HEAD[:7]}.",
        ],
    )
    face.curves = _curve_template()
    face = compute_measurements(canonicalize(face))
    _refresh_estimated_fields(face)
    missing_fields = validate_required_fields(face)

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
    if "similarity_matrix" in calibration.diagnostics:
        write_json(
            out_dir / "face_transform.json",
            {
                "facial_transformation_matrix": calibration.diagnostics["similarity_matrix"],
                "scale_source": face.scale_source,
                "metric_ready": face.metric_ready,
                "pose_deg": face.pose_deg,
                "backend_status": face.backend_status,
                "calibration_status": calibration.status,
                "residual_rms_mm": calibration.diagnostics.get("residual_rms_mm"),
            },
        )
    write_json(
        out_dir / "calibration.json",
        {
            "scale_source": face.scale_source,
            "status": calibration.status if backend_ran else "approximate_template_proxy",
            "backend_status": backend_status,
            "metric_ready": face.metric_ready,
            "assumed_iris_diameter_mm": 11.7,
            "diagnostics": calibration.diagnostics,
            "upstream_repository": UPSTREAM_REPO,
            "upstream_expected_commit": UPSTREAM_HEAD,
            "upstream_detected_commit": upstream_check.commit,
            "missing_upstream_files": upstream_check.missing_files,
            "missing_manual_assets": upstream_check.missing_assets,
            "prepared_input": {
                "image_name": upstream_result.prepared_input.image_name,
                "image_path": str(upstream_result.prepared_input.image_path),
                "landmark_path": str(upstream_result.prepared_input.landmark_path),
                "mask_path": str(upstream_result.prepared_input.mask_path),
            } if upstream_result and upstream_result.prepared_input else None,
            "imported_outputs": imported,
            "stdout_tail": upstream_result.stdout_tail if upstream_result else "",
            "stderr_tail": upstream_result.stderr_tail if upstream_result else "",
            "notes": (
                "Method B is metric-ready only when post-hoc iris calibration succeeds under the residual threshold. "
                "Ear/back-of-ear geometry remains estimated even when metric_ready is true."
            ),
        },
    )

    return {
        "method": "photometric",
        "output_dir": str(out_dir),
        "missing_fields": missing_fields,
        "backend_status": backend_status,
        "upstream_ran": backend_ran,
        "runtime_sec": round(time.time() - t0, 3),
        "success": len(missing_fields) == 0,
    }

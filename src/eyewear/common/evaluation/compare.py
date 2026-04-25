from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required comparison input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_profile(metadata: dict, landmarks: dict) -> dict:
    run = metadata.get("run", {})
    quality = metadata.get("quality", {})
    factors = metadata.get("comparison_factors", {})
    return {
        "success": run.get("success"),
        "backend_status": run.get("backend_status"),
        "landmark_count": run.get("landmark_count", len(landmarks.get("points", {}))),
        "missing_landmarks": run.get("missing_landmarks", []),
        "missing_measurements": run.get("missing_measurements", []),
        "estimated_fields": run.get("estimated_fields", metadata.get("estimated_fields", [])),
        "quality_notes": run.get("quality_notes", [quality.get("notes", "")]),
        "metric_scale_source": metadata.get("scale_source"),
        "metric_ready": metadata.get("metric_ready"),
        "input_burden": factors.get("input_burden", metadata.get("capture", {}).get("input_mode")),
        "dependency_burden": metadata.get("dependency_burden"),
        "gpu_required": metadata.get("gpu_required"),
        "runtime_sec": metadata.get("runtime_sec"),
    }


def compare_subject(output_root: Path, subject_id: str) -> dict:
    m_dir = output_root / subject_id / "mediapipe"
    p_dir = output_root / subject_id / "photometric"

    mm = _load(m_dir / "measurements.json")
    pm = _load(p_dir / "measurements.json")
    md = _load(m_dir / "metadata.json")
    pd = _load(p_dir / "metadata.json")
    ml = _load(m_dir / "eyewear_landmarks.json")
    pl = _load(p_dir / "eyewear_landmarks.json")

    keys = [k for k in mm.keys() if isinstance(mm[k], dict) and "value" in mm[k] and k in pm and isinstance(pm[k], dict)]
    deltas = {k: round(pm[k]["value"] - mm[k]["value"], 3) for k in keys}
    profiles = {
        "mediapipe": _run_profile(md, ml),
        "photometric": _run_profile(pd, pl),
    }

    summary = {
        "subject_id": subject_id,
        "methods": ["mediapipe", "photometric"],
        "runtime_sec": {"mediapipe": md.get("runtime_sec"), "photometric": pd.get("runtime_sec")},
        "scale_source": {"mediapipe": md.get("scale_source"), "photometric": pd.get("scale_source")},
        "metric_ready": {"mediapipe": md.get("metric_ready"), "photometric": pd.get("metric_ready")},
        "per_run": profiles,
        "estimated_fields_count": {
            "mediapipe": len(md.get("estimated_fields", [])),
            "photometric": len(pd.get("estimated_fields", [])),
        },
        "measurement_deltas_photometric_minus_mediapipe": deltas,
        "comparison_dimensions": {
            "accuracy": "Cannot be validated without scanner/manual ground truth; use measurement deltas and missing fields as proxy indicators.",
            "cost": "MediaPipe is low-cost and CPU-friendly; photometric fitting has higher setup and compute burden.",
            "time": {"mediapipe_sec": md.get("runtime_sec"), "photometric_sec": pd.get("runtime_sec")},
            "engineering_complexity": {"mediapipe": "low", "photometric": "high"},
            "setup_burden": {"mediapipe": md.get("dependency_burden"), "photometric": pd.get("dependency_burden")},
            "output_completeness": {
                "mediapipe": "complete" if profiles["mediapipe"]["success"] else "partial",
                "photometric": "complete" if profiles["photometric"]["success"] else "partial",
            },
            "repeatability": "Compare repeated captures with the same canonical schema and review scale_source/metric_ready.",
        },
    }

    rows = [
        "| Dimension | MediaPipe | Photometric |",
        "| --- | --- | --- |",
        f"| Backend status | {profiles['mediapipe']['backend_status']} | {profiles['photometric']['backend_status']} |",
        f"| Metric ready | {profiles['mediapipe']['metric_ready']} | {profiles['photometric']['metric_ready']} |",
        f"| Runtime sec | {profiles['mediapipe']['runtime_sec']} | {profiles['photometric']['runtime_sec']} |",
        f"| Input burden | {profiles['mediapipe']['input_burden']} | {profiles['photometric']['input_burden']} |",
        f"| Dependency burden | {profiles['mediapipe']['dependency_burden']} | {profiles['photometric']['dependency_burden']} |",
        f"| GPU | {profiles['mediapipe']['gpu_required']} | {profiles['photometric']['gpu_required']} |",
        f"| Missing landmarks | {len(profiles['mediapipe']['missing_landmarks'])} | {len(profiles['photometric']['missing_landmarks'])} |",
        f"| Estimated fields | {len(profiles['mediapipe']['estimated_fields'])} | {len(profiles['photometric']['estimated_fields'])} |",
    ]

    report_md = f"""# Comparison Report ({subject_id})\n\n{chr(10).join(rows)}\n\n## Measurement deltas (photometric - mediapipe)\n```json\n{json.dumps(deltas, indent=2, ensure_ascii=False)}\n```\n\n## Interpretation\nAccuracy needs scanner/manual ground truth; this report keeps both methods on the same canonical schema so deltas, missing fields, runtime, cost, and setup burden can be compared fairly.\n\n## Limitations\n- RGB-only limitations remain.\n- Ear/back-of-ear geometry is estimated/proxy-level.\n- Method B setup burden is higher due to external dependencies/assets.\n- Template/proxy backend statuses are not subject-specific evidence.\n"""

    out_dir = output_root / subject_id / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "comparison_report.md").write_text(report_md, encoding="utf-8")
    return summary

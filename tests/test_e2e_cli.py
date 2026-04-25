import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _run(cmd: list[str], cwd: Path) -> str:
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    return result.stdout


def test_cli_e2e_generates_required_outputs(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    input_img = tmp_path / "face.jpg"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(input_img)

    _run([sys.executable, "-m", "eyewear.cli", "run", "mediapipe", "--input", str(input_img), "--subject-id", "t01", "--output-root", str(tmp_path)], repo)
    _run([sys.executable, "-m", "eyewear.cli", "run", "photometric", "--input", str(input_img), "--subject-id", "t01", "--output-root", str(tmp_path)], repo)
    _run([sys.executable, "-m", "eyewear.cli", "compare", "--subject-id", "t01", "--output-root", str(tmp_path)], repo)

    common = ["metadata.json", "eyewear_landmarks.json", "measurements.json", "preview_front.png", "preview_side.png", "face_transform.json", "design_curves.json"]
    for method in ["mediapipe", "photometric"]:
        base = tmp_path / "t01" / method
        for name in common:
            assert (base / name).exists(), f"missing {method}/{name}"

    assert (tmp_path / "t01" / "mediapipe" / "raw_landmarks.csv").exists()
    assert (tmp_path / "t01" / "photometric" / "raw_mesh.obj").exists()
    assert (tmp_path / "t01" / "photometric" / "flame_params.npz").exists()
    assert (tmp_path / "t01" / "photometric" / "calibration.json").exists()

    summary = json.loads((tmp_path / "t01" / "comparison" / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["subject_id"] == "t01"
    assert "per_run" in summary
    assert summary["per_run"]["photometric"]["backend_status"] in {
        "not_vendored_placeholder",
        "setup_blocked_missing_assets",
        "upstream_failed",
        "upstream_success_scale_unverified",
    }

    calibration = json.loads((tmp_path / "t01" / "photometric" / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["upstream_expected_commit"].startswith("83f84b8")
    assert calibration["metric_ready"] is False


def test_cli_photo_set_records_image_count(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for name in ["front.jpg", "left.jpg", "right.jpg"]:
        Image.new("RGB", (64, 64), (200, 200, 200)).save(photo_dir / name)

    _run([sys.executable, "-m", "eyewear.cli", "run", "mediapipe", "--input", str(photo_dir), "--subject-id", "t02", "--output-root", str(tmp_path), "--input-mode", "photo_set"], repo)

    metadata = json.loads((tmp_path / "t02" / "mediapipe" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["capture"]["image_count"] == 3
    assert metadata["capture"]["user_input_burden"] == "multi-view photo set (3 images)"


def test_cli_pipeline_runs_both_methods_and_compare(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    input_img = tmp_path / "face.jpg"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(input_img)

    stdout = _run([
        sys.executable,
        "-m",
        "eyewear.cli",
        "pipeline",
        "--input",
        str(input_img),
        "--subject-id",
        "t03",
        "--output-root",
        str(tmp_path),
    ], repo)

    payload = json.loads(stdout)
    assert payload["subject_id"] == "t03"
    assert (tmp_path / "t03" / "mediapipe" / "metadata.json").exists()
    assert (tmp_path / "t03" / "photometric" / "metadata.json").exists()
    assert (tmp_path / "t03" / "comparison" / "comparison_summary.json").exists()

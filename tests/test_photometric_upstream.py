from pathlib import Path

import numpy as np
from PIL import Image

from eyewear.common.io.input_check import inspect_input_path
from eyewear.methods.photometric.upstream import (
    REQUIRED_MANUAL_ASSETS,
    REQUIRED_UPSTREAM_FILES,
    inspect_upstream,
    prepare_upstream_input,
)


def test_inspect_upstream_reports_missing_assets_after_code_present(tmp_path: Path):
    upstream = tmp_path / "photometric_optimization"
    upstream.mkdir()
    for rel in REQUIRED_UPSTREAM_FILES:
        path = upstream / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    check = inspect_upstream(upstream)

    assert check.status == "setup_blocked_missing_assets"
    assert check.missing_assets == list(REQUIRED_MANUAL_ASSETS)


def test_prepare_upstream_input_creates_ffhq_style_files(tmp_path: Path):
    upstream = tmp_path / "photometric_optimization"
    input_img = tmp_path / "front.jpg"
    Image.new("RGB", (96, 128), (220, 200, 180)).save(input_img)
    input_info = inspect_input_path(str(input_img), "single_image")

    prepared = prepare_upstream_input(upstream, input_info, "subject 01")

    assert prepared.image_name == "subject_01"
    assert prepared.image_path.exists()
    assert prepared.landmark_path.exists()
    assert prepared.mask_path.exists()
    assert np.load(prepared.landmark_path).shape == (68, 2)
    assert np.load(prepared.mask_path).shape == (256, 256)

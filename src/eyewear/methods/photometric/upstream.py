from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from eyewear.common.io.input_check import InputInfo


UPSTREAM_REPO = "https://github.com/HavenFeng/photometric_optimization"
UPSTREAM_HEAD = "83f84b804938af6fe7472e492ec997e38552069d"
REQUIRED_UPSTREAM_FILES = (
    "photometric_fitting.py",
    "renderer.py",
    "util.py",
    "models/FLAME.py",
    "models/lbs.py",
    "data/head_template_mesh.obj",
    "data/landmark_embedding.npy",
)
REQUIRED_MANUAL_ASSETS = (
    "data/generic_model.pkl",
    "data/FLAME_texture.npz",
)


@dataclass(frozen=True)
class UpstreamCheck:
    status: str
    upstream_dir: Path
    notes: list[str]
    missing_files: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    commit: str | None = None

    @property
    def runnable(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class PreparedInput:
    image_name: str
    image_path: Path
    landmark_path: Path
    mask_path: Path
    notes: list[str]


@dataclass(frozen=True)
class UpstreamRunResult:
    success: bool
    status: str
    notes: list[str]
    prepared_input: PreparedInput | None = None
    mesh_path: Path | None = None
    params_path: Path | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def default_upstream_dir() -> Path:
    return repo_root_from_here() / "third_party" / "photometric_optimization"


def sanitize_image_name(subject_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id).strip("._")
    return stem or "subject"


def _git_commit(upstream_dir: Path) -> str | None:
    if not (upstream_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def inspect_upstream(upstream_dir: Path | None = None) -> UpstreamCheck:
    upstream = upstream_dir or default_upstream_dir()
    if not upstream.exists():
        return UpstreamCheck(
            status="not_present",
            upstream_dir=upstream,
            notes=["third_party/photometric_optimization is missing."],
        )

    missing_files = [rel for rel in REQUIRED_UPSTREAM_FILES if not (upstream / rel).exists()]
    if missing_files:
        return UpstreamCheck(
            status="not_vendored_placeholder",
            upstream_dir=upstream,
            notes=[
                "HavenFeng/photometric_optimization code is not fully present.",
                "Add the upstream repository as a submodule or vendored copy before real Method B fitting.",
            ],
            missing_files=missing_files,
            commit=_git_commit(upstream),
        )

    missing_assets = [rel for rel in REQUIRED_MANUAL_ASSETS if not (upstream / rel).exists()]
    if missing_assets:
        return UpstreamCheck(
            status="setup_blocked_missing_assets",
            upstream_dir=upstream,
            notes=[
                "Upstream code is present, but FLAME/texture assets are missing.",
                "The assets require manual download and license acceptance from the FLAME project website.",
            ],
            missing_assets=missing_assets,
            commit=_git_commit(upstream),
        )

    return UpstreamCheck(
        status="ready",
        upstream_dir=upstream,
        notes=["Upstream code and required manual assets are present."],
        commit=_git_commit(upstream),
    )


def _first_image(input_info: InputInfo) -> Path:
    if not input_info.files:
        raise ValueError("No input files available for photometric staging.")
    return input_info.files[0]


def _approximate_68_landmarks(width: int, height: int) -> np.ndarray:
    """Create a coarse 68-point FAN-style layout only for upstream smoke staging."""
    cx, cy = width / 2.0, height / 2.0
    face_w, face_h = width * 0.62, height * 0.78
    points: list[tuple[float, float]] = []

    for t in np.linspace(np.pi * 0.92, np.pi * 2.08, 17):
        points.append((cx + np.cos(t) * face_w * 0.50, cy + np.sin(t) * face_h * 0.53 + face_h * 0.06))

    for side in (-1, 1):
        brow_cx = cx + side * face_w * 0.20
        for dx in np.linspace(-0.14, 0.14, 5):
            points.append((brow_cx + side * dx * width, cy - face_h * 0.20 - abs(dx) * height * 0.20))

    for y in np.linspace(cy - face_h * 0.12, cy + face_h * 0.15, 4):
        points.append((cx, y))
    for dx in np.linspace(-0.14, 0.14, 5):
        points.append((cx + dx * width, cy + face_h * 0.18 + abs(dx) * height * 0.05))

    for side in (-1, 1):
        eye_cx = cx + side * face_w * 0.22
        for t in np.linspace(0, np.pi * 2, 6, endpoint=False):
            points.append((eye_cx + np.cos(t) * face_w * 0.10, cy - face_h * 0.05 + np.sin(t) * face_h * 0.045))

    mouth_cx, mouth_cy = cx, cy + face_h * 0.31
    for t in np.linspace(0, np.pi * 2, 12, endpoint=False):
        points.append((mouth_cx + np.cos(t) * face_w * 0.18, mouth_cy + np.sin(t) * face_h * 0.07))
    for t in np.linspace(0, np.pi * 2, 8, endpoint=False):
        points.append((mouth_cx + np.cos(t) * face_w * 0.10, mouth_cy + np.sin(t) * face_h * 0.035))

    arr = np.array(points[:68], dtype=np.float32)
    if arr.shape != (68, 2):
        raise RuntimeError(f"Expected 68 approximate landmarks, got {arr.shape}")
    return arr


def prepare_upstream_input(upstream_dir: Path, input_info: InputInfo, subject_id: str) -> PreparedInput:
    image_name = sanitize_image_name(subject_id)
    source = _first_image(input_info)
    image_dir = upstream_dir / "FFHQ"
    mask_dir = upstream_dir / "FFHQ_seg"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source).convert("RGB")
    image_path = image_dir / f"{image_name}.png"
    image.save(image_path)

    landmarks = _approximate_68_landmarks(*image.size)
    landmark_path = image_dir / f"{image_name}.npy"
    np.save(landmark_path, landmarks)

    mask_path = mask_dir / f"{image_name}.npy"
    np.save(mask_path, np.ones((256, 256), dtype=np.float32))

    return PreparedInput(
        image_name=image_name,
        image_path=image_path,
        landmark_path=landmark_path,
        mask_path=mask_path,
        notes=[
            "Input was staged into the upstream FFHQ-style folder layout.",
            "A coarse 68-point landmark proxy and full-face mask were generated because the upstream demo expects precomputed FAN landmarks and segmentation masks.",
        ],
    )


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_upstream_fitting(
    input_info: InputInfo,
    subject_id: str,
    device: str = "cpu",
    timeout_sec: int = 1800,
    upstream_dir: Path | None = None,
) -> UpstreamRunResult:
    check = inspect_upstream(upstream_dir)
    if not check.runnable:
        return UpstreamRunResult(
            success=False,
            status=check.status,
            notes=[*check.notes, *check.missing_files, *check.missing_assets],
        )

    prepared = prepare_upstream_input(check.upstream_dir, input_info, subject_id)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(check.upstream_dir), str(check.upstream_dir / "models"), os.environ.get("PYTHONPATH", "")]),
    }
    command = [sys.executable, "photometric_fitting.py", prepared.image_name, device]

    try:
        result = subprocess.run(
            command,
            cwd=check.upstream_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return UpstreamRunResult(
            success=False,
            status="upstream_timeout",
            notes=[*check.notes, *prepared.notes, f"Upstream fitting exceeded timeout_sec={timeout_sec}."],
            prepared_input=prepared,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
        )

    mesh_path = check.upstream_dir / "test_results" / f"{prepared.image_name}.obj"
    params_path = check.upstream_dir / "test_results" / f"{prepared.image_name}.npy"
    if result.returncode != 0:
        return UpstreamRunResult(
            success=False,
            status="upstream_failed",
            notes=[*check.notes, *prepared.notes, f"Upstream command failed with returncode={result.returncode}."],
            prepared_input=prepared,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )

    if not mesh_path.exists() or not params_path.exists():
        return UpstreamRunResult(
            success=False,
            status="upstream_missing_outputs",
            notes=[*check.notes, *prepared.notes, "Upstream finished but expected OBJ/NPY outputs were not found."],
            prepared_input=prepared,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )

    return UpstreamRunResult(
        success=True,
        status="upstream_success_scale_unverified",
        notes=[*check.notes, *prepared.notes, "Upstream fitting completed; dense mesh absolute metric scale is still unverified."],
        prepared_input=prepared,
        mesh_path=mesh_path,
        params_path=params_path,
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def copy_upstream_outputs(result: UpstreamRunResult, out_dir: Path) -> dict:
    if not result.success or result.mesh_path is None or result.params_path is None:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh = out_dir / "raw_mesh.obj"
    shutil.copyfile(result.mesh_path, raw_mesh)

    params = np.load(result.params_path, allow_pickle=True).item()
    flame_params = out_dir / "flame_params.npz"
    np.savez(
        flame_params,
        shape=params.get("shape", np.array([])),
        expression=params.get("exp", np.array([])),
        pose=params.get("pose", np.array([])),
        camera=params.get("cam", np.array([])),
        texture_or_albedo=params.get("tex", params.get("albedos", np.array([]))),
        lighting=params.get("lit", np.array([])),
        vertices_projected=params.get("verts", np.array([])),
    )

    diagnostics = {
        "raw_upstream_params": str(result.params_path),
        "raw_upstream_mesh": str(result.mesh_path),
        "imported_mesh": str(raw_mesh),
        "imported_flame_params": str(flame_params),
    }
    (out_dir / "photometric_backend_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return diagnostics

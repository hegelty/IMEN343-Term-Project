from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_INPUT_MODES = {"single_image", "photo_set", "video"}


@dataclass(frozen=True)
class InputInfo:
    path: Path
    input_mode: str
    files: tuple[Path, ...]
    image_count: int
    video_count: int
    user_input_burden: str
    notes: tuple[str, ...] = ()


def validate_input_path(input_path: str, input_mode: str) -> Path:
    return inspect_input_path(input_path, input_mode).path


def inspect_input_path(input_path: str, input_mode: str, allowed_modes: set[str] | None = None) -> InputInfo:
    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(f"Unsupported input_mode={input_mode}; expected one of {sorted(SUPPORTED_INPUT_MODES)}")
    if allowed_modes is not None and input_mode not in allowed_modes:
        raise ValueError(f"input_mode={input_mode} is not supported for this method; allowed: {sorted(allowed_modes)}")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_dir() and input_mode != "photo_set":
        raise ValueError("Directory input is only allowed with --input-mode photo_set")

    notes: list[str] = []
    files: tuple[Path, ...]
    image_count = 0
    video_count = 0

    if path.is_dir():
        images = tuple(sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXT))
        if not images:
            raise ValueError(f"Photo set directory contains no supported images: {path}")
        files = images
        image_count = len(images)
        if image_count < 3:
            notes.append("Photo set has fewer than the preferred front/left/right views.")
    elif path.is_file():
        ext = path.suffix.lower()
        files = (path,)
        if input_mode == "video":
            if ext not in SUPPORTED_VIDEO_EXT:
                raise ValueError(f"Expected a video file for input_mode=video, got: {ext}")
            video_count = 1
        else:
            if ext not in SUPPORTED_IMAGE_EXT:
                raise ValueError(f"Expected an image file for input_mode={input_mode}, got: {ext}")
            image_count = 1
            if input_mode == "photo_set":
                notes.append("Photo set mode received one image file; treating it as a degenerate one-view set.")
    else:
        raise ValueError(f"Input is neither a file nor a directory: {path}")

    if path.is_file():
        resolved_path = path.resolve()
    else:
        resolved_path = path.resolve()

    if input_mode == "single_image":
        burden = "single portrait image"
    elif input_mode == "photo_set":
        burden = f"multi-view photo set ({image_count} image{'s' if image_count != 1 else ''})"
    else:
        burden = "short face video"

    return InputInfo(
        path=resolved_path,
        input_mode=input_mode,
        files=tuple(p.resolve() for p in files),
        image_count=image_count,
        video_count=video_count,
        user_input_burden=burden,
        notes=tuple(notes),
    )

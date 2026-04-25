from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

from eyewear.common.schema.models import CanonicalFace


def _project(xyz: list[float], view: str) -> tuple[float, float]:
    x, y, z = xyz
    if view == "side":
        return z, y
    return x, y


def write_preview(path: Path, title: str, face: CanonicalFace | None = None, view: str = "front") -> None:
    img = Image.new("RGB", (800, 600), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((24, 24), title, fill=(20, 20, 20))
    draw.line((400, 80, 400, 540), fill=(210, 210, 210))
    draw.line((80, 300, 720, 300), fill=(210, 210, 210))

    if face is not None:
        scale = 2.6
        for name, point in face.landmarks.items():
            px, py = _project(point.xyz, view)
            sx = 400 + px * scale
            sy = 300 - py * scale
            color = (30, 95, 150) if point.source != "estimated" else (185, 105, 45)
            draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=color)
            if name in {"left_iris_center", "right_iris_center", "sellion_or_nasion", "pronasale"}:
                draw.text((sx + 5, sy - 5), name, fill=(30, 30, 30))

        draw.text((24, 558), "blue=observed/derived, orange=estimated proxy", fill=(80, 80, 80))

    img.save(path)

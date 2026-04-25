from __future__ import annotations

from eyewear.common.schema.models import REQUIRED_LANDMARKS, REQUIRED_MEASUREMENTS, CanonicalFace


def validate_required_fields(face: CanonicalFace) -> list[str]:
    missing = [k for k in REQUIRED_LANDMARKS if k not in face.landmarks]
    missing += [k for k in REQUIRED_MEASUREMENTS if k not in face.measurements and not k.startswith("face_")]
    return missing


def split_missing_fields(missing: list[str]) -> dict[str, list[str]]:
    return {
        "landmarks": [k for k in missing if k in REQUIRED_LANDMARKS],
        "measurements": [k for k in missing if k in REQUIRED_MEASUREMENTS],
    }

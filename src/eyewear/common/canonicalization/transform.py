from __future__ import annotations

from eyewear.common.geometry.math3d import midpoint
from eyewear.common.schema.models import CanonicalFace, LandmarkPoint


def _shift_xyz(xyz: list[float], origin: list[float]) -> list[float]:
    return [xyz[0] - origin[0], xyz[1] - origin[1], xyz[2] - origin[2]]


def _looks_like_xyz(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _canonicalize_curve_payload(value: object, origin: list[float]) -> object:
    if _looks_like_xyz(value):
        return _shift_xyz(value, origin)  # type: ignore[arg-type]
    if isinstance(value, list):
        return [_canonicalize_curve_payload(item, origin) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_curve_payload(item, origin) for key, item in value.items()}
    return value


def canonicalize(face: CanonicalFace) -> CanonicalFace:
    li = face.landmarks["left_iris_center"].xyz
    ri = face.landmarks["right_iris_center"].xyz
    origin = midpoint(li, ri)
    for key, point in list(face.landmarks.items()):
        face.landmarks[key] = LandmarkPoint(
            xyz=_shift_xyz(point.xyz, origin),
            source=point.source,
            method=point.method,
            confidence=point.confidence,
        )
    face.curves = {
        key: _canonicalize_curve_payload(payload, origin)
        for key, payload in face.curves.items()
    }
    return face

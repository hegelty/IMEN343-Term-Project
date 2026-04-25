from __future__ import annotations

import math
from typing import Iterable


def dist3(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def midpoint(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az = a
    bx, by, bz = b
    return [(ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0]


def polyline_length(points: Iterable[Iterable[float]]) -> float:
    seq = [list(p) for p in points]
    if len(seq) < 2:
        return 0.0
    return sum(dist3(a, b) for a, b in zip(seq, seq[1:]))

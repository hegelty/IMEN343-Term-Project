from __future__ import annotations

from eyewear.common.schema.models import LandmarkPoint


OBSERVED_LANDMARKS = {
    "left_iris_center", "right_iris_center",
    "left_inner_canthus", "right_inner_canthus",
    "left_outer_canthus", "right_outer_canthus",
    "pronasale", "left_alare", "right_alare",
}


ESTIMATED_LANDMARKS = {
    "left_ear_root_upper", "left_ear_root_center", "left_ear_root_lower",
    "right_ear_root_upper", "right_ear_root_center", "right_ear_root_lower",
}


def landmark_source(name: str) -> str:
    if name in OBSERVED_LANDMARKS:
        return "observed"
    if name in ESTIMATED_LANDMARKS:
        return "estimated"
    return "derived"


def template_landmarks(method: str) -> dict[str, LandmarkPoint]:
    d = {
        "left_iris_center": [31, 0, 4], "right_iris_center": [-31, 0, 4],
        "left_inner_canthus": [20, 1, 5], "right_inner_canthus": [-20, 1, 5],
        "left_outer_canthus": [38, 1, 4], "right_outer_canthus": [-38, 1, 4],
        "sellion_or_nasion": [0, 12, 7], "bridge_top": [0, 14, 6.5], "bridge_mid": [0, 10, 7], "bridge_low": [0, 7, 7.5],
        "bridge_left_contact": [7, 8, 8], "bridge_right_contact": [-7, 8, 8],
        "pronasale": [0, 5.5, 11], "left_alare": [11, 4, 8], "right_alare": [-11, 4, 8],
        "forehead_center": [0, 32, 3], "left_forehead_wrap": [42, 28, 1], "right_forehead_wrap": [-42, 28, 1],
        "left_front_width_point": [68, 5, -2], "right_front_width_point": [-68, 5, -2],
        "left_temple_start": [66, 7, -6], "right_temple_start": [-66, 7, -6],
        "left_ear_root_upper": [74, 18, -12], "left_ear_root_center": [74, 8, -12], "left_ear_root_lower": [74, -2, -11],
        "right_ear_root_upper": [-74, 18, -12], "right_ear_root_center": [-74, 8, -12], "right_ear_root_lower": [-74, -2, -11],
    }
    return {
        k: LandmarkPoint(
            xyz=v,
            source=landmark_source(k),
            method=method,
            confidence=0.9 if landmark_source(k) != "estimated" else 0.45,
        )
        for k, v in d.items()
    }

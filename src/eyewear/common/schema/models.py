from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

REQUIRED_LANDMARKS = [
    "left_iris_center", "right_iris_center", "left_inner_canthus", "right_inner_canthus",
    "left_outer_canthus", "right_outer_canthus", "sellion_or_nasion", "bridge_top",
    "bridge_mid", "bridge_low", "bridge_left_contact", "bridge_right_contact", "pronasale",
    "left_alare", "right_alare", "forehead_center", "left_forehead_wrap", "right_forehead_wrap",
    "left_front_width_point", "right_front_width_point", "left_temple_start", "right_temple_start",
    "left_ear_root_upper", "left_ear_root_center", "left_ear_root_lower", "right_ear_root_upper",
    "right_ear_root_center", "right_ear_root_lower",
]

REQUIRED_MEASUREMENTS = [
    "pd_mm", "bridge_width_mm_at_eye_line", "bridge_width_mm_at_pad_line", "nasion_tilt_deg",
    "front_width_mm", "head_breadth_proxy_mm", "temple_start_width_mm",
    "face_roll_deg", "face_pitch_deg", "face_yaw_deg",
]

RECOMMENDED_MEASUREMENTS = [
    "eye_orbit_width_mm_left", "eye_orbit_width_mm_right",
    "eye_orbit_height_mm_left", "eye_orbit_height_mm_right",
    "nose_protrusion_mm", "bridge_curve_length_mm",
    "temple_length_estimated_mm", "vertex_distance_proxy_mm",
]

@dataclass
class LandmarkPoint:
    xyz: List[float]
    source: str
    method: str
    confidence: float

@dataclass
class Measurement:
    value: float
    status: str
    confidence: float
    basis: str | None = None

@dataclass
class CanonicalFace:
    subject_id: str
    method_name: str
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    measurements: Dict[str, Measurement] = field(default_factory=dict)
    curves: Dict[str, Any] = field(default_factory=dict)
    scale_source: str = "unknown"
    metric_ready: bool = True
    pose_normalized: bool = False
    pose_deg: Dict[str, float] = field(default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
    estimated_fields: List[str] = field(default_factory=list)
    backend_name: str = "unknown"
    backend_status: str = "unknown"
    quality_notes: List[str] = field(default_factory=list)

    def to_landmarks_json(self) -> Dict[str, Any]:
        return {
            "unit": "mm",
            "coordinate_system": {
                "origin": "midpoint_between_iris_centers",
                "x": "subject_right",
                "y": "up",
                "z": "forward",
                "handedness": "right-handed",
            },
            "points": {k: asdict(v) for k, v in self.landmarks.items()},
        }

    def to_measurements_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "unit": "mm",
            **{k: asdict(v) for k, v in self.measurements.items()},
        }
        payload["face_roll_deg"] = self.pose_deg["roll"]
        payload["face_pitch_deg"] = self.pose_deg["pitch"]
        payload["face_yaw_deg"] = self.pose_deg["yaw"]
        return payload

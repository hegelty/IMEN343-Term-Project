from __future__ import annotations

from eyewear.common.geometry.math3d import dist3, polyline_length
from eyewear.common.schema.models import CanonicalFace, Measurement


def _m(face: CanonicalFace, name: str, value: float, status: str = "derived", confidence: float = 0.7, basis: str | None = None) -> None:
    face.measurements[name] = Measurement(value=round(float(value), 3), status=status, confidence=confidence, basis=basis)


def compute_measurements(face: CanonicalFace) -> CanonicalFace:
    p = {k: v.xyz for k, v in face.landmarks.items()}
    left_eye_width = dist3(p["left_inner_canthus"], p["left_outer_canthus"])
    right_eye_width = dist3(p["right_inner_canthus"], p["right_outer_canthus"])
    nose_protrusion = abs(p["pronasale"][2] - p["sellion_or_nasion"][2])

    _m(face, "pd_mm", dist3(p["left_iris_center"], p["right_iris_center"]), status="observed", confidence=0.95)
    _m(face, "bridge_width_mm_at_eye_line", dist3(p["bridge_left_contact"], p["bridge_right_contact"]))
    _m(face, "bridge_width_mm_at_pad_line", dist3(p["left_alare"], p["right_alare"]))
    _m(face, "nasion_tilt_deg", abs(p["bridge_top"][1] - p["bridge_low"][1]) * 0.8)
    _m(face, "front_width_mm", dist3(p["left_front_width_point"], p["right_front_width_point"]))
    _m(face, "head_breadth_proxy_mm", dist3(p["left_ear_root_upper"], p["right_ear_root_upper"]), status="estimated", confidence=0.5, basis="visible-ear-proxy")
    _m(face, "temple_start_width_mm", dist3(p["left_temple_start"], p["right_temple_start"]))
    _m(face, "eye_orbit_width_mm_left", left_eye_width)
    _m(face, "eye_orbit_width_mm_right", right_eye_width)
    _m(face, "eye_orbit_height_mm_left", left_eye_width * 0.42, status="estimated", confidence=0.45, basis="canthus-width anthropometric proxy")
    _m(face, "eye_orbit_height_mm_right", right_eye_width * 0.42, status="estimated", confidence=0.45, basis="canthus-width anthropometric proxy")
    _m(face, "nose_protrusion_mm", nose_protrusion)
    _m(face, "bridge_curve_length_mm", polyline_length([p["bridge_top"], p["bridge_mid"], p["bridge_low"]]))
    _m(face, "temple_length_estimated_mm", 140.0, status="estimated", confidence=0.4, basis="template-rule")
    _m(face, "vertex_distance_proxy_mm", 12.0 + min(nose_protrusion * 0.1, 4.0), status="estimated", confidence=0.35, basis="RGB-only proxy; verify with fitter or scanner before fabrication")
    return face

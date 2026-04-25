from eyewear.common.schema.models import CanonicalFace
from eyewear.methods.template import template_landmarks
from eyewear.common.canonicalization.transform import canonicalize
from eyewear.common.measurements.compute import compute_measurements
from eyewear.common.schema.validation import validate_required_fields


def test_required_fields_present():
    face = CanonicalFace(subject_id="t", method_name="mediapipe_iris", landmarks=template_landmarks("mediapipe_iris"))
    face = compute_measurements(canonicalize(face))
    missing = validate_required_fields(face)
    assert missing == []


def test_recommended_measurements_and_curves_are_metric_canonical():
    face = CanonicalFace(subject_id="t", method_name="mediapipe_iris", landmarks=template_landmarks("mediapipe_iris"))
    face.curves = {
        "nose_bridge_curve": {"type": "polyline", "unit": "mm", "points": [[0, 14, 6.5], [0, 10, 7], [0, 7, 7.5]]}
    }
    face = compute_measurements(canonicalize(face))
    for key in ["eye_orbit_height_mm_left", "bridge_curve_length_mm", "vertex_distance_proxy_mm"]:
        assert key in face.measurements

    first_curve_point = face.curves["nose_bridge_curve"]["points"][0]
    assert first_curve_point == [0.0, 14.0, 2.5]

from pathlib import Path

import numpy as np

from eyewear.methods.photometric.calibration import (
    apply_similarity,
    fit_similarity_transform,
    load_obj_vertices,
    mesh_bbox_proxy_landmarks,
)
from eyewear.methods.template import template_landmarks


def test_similarity_transform_recovers_scale_rotation_translation():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale_expected = 12.5
    translation_expected = np.array([3.0, -4.0, 7.0])
    target = np.asarray([scale_expected * (rotation @ point) + translation_expected for point in source])

    scale, recovered_rotation, translation = fit_similarity_transform(source, target)
    transformed = np.asarray([apply_similarity(point, scale, recovered_rotation, translation) for point in source])

    assert np.allclose(transformed, target)
    assert np.isclose(scale, scale_expected)


def test_mesh_bbox_proxy_landmarks_maps_template_extents(tmp_path: Path):
    obj = tmp_path / "mesh.obj"
    obj.write_text(
        "v 10 20 30\n"
        "v 20 20 30\n"
        "v 10 40 30\n"
        "v 10 20 60\n",
        encoding="utf-8",
    )
    vertices = load_obj_vertices(obj)
    proxies = mesh_bbox_proxy_landmarks(vertices, "test_proxy")
    template = template_landmarks("test_proxy")

    left = proxies["left_ear_root_upper"]
    right = proxies["right_ear_root_upper"]

    assert vertices.shape == (4, 3)
    assert left[0] > right[0]
    assert np.all(proxies["forehead_center"] >= vertices.min(axis=0))
    assert np.all(proxies["forehead_center"] <= vertices.max(axis=0))
    assert set(proxies) == set(template)

import dataclasses

import numpy as np
import pytest

from napari_ooctyle_analysis._regions import Sphere, sphere_from_line


class TestSphereDataclass:
    def test_construction(self):
        s = Sphere(
            center_px=np.array([1.0, 2.0, 3.0]),
            radius_physical=4.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        assert s.radius_physical == 4.0
        np.testing.assert_array_equal(s.center_px, [1.0, 2.0, 3.0])

    def test_is_frozen(self):
        s = Sphere(
            center_px=np.array([0.0, 0.0, 0.0]),
            radius_physical=1.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.radius_physical = 9.0  # type: ignore[misc]

    def test_arrays_are_immutable(self):
        s = Sphere(
            center_px=np.array([1.0, 2.0, 3.0]),
            radius_physical=4.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(ValueError):
            s.center_px[0] = 99.0
        with pytest.raises(ValueError):
            s.scale[0] = 99.0


class TestSphereFromLine:
    SCALE = np.array([1.0, 1.0, 1.0])

    def test_3d_line(self):
        line = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 20.0]])
        s = sphere_from_line(line, self.SCALE)
        assert s is not None
        np.testing.assert_allclose(s.center_px, [5.0, 0.0, 10.0])
        assert s.radius_physical == 10.0

    def test_2d_line_pads_with_viewer_point(self):
        line = np.array([[0.0, 0.0], [0.0, 20.0]])
        s = sphere_from_line(
            line, self.SCALE, viewer_point=np.array([7.0]), ndim=3
        )
        assert s is not None
        np.testing.assert_allclose(s.center_px, [7.0, 0.0, 10.0])
        assert s.radius_physical == 10.0

    def test_anisotropic_scale(self):
        line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
        scale = np.array([2.0, 1.0, 0.5])
        s = sphere_from_line(line, scale)
        assert s is not None
        # Physical length along X = 20 px * 0.5 um/px = 10 um, so radius = 5 um
        assert s.radius_physical == 5.0
        np.testing.assert_allclose(s.center_px, [0.0, 0.0, 10.0])

    def test_wrong_vertex_count_returns_none(self):
        line = np.array([[0.0, 0.0, 0.0]])  # only 1 vertex
        assert sphere_from_line(line, self.SCALE) is None

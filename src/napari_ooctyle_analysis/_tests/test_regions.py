import dataclasses

import numpy as np
import pytest

from napari_ooctyle_analysis._regions import (
    Sphere,
    apply_sphere_to_mask,
    build_sphere_mesh,
    contains_sphere,
    filter_spots,
    sphere_from_line,
    sphere_to_mask,
)


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


class TestContainsSphere:
    SCALE = np.array([1.0, 1.0, 1.0])

    def _sphere(self, center, radius):
        return Sphere(
            center_px=np.array(center, dtype=np.float64),
            radius_physical=float(radius),
            scale=self.SCALE,
        )

    def test_fully_inside(self):
        outer = self._sphere([0, 0, 0], 10.0)
        inner = self._sphere([0, 0, 0], 5.0)
        assert contains_sphere(outer, inner) is True

    def test_concentric_equal_radius(self):
        outer = self._sphere([0, 0, 0], 5.0)
        inner = self._sphere([0, 0, 0], 5.0)
        assert contains_sphere(outer, inner) is True

    def test_inner_pokes_out(self):
        outer = self._sphere([0, 0, 0], 5.0)
        inner = self._sphere([3, 0, 0], 3.0)
        assert contains_sphere(outer, inner) is False

    def test_disjoint(self):
        outer = self._sphere([0, 0, 0], 1.0)
        inner = self._sphere([100, 0, 0], 1.0)
        assert contains_sphere(outer, inner) is False

    def test_anisotropic_scale(self):
        outer = Sphere(
            center_px=np.array([0.0, 0.0, 0.0]),
            radius_physical=10.0,
            scale=np.array([2.0, 1.0, 1.0]),
        )
        inner = Sphere(
            center_px=np.array([4.0, 0.0, 0.0]),  # 4 px * 2 um/px = 8 um away
            radius_physical=1.0,
            scale=np.array([2.0, 1.0, 1.0]),
        )
        assert contains_sphere(outer, inner) is True


class TestApplySphereToMask:
    def test_zero_inside(self):
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        s = Sphere(
            center_px=np.array([5.0, 5.0, 5.0]),
            radius_physical=2.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        apply_sphere_to_mask(mask, s, mode="zero_inside")
        assert mask[5, 5, 5] == 0
        assert mask[0, 0, 0] == 1

    def test_zero_outside(self):
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        s = Sphere(
            center_px=np.array([5.0, 5.0, 5.0]),
            radius_physical=2.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        apply_sphere_to_mask(mask, s, mode="zero_outside")
        assert mask[5, 5, 5] == 1
        assert mask[0, 0, 0] == 0


class TestFilterSpots:
    SCALE = np.array([1.0, 1.0, 1.0])

    def test_keep_outside(self):
        spots = np.array([[5.0, 5.0, 5.0], [50.0, 50.0, 50.0]])
        s = Sphere(np.array([5.0, 5.0, 5.0]), 10.0, self.SCALE)
        kept = filter_spots(spots, s, keep="outside")
        assert len(kept) == 1
        np.testing.assert_array_equal(kept[0], [50.0, 50.0, 50.0])

    def test_keep_inside(self):
        spots = np.array([[5.0, 5.0, 5.0], [50.0, 50.0, 50.0]])
        s = Sphere(np.array([5.0, 5.0, 5.0]), 10.0, self.SCALE)
        kept = filter_spots(spots, s, keep="inside")
        assert len(kept) == 1
        np.testing.assert_array_equal(kept[0], [5.0, 5.0, 5.0])

    def test_empty_spots(self):
        spots = np.zeros((0, 3))
        s = Sphere(np.array([0.0, 0.0, 0.0]), 1.0, self.SCALE)
        assert filter_spots(spots, s, keep="outside").shape == (0, 3)


class TestSphereToMask:
    def test_boolean_mask_inside_sphere(self):
        s = Sphere(
            center_px=np.array([5.0, 5.0, 5.0]),
            radius_physical=2.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        m = sphere_to_mask(s, (10, 10, 10))
        assert m.dtype == bool
        assert m[5, 5, 5]
        assert not m[0, 0, 0]

    def test_anisotropic_scale(self):
        s = Sphere(
            center_px=np.array([5.0, 5.0, 5.0]),
            radius_physical=2.0,
            scale=np.array([3.0, 1.0, 1.0]),
        )
        m = sphere_to_mask(s, (10, 10, 10))
        assert m[5, 5, 5]
        assert not m[4, 5, 5]  # 1 px in Z = 3 um, > 2 um radius
        assert not m[6, 5, 5]


class TestBuildSphereMesh:
    def test_returns_vertices_and_faces(self):
        s = Sphere(
            center_px=np.array([10.0, 20.0, 20.0]),
            radius_physical=5.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        vertices, faces = build_sphere_mesh(s)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert vertices.shape[0] > 0 and faces.shape[0] > 0
        for axis in range(3):
            extent = vertices[:, axis].max() - vertices[:, axis].min()
            assert abs(extent - 2 * 5.0) < 0.5

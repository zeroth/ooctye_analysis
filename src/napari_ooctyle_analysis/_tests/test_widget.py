import napari
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import date

from napari_ooctyle_analysis._widget import OoctyleAnalysisWidget
from napari_ooctyle_analysis._regions import Sphere, compute_perinuclear
from napari_ooctyle_analysis._segmentation import (
    reorder_to_zyx,
    filter_spots_by_sphere,
    generate_save_path,
)
from napari_ooctyle_analysis._analysis import (
    compute_overlap,
    compute_zonal_voxels,
    create_zonal_figure,
)


def test_widget_creation(make_napari_viewer):
    """Widget instantiates and has two tabs."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)

    assert widget.viewer is viewer
    assert widget.tabs.count() == 2
    assert widget.tabs.tabText(0) == "Segmentation"
    assert widget.tabs.tabText(1) == "Analysis"


def test_image_layer_sync(make_napari_viewer):
    """Image combo updates when layers are added/removed."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)

    assert widget._image_combo.count() == 0

    viewer.add_image(np.random.random((10, 64, 64)), name="test_3d")
    assert widget._image_combo.count() == 1
    assert widget._image_combo.itemText(0) == "test_3d"

    viewer.layers.remove("test_3d")
    assert widget._image_combo.count() == 0


def test_model_combo_has_pretrained(make_napari_viewer):
    """Model dropdown contains expected pretrained models."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)

    model_names = [
        widget._model_combo.itemText(i)
        for i in range(widget._model_combo.count())
    ]
    assert "synth_3d" in model_names
    assert "smfish_3d" in model_names


def test_gpu_checkbox_reflects_cuda(make_napari_viewer):
    """GPU checkbox is disabled when CUDA is unavailable."""
    viewer = make_napari_viewer()

    with patch(
        "napari_ooctyle_analysis._widget.OoctyleAnalysisWidget._cuda_available",
        return_value=False,
    ):
        widget = OoctyleAnalysisWidget(viewer)
        assert not widget._use_gpu.isEnabled()
        assert not widget._use_gpu.isChecked()


def test_detection_rejects_2d(make_napari_viewer):
    """Detection refuses to run on a 2D image."""
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((64, 64)), name="flat")

    widget = OoctyleAnalysisWidget(viewer)
    widget._run_detection()

    assert "3D" in widget._status_label.text()


def test_detection_no_image(make_napari_viewer):
    """Detection shows error when no image is selected."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)
    widget._run_detection()

    assert "No image" in widget._status_label.text()


def test_generate_save_path(tmp_path):
    """Auto-versioning increments correctly."""
    today = date.today().isoformat()

    # First call → v1
    path1 = generate_save_path(str(tmp_path))
    assert path1.name == f"{today}_v1"

    # Create v1, next call → v2
    path1.mkdir()
    path2 = generate_save_path(str(tmp_path))
    assert path2.name == f"{today}_v2"

    # Create v2, next call → v3
    path2.mkdir()
    path3 = generate_save_path(str(tmp_path))
    assert path3.name == f"{today}_v3"


def test_finetune_validates_inputs(make_napari_viewer):
    """Fine-tuning requires all directories to be filled."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)
    widget._run_finetuning()

    assert "fill all" in widget._ft_status_label.text().lower()


# ------------------------------------------------------------------
# Channel splitter tests
# ------------------------------------------------------------------


class TestReorderToZyx:
    """Unit tests for the static _reorder_to_zyx method."""

    def test_czyx(self):
        """CZYX input splits into C separate ZYX arrays."""
        data = np.random.random((3, 10, 64, 64))  # C=3, Z=10
        channels, multi = reorder_to_zyx(data, "CZYX")
        assert multi is True
        assert len(channels) == 3
        for ch in channels:
            assert ch.shape == (10, 64, 64)
        # First channel should match original data[0]
        np.testing.assert_array_equal(channels[0], data[0])

    def test_zyxc(self):
        """ZYXC input correctly transposes to get ZYX per channel."""
        data = np.random.random((10, 64, 64, 3))  # Z=10, C=3 at end
        channels, multi = reorder_to_zyx(data, "ZYXC")
        assert multi is True
        assert len(channels) == 3
        for ch in channels:
            assert ch.shape == (10, 64, 64)
        # Channel 0 should be data[:, :, :, 0]
        np.testing.assert_array_equal(channels[0], data[:, :, :, 0])

    def test_zcyx(self):
        """ZCYX input (channel between Z and YX) is handled."""
        data = np.random.random((10, 2, 64, 64))  # Z=10, C=2
        channels, multi = reorder_to_zyx(data, "ZCYX")
        assert multi is True
        assert len(channels) == 2
        for ch in channels:
            assert ch.shape == (10, 64, 64)
        # Channel 1 should be data[:, 1, :, :]
        np.testing.assert_array_equal(channels[1], data[:, 1, :, :])

    def test_zyx_no_channel(self):
        """ZYX (no C) returns single array, was_multichannel=False."""
        data = np.random.random((10, 64, 64))
        channels, multi = reorder_to_zyx(data, "ZYX")
        assert multi is False
        assert len(channels) == 1
        np.testing.assert_array_equal(channels[0], data)

    def test_ndim_mismatch_raises(self):
        """Mismatched ndim raises ValueError."""
        data = np.random.random((10, 64, 64))  # 3D
        with pytest.raises(ValueError, match="3 dimensions.*CZYX.*4"):
            reorder_to_zyx(data, "CZYX")

    def test_invalid_axis_raises(self):
        """Unknown axis letter raises ValueError."""
        data = np.random.random((10, 64, 64, 3))
        with pytest.raises(ValueError, match="Unknown axis"):
            reorder_to_zyx(data, "ZTXC")


def test_split_channels_adds_layers(make_napari_viewer):
    """Splitting a CZYX image adds one layer per channel."""
    viewer = make_napari_viewer()
    data = np.random.random((3, 10, 64, 64))  # CZYX, 3 channels
    viewer.add_image(data, name="multi_ch")

    widget = OoctyleAnalysisWidget(viewer)
    widget._axis_order.setCurrentText("CZYX")
    widget._split_channels()

    # Original layer + 3 channel layers
    image_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Image)]
    assert len(image_layers) == 4
    assert "Channel 0" in [l.name for l in image_layers]
    assert "Channel 1" in [l.name for l in image_layers]
    assert "Channel 2" in [l.name for l in image_layers]
    assert "3 channels" in widget._split_status.text()


def test_split_no_channel_axis(make_napari_viewer):
    """Splitting with no C in axis order shows info message."""
    viewer = make_napari_viewer()
    data = np.random.random((10, 64, 64))
    viewer.add_image(data, name="single")

    widget = OoctyleAnalysisWidget(viewer)
    widget._axis_order.setCurrentText("ZYX")
    widget._split_channels()

    assert "nothing to split" in widget._split_status.text().lower()


# ------------------------------------------------------------------
# Exclude region tests
# ------------------------------------------------------------------


class TestFilterSpotsBySphere:
    """Unit tests for the static _filter_spots_by_sphere method."""

    ISOTROPIC = np.array([1.0, 1.0, 1.0])

    def test_removes_spots_inside_sphere(self):
        """Spots inside the sphere are removed."""
        spots = np.array([
            [5.0, 5.0, 5.0],   # center — inside
            [5.0, 5.0, 5.1],   # near center — inside
            [50.0, 50.0, 50.0], # far away — outside
        ])
        center = np.array([5.0, 5.0, 5.0])
        radius = 10.0
        result = filter_spots_by_sphere(
            spots, center, radius, self.ISOTROPIC
        )
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [50.0, 50.0, 50.0])

    def test_keeps_all_if_outside(self):
        """All spots outside the sphere are kept."""
        spots = np.array([
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
        ])
        center = np.array([0.0, 0.0, 0.0])
        radius = 5.0
        result = filter_spots_by_sphere(
            spots, center, radius, self.ISOTROPIC
        )
        assert len(result) == 2

    def test_empty_spots(self):
        """Empty spots array returns empty."""
        spots = np.zeros((0, 3))
        center = np.array([0.0, 0.0, 0.0])
        result = filter_spots_by_sphere(
            spots, center, 10.0, self.ISOTROPIC
        )
        assert len(result) == 0

    def test_boundary_spot(self):
        """A spot exactly on the radius boundary is excluded (> not >=)."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 10.0
        spots = np.array([[10.0, 0.0, 0.0]])  # exactly at radius
        result = filter_spots_by_sphere(
            spots, center, radius, self.ISOTROPIC
        )
        assert len(result) == 0

    def test_anisotropic_scale(self):
        """Scale stretches the exclusion: spot outside in px but inside in um."""
        # Spot is 5 pixels away in Z from center, but Z scale = 3.0 um/px
        # Physical distance = 5 * 3.0 = 15 um, which is inside radius=20 um
        center = np.array([10.0, 50.0, 50.0])
        spots = np.array([[15.0, 50.0, 50.0]])  # 5 px away in Z
        scale = np.array([3.0, 1.0, 1.0])  # anisotropic
        result = filter_spots_by_sphere(
            spots, center, 20.0, scale
        )
        assert len(result) == 0  # 15 um < 20 um → excluded

    def test_anisotropic_scale_keeps_far_spot(self):
        """With small Z-scale a large pixel offset stays outside the sphere."""
        center = np.array([10.0, 50.0, 50.0])
        spots = np.array([[15.0, 50.0, 50.0]])  # 5 px away in Z
        scale = np.array([0.5, 1.0, 1.0])  # small Z scale
        # Physical distance = 5 * 0.5 = 2.5 um, outside radius=2 um
        result = filter_spots_by_sphere(
            spots, center, 2.0, scale
        )
        assert len(result) == 1  # 2.5 um > 2.0 um → kept


# ------------------------------------------------------------------
# Fitting + mask tests (single-pass via _fitting module)
# ------------------------------------------------------------------


from napari_ooctyle_analysis._fitting import (
    fit_and_mask_3d,
    SpotFit3D,
)


class TestSpotFit3DPaintMask:
    """Unit tests for SpotFit3D.paint_mask."""

    @staticmethod
    def _make_fit(fwhm_z=4.0, fwhm_yx=6.0):
        return SpotFit3D(
            fwhm_z=fwhm_z, fwhm_yx=fwhm_yx,
            offset_z=0.0, offset_y=0.0, offset_x=0.0,
            intens_A=1.0, intens_B=0.0, r_squared=1.0,
        )

    def test_center_voxel_set(self):
        mask = np.zeros((20, 40, 40), dtype=np.uint8)
        fit = self._make_fit(fwhm_z=4.0, fwhm_yx=6.0)
        fit.paint_mask(mask, np.array([10.0, 20.0, 20.0]))
        assert mask[10, 20, 20] == 1
        assert mask.sum() > 0

    def test_larger_fwhm_more_voxels(self):
        mask_s = np.zeros((30, 60, 60), dtype=np.uint8)
        mask_l = np.zeros((30, 60, 60), dtype=np.uint8)
        spot = np.array([15.0, 30.0, 30.0])
        self._make_fit(fwhm_z=2.0, fwhm_yx=2.0).paint_mask(mask_s, spot)
        self._make_fit(fwhm_z=8.0, fwhm_yx=8.0).paint_mask(mask_l, spot)
        assert mask_l.sum() > mask_s.sum()

    def test_edge_clipped(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        fit = self._make_fit(fwhm_z=6.0, fwhm_yx=6.0)
        fit.paint_mask(mask, np.array([0.0, 0.0, 0.0]))
        assert mask[0, 0, 0] == 1
        assert mask.sum() > 0

    def test_nan_fwhm_skipped(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        fit = SpotFit3D(
            fwhm_z=np.nan, fwhm_yx=np.nan,
            offset_z=0.0, offset_y=0.0, offset_x=0.0,
            intens_A=1.0, intens_B=0.0, r_squared=0.0,
        )
        fit.paint_mask(mask, np.array([5.0, 5.0, 5.0]))
        assert mask.sum() == 0


class TestFitAndMask3D:
    """Unit tests for fit_and_mask_3d."""

    def _make_spot_image(self, shape, center, sigma_z=1.5, sigma_yx=2.0):
        """Create a synthetic 3D image with a single Gaussian spot."""
        z, y, x = np.indices(shape, dtype=np.float64)
        img = np.exp(-(
            (z - center[0]) ** 2 / (2 * sigma_z ** 2)
            + (y - center[1]) ** 2 / (2 * sigma_yx ** 2)
            + (x - center[2]) ** 2 / (2 * sigma_yx ** 2)
        )).astype(np.float32)
        return img

    def test_single_spot(self):
        """Single Gaussian spot produces mask and fit_params."""
        shape = (20, 40, 40)
        center = [10.0, 20.0, 20.0]
        img = self._make_spot_image(shape, center)
        spots = np.array([center])

        result = fit_and_mask_3d(img, spots, mask_shape=shape)
        assert result.mask.shape == shape
        assert result.mask[10, 20, 20] == 1
        assert result.mask.sum() > 0
        assert len(result.fit_params.fwhm_z) == 1
        assert np.isfinite(result.fit_params.fwhm_z[0])
        assert np.isfinite(result.fit_params.fwhm_yx[0])

    def test_empty_spots(self):
        """No spots → empty fit_params and zero mask."""
        shape = (10, 10, 10)
        img = np.zeros(shape, dtype=np.float32)
        spots = np.zeros((0, 3))

        result = fit_and_mask_3d(img, spots, mask_shape=shape)
        assert result.mask.sum() == 0
        assert len(result.fit_params.fwhm_z) == 0

    def test_multiple_spots(self):
        """Two spots both get mask regions."""
        shape = (30, 60, 60)
        c1 = [10.0, 20.0, 20.0]
        c2 = [20.0, 40.0, 40.0]
        img = self._make_spot_image(shape, c1) + self._make_spot_image(shape, c2)
        spots = np.array([c1, c2])

        result = fit_and_mask_3d(img, spots, mask_shape=shape)
        assert result.mask[10, 20, 20] == 1
        assert result.mask[20, 40, 40] == 1
        assert len(result.fit_params.fwhm_z) == 2

    def test_spot_at_edge(self):
        """Spot near the image boundary is handled without error."""
        shape = (10, 10, 10)
        img = self._make_spot_image(shape, [0.0, 0.0, 0.0])
        spots = np.array([[0.0, 0.0, 0.0]])

        result = fit_and_mask_3d(img, spots, mask_shape=shape)
        assert result.mask.sum() > 0
        assert result.mask[0, 0, 0] == 1


# ------------------------------------------------------------------
# Overlap analysis tests
# ------------------------------------------------------------------


class TestComputeOverlap:
    """Unit tests for _compute_overlap."""

    def test_full_overlap(self):
        """Identical masks produce 100% overlap."""
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        result = compute_overlap(mask, mask)
        assert result["n_overlap"] == 1000
        assert result["pct_a"] == 100.0
        assert result["pct_b"] == 100.0

    def test_no_overlap(self):
        """Non-overlapping masks produce 0% overlap."""
        mask_a = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_b = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_a[:5] = 1
        mask_b[5:] = 1
        result = compute_overlap(mask_a, mask_b)
        assert result["n_overlap"] == 0
        assert result["pct_a"] == 0.0
        assert result["pct_b"] == 0.0

    def test_partial_overlap(self):
        """Partial overlap computes correct percentages."""
        mask_a = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_b = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_a[:6] = 1  # 600 voxels
        mask_b[4:] = 1  # 600 voxels
        # Overlap is slices 4-5 → 200 voxels
        result = compute_overlap(mask_a, mask_b)
        assert result["n_overlap"] == 200
        assert abs(result["pct_a"] - 200 / 600 * 100) < 0.1
        assert abs(result["pct_b"] - 200 / 600 * 100) < 0.1


def test_analysis_tab_has_overlap_controls(make_napari_viewer):
    """Analysis tab has the expected overlap UI widgets."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)

    assert widget.tabs.tabText(1) == "Analysis"
    assert widget._mask_a_combo is not None
    assert widget._mask_b_combo is not None
    assert widget._overlap_btn is not None
    assert widget._clear_charts_btn is not None


def test_overlap_analysis_validates_inputs(make_napari_viewer):
    """Overlap analysis shows error when layers not selected."""
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)
    widget._run_overlap_analysis()
    assert "select" in widget._overlap_status.text().lower()


def test_overlap_analysis_rejects_same_layer(make_napari_viewer):
    """Overlap analysis rejects same layer selected for both channels."""
    viewer = make_napari_viewer()
    mask = np.ones((10, 10, 10), dtype=np.uint8)
    viewer.add_labels(mask, name="mask1")

    widget = OoctyleAnalysisWidget(viewer)
    widget._mask_a_combo.setCurrentText("mask1")
    widget._mask_b_combo.setCurrentText("mask1")
    widget._run_overlap_analysis()
    assert "different" in widget._overlap_status.text().lower()


def test_overlap_end_to_end(make_napari_viewer):
    """Full overlap analysis produces chart and overlap layer."""
    viewer = make_napari_viewer()

    mask_a = np.zeros((10, 20, 20), dtype=np.uint8)
    mask_b = np.zeros((10, 20, 20), dtype=np.uint8)
    mask_a[:6] = 1
    mask_b[4:] = 1

    viewer.add_labels(mask_a, name="Mask A")
    viewer.add_labels(mask_b, name="Mask B")

    widget = OoctyleAnalysisWidget(viewer)
    widget._mask_a_combo.setCurrentText("Mask A")
    widget._mask_b_combo.setCurrentText("Mask B")
    widget._run_overlap_analysis()

    assert "800" in widget._overlap_status.text()
    # A chart was added (layout has stretch + 1 chart = 2 items)
    assert widget._charts_layout.count() == 2


def test_overlap_charts_accumulate(make_napari_viewer):
    """Multiple overlap runs add multiple charts."""
    viewer = make_napari_viewer()

    mask_a = np.ones((5, 5, 5), dtype=np.uint8)
    mask_b = np.ones((5, 5, 5), dtype=np.uint8)
    viewer.add_labels(mask_a, name="A")
    viewer.add_labels(mask_b, name="B")

    widget = OoctyleAnalysisWidget(viewer)
    widget._mask_a_combo.setCurrentText("A")
    widget._mask_b_combo.setCurrentText("B")

    widget._run_overlap_analysis()
    widget._run_overlap_analysis()
    widget._run_overlap_analysis()

    # 3 charts + 1 stretch = 4 items
    assert widget._charts_layout.count() == 4


def test_clear_charts(make_napari_viewer):
    """Clear button removes all charts."""
    viewer = make_napari_viewer()

    mask = np.ones((5, 5, 5), dtype=np.uint8)
    viewer.add_labels(mask, name="A")
    viewer.add_labels(mask.copy(), name="B")

    widget = OoctyleAnalysisWidget(viewer)
    widget._mask_a_combo.setCurrentText("A")
    widget._mask_b_combo.setCurrentText("B")

    widget._run_overlap_analysis()
    widget._run_overlap_analysis()
    assert widget._charts_layout.count() == 3  # 2 charts + stretch

    widget._clear_overlap_charts()
    assert widget._charts_layout.count() == 1  # only stretch remains


# ------------------------------------------------------------------
# Three-region tests
# ------------------------------------------------------------------


class TestRegionWidgets:
    REGION_KEYS = ("oocyte", "nucleus")
    LAYER_NAMES = {
        "oocyte": "Oocyte line",
        "nucleus": "Nucleus line",
    }

    def test_region_combos_exist(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        for key in self.REGION_KEYS:
            assert key in widget._region_combos
            assert key in widget._region_show
            assert key in widget._add_region_buttons

    def test_new_buttons_create_named_layers(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        for key in self.REGION_KEYS:
            widget._add_region_shapes_layer(key)
        names = {l.name for l in viewer.layers if isinstance(l, napari.layers.Shapes)}
        assert names == set(self.LAYER_NAMES.values())
        for key in self.REGION_KEYS:
            assert widget._region_combos[key].currentText() == self.LAYER_NAMES[key]

    def test_get_region_sphere_returns_none_when_no_layer(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        for key in self.REGION_KEYS:
            assert widget._get_region_sphere(key, ndim=3) is None

    def test_get_region_sphere_from_line(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        viewer.add_shapes(
            [np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 20.0]])],
            shape_type="line",
            name="Nucleus line",
        )
        widget._refresh_image_layers()
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        s = widget._get_region_sphere("nucleus", ndim=3)
        assert s is not None
        np.testing.assert_allclose(s.center_px, [5.0, 0.0, 10.0])
        assert abs(s.radius_physical - 10.0) < 1e-6

    def test_region_sphere_with_anisotropic_scale(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        viewer.add_shapes(
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]])],
            shape_type="line",
            name="Nucleus line",
        )
        widget._refresh_image_layers()
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._scale_x.setValue(0.5)
        widget._scale_y.setValue(1.0)
        widget._scale_z.setValue(2.0)
        s = widget._get_region_sphere("nucleus", ndim=3)
        assert s is not None
        assert abs(s.radius_physical - 5.0) < 1e-6
        np.testing.assert_allclose(s.scale, [2.0, 1.0, 0.5])

    def test_region_sphere_2d_line_on_3d(self, make_napari_viewer):
        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((20, 64, 64)), name="vol")
        widget = OoctyleAnalysisWidget(viewer)
        viewer.dims.set_point(0, 10.0)
        viewer.add_shapes(
            [np.array([[0.0, 0.0], [0.0, 20.0]])],
            shape_type="line",
            name="Nucleus line",
        )
        widget._refresh_image_layers()
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        s = widget._get_region_sphere("nucleus", ndim=3)
        assert s is not None
        np.testing.assert_allclose(s.center_px, [10.0, 0.0, 10.0])
        assert abs(s.radius_physical - 10.0) < 1e-6

    def test_fraction_spinbox_drives_perinuclear_radius(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        # Oocyte radius 20, nucleus radius 10, shared center [5,5,20], unit scale.
        viewer.add_shapes(
            [np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 40.0]])],
            shape_type="line", name="Oocyte line",
        )
        viewer.add_shapes(
            [np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 30.0]])],
            shape_type="line", name="Nucleus line",
        )
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._peri_fraction.setValue(60.0)
        spheres = widget._get_all_region_spheres(ndim=3)
        # R_p = 10 + 0.6*(20-10) = 16
        assert spheres["perinuclear"] is not None
        assert abs(spheres["perinuclear"].radius_physical - 16.0) < 1e-6


# ------------------------------------------------------------------
# compute_perinuclear tests
# ------------------------------------------------------------------


class TestComputePerinuclear:
    def _sphere(self, center, radius, scale=(1.0, 1.0, 1.0)):
        return Sphere(
            center_px=np.array(center, dtype=np.float64),
            radius_physical=float(radius),
            scale=np.array(scale, dtype=np.float64),
        )

    def test_forty_percent_midpoint(self):
        nucleus = self._sphere([50, 50, 50], 10.0)
        oocyte = self._sphere([50, 50, 50], 20.0)
        peri = compute_perinuclear(nucleus, oocyte, 0.40)
        # R_p = 10 + 0.4*(20-10) = 14
        assert abs(peri.radius_physical - 14.0) < 1e-9

    def test_frac_zero_equals_nucleus_radius(self):
        nucleus = self._sphere([0, 0, 0], 7.0)
        oocyte = self._sphere([0, 0, 0], 30.0)
        peri = compute_perinuclear(nucleus, oocyte, 0.0)
        assert abs(peri.radius_physical - 7.0) < 1e-9

    def test_frac_one_equals_oocyte_radius(self):
        nucleus = self._sphere([0, 0, 0], 7.0)
        oocyte = self._sphere([0, 0, 0], 30.0)
        peri = compute_perinuclear(nucleus, oocyte, 1.0)
        assert abs(peri.radius_physical - 30.0) < 1e-9

    def test_centered_on_nucleus_even_when_offset(self):
        nucleus = self._sphere([10, 20, 30], 5.0)
        oocyte = self._sphere([0, 0, 0], 25.0)
        peri = compute_perinuclear(nucleus, oocyte, 0.5)
        np.testing.assert_allclose(peri.center_px, [10, 20, 30])
        # R_p = 5 + 0.5*(25-5) = 15
        assert abs(peri.radius_physical - 15.0) < 1e-9

    def test_inherits_nucleus_scale(self):
        nucleus = self._sphere([0, 0, 0], 10.0, scale=(2.0, 1.0, 0.5))
        oocyte = self._sphere([0, 0, 0], 20.0)
        peri = compute_perinuclear(nucleus, oocyte, 0.4)
        np.testing.assert_allclose(peri.scale, [2.0, 1.0, 0.5])


# ------------------------------------------------------------------
# Containment validation tests
# ------------------------------------------------------------------


class TestContainmentValidation:
    def _add_line(self, viewer, name, p1, p2):
        viewer.add_shapes(
            [np.array([p1, p2])], shape_type="line", name=name,
        )

    def test_no_warning_when_nucleus_inside_oocyte(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        # Oocyte radius=50, Nucleus radius=5, shared center → nucleus well inside.
        self._add_line(viewer, "Oocyte line",  [50, 50, 0],  [50, 50, 100])
        self._add_line(viewer, "Nucleus line", [50, 50, 45], [50, 50, 55])
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._update_containment_status()
        assert widget._region_status.text() == ""

    def test_warning_when_nucleus_outside_oocyte(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        # Nucleus centered far from oocyte → not contained.
        self._add_line(viewer, "Oocyte line",  [50, 50, 0],   [50, 50, 20])
        self._add_line(viewer, "Nucleus line", [50, 50, 100], [50, 50, 110])
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._update_containment_status()
        text = widget._region_status.text().lower()
        assert "nucleus" in text and "oocyte" in text

    def test_warning_when_oocyte_smaller_than_nucleus(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        # Shared center, oocyte radius 5 < nucleus radius 25.
        self._add_line(viewer, "Oocyte line",  [50, 50, 45], [50, 50, 55])
        self._add_line(viewer, "Nucleus line", [50, 50, 25], [50, 50, 75])
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._update_containment_status()
        assert "larger" in widget._region_status.text().lower()

    def test_missing_region_skips_check(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        self._add_line(viewer, "Nucleus line", [0, 0, 0], [0, 0, 10])
        widget._refresh_image_layers()
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        # No oocyte layer selected -> containment check is skipped.
        widget._region_combos["oocyte"].setCurrentIndex(-1)
        widget._update_containment_status()
        assert widget._region_status.text() == ""


class TestOocyteClipping:
    def test_clip_spots_and_mask(self, monkeypatch, make_napari_viewer):
        viewer = make_napari_viewer()
        img = np.zeros((20, 40, 40), dtype=np.float32)
        viewer.add_image(img, name="vol")
        widget = OoctyleAnalysisWidget(viewer)
        widget._image_combo.setCurrentText("vol")

        viewer.add_shapes(
            [np.array([[10.0, 20.0, 10.0], [10.0, 20.0, 30.0]])],
            shape_type="line", name="Oocyte line",
        )
        viewer.add_shapes(
            [np.array([[10.0, 20.0, 19.0], [10.0, 20.0, 21.0]])],
            shape_type="line", name="Nucleus line",
        )
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")

        fake_spots = np.array([
            [10.0, 20.0, 25.0],  # inside oocyte, outside nucleus
            [10.0, 20.0, 20.0],  # inside nucleus
            [0.0, 0.0, 0.0],     # outside oocyte
        ], dtype=np.float64)
        fake_mask = np.ones_like(img, dtype=np.uint8)

        from napari_ooctyle_analysis._workers import PredictWorker
        captured = {}

        def fake_run(self):
            from napari_ooctyle_analysis._regions import (
                apply_sphere_to_mask, filter_spots,
            )
            spots = fake_spots.copy()
            mask = fake_mask.copy()
            oocyte = self.regions.get("oocyte")
            nucleus = self.regions.get("nucleus")
            if oocyte is not None:
                spots = filter_spots(spots, oocyte, keep="inside")
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if nucleus is not None:
                spots = filter_spots(spots, nucleus, keep="outside")
                apply_sphere_to_mask(mask, nucleus, mode="zero_inside")
            captured["spots"] = spots
            captured["mask"] = mask

        monkeypatch.setattr(PredictWorker, "run", fake_run)

        region_spheres = widget._get_all_region_spheres(ndim=3)
        worker = PredictWorker(
            model=None, image=img,
            prob_thresh=0.5, min_distance=2, exclude_border=False,
            device="cpu", regions=region_spheres,
        )
        worker.run()

        assert len(captured["spots"]) == 1
        np.testing.assert_allclose(captured["spots"][0], [10.0, 20.0, 25.0])
        assert captured["mask"][0, 0, 0] == 0
        assert captured["mask"][10, 20, 20] == 0
        assert captured["mask"][10, 20, 25] == 1


class TestComputeZonalVoxels:
    def test_partitions_oocyte_into_peri_and_rest(self):
        shape = (10, 10, 10)
        oocyte = np.ones(shape, dtype=bool)
        peri = np.zeros(shape, dtype=bool); peri[3:7, 3:7, 3:7] = True
        excl = np.zeros(shape, dtype=bool); excl[4:6, 4:6, 4:6] = True
        channel = np.ones(shape, dtype=np.uint8)

        result = compute_zonal_voxels(channel, oocyte, peri, excl)
        # Perinuclear minus nucleus: 4^3 - 2^3 = 56
        # Rest of oocyte (oocyte minus peri): 10^3 - 4^3 = 936
        assert result["n_perinuclear"] == 56
        assert result["n_rest_oocyte"] == 936
        assert result["n_total"] == result["n_perinuclear"] + result["n_rest_oocyte"]

    def test_channel_outside_oocyte_ignored(self):
        shape = (10, 10, 10)
        oocyte = np.zeros(shape, dtype=bool); oocyte[:5] = True
        peri = np.zeros(shape, dtype=bool); peri[:2] = True
        excl = np.zeros(shape, dtype=bool)
        channel = np.ones(shape, dtype=np.uint8)

        result = compute_zonal_voxels(channel, oocyte, peri, excl)
        assert result["n_perinuclear"] == 200
        assert result["n_rest_oocyte"] == 300
        assert result["n_total"] == 500


class TestZonalFigure:
    def test_returns_figure_with_two_axes(self):
        results = [
            {"n_perinuclear": 100, "n_rest_oocyte": 400, "n_total": 500,
             "pct_perinuclear": 20.0, "pct_rest_oocyte": 80.0},
            {"n_perinuclear": 250, "n_rest_oocyte": 250, "n_total": 500,
             "pct_perinuclear": 50.0, "pct_rest_oocyte": 50.0},
        ]
        fig = create_zonal_figure(["Mask A", "Mask B"], results)
        assert fig is not None
        assert len(fig.axes) == 2


class TestZonalChartWiring:
    def test_no_zonal_chart_when_regions_missing(self, make_napari_viewer):
        viewer = make_napari_viewer()
        mask_a = np.ones((10, 10, 10), dtype=np.uint8)
        mask_b = np.ones((10, 10, 10), dtype=np.uint8)
        viewer.add_labels(mask_a, name="A")
        viewer.add_labels(mask_b, name="B")
        widget = OoctyleAnalysisWidget(viewer)
        widget._mask_a_combo.setCurrentText("A")
        widget._mask_b_combo.setCurrentText("B")
        widget._run_overlap_analysis()
        assert widget._charts_layout.count() == 2

    def test_zonal_chart_added_when_oocyte_and_nucleus_drawn(self, make_napari_viewer):
        viewer = make_napari_viewer()
        mask_a = np.ones((10, 10, 10), dtype=np.uint8)
        mask_b = np.ones((10, 10, 10), dtype=np.uint8)
        viewer.add_labels(mask_a, name="A")
        viewer.add_labels(mask_b, name="B")
        viewer.add_shapes(
            [np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 10.0]])],
            shape_type="line", name="Oocyte line",
        )
        viewer.add_shapes(
            [np.array([[5.0, 5.0, 4.0], [5.0, 5.0, 6.0]])],
            shape_type="line", name="Nucleus line",
        )
        widget = OoctyleAnalysisWidget(viewer)
        widget._mask_a_combo.setCurrentText("A")
        widget._mask_b_combo.setCurrentText("B")
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["nucleus"].setCurrentText("Nucleus line")
        widget._run_overlap_analysis()
        assert widget._charts_layout.count() == 3


class TestComputeSpotRegionprops:
    def test_intensity_mean_per_label(self):
        from napari_ooctyle_analysis._analysis import compute_spot_regionprops
        label_img = np.zeros((4, 4, 4), dtype=np.int32)
        label_img[0:2, 0:2, 0:2] = 1
        label_img[2:4, 2:4, 2:4] = 2
        intensity = np.zeros((4, 4, 4), dtype=np.float32)
        intensity[0:2, 0:2, 0:2] = 10.0
        intensity[2:4, 2:4, 2:4] = 20.0
        table = compute_spot_regionprops(label_img, intensity)
        assert list(table["label"]) == [1, 2]
        np.testing.assert_allclose(table["intensity_mean"], [10.0, 20.0])
        assert list(table["area"]) == [8, 8]
        # 3D centroids present
        assert "centroid-0" in table and "centroid-1" in table and "centroid-2" in table

    def test_empty_label_image_yields_empty_table(self):
        from napari_ooctyle_analysis._analysis import compute_spot_regionprops
        label_img = np.zeros((3, 3, 3), dtype=np.int32)
        intensity = np.zeros((3, 3, 3), dtype=np.float32)
        table = compute_spot_regionprops(label_img, intensity)
        assert len(table["label"]) == 0
        assert len(table["intensity_mean"]) == 0


class TestSplitSpotIntensities:
    def test_overlap_vs_non_overlap(self):
        from napari_ooctyle_analysis._analysis import split_spot_intensities
        # B has two labels: label 1 (top-left) and label 2 (bottom-right).
        label_b = np.zeros((4, 4), dtype=np.int32)
        label_b[0:2, 0:2] = 1
        label_b[2:4, 2:4] = 2
        # A covers only the top-left quadrant -> label 1 overlaps, label 2 does not.
        mask_a = np.zeros((4, 4), dtype=np.int32)
        mask_a[0:2, 0:2] = 1
        table = {"label": np.array([1, 2]), "intensity_mean": np.array([10.0, 20.0])}
        split = split_spot_intensities(label_b, mask_a, table)
        np.testing.assert_allclose(split["overlap"], [10.0])
        np.testing.assert_allclose(split["non_overlap"], [20.0])
        assert split["n_overlap"] == 1
        assert split["n_non_overlap"] == 1

    def test_no_overlap_when_a_empty(self):
        from napari_ooctyle_analysis._analysis import split_spot_intensities
        label_b = np.zeros((4, 4), dtype=np.int32)
        label_b[0:2, 0:2] = 1
        mask_a = np.zeros((4, 4), dtype=np.int32)
        table = {"label": np.array([1]), "intensity_mean": np.array([7.0])}
        split = split_spot_intensities(label_b, mask_a, table)
        assert split["n_overlap"] == 0
        np.testing.assert_allclose(split["non_overlap"], [7.0])


class TestIntensityHistogramFigure:
    def test_two_axes_with_data(self):
        from napari_ooctyle_analysis._analysis import create_intensity_histogram_figure
        split = {
            "overlap": np.array([1.0, 2.0, 2.0, 3.0]),
            "non_overlap": np.array([4.0, 5.0]),
            "n_overlap": 4, "n_non_overlap": 2,
        }
        fig = create_intensity_histogram_figure("Channel 1 mask", split)
        assert fig is not None
        assert len(fig.axes) == 2

    def test_empty_arrays_do_not_crash(self):
        from napari_ooctyle_analysis._analysis import create_intensity_histogram_figure
        split = {"overlap": np.array([]), "non_overlap": np.array([]),
                 "n_overlap": 0, "n_non_overlap": 0}
        fig = create_intensity_histogram_figure("B", split)
        assert len(fig.axes) == 2

    def test_all_equal_values_do_not_crash(self):
        from napari_ooctyle_analysis._analysis import create_intensity_histogram_figure
        split = {
            "overlap": np.array([7.0, 7.0, 7.0]),
            "non_overlap": np.array([7.0]),
            "n_overlap": 3, "n_non_overlap": 1,
        }
        fig = create_intensity_histogram_figure("B", split)
        assert len(fig.axes) == 2


class TestSpotTableToRows:
    def test_header_order_and_rows(self):
        from napari_ooctyle_analysis._analysis import spot_table_to_rows
        table = {
            "label": np.array([1, 2]),
            "centroid-0": np.array([0.5, 2.5]),
            "centroid-1": np.array([0.5, 2.5]),
            "centroid-2": np.array([0.5, 2.5]),
            "area": np.array([8, 8]),
            "intensity_mean": np.array([10.0, 20.0]),
        }
        header, rows = spot_table_to_rows(table)
        assert header == ["label", "centroid-0", "centroid-1", "centroid-2", "area", "intensity_mean"]
        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[1][5] == 20.0

    def test_empty_table_has_header_no_rows(self):
        from napari_ooctyle_analysis._analysis import spot_table_to_rows
        table = {"label": np.array([]), "centroid-0": np.array([]),
                 "centroid-1": np.array([]), "centroid-2": np.array([]),
                 "area": np.array([]), "intensity_mean": np.array([])}
        header, rows = spot_table_to_rows(table)
        assert header[0] == "label"
        assert rows == []


class TestWorkerRegionpropsPipeline:
    def test_nucleus_spot_excluded_from_table(self):
        from napari_ooctyle_analysis._analysis import compute_spot_regionprops
        from napari_ooctyle_analysis._regions import Sphere, apply_sphere_to_mask
        from scipy.ndimage import label as ndlabel

        shape = (1, 20, 20)
        intensity = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        # Spot A: outside nucleus (far corner). Spot B: at the nucleus center.
        mask[0, 2:4, 2:4] = 1;  intensity[0, 2:4, 2:4] = 50.0
        mask[0, 9:11, 9:11] = 1; intensity[0, 9:11, 9:11] = 99.0
        nucleus = Sphere(
            center_px=np.array([0.0, 10.0, 10.0]),
            radius_physical=3.0,
            scale=np.array([1.0, 1.0, 1.0]),
        )
        # Worker pipeline: zero inside nucleus, then label, then regionprops.
        apply_sphere_to_mask(mask, nucleus, mode="zero_inside")
        labeled, _ = ndlabel(mask)
        table = compute_spot_regionprops(labeled, intensity)
        # Only the far-corner spot survives; the 99.0 (nucleus) spot is gone.
        assert table["intensity_mean"].size == 1
        np.testing.assert_allclose(table["intensity_mean"], [50.0])


class TestDetectionMetadata:
    def _widget(self, make_napari_viewer):
        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="vol")
        widget = OoctyleAnalysisWidget(viewer)
        widget._image_combo.setCurrentText("vol")
        return viewer, widget

    def test_metadata_attached_when_present(self, make_napari_viewer):
        viewer, widget = self._widget(make_napari_viewer)
        labeled = np.zeros((2, 8, 8), dtype=np.int32)
        labeled[0, 1:3, 1:3] = 1
        table = {"label": np.array([1]), "intensity_mean": np.array([5.0]),
                 "centroid-0": np.array([0.0]), "centroid-1": np.array([2.0]),
                 "centroid-2": np.array([2.0]), "area": np.array([4])}
        meta = {"image_shape": (2, 8, 8), "n_excluded": 0, "mask": labeled,
                "labeled_mask": labeled, "n_labels": 1, "spot_intensity": table}
        widget._on_detection_finished(np.zeros((0, 3)), None, meta)
        layer = viewer.layers["vol mask"]
        assert "spot_intensity" in layer.metadata
        assert list(layer.metadata["spot_intensity"]["label"]) == [1]

    def test_fallback_without_metadata(self, make_napari_viewer):
        viewer, widget = self._widget(make_napari_viewer)
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[0, 1:3, 1:3] = 1
        meta = {"image_shape": (2, 8, 8), "n_excluded": 0, "mask": mask}
        widget._on_detection_finished(np.zeros((0, 3)), None, meta)
        layer = viewer.layers["vol mask"]
        assert "spot_intensity" not in layer.metadata


class TestIntensityHistogramWiring:
    def _setup(self, make_napari_viewer, with_meta):
        viewer = make_napari_viewer()
        # A mask: label 1 in top-left. B mask: label 1 top-left (overlaps A),
        # label 2 bottom-right (no overlap).
        a = np.zeros((1, 8, 8), dtype=np.int32); a[0, 0:2, 0:2] = 1
        b = np.zeros((1, 8, 8), dtype=np.int32); b[0, 0:2, 0:2] = 1; b[0, 6:8, 6:8] = 2
        viewer.add_labels(a, name="A")
        b_layer = viewer.add_labels(b, name="B")
        if with_meta:
            b_layer.metadata["spot_intensity"] = {
                "label": np.array([1, 2]), "intensity_mean": np.array([10.0, 20.0]),
            }
        widget = OoctyleAnalysisWidget(viewer)
        widget._mask_a_combo.setCurrentText("A")
        widget._mask_b_combo.setCurrentText("B")
        return widget

    def test_histogram_added_when_b_has_metadata(self, make_napari_viewer):
        widget = self._setup(make_napari_viewer, with_meta=True)
        widget._run_overlap_analysis()
        # stretch (1) + overlap chart (1) + intensity histogram (1) = 3
        assert widget._charts_layout.count() == 3

    def test_no_histogram_without_metadata(self, make_napari_viewer):
        widget = self._setup(make_napari_viewer, with_meta=False)
        widget._run_overlap_analysis()
        # stretch (1) + overlap chart (1) only
        assert widget._charts_layout.count() == 2
        assert "intensity" in widget._overlap_status.text().lower()

    def test_round_trip_real_regionprops_table(self, make_napari_viewer):
        from napari_ooctyle_analysis._analysis import (
            compute_spot_regionprops, split_spot_intensities,
        )
        viewer = make_napari_viewer()
        # Build a real labeled B volume + intensity image, then derive the table
        # exactly as the worker does (compute_spot_regionprops on the labeled mask).
        label_b = np.zeros((1, 8, 8), dtype=np.int32)
        label_b[0, 0:2, 0:2] = 1        # overlaps A
        label_b[0, 6:8, 6:8] = 2        # does not overlap A
        intensity = np.zeros((1, 8, 8), dtype=np.float32)
        intensity[0, 0:2, 0:2] = 30.0
        intensity[0, 6:8, 6:8] = 70.0
        table = compute_spot_regionprops(label_b, intensity)

        a = np.zeros((1, 8, 8), dtype=np.int32)
        a[0, 0:2, 0:2] = 1              # A covers only B's label 1
        viewer.add_labels(a, name="A")
        b_layer = viewer.add_labels(label_b, name="B")
        b_layer.metadata["spot_intensity"] = table

        widget = OoctyleAnalysisWidget(viewer)
        widget._mask_a_combo.setCurrentText("A")
        widget._mask_b_combo.setCurrentText("B")
        widget._run_overlap_analysis()

        # Histogram chart was added (stretch + overlap + histogram = 3).
        assert widget._charts_layout.count() == 3
        # The real table's labels index the B layer data: label 1 (intensity 30)
        # overlaps A; label 2 (intensity 70) does not.
        split = split_spot_intensities(label_b, a, table)
        assert split["n_overlap"] == 1
        assert split["n_non_overlap"] == 1
        np.testing.assert_allclose(sorted(split["overlap"]), [30.0])
        np.testing.assert_allclose(sorted(split["non_overlap"]), [70.0])


class TestSpotTableExport:
    def test_export_combo_lists_labels_layers(self, make_napari_viewer):
        viewer = make_napari_viewer()
        viewer.add_labels(np.zeros((1, 4, 4), dtype=np.int32), name="A")
        viewer.add_image(np.zeros((1, 4, 4), dtype=np.float32), name="img")
        widget = OoctyleAnalysisWidget(viewer)
        items = [widget._export_combo.itemText(i) for i in range(widget._export_combo.count())]
        assert "A" in items
        assert "img" not in items

    def test_export_writes_csv(self, tmp_path, monkeypatch, make_napari_viewer):
        from qtpy.QtWidgets import QFileDialog
        viewer = make_napari_viewer()
        layer = viewer.add_labels(np.zeros((1, 4, 4), dtype=np.int32), name="A")
        layer.metadata["spot_intensity"] = {
            "label": np.array([1, 2]),
            "centroid-0": np.array([0.0, 0.0]),
            "centroid-1": np.array([1.0, 3.0]),
            "centroid-2": np.array([1.0, 3.0]),
            "area": np.array([4, 4]),
            "intensity_mean": np.array([10.0, 20.0]),
        }
        widget = OoctyleAnalysisWidget(viewer)
        widget._export_combo.setCurrentText("A")
        out = tmp_path / "spots.csv"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")),
        )
        widget._export_spot_table_csv()
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert lines[0].split(",")[0] == "label"
        assert len(lines) == 3  # header + 2 spots

    def test_export_without_metadata_writes_nothing(self, tmp_path, monkeypatch, make_napari_viewer):
        from qtpy.QtWidgets import QFileDialog
        viewer = make_napari_viewer()
        viewer.add_labels(np.zeros((1, 4, 4), dtype=np.int32), name="A")
        widget = OoctyleAnalysisWidget(viewer)
        widget._export_combo.setCurrentText("A")
        out = tmp_path / "nope.csv"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")),
        )
        widget._export_spot_table_csv()
        assert not out.exists()
        assert "no per-spot" in widget._export_status.text().lower()

    def test_export_handles_write_error(self, tmp_path, monkeypatch, make_napari_viewer):
        from qtpy.QtWidgets import QFileDialog
        viewer = make_napari_viewer()
        layer = viewer.add_labels(np.zeros((1, 4, 4), dtype=np.int32), name="A")
        layer.metadata["spot_intensity"] = {
            "label": np.array([1]),
            "centroid-0": np.array([0.0]),
            "centroid-1": np.array([1.0]),
            "centroid-2": np.array([1.0]),
            "area": np.array([4]),
            "intensity_mean": np.array([10.0]),
        }
        widget = OoctyleAnalysisWidget(viewer)
        widget._export_combo.setCurrentText("A")
        bad = tmp_path / "does_not_exist" / "spots.csv"  # parent dir missing -> open() raises
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(bad), "CSV (*.csv)")),
        )
        widget._export_spot_table_csv()
        assert not bad.exists()
        assert "export failed" in widget._export_status.text().lower()

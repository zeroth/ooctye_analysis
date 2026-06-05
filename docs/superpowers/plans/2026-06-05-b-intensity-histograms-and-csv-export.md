# Channel B Per-Spot Intensity Histograms + Spot-Table CSV Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture per-spot intensity at detection time, then (a) draw two side-by-side histograms of Channel B's per-spot mean intensity split by overlap-with-A, and (b) add a main-widget control to export any detected layer's spot table as CSV (nucleus excluded by construction).

**Architecture:** `skimage.measure.regionprops_table` runs in `PredictWorker` on the nucleus-clipped, labeled mask using the channel's own image; the table is stored on the Labels layer's `metadata["spot_intensity"]`. Pure functions in `_analysis.py` do the splitting, charting, and CSV-row formatting; the widget wires them to the Analysis tab and a top-level export control.

**Tech Stack:** Python, NumPy, scikit-image (regionprops), scipy.ndimage (labeling), matplotlib, qtpy/napari, pytest. Spec: `docs/superpowers/specs/2026-06-05-b-intensity-histograms-and-csv-export-design.md`.

---

## Conventions

- **Run tests with the project venv** (a bare `pytest` is the wrong interpreter):
  `env/Scripts/python.exe -m pytest <args>` — from the project root `d:\Code\Lab\ooctye_analysis`.
- **Commit with the D: drive fsync workaround:** `git -c core.fsync=none add ...` and
  `git -c core.fsync=none commit -m "..."`.
- These changes are **additive** — the suite should stay green after every task.
- Test file: `src/napari_ooctyle_analysis/_tests/test_widget.py`. New `_analysis` functions must be
  added to its imports. Check the existing import line near the top
  (`from napari_ooctyle_analysis._analysis import ...`) and append the new names to it.

## File Structure

- `pyproject.toml` — add `scikit-image` to `dependencies`.
- `src/napari_ooctyle_analysis/_analysis.py` — 4 new pure functions:
  `compute_spot_regionprops`, `split_spot_intensities`, `create_intensity_histogram_figure`,
  `spot_table_to_rows`.
- `src/napari_ooctyle_analysis/_workers.py` — `PredictWorker.run` computes regionprops and adds
  `labeled_mask`/`n_labels`/`spot_intensity` to `model_meta`.
- `src/napari_ooctyle_analysis/_widget.py` — `_on_detection_finished` attaches metadata; new
  `_maybe_add_intensity_histogram`; new top-level export control + `_export_spot_table_csv`;
  `_export_combo` populated in `_refresh_image_layers`.
- `src/napari_ooctyle_analysis/_tests/test_widget.py` — tests for every task.

---

## Task 1: `compute_spot_regionprops` + scikit-image dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/napari_ooctyle_analysis/_analysis.py`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the test**

Add to `test_widget.py`. First ensure the import line for `_analysis` includes `compute_spot_regionprops` (append it to the existing `from napari_ooctyle_analysis._analysis import (...)` block).

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputeSpotRegionprops -v`
Expected: FAIL with ImportError for `compute_spot_regionprops`.

- [ ] **Step 3: Implement the function**

Add to `_analysis.py` (after `compute_zonal_voxels`, before `create_zonal_figure`):

```python
SPOT_PROPERTIES = ("label", "centroid", "area", "intensity_mean")


def compute_spot_regionprops(label_img: np.ndarray, intensity_img: np.ndarray) -> dict:
    """Per-spot region properties measured on ``intensity_img``.

    Wraps ``skimage.measure.regionprops_table`` over a labeled image. Background
    (label 0) is excluded. Returns a dict of equal-length arrays with keys
    ``label``, ``centroid-0/1/2`` (per image dim), ``area``, ``intensity_mean``.
    An all-zero label image yields the same keys mapping to empty arrays.
    """
    from skimage.measure import regionprops_table

    table = regionprops_table(
        label_img.astype(np.int32, copy=False),
        intensity_image=intensity_img,
        properties=SPOT_PROPERTIES,
    )
    return {key: np.asarray(value) for key, value in table.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputeSpotRegionprops -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add scikit-image to dependencies**

In `pyproject.toml`, change the `dependencies` list from:

```toml
dependencies = [
    "numpy",
    "spotiflow>=0.6.0",
    "qtpy",
    "matplotlib",
]
```

to:

```toml
dependencies = [
    "numpy",
    "scikit-image",
    "spotiflow>=0.6.0",
    "qtpy",
    "matplotlib",
]
```

- [ ] **Step 6: Commit**

```bash
git -c core.fsync=none add pyproject.toml src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_analysis): add compute_spot_regionprops + scikit-image dep"
```

---

## Task 2: `split_spot_intensities`

**Files:**
- Modify: `src/napari_ooctyle_analysis/_analysis.py`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the test** (append `split_spot_intensities` to the `_analysis` import block)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSplitSpotIntensities -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

Add to `_analysis.py` (after `compute_spot_regionprops`):

```python
def split_spot_intensities(
    label_img_b: np.ndarray,
    mask_a: np.ndarray,
    table: dict,
) -> dict:
    """Split B's per-spot ``intensity_mean`` into overlap-with-A vs non-overlap.

    A B spot overlaps A if ANY of its voxels lie in ``mask_a`` (> 0). Returns
    ``{"overlap": ndarray, "non_overlap": ndarray, "n_overlap": int,
    "n_non_overlap": int}``. Labels in ``table`` must match ``label_img_b``.
    """
    labels = np.asarray(table["label"])
    means = np.asarray(table["intensity_mean"], dtype=np.float64)
    overlapping = np.unique(label_img_b[(mask_a > 0) & (label_img_b > 0)])
    is_overlap = np.isin(labels, overlapping)
    overlap = means[is_overlap]
    non_overlap = means[~is_overlap]
    return {
        "overlap": overlap,
        "non_overlap": non_overlap,
        "n_overlap": int(overlap.size),
        "n_non_overlap": int(non_overlap.size),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSplitSpotIntensities -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_analysis): add split_spot_intensities"
```

---

## Task 3: `create_intensity_histogram_figure`

**Files:**
- Modify: `src/napari_ooctyle_analysis/_analysis.py`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the test** (append `create_intensity_histogram_figure` to the import block)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestIntensityHistogramFigure -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

Add to `_analysis.py` (after `create_zonal_figure`):

```python
def create_intensity_histogram_figure(name_b: str, split: dict):
    """Two side-by-side histograms of B per-spot mean intensity.

    Left = spots overlapping A, right = non-overlapping spots. Shared x-range and
    bins computed from the pooled values so the panels are directly comparable.
    Empty arrays render an empty axes (no crash). Returns a matplotlib Figure.
    """
    from matplotlib.figure import Figure

    overlap = np.asarray(split["overlap"], dtype=np.float64)
    non_overlap = np.asarray(split["non_overlap"], dtype=np.float64)
    short_b = short_label(name_b)

    fig = Figure(figsize=(6.0, 3.2), dpi=100)
    fig.suptitle(f"{short_b} per-spot mean intensity", fontsize=10, fontweight="bold")

    pooled = np.concatenate([overlap, non_overlap]) if (overlap.size + non_overlap.size) else np.array([])
    if pooled.size:
        lo, hi = float(pooled.min()), float(pooled.max())
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 21)
    else:
        bins = 10

    ax1 = fig.add_subplot(121)
    if overlap.size:
        ax1.hist(overlap, bins=bins, color="#4C72B0")
    ax1.set_title(f"Overlapping A (n={split['n_overlap']})", fontsize=9)
    ax1.set_xlabel("Mean intensity", fontsize=8)
    ax1.set_ylabel("Spot count", fontsize=8)

    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    if non_overlap.size:
        ax2.hist(non_overlap, bins=bins, color="#DD8452")
    ax2.set_title(f"Non-overlapping (n={split['n_non_overlap']})", fontsize=9)
    ax2.set_xlabel("Mean intensity", fontsize=8)

    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.3)
    return fig
```

- [ ] **Step 4: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestIntensityHistogramFigure -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_analysis): add create_intensity_histogram_figure"
```

---

## Task 4: `spot_table_to_rows`

**Files:**
- Modify: `src/napari_ooctyle_analysis/_analysis.py`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the test** (append `spot_table_to_rows` to the import block)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSpotTableToRows -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

Add to `_analysis.py` (after `spot`-related functions, e.g. after `split_spot_intensities`):

```python
_SPOT_TABLE_COLUMN_ORDER = (
    "label", "centroid-0", "centroid-1", "centroid-2", "area", "intensity_mean",
)


def spot_table_to_rows(table: dict) -> tuple[list, list]:
    """Convert a regionprops table dict into (header, rows) for CSV writing.

    Columns follow ``_SPOT_TABLE_COLUMN_ORDER``, keeping only keys present in the
    table (any extra keys are appended after, in sorted order). ``rows`` is the
    column-wise zip; an empty table yields the header and an empty row list.
    """
    ordered = [c for c in _SPOT_TABLE_COLUMN_ORDER if c in table]
    extra = sorted(k for k in table if k not in ordered)
    header = ordered + extra
    columns = [np.asarray(table[h]).tolist() for h in header]
    rows = [list(r) for r in zip(*columns)] if columns and len(columns[0]) else []
    return header, rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSpotTableToRows -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_analysis): add spot_table_to_rows for CSV export"
```

---

## Task 5: Worker captures regionprops (incl. nucleus-exclusion)

**Files:**
- Modify: `src/napari_ooctyle_analysis/_workers.py:84-98`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

The worker's spotiflow path can't run in tests, so the test verifies the exact pipeline the
worker performs (clip nucleus → label → regionprops) and asserts an in-nucleus spot is absent.

- [ ] **Step 1: Add the pipeline / nucleus-exclusion test**

```python
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
```

- [ ] **Step 2: Run to verify it passes already**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestWorkerRegionpropsPipeline -v`
Expected: PASS — this test exercises only already-implemented functions and documents the
invariant. (It is the regression guard for nucleus exclusion.)

- [ ] **Step 3: Wire regionprops into the worker**

In `_workers.py`, the current tail of `run()` is:

```python
            mask = result.mask
            if oocyte is not None:
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if nucleus is not None:
                apply_sphere_to_mask(mask, nucleus, mode="zero_inside")

            model_meta = {
                "sigma": self.model.config.sigma,
                "grid": tuple(self.model.config.grid) if self.model.config.is_3d else (1, 1),
                "image_shape": self.image.shape,
                "n_excluded": n_excluded,
                "mask": mask,
            }
            self.finished.emit(spots, details, model_meta)
```

Replace it with (adds labeling + regionprops on the final clipped mask, using the channel image
`img_for_fit` already computed above):

```python
            mask = result.mask
            if oocyte is not None:
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if nucleus is not None:
                apply_sphere_to_mask(mask, nucleus, mode="zero_inside")

            from scipy.ndimage import label as ndlabel
            from napari_ooctyle_analysis._analysis import compute_spot_regionprops

            self.progress.emit("Measuring spot intensities", 0, 0)
            labeled_mask, n_labels = ndlabel(mask)
            spot_intensity = compute_spot_regionprops(labeled_mask, img_for_fit)

            model_meta = {
                "sigma": self.model.config.sigma,
                "grid": tuple(self.model.config.grid) if self.model.config.is_3d else (1, 1),
                "image_shape": self.image.shape,
                "n_excluded": n_excluded,
                "mask": mask,
                "labeled_mask": labeled_mask,
                "n_labels": n_labels,
                "spot_intensity": spot_intensity,
            }
            self.finished.emit(spots, details, model_meta)
```

- [ ] **Step 4: Run the worker-related tests**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestWorkerRegionpropsPipeline src/napari_ooctyle_analysis/_tests/test_widget.py::TestOocyteClipping -v`
Expected: PASS (existing TestOocyteClipping still green — it monkeypatches `run`, so it is unaffected).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_workers.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_workers): measure per-spot regionprops on clipped mask"
```

---

## Task 6: `_on_detection_finished` attaches metadata (with fallback)

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py:705-706`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the widget tests**

```python
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
```

- [ ] **Step 2: Run to verify the first test fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestDetectionMetadata -v`
Expected: `test_metadata_attached_when_present` FAILS (metadata not attached yet);
`test_fallback_without_metadata` may already pass.

- [ ] **Step 3: Update `_on_detection_finished`**

Replace the labeling + add_labels block (currently lines 705-706):

```python
        labeled_mask, n_labels = ndlabel(mask)
        self.viewer.add_labels(labeled_mask, name=f"{image_name} mask", opacity=0.4)
```

with (prefer the worker's labeled mask + metadata; fall back to local labeling):

```python
        labeled_mask = model_meta.get("labeled_mask")
        spot_intensity = model_meta.get("spot_intensity")
        if labeled_mask is None:
            labeled_mask, n_labels = ndlabel(mask)
        else:
            n_labels = model_meta.get("n_labels", int(labeled_mask.max()))
        mask_layer = self.viewer.add_labels(
            labeled_mask, name=f"{image_name} mask", opacity=0.4,
        )
        if spot_intensity is not None:
            mask_layer.metadata["spot_intensity"] = spot_intensity
```

- [ ] **Step 4: Run to verify both pass**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestDetectionMetadata -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_widget): store per-spot intensity table on mask layer metadata"
```

---

## Task 7: Analysis-tab intensity histogram wiring

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py` (`_run_overlap_analysis` tail; new
  `_maybe_add_intensity_histogram`)
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the wiring tests**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestIntensityHistogramWiring -v`
Expected: FAIL (no histogram added; count is 2 in the first test).

- [ ] **Step 3: Add the call at the end of `_run_overlap_analysis`**

The method currently ends with:

```python
        self._maybe_add_zonal_chart(mask_a, mask_b, name_a, name_b)
```

Add immediately after it:

```python
        self._maybe_add_intensity_histogram(mask_a, mask_b, name_b)
```

- [ ] **Step 4: Implement `_maybe_add_intensity_histogram`**

Add this method right after `_maybe_add_zonal_chart` in `_widget.py`:

```python
    def _maybe_add_intensity_histogram(self, mask_a, mask_b, name_b: str) -> None:
        """Histogram of Channel B per-spot mean intensity, split by overlap with A.

        Requires the Channel B layer to carry per-spot intensity metadata from a
        detection run; otherwise the histogram is skipped with a status hint.
        """
        try:
            b_layer = self.viewer.layers[name_b]
        except KeyError:
            return
        table = b_layer.metadata.get("spot_intensity")
        if table is None:
            self._overlap_status.setText(
                self._overlap_status.text()
                + "  (Channel B has no per-spot intensity data — "
                "re-run detection on Channel B to enable intensity histograms.)"
            )
            return

        split = analysis.split_spot_intensities(mask_b, mask_a, table)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        fig = analysis.create_intensity_histogram_figure(name_b, split)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(260)
        count = self._charts_layout.count()
        self._charts_layout.insertWidget(count - 1, canvas)
```

- [ ] **Step 5: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestIntensityHistogramWiring -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_widget): add Channel B intensity histogram to analysis"
```

---

## Task 8: Main-widget CSV export control

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py` (`__init__` pre-create + top-level row;
  `_refresh_image_layers`; new `_export_spot_table_csv`)
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Add the export tests**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSpotTableExport -v`
Expected: FAIL (`_export_combo` does not exist yet).

- [ ] **Step 3: Pre-create the export combo**

In `__init__`, the pre-create block currently is:

```python
        self._mask_a_combo = QComboBox()
        self._mask_b_combo = QComboBox()
```

Add the export combo right after:

```python
        self._mask_a_combo = QComboBox()
        self._mask_b_combo = QComboBox()
        self._export_combo = QComboBox()
```

- [ ] **Step 4: Add the top-level export row after the tabs**

In `__init__`, after:

```python
        self.tabs.addTab(self._build_segmentation_tab(), "Segmentation")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
```

insert:

```python
        # Top-level spot-table CSV export (available regardless of active tab).
        export_group = QGroupBox("Export spot table")
        export_layout = QHBoxLayout()
        export_group.setLayout(export_layout)
        export_layout.addWidget(QLabel("Layer:"))
        export_layout.addWidget(self._export_combo)
        export_refresh = QPushButton("↻")
        export_refresh.setFixedWidth(28)
        export_refresh.setToolTip("Refresh layer list")
        export_refresh.clicked.connect(self._refresh_image_layers)
        export_layout.addWidget(export_refresh)
        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self._export_spot_table_csv)
        export_layout.addWidget(export_btn)
        self._export_status = QLabel("")
        layout.addWidget(export_group)
        layout.addWidget(self._export_status)
```

- [ ] **Step 5: Populate `_export_combo` in `_refresh_image_layers`**

In `_refresh_image_layers`, add `self._export_combo` to the `all_combos` list:

```python
        all_combos = [
            self._image_combo, self._mask_a_combo, self._mask_b_combo,
            self._export_combo,
            *self._region_combos.values(),
        ]
```

and in the layer-type loop, the Labels branch currently is:

```python
            elif isinstance(layer, napari.layers.Labels):
                self._mask_a_combo.addItem(layer.name)
                self._mask_b_combo.addItem(layer.name)
```

change it to also feed the export combo:

```python
            elif isinstance(layer, napari.layers.Labels):
                self._mask_a_combo.addItem(layer.name)
                self._mask_b_combo.addItem(layer.name)
                self._export_combo.addItem(layer.name)
```

- [ ] **Step 6: Implement `_export_spot_table_csv`**

Add this method to the widget (e.g. right after `_export`-adjacent analysis methods, or after
`_maybe_add_intensity_histogram`):

```python
    def _export_spot_table_csv(self) -> None:
        """Write the selected layer's per-spot regionprops table to CSV.

        The table was computed on the nucleus-clipped mask, so nucleus spots are
        absent by construction.
        """
        import csv
        from qtpy.QtWidgets import QFileDialog

        name = self._export_combo.currentText()
        if not name:
            self._export_status.setText("No layer selected to export.")
            return
        try:
            layer = self.viewer.layers[name]
        except KeyError:
            self._export_status.setText(f"Layer not found: {name}")
            return
        table = getattr(layer, "metadata", {}).get("spot_intensity")
        if table is None:
            self._export_status.setText(
                f"'{name}' has no per-spot data; it must come from a detection run."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export spot table", f"{name}.csv", "CSV (*.csv)"
        )
        if not path:
            return

        header, rows = analysis.spot_table_to_rows(table)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        self._export_status.setText(f"Exported {len(rows)} spots to {path}")
```

- [ ] **Step 7: Run to verify it passes**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestSpotTableExport -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_widget): main-widget CSV export of per-spot table"
```

---

## Task 9: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole package**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis -q`
Expected: all tests pass (previous 81 + the new tests from Tasks 1-8).

- [ ] **Step 2: If anything fails**, fix the offending task's code/test and re-run until green.
  No commit needed if Tasks 1-8 each committed cleanly and this run is green.

---

## Self-Review Notes

- **Spec coverage:** regionprops capture (Task 1,5) ✔; metadata storage (Task 6) ✔; per-spot
  overlap/non-overlap split with "any voxel" rule (Task 2) ✔; two-subplot histogram (Task 3,7) ✔;
  nucleus exclusion by construction + test (Task 5) ✔; main-widget CSV export with layer dropdown
  (Task 8) ✔; `spot_table_to_rows` column order (Task 4) ✔; scikit-image dependency (Task 1) ✔;
  graceful skip when B lacks metadata (Task 7) ✔; export validation when no metadata (Task 8) ✔.
- **Type/name consistency:** `compute_spot_regionprops(label_img, intensity_img)`,
  `split_spot_intensities(label_img_b, mask_a, table)`, `create_intensity_histogram_figure(name_b,
  split)`, `spot_table_to_rows(table)`, metadata key `"spot_intensity"`, model_meta keys
  `labeled_mask`/`n_labels`/`spot_intensity`, widget members `_export_combo`/`_export_status`/
  `_maybe_add_intensity_histogram`/`_export_spot_table_csv` are used identically across tasks.
- **Additive:** every task keeps the suite green; commits are independent (no coupled multi-task
  commit needed this time).
```

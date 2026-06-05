# Channel B Per-Spot Intensity Histograms + Spot-Table CSV Export — Design

**Date:** 2026-06-05
**Status:** Approved

## Problem

The Analysis tab computes voxel-level overlap between two masks (Channel A, Channel B) and
charts it. Researchers also want to know whether Channel B spots that **colocalize** with
Channel A differ in intensity from B spots that do **not**. They also need to export the
per-spot measurements as CSV for downstream analysis, with the excluded nucleus region absent.

## Goal

1. Keep the existing two-mask overlap analysis unchanged.
2. Capture per-spot intensity at detection time and store it on the mask Labels layer.
3. In the Analysis tab, after the overlap chart, draw two side-by-side histograms of Channel B's
   per-spot mean intensity: one for B spots overlapping A, one for B spots not overlapping A.
4. Add a top-level (main-widget) control to export any detected layer's per-spot table as CSV,
   with the nucleus-excluded by construction.

## Key Decisions (locked)

- **Intensity unit:** per **spot** (one value per detected region), not per voxel.
- **Intensity measure:** `intensity_mean` (regionprops).
- **Overlap rule for a B spot:** overlap if **any** of its voxels fall inside A's mask.
- **Non-overlap definition:** B spots whose label is not in the overlapping set (B \ A).
- **Intensity source:** the channel's own image at detection time (`img_for_fit` in the worker);
  no separate intensity picker is needed.
- **Histogram layout:** two side-by-side subplots, shared x-axis.
- **CSV export:** a control on the **main widget** (outside the tabs), with its own layer-picker
  dropdown and an "Export CSV…" button.
- **Nucleus exclusion:** regionprops runs on the mask **after** nucleus clipping, so nucleus
  voxels are gone before labeling — the table inherently excludes the nucleus.

## Architecture / Data Flow

```
PredictWorker.run()
  fit -> mask -> clip to oocyte -> zero nucleus     (existing)
  labeled_mask = ndimage.label(mask)                (moved here from _on_detection_finished)
  table = compute_spot_regionprops(labeled_mask, img_for_fit)
  model_meta += { labeled_mask, n_labels, spot_intensity: table }

_on_detection_finished(model_meta)
  layer = add_labels(labeled_mask)                  (use worker's labeled_mask if present)
  layer.metadata["spot_intensity"] = table          (the regionprops table)
  # fallback: if model_meta has no labeled_mask, ndlabel(mask) as today, no metadata

_run_overlap_analysis()                              (existing overlap, unchanged)
  ... existing overlap chart + zonal chart ...
  _maybe_add_intensity_histogram(mask_a_labels, mask_b_labels, name_b)

Main widget (outside tabs)
  [Export spot table:  (layer dropdown)  ⟳ ]  [ Export CSV… ]
    -> spot_table_to_rows(layer.metadata["spot_intensity"]) -> csv.writer
```

## Components

### 1. `_analysis.py` — new pure functions (no Qt, unit-tested)

```python
def compute_spot_regionprops(label_img, intensity_img) -> dict:
    """regionprops_table over labeled spots, measured on intensity_img.

    Returns a dict of equal-length arrays with keys:
    'label', 'centroid-0', 'centroid-1', 'centroid-2', 'area', 'intensity_mean'.
    Background (label 0) is excluded by regionprops. For an empty/all-zero label
    image, returns the same keys mapping to empty arrays.
    """
```
- Implemented with `skimage.measure.regionprops_table(label_img, intensity_image=intensity_img,
  properties=("label", "centroid", "area", "intensity_mean"))`.
- Handles 2D as well as 3D (centroid columns follow the image ndim); the worker always passes a
  3D label image, so centroid-0/1/2 are present in practice.

```python
def split_spot_intensities(label_img_b, mask_a, table) -> dict:
    """Split B's per-spot intensity_mean into overlap (with A) vs non-overlap.

    A B spot overlaps A if ANY of its voxels lie in mask_a. Returns:
    {"overlap": np.ndarray, "non_overlap": np.ndarray,
     "n_overlap": int, "n_non_overlap": int}.
    """
    labels = np.asarray(table["label"])
    means = np.asarray(table["intensity_mean"])
    overlapping = np.unique(label_img_b[(mask_a > 0) & (label_img_b > 0)])
    is_overlap = np.isin(labels, overlapping)
    ...
```

```python
def create_intensity_histogram_figure(name_b, split) -> Figure:
    """Two side-by-side histograms (overlap | non-overlap) of B per-spot mean intensity.

    Shared x-range computed from the pooled values so the two panels are comparable.
    Titles include counts: "Overlapping A (n=..)" and "Non-overlapping (n=..)".
    Empty arrays render an empty axes with the n=0 title (no crash).
    """
```

```python
def spot_table_to_rows(table) -> tuple[list[str], list[list]]:
    """Return (header, rows) for CSV writing from a regionprops table dict.

    Column order: label, centroid-0, centroid-1, centroid-2, area, intensity_mean
    (only the keys present in the table, in that order). rows = column-wise zip.
    """
```

### 2. `_workers.py` — capture at detection

- After nucleus/oocyte clipping of `mask`, label it (`from scipy.ndimage import label`) and call
  `compute_spot_regionprops(labeled_mask, img_for_fit)`.
- `img_for_fit` is the already-squeezed 3D channel image used for fitting (the right intensity
  source for this channel's spots).
- Add to `model_meta`: `"labeled_mask"`, `"n_labels"`, `"spot_intensity"` (the table). Keep the
  existing keys (`mask`, `n_excluded`, ...) for backward compatibility.

### 3. `_widget.py` — `_on_detection_finished`

- If `model_meta` has `labeled_mask`, use it and `n_labels` directly; attach
  `layer.metadata["spot_intensity"] = model_meta["spot_intensity"]`.
- Else fall back to the current `ndlabel(mask)` path with no metadata (keeps existing tests green).

### 4. `_widget.py` — Analysis tab histogram

- New `_maybe_add_intensity_histogram(self, mask_a, mask_b, name_b)` called at the end of
  `_run_overlap_analysis`:
  - `b_layer = self.viewer.layers[name_b]`; read `table = b_layer.metadata.get("spot_intensity")`.
  - If `table is None`: set a hint on `_overlap_status` ("Channel B has no per-spot intensity
    data — re-run detection on Channel B to enable intensity histograms") and return (do not
    disturb the overlap chart already drawn).
  - Else: `split = analysis.split_spot_intensities(mask_b, mask_a > 0, table)`, build the figure,
    wrap in `FigureCanvasQTAgg`, insert into `_charts_layout` (same pattern as overlap/zonal).
  - `mask_b` here is the Channel B **Labels data** (integer labels), which matches the labels in
    `table`. `mask_a` is Channel A's Labels data.

### 5. `_widget.py` — main-widget CSV export (outside the tabs)

- In `__init__`, after `self.tabs` is added to the top-level `layout`, add a small export row:
  `QLabel("Export spot table:")`, `self._export_combo` (QComboBox), a refresh button, and
  `QPushButton("Export CSV…")` wired to `_export_spot_table_csv`.
- `self._export_combo` is created alongside the other combos and populated in
  `_refresh_image_layers`: list every **Labels** layer (validation happens on export).
- `_export_spot_table_csv`:
  - Resolve the selected layer; if missing → status/message and return.
  - `table = layer.metadata.get("spot_intensity")`; if None → message
    ("Selected layer has no per-spot table; it must come from a detection run") and return.
  - `header, rows = analysis.spot_table_to_rows(table)`.
  - `QFileDialog.getSaveFileName(... "CSV (*.csv)")`; if a path is chosen, write with the stdlib
    `csv` module (`newline=""`), header first. Report success/row count in a status label.

### 6. `pyproject.toml`

- Add `"scikit-image"` to `dependencies` (already present transitively via spotiflow; now used
  directly).

## Error Handling

- No spots / empty table → empty histograms (n=0 titles), CSV with header and zero rows.
- B layer lacks `spot_intensity` metadata → histogram skipped with a clear hint; overlap analysis
  still completes.
- Export with no selection / no metadata → user-facing message, no file written.
- Shape mismatch between A and B is already guarded in `_run_overlap_analysis` before the split.

## Testing

Pure-logic tests in `_tests/test_widget.py` (or a sibling), no Qt where avoidable:

1. `compute_spot_regionprops`: a small labeled volume + intensity image yields expected
   `intensity_mean` per label; background excluded; empty image → empty arrays.
2. **Nucleus exclusion:** build a mask where one spot sits inside the nucleus sphere; after the
   worker's clip-then-label-then-regionprops pipeline, that spot's label is **absent** from the
   table (assert via the worker path or a focused helper mirroring it).
3. `split_spot_intensities`: a B label touching `mask_a` lands in `overlap`; one not touching
   lands in `non_overlap`; counts correct; label 0 ignored.
4. `create_intensity_histogram_figure`: returns a Figure with two axes; empty arrays don't crash.
5. `spot_table_to_rows`: header order is `label, centroid-0, centroid-1, centroid-2, area,
   intensity_mean`; row count equals number of spots; values align column-wise.
6. Widget wiring: `_export_combo` lists Labels layers; export with no metadata shows a message and
   writes nothing; export with metadata writes a CSV whose row count matches the table.

## Out of Scope

- Per-voxel intensity distributions (we chose per-spot).
- Intensity measures other than mean (max/integrated) — `compute_spot_regionprops` already
  collects `area`, so integrated could be derived later without re-detection.
- Channel A intensity histograms (B is the focus; A is the colocalization reference).
- A separate intensity-image picker (the channel's own image is the source).
```

# Export Underlying Points for Overlap / Non-Overlap Mask Layers — Design

**Date:** 2026-06-05
**Status:** Approved

## Problem

The overlap (A∩B) and non-overlap (A\B) layers are voxel masks with no per-spot data, so the
existing "Export spot table (CSV)" control (which reads a layer's `spot_intensity` metadata)
produces nothing useful for them. Users want to export the detected spots that fall in each
region.

## Goal

Attach a per-region spot table — drawn from both channels' detected spots — to the overlap and
non-overlap mask layers, so the existing CSV export works on them and emits the underlying points.

## Decisions (locked)

- **Whose spots:** both channels (A and B), each row tagged with a `channel` column (the layer
  name).
- **Membership rule:** a spot belongs to a region if **any** of its labeled voxels fall in the
  region mask (`np.unique(label_img[(region_mask>0) & (label_img>0)])`). Consistent with the
  intensity histogram.

## Consequences (intended)

- The non-overlap layer (A\B) contains no B voxels, so **only A spots** can qualify there; B
  spots only appear in the overlap CSV.
- Under the any-voxel rule, a spot straddling the A∩B boundary may appear in **both** the overlap
  and non-overlap CSVs. Expected.
- Region spot export requires the mask layer to exist — i.e. the "Show overlap & non-overlap mask
  layers" checkbox must be on (the metadata is attached when the layer is created).

## Changes

### 1. `_analysis.py` — new pure function

```python
def compute_region_spot_table(region_mask, channels) -> dict:
    """Combine per-channel spot tables for spots whose labels intersect region_mask.

    channels: list of (channel_name, label_img, spot_table). For each channel, keeps
    spots whose label intersects region_mask (any voxel), tags kept rows with a
    'channel' value (channel_name), and concatenates. Returns a dict-of-arrays with a
    leading 'channel' column plus the regionprops columns (label, centroid-0/1/2,
    area, intensity_mean). Channels with None/empty tables or zero matching spots
    contribute nothing. If nothing matches, returns {} (no columns).
    """
```
- Assumes all provided `spot_table`s share the same column keys (they do — all come from
  `compute_spot_regionprops`).
- Per channel: `in_region = np.unique(label_img[(region_mask > 0) & (label_img > 0)])`,
  `keep = np.isin(np.asarray(table["label"]), in_region)`; extend `channel` with
  `[channel_name] * keep.sum()` and each regionprops column with its kept values.

### 2. `_analysis.py` — `spot_table_to_rows`

Prepend `"channel"` to `_SPOT_TABLE_COLUMN_ORDER`:
`("channel", "label", "centroid-0", "centroid-1", "centroid-2", "area", "intensity_mean")`.
Because the function keeps only keys present in the table, detection-mask exports (no `channel`
key) are unchanged; region exports get `channel` as the first column.

### 3. `_widget.py` — `_run_overlap_analysis`

Inside the existing `if self._show_overlap_layer.isChecked():` block, after each mask layer is
added (capture the returned layer object):
- Build `channels` from the selected layers' metadata:
  `a_table = self.viewer.layers[name_a].metadata.get("spot_intensity")` (and B). Include
  `(name_a, mask_a, a_table)` / `(name_b, mask_b, b_table)` only when the table is not None.
- `region_table = analysis.compute_region_spot_table(result["overlap_mask"], channels)`; if it has
  rows, set `overlap_layer.metadata["spot_intensity"] = region_table`. Same for the non-overlap
  layer with `result["non_overlap_mask"]`.

No change to `_export_spot_table_csv` or the export dropdown — they already read
`metadata["spot_intensity"]` and serialize via `spot_table_to_rows`.

## Error Handling

- Neither channel has spot metadata → no region table attached; exporting that layer reports the
  existing "no per-spot data" message.
- Empty region (no matching spots) → no metadata attached.

## Testing

1. `compute_region_spot_table`: with A and B labeled volumes + tables and a region mask, kept
   spots are exactly those whose label intersects the region; rows carry the right `channel`
   value; B spots are absent from an A\B region; empty region → `{}`.
2. `spot_table_to_rows`: a table containing `channel` emits it as the first column; a table
   without `channel` (detection layer) is unchanged.
3. Widget end-to-end: after `_run_overlap_analysis` with both A and B carrying `spot_intensity`,
   the overlap layer's `metadata["spot_intensity"]` has a `channel` column with both names; the
   non-overlap layer's table has only A. Exporting the overlap layer writes a CSV whose header
   starts with `channel`.

## Out of Scope

- Centroid-based membership (we use any-voxel).
- Changing the histogram, the per-channel detection export, or the mask geometry.
- Auto-creating the mask layers when the checkbox is off.

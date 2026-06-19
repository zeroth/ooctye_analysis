# Channel C Overlap Analysis — Design

**Date:** 2026-06-05
**Status:** Approved

## Problem

The Analysis tab overlaps two channels (A, B) and produces A∩B and A\B regions. Researchers
want to further intersect those A-centric regions with a third channel C — i.e. how much of the
"A overlapping B" and "A not overlapping B" populations also coincide with C — and export the
underlying points.

## Goal

Add an optional Channel C. On Compute Overlap, after the existing A∩B / A\B work, also compute:
- **(A∩B) ∩ C** — the A∩B region intersected with C
- **(A\B) ∩ C** — the A\B region intersected with C

For each: an overlap bar chart, an exportable mask layer, and a channel-tagged underlying-points
spot table (CSV via the existing export control).

## Decisions (locked)

- **C is optional**: a leading "(none)" entry. When C is "(none)" (or empty), behavior is exactly
  as today. When C is a layer, the C step runs.
- **Underlying spots for the C-intersections**: A, B, and C spots (channel-tagged), any-voxel
  membership; channels lacking `spot_intensity` metadata are skipped.
- **Charts**: yes — two overlap bar charts, like the existing A-vs-B chart.
- The existing A∩B / A\B layers and their spot tables stay **A+B only** (unchanged).

## Reuse (no new pure logic)

- `(A∩B)∩C` and `(A\B)∩C` are produced by `compute_overlap(region_mask, mask_c)` — its
  `overlap_mask` key is `region ∩ C`, and its counts/figure feed the existing chart path.
- Underlying-points tables use the existing `compute_region_spot_table(region_mask, channels)`.
- Mask layers + tables use the existing `_add_region_mask_layer`.
- Charts use the existing `_add_overlap_chart` / `analysis.create_overlap_figure`.

## Changes

### 1. UI — `_widget.py`

- Pre-create `self._mask_c_combo = QComboBox()` alongside the A/B combos.
- Add a "Channel C:" row to the Analysis tab controls (after Channel B).
- In `_refresh_image_layers`: add `_mask_c_combo` to `all_combos`; after clearing, prepend the
  sentinel `"(none)"` to `_mask_c_combo`, then add each Labels layer name to it (alongside A/B).
  Define a module/class constant `C_NONE = "(none)"`.

### 2. `_run_overlap_analysis` — C step (`_widget.py`)

After the existing A∩B/A\B charts, region layers, and status text:

```python
name_c = self._mask_c_combo.currentText()
if name_c and name_c != C_NONE:
    mask_c = np.asarray(self.viewer.layers[name_c].data)   # KeyError-guarded
    if mask_c.shape != mask_a.shape:
        self._overlap_status.setText("Shape mismatch: ... C ...")
        return
    res_abc = analysis.compute_overlap(result["overlap_mask"], mask_c)
    res_anb_c = analysis.compute_overlap(result["non_overlap_mask"], mask_c)
    self._add_overlap_chart(f"{name_a}∩{name_b}", name_c, res_abc)
    self._add_overlap_chart(f"{name_a}\\{name_b}", name_c, res_anb_c)
    if self._show_overlap_layer.isChecked():
        channels_abc = self._region_channels(name_a, mask_a, name_b, mask_b, name_c, mask_c)
        if res_abc["n_overlap"] > 0:
            self._add_region_mask_layer(
                f"{name_a} & {name_b} & {name_c} Overlap Mask",
                res_abc["overlap_mask"], channels_abc)
        if res_anb_c["n_overlap"] > 0:
            self._add_region_mask_layer(
                f"({name_a} \\ {name_b}) & {name_c} Overlap Mask",
                res_anb_c["overlap_mask"], channels_abc)
```

- `channels_abc` is built from the A/B/C selected layers' `metadata["spot_intensity"]`, including
  only channels whose table is not None. (The existing A/B region layers keep using just A+B; a
  small helper or inline build is fine — keep the A+B build and add a 3-channel build.)
- The C step runs **before** `_maybe_add_zonal_chart` / `_maybe_add_intensity_histogram` so the C
  charts appear right after the A-vs-B chart.

### 3. No changes needed

`compute_overlap`, `compute_region_spot_table`, `spot_table_to_rows`, `_add_region_mask_layer`,
`_add_overlap_chart`, and the export control are reused unchanged.

## Error Handling

- C "(none)"/empty → C step skipped entirely.
- C shape mismatch with A → status message, no C step (mirrors the A/B shape guard).
- C selected but no spot metadata on any channel → mask layers still added; their spot tables
  include whichever channels have metadata (possibly none → no table attached, export reports
  "no per-spot data").

## Consequences (intended)

- `(A\B)∩C` underlying-points CSV contains only A and C spots (B has no voxels in A\B).
- Any-voxel membership means a spot straddling a boundary can appear in multiple region CSVs.

## Testing

1. C combo exists and its first item is `"(none)"`; with C = "(none)", Compute Overlap adds no
   C layers and the chart count matches the no-C case.
2. With A, B, C selected (all carrying `spot_intensity`): the two C-intersection mask layers are
   added; `(A∩B)∩C` table has channels {A,B,C} as applicable; `(A\B)∩C` table excludes B; the
   charts layout count increases by 2 versus the no-C case.
3. C shape mismatch → status message contains "shape", no C layers added.

## Out of Scope

- Changing the existing A∩B / A\B layers or the B intensity histogram.
- A 4th channel, or intersections other than the two specified.
- Centroid-based membership (any-voxel only).

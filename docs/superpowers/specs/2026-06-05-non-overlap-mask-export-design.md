# Non-Overlap Mask Layer for Export — Design

**Date:** 2026-06-05
**Status:** Approved

## Problem

`_run_overlap_analysis` adds an **overlap** mask (A∩B) as a napari Labels layer, which the user
can save/export via napari's native layer save. The complementary **non-overlap** region is
already computed implicitly but never surfaced as a layer, so it can't be exported the same way.

## Goal

Add an exportable **non-overlap** mask layer alongside the existing overlap mask.

## Definition (locked)

`non_overlap_mask = (A > 0) & ~(B > 0)` — voxels in **A that do not overlap B** (A \ B).

Note: this is the opposite side from the intensity histogram's non-overlap concept (B \ A). That
is intentional per the user's explicit request; the two features are not meant to match here.

## Changes

### 1. `_analysis.py` — `compute_overlap`

Add two keys to the returned dict (all existing keys unchanged):

- `"non_overlap_mask"`: `(bool_a & ~bool_b).astype(np.uint8)`
- `"n_non_overlap"`: `int((bool_a & ~bool_b).sum())` (equals `n_a - n_overlap`)

### 2. `_widget.py` — `_run_overlap_analysis`

- Relabel the existing checkbox `_show_overlap_layer` text to
  **"Show overlap & non-overlap mask layers"** (same widget/attribute, gate unchanged).
- Under the same `if self._show_overlap_layer.isChecked()` block, after adding the overlap layer,
  also add the non-overlap layer when `result["n_non_overlap"] > 0`:
  - Layer name: `f"{name_a} \\ {name_b} Non-overlap Mask"`.
  - Same remove-existing-then-add pattern used for the overlap layer.
  - `opacity=0.5`, consistent with the overlap layer.

## Error Handling

- `n_non_overlap == 0` → no non-overlap layer added (mirrors the overlap-layer guard).
- Shape mismatch is already guarded earlier in `_run_overlap_analysis`.

## Testing

1. `compute_overlap`: for known A/B masks, `non_overlap_mask == A & ~B` and
   `n_non_overlap == n_a - n_overlap`.
2. Widget: with the checkbox on and a non-empty A\B, the `"{A} \\ {B} Non-overlap Mask"` layer is
   added; with the checkbox off, it is not.

## Out of Scope

- B-only (B\A) or symmetric-difference masks (the user chose A\B only).
- Changing the existing overlap layer or the intensity histogram.

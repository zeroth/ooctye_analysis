# Auto-Computed Perinuclear Region — Design

**Date:** 2026-06-05
**Status:** Approved

## Problem

The Regions group currently has three **manually drawn** spherical regions: `oocyte`,
`perinuclear`, and `exclude`. In practice the `exclude` region *is* the nucleus, and the
perinuclear region should not be drawn by hand — it is a fixed geometric shell between the
nucleus and the oocyte. Drawing it manually is error-prone and inconsistent.

## Goal

- Treat the existing "exclude" region as the **nucleus** (rename throughout for clarity).
- **Auto-compute** the perinuclear region from the nucleus and oocyte geometry.
- Remove the manual perinuclear drawing UI (combo + "New" button), keeping only a "show"
  toggle for its surface overlay.

## Geometry

Given nucleus radius `R_n` and oocyte radius `R_o` (physical units), the perinuclear
boundary radius is:

```
R_p = R_n + frac * (R_o - R_n)
```

where `frac` defaults to `0.40` (the diagram's "A is 40% of B", with B = the full
nucleus-surface → oocyte-surface gap). The perinuclear sphere is **centered on the nucleus
center**, using the nucleus scale.

The perinuclear *zone* used in analysis remains the shell:

```
peri_zone = perinuclear_sphere & ~nucleus & oocyte
rest_zone = oocyte & ~perinuclear_sphere
```

## Changes

### 1. Geometry primitive — `_regions.py`

Add a pure function:

```python
def compute_perinuclear(nucleus: Sphere, oocyte: Sphere, frac: float) -> Sphere:
    r_p = nucleus.radius_physical + frac * (oocyte.radius_physical - nucleus.radius_physical)
    return Sphere(center_px=nucleus.center_px, radius_physical=r_p, scale=nucleus.scale)
```

No clamping of `frac` here — the widget supplies a validated value (0–1).

### 2. Region model — `_widget.py`

- `REGION_DESCRIPTORS` lists only the two **drawn** regions:
  `("oocyte", "Oocyte", "cyan")` and `("nucleus", "Nucleus", "yellow")`
  (renamed from `exclude` / "Exclude").
- Perinuclear (magenta) is handled as a **computed** region: it keeps a "show" checkbox
  and surface visualization, but has no combo or "New" button.
- `_get_all_region_spheres` returns `{"oocyte", "nucleus", "perinuclear"}` where
  `perinuclear` is computed via `compute_perinuclear` (or `None` when nucleus or oocyte
  is missing).

### 3. UI — Regions group (`_widget.py`)

- Iterate `REGION_DESCRIPTORS` to build the drawn-region rows (combo + New + show), as today.
- Add a separate row for perinuclear: a "Perinuclear:" label, a "show" checkbox, and a
  `QDoubleSpinBox` "fraction (%)" (range 0–100, default 40, suffix "%"). The spinbox value
  divided by 100 is `frac`.
- Containment validation (`_update_containment_status`) collapses to:
  - warn if `nucleus` is not inside `oocyte`;
  - warn if `R_o <= R_n` (perinuclear undefined / inverted).
  Perinuclear nesting is guaranteed by construction (0 ≤ frac ≤ 1, concentric with nucleus).

### 4. Detection worker — `_workers.py`

- Rename the region dict key `"exclude"` → `"nucleus"`. Behavior is unchanged: spots inside
  the nucleus are filtered out (`keep="outside"`) and the mask is zeroed inside the nucleus
  (`mode="zero_inside"`). `n_excluded` count semantics unchanged.

### 5. Status text — `_widget.py`

- In `_on_detection_finished`, read `region_spheres["nucleus"]` instead of `["exclude"]`;
  status string wording updated to reference the nucleus.

### 6. Analysis — `_analysis.py`

- Rename `compute_zonal_voxels` parameter `exclude_mask` → `nucleus_mask`. Internal zonal
  logic is identical (`peri_zone = perinuclear_mask & ~nucleus_mask & oocyte_mask`).
- `_maybe_add_zonal_chart` now sources the perinuclear mask from the **computed** perinuclear
  sphere (already returned by `_get_all_region_spheres`), and the nucleus mask from the
  `nucleus` sphere.

### 7. Tests — `_tests/test_widget.py`

- Update `REGION_KEYS` and the layer-name map: drop the drawn perinuclear; rename
  `exclude` → `nucleus` ("Nucleus line").
- Add unit tests for `compute_perinuclear`:
  - 40% case: `R_n=10, R_o=20, frac=0.4 → R_p=14`;
  - edge cases `frac=0 → R_p=R_n`, `frac=1 → R_p=R_o`;
  - center equals nucleus center even when nucleus and oocyte are off-center.
- Update zonal-voxel tests to pass a computed perinuclear mask and the renamed
  `nucleus_mask` argument.

## Out of Scope

- Non-spherical / anisotropic perinuclear shapes.
- Per-direction (non-concentric) gap measurement — the shell is concentric with the nucleus.
- Changing the spots/mask exclusion behavior (still excludes the nucleus interior).

## Data Flow Summary

The only conceptual shift: `spheres["perinuclear"]` is now **derived** from nucleus + oocyte
+ fraction rather than read from a Shapes layer. Everything downstream (zonal chart, mask
clipping, surface overlay) consumes the same `Sphere` object as before.

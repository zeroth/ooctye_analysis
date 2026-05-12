# Three Regions (Oocyte / Perinuclear / Exclude) — Design

**Date:** 2026-05-11
**Status:** Draft — pending implementation

## Goal

Extend the plugin from a single Exclude region to three nested spherical regions:

- **Oocyte** — outer bound of the analyzable volume.
- **Perinuclear** — middle shell around the nucleus.
- **Exclude** — innermost, what we already have (the nucleus).

The three regions are concentric in intent but each is defined independently by the user, so the UI validates the expected nesting and warns when it is violated.

## Requirements

1. User can draw three independent regions, each with the same line-on-Shapes-layer convention used today (midpoint = center, half physical length = radius).
2. Each region is stored on its own Shapes layer with a distinct color.
3. All three regions are optional. Missing regions produce graceful fallbacks rather than errors.
4. Containment validation: `exclude ⊆ perinuclear ⊆ oocyte`. Violations show a red warning label but do not block detection or analysis.
5. Detection clips both spots and mask to `oocyte − exclude` (i.e., drops anything outside the oocyte; zeros anything inside the exclude).
6. Overlap analysis (existing A-vs-B chart) is unchanged in math, but operates on the clipped masks.
7. New per-channel zonal chart: voxel counts of each channel's mask in **(perinuclear − exclude)** vs **(oocyte − perinuclear)**. Skipped when oocyte or perinuclear is not drawn.

## Non-goals

- Non-spherical regions (ellipsoid / freehand). Future work; sphere parity with the current exclude region is sufficient for now.
- Hard-blocking validation. The biologist user knows their data better than the plugin.
- Restructuring the analysis tab beyond adding the new chart.

## Architecture

### New module: `_regions.py`

A small, focused module encapsulating sphere geometry and per-region operations. Everything is pure functions / dataclasses — no Qt, no napari imports — so it is unit-testable in isolation.

```python
@dataclass(frozen=True)
class Sphere:
    center_px: np.ndarray  # shape (ndim,)
    radius_physical: float
    scale: np.ndarray      # shape (ndim,), µm/px per axis

def sphere_from_line(line: np.ndarray, scale: np.ndarray,
                     viewer_point: np.ndarray | None = None,
                     ndim: int = 3) -> Sphere | None:
    """Build a Sphere from a 2-vertex line on a Shapes layer.
    viewer_point fills in missing leading dims (e.g. a 2D line in a 3D view)."""

def contains_sphere(outer: Sphere, inner: Sphere, tol: float = 0.0) -> bool:
    """True if every point of inner is inside outer (physical distance check)."""

def apply_sphere_to_mask(mask: np.ndarray, sphere: Sphere, *,
                          mode: Literal["zero_inside", "zero_outside"]) -> None:
    """In-place: zero voxels inside (exclude) or outside (oocyte) the sphere."""

def filter_spots(spots: np.ndarray, sphere: Sphere, *,
                  keep: Literal["outside", "inside"]) -> np.ndarray:
    """Return spots whose physical distance to center is outside/inside radius."""

def build_sphere_mesh(sphere: Sphere) -> tuple[np.ndarray, np.ndarray]:
    """Vertices/faces for napari add_surface (current ellipsoid mesh logic)."""

def sphere_to_mask(sphere: Sphere, shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of voxels inside the sphere — used by zonal analysis."""
```

The existing helpers in `_segmentation.py` (`filter_spots_by_sphere`, `apply_exclusion_to_mask`, `build_exclusion_sphere_mesh`) become thin wrappers over `_regions` or are removed and callers updated. No behavior change for the exclude region.

### `_widget.py` refactor

Replace the single "Exclude Region" `QGroupBox` with a "Regions" group containing one row per region. Each row is driven by a small descriptor:

```python
REGIONS = [
    ("oocyte",      "Oocyte",       "cyan"),
    ("perinuclear", "Perinuclear",  "magenta"),
    ("exclude",     "Exclude",      "yellow"),
]
```

For each `(key, label, color)`:
- `self._region_combos[key]` — Shapes-layer combo
- `self._region_show[key]`  — show-mesh checkbox
- `New` button creates a Shapes layer named `f"{label} line"` with the matching edge color
- Hidden via the existing `_refresh_image_layers` (all Shapes layers appear in every combo)

The shared **Scale (µm/px)** Z/Y/X spinboxes stay where they are; they are a property of the image, not any single region.

A single `self._region_status: QLabel` below the group shows containment warnings (red text via stylesheet).

### Detection pipeline changes

In `_run_detection`:
- Build `spheres: dict[str, Sphere | None]` for all three regions.
- Pass the full dict to `PredictWorker` (replacing the current single `exclusion=` arg).

In `_workers.PredictWorker`:
- After spotiflow returns spots: drop spots outside oocyte (if present), then drop spots inside exclude (current behavior).
- Mask handling moves out of `_on_detection_finished` and into the worker: apply `zero_outside(oocyte)` then `zero_inside(exclude)`.
- Perinuclear has no detection-time effect.

In `_on_detection_finished`: render mesh overlays for whichever regions have `show` checked.

### Analysis changes

`_analysis.py`:
- `compute_overlap(mask_a, mask_b)` — unchanged.
- New: `compute_zonal_voxels(mask, oocyte_mask, perinuclear_mask, exclude_mask) -> dict` returning voxel counts and percentages for `peri − exclude` and `oocyte − peri` zones.
- New: `create_zonal_figure(channel_names: list[str], zonal_results: list[dict])` — grouped bar chart, one group per channel, two bars per group.

`_widget._run_overlap_analysis`:
- Compute the existing A-vs-B chart.
- If both `oocyte` and `perinuclear` are drawn, voxelize them once via `_regions.sphere_to_mask(sphere, shape)` (new helper) and call `compute_zonal_voxels` for each Labels layer in `_mask_a_combo` and `_mask_b_combo`. Append a single zonal figure with both channels.

### Containment check

Triggered on `Detect Spots` and `Compute Overlap`. Uses `contains_sphere` from `_regions`:

```python
issues = []
if exclude and perinuclear and not contains_sphere(perinuclear, exclude):
    issues.append("exclude is not inside perinuclear")
if perinuclear and oocyte and not contains_sphere(oocyte, perinuclear):
    issues.append("perinuclear is not inside oocyte")
if exclude and oocyte and not contains_sphere(oocyte, exclude):
    issues.append("exclude is not inside oocyte")
```

Joined issues are written to `_region_status` with `color: red`. Empty issues → clear the label.

## Data flow

```
Shapes layers (3) ──► sphere_from_line ──► dict[str, Sphere]
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                  PredictWorker         containment check    overlay meshes
                          │                   │
        clip spots & mask │             warn-only label
                          ▼
                  Labels layer (clipped)
                          │
                          ▼
                  compute_overlap + compute_zonal_voxels
                          │
                          ▼
                  matplotlib charts in Analysis tab
```

## Backwards compatibility

The single existing Shapes layer combo box is replaced. Old saved napari sessions that referenced "Exclusion line" still work — the new Exclude row picks up any layer named that way via the standard combo refresh.

`_segmentation.py`'s public helpers are kept (re-exported from `_regions` if needed) so external tests do not break.

## Testing

Unit tests in `_tests/`:

- `test_regions.py`
  - `sphere_from_line` with 2D and 3D inputs, with and without `viewer_point` padding.
  - `contains_sphere`: fully inside, edge-touching (tol behavior), partially out, identical, disjoint.
  - `apply_sphere_to_mask` for both modes (anisotropic scale).
  - `filter_spots` both directions.
- `test_analysis.py`
  - `compute_zonal_voxels` with hand-built masks where the answer is obvious.
- `test_widget.py` (extend existing)
  - Adding three regions creates three Shapes layers with the expected names and colors.
  - Drawing geometrically invalid regions produces the expected warning text.
  - Detection with oocyte clipping removes voxels outside the oocyte.

## Open questions

None at design time. UI labels and exact bar-chart styling can be iterated during implementation.

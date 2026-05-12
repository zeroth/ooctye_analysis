# Three Regions (Oocyte / Perinuclear / Exclude) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the existing single Exclude region into three optional, nested spherical regions (Oocyte contains Perinuclear contains Exclude) with containment validation, oocyte clipping at detection time, and a new per-channel zonal voxel-count chart.

**Architecture:** A new `_regions.py` module centralizes sphere geometry (Sphere dataclass + pure helpers). `_segmentation.py` keeps its public API but delegates to `_regions`. The widget grows a three-row "Regions" group that replaces the single exclude-only block. The PredictWorker clips spots and mask to (oocyte minus exclude). The Analysis tab gets a second chart per Compute-Overlap run.

**Tech Stack:** Python 3, NumPy, Qt (via qtpy), napari (Image/Shapes/Labels/Surface layers), matplotlib (backend_qtagg), pytest, napari pytest fixtures (`make_napari_viewer`).

**Spec:** [docs/superpowers/specs/2026-05-11-three-regions-design.md](../specs/2026-05-11-three-regions-design.md)

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/napari_ooctyle_analysis/_regions.py` | **Create** | `Sphere` dataclass + pure geometry helpers (no Qt, no napari). |
| `src/napari_ooctyle_analysis/_tests/test_regions.py` | **Create** | Unit tests for everything in `_regions.py`. |
| `src/napari_ooctyle_analysis/_segmentation.py` | **Modify** | Keep public function names; rewrite bodies as thin wrappers over `_regions`. |
| `src/napari_ooctyle_analysis/_workers.py` | **Modify** | `PredictWorker` accepts a `dict[str, Sphere or None]` of regions; clips mask + spots inside `run()`. |
| `src/napari_ooctyle_analysis/_widget.py` | **Modify** | Replace single Exclude group with three-row Regions group; add containment status label; wire dict to worker; add zonal chart call. |
| `src/napari_ooctyle_analysis/_analysis.py` | **Modify** | Add `compute_zonal_voxels` and `create_zonal_figure`. |
| `src/napari_ooctyle_analysis/_tests/test_widget.py` | **Modify** | Update tests using renamed widget attributes; add containment + zonal + clipping tests. |

---

## Task 1: `_regions.py` — `Sphere` dataclass and `sphere_from_line`

**Files:**
- Create: `src/napari_ooctyle_analysis/_regions.py`
- Create: `src/napari_ooctyle_analysis/_tests/test_regions.py`

- [ ] **Step 1.1: Write the failing tests for `Sphere` and `sphere_from_line`**

Create `src/napari_ooctyle_analysis/_tests/test_regions.py`:

```python
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
        with pytest.raises(Exception):
            s.radius_physical = 9.0  # type: ignore[misc]


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
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py -v`
Expected: FAIL with `ImportError` (module does not exist yet).

- [ ] **Step 1.3: Implement `_regions.py` with `Sphere` and `sphere_from_line`**

Create `src/napari_ooctyle_analysis/_regions.py`:

```python
"""Pure geometry primitives for spherical regions of interest.

This module has no Qt and no napari imports - it is pure NumPy so it can be
unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sphere:
    """A sphere in pixel-space center / physical-space radius."""
    center_px: np.ndarray
    radius_physical: float
    scale: np.ndarray


def sphere_from_line(
    line: np.ndarray,
    scale: np.ndarray,
    viewer_point: np.ndarray | None = None,
    ndim: int = 3,
) -> Sphere | None:
    """Build a Sphere from a 2-vertex line."""
    line = np.asarray(line, dtype=np.float64)
    if line.shape[0] != 2:
        return None

    p1, p2 = line[0], line[1]
    if len(p1) < ndim:
        missing = ndim - len(p1)
        if viewer_point is None:
            viewer_point = np.zeros(missing, dtype=np.float64)
        p1 = np.concatenate([viewer_point[:missing], p1])
        p2 = np.concatenate([viewer_point[:missing], p2])

    center_px = (p1 + p2) / 2.0
    delta_physical = (p2 - p1) * scale
    radius_physical = float(np.linalg.norm(delta_physical)) / 2.0
    return Sphere(center_px=center_px, radius_physical=radius_physical, scale=scale)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py -v`
Expected: 6 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_regions.py src/napari_ooctyle_analysis/_tests/test_regions.py
git commit -m "feat(_regions): add Sphere dataclass and sphere_from_line helper"
```

---

## Task 2: `_regions.py` — `contains_sphere`

**Files:**
- Modify: `src/napari_ooctyle_analysis/_regions.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_regions.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `src/napari_ooctyle_analysis/_tests/test_regions.py`:

```python
from napari_ooctyle_analysis._regions import contains_sphere


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
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py::TestContainsSphere -v`
Expected: FAIL with `ImportError: cannot import name 'contains_sphere'`.

- [ ] **Step 2.3: Implement `contains_sphere`**

Append to `src/napari_ooctyle_analysis/_regions.py`:

```python
def contains_sphere(outer: Sphere, inner: Sphere) -> bool:
    """True iff every point of inner lies inside (or on the surface of) outer.

    Physical-space distance check (anisotropic-scale aware), using the outer
    sphere's scale (both spheres are built from the same image scale).
    """
    delta_px = inner.center_px - outer.center_px
    distance_physical = float(np.linalg.norm(delta_px * outer.scale))
    return distance_physical + inner.radius_physical <= outer.radius_physical + 1e-9
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py::TestContainsSphere -v`
Expected: 5 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_regions.py src/napari_ooctyle_analysis/_tests/test_regions.py
git commit -m "feat(_regions): add contains_sphere containment check"
```

---

## Task 3: `_regions.py` — mask + spot filtering helpers

**Files:**
- Modify: `src/napari_ooctyle_analysis/_regions.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_regions.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `src/napari_ooctyle_analysis/_tests/test_regions.py`:

```python
from napari_ooctyle_analysis._regions import (
    apply_sphere_to_mask,
    filter_spots,
    sphere_to_mask,
)


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
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py -v`
Expected: FAIL with `ImportError` for `apply_sphere_to_mask`, `filter_spots`, `sphere_to_mask`.

- [ ] **Step 3.3: Implement the three helpers**

Append to `src/napari_ooctyle_analysis/_regions.py`:

```python
from typing import Literal


def _distance_sq_physical(
    shape: tuple[int, ...],
    center_px: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Per-voxel squared physical distance from center_px."""
    grids = np.indices(shape, dtype=np.float64)
    dist_sq = np.zeros(shape, dtype=np.float64)
    for d in range(len(shape)):
        dist_sq += ((grids[d] - center_px[d]) * scale[d]) ** 2
    return dist_sq


def apply_sphere_to_mask(
    mask: np.ndarray,
    sphere: Sphere,
    *,
    mode: Literal["zero_inside", "zero_outside"],
) -> None:
    """In-place: zero voxels either inside or outside the sphere."""
    dist_sq = _distance_sq_physical(mask.shape, sphere.center_px, sphere.scale)
    r2 = sphere.radius_physical ** 2
    if mode == "zero_inside":
        mask[dist_sq <= r2] = 0
    elif mode == "zero_outside":
        mask[dist_sq > r2] = 0
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def filter_spots(
    spots: np.ndarray,
    sphere: Sphere,
    *,
    keep: Literal["outside", "inside"],
) -> np.ndarray:
    """Return spots whose physical distance from sphere center is outside / inside."""
    if len(spots) == 0:
        return spots
    delta_physical = (spots - sphere.center_px) * sphere.scale
    distances = np.linalg.norm(delta_physical, axis=1)
    if keep == "outside":
        return spots[distances > sphere.radius_physical]
    if keep == "inside":
        return spots[distances <= sphere.radius_physical]
    raise ValueError(f"Unknown keep value: {keep!r}")


def sphere_to_mask(sphere: Sphere, shape: tuple[int, ...]) -> np.ndarray:
    """Return a boolean mask True for voxels inside the sphere."""
    dist_sq = _distance_sq_physical(shape, sphere.center_px, sphere.scale)
    return dist_sq <= sphere.radius_physical ** 2
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py -v`
Expected: all PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_regions.py src/napari_ooctyle_analysis/_tests/test_regions.py
git commit -m "feat(_regions): add apply_sphere_to_mask, filter_spots, sphere_to_mask"
```

---

## Task 4: `_regions.py` — `build_sphere_mesh`

Port the existing ellipsoid mesh builder from `_segmentation.py`.

**Files:**
- Modify: `src/napari_ooctyle_analysis/_regions.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_regions.py`

- [ ] **Step 4.1: Write the failing test**

Append to `src/napari_ooctyle_analysis/_tests/test_regions.py`:

```python
from napari_ooctyle_analysis._regions import build_sphere_mesh


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
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py::TestBuildSphereMesh -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4.3: Implement `build_sphere_mesh`**

Append to `src/napari_ooctyle_analysis/_regions.py`:

```python
def build_sphere_mesh(sphere: Sphere) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for a triangulated ellipsoid surface."""
    radii_px = sphere.radius_physical / sphere.scale

    n_phi, n_theta = 30, 30
    phi = np.linspace(0, np.pi, n_phi)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)

    z = radii_px[0] * np.cos(phi) + sphere.center_px[0]
    y = radii_px[1] * np.sin(phi) * np.cos(theta) + sphere.center_px[1]
    x = radii_px[2] * np.sin(phi) * np.sin(theta) + sphere.center_px[2]

    vertices = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1)

    faces = []
    for i in range(n_theta):
        for j in range(n_phi - 1):
            idx = i * n_phi + j
            next_i = ((i + 1) % n_theta) * n_phi + j
            faces.append([idx, next_i, idx + 1])
            faces.append([next_i, next_i + 1, idx + 1])
    faces = np.array(faces)
    return vertices, faces
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_regions.py -v`
Expected: all PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_regions.py src/napari_ooctyle_analysis/_tests/test_regions.py
git commit -m "feat(_regions): port ellipsoid mesh builder as build_sphere_mesh"
```

---

## Task 5: Rewire `_segmentation.py` to delegate to `_regions`

**Files:**
- Modify: `src/napari_ooctyle_analysis/_segmentation.py`

- [ ] **Step 5.1: Run the full test suite to confirm it passes before refactor**

Run: `pytest src/napari_ooctyle_analysis/ -v`
Expected: all pre-existing tests pass.

- [ ] **Step 5.2: Rewrite the three sphere helpers as thin wrappers**

Replace the bodies of `filter_spots_by_sphere`, `apply_exclusion_to_mask`, and `build_exclusion_sphere_mesh` in `src/napari_ooctyle_analysis/_segmentation.py` with:

```python
from napari_ooctyle_analysis._regions import (
    Sphere,
    apply_sphere_to_mask,
    build_sphere_mesh,
    filter_spots,
)


def filter_spots_by_sphere(
    spots: np.ndarray,
    center_px: np.ndarray,
    radius_physical: float,
    scale: np.ndarray,
) -> np.ndarray:
    """Remove spots inside the exclusion sphere (in physical space)."""
    if len(spots) == 0:
        return spots
    sphere = Sphere(
        center_px=np.asarray(center_px, dtype=np.float64),
        radius_physical=float(radius_physical),
        scale=np.asarray(scale, dtype=np.float64),
    )
    return filter_spots(spots, sphere, keep="outside")


def apply_exclusion_to_mask(
    mask: np.ndarray,
    center_px: np.ndarray,
    radius_physical: float,
    scale: np.ndarray,
) -> None:
    """Zero out voxels inside the exclusion sphere (in-place)."""
    sphere = Sphere(
        center_px=np.asarray(center_px, dtype=np.float64),
        radius_physical=float(radius_physical),
        scale=np.asarray(scale, dtype=np.float64),
    )
    apply_sphere_to_mask(mask, sphere, mode="zero_inside")


def build_exclusion_sphere_mesh(
    center_px: np.ndarray,
    radius_physical: float,
    scale: np.ndarray,
) -> tuple:
    """Generate ellipsoid surface mesh (delegates to _regions.build_sphere_mesh)."""
    sphere = Sphere(
        center_px=np.asarray(center_px, dtype=np.float64),
        radius_physical=float(radius_physical),
        scale=np.asarray(scale, dtype=np.float64),
    )
    return build_sphere_mesh(sphere)
```

Note: keep `reorder_to_zyx`, `generate_save_path`, and `load_data_from_dirs` unchanged.

- [ ] **Step 5.3: Run the full test suite again**

Run: `pytest src/napari_ooctyle_analysis/ -v`
Expected: every previously passing test still passes (especially `TestFilterSpotsBySphere`).

- [ ] **Step 5.4: Commit**

```bash
git add src/napari_ooctyle_analysis/_segmentation.py
git commit -m "refactor(_segmentation): delegate sphere helpers to _regions module"
```

---

## Task 6: Widget — refactor exclude UI block into three-region block

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 6.1: Write failing tests for the new widget surface**

In `src/napari_ooctyle_analysis/_tests/test_widget.py`, delete the old single-region tests `test_add_exclusion_shapes_layer`, `test_exclusion_sphere_from_line`, `test_exclusion_sphere_with_anisotropic_scale`, `test_exclusion_sphere_2d_line_on_3d`, and `test_exclusion_no_shapes` (they will be replaced). Then append:

```python
# ------------------------------------------------------------------
# Three-region tests
# ------------------------------------------------------------------


class TestRegionWidgets:
    REGION_KEYS = ("oocyte", "perinuclear", "exclude")
    LAYER_NAMES = {
        "oocyte": "Oocyte line",
        "perinuclear": "Perinuclear line",
        "exclude": "Exclusion line",
    }

    def test_three_combos_exist(self, make_napari_viewer):
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
            name="Exclusion line",
        )
        widget._refresh_image_layers()
        widget._region_combos["exclude"].setCurrentText("Exclusion line")
        s = widget._get_region_sphere("exclude", ndim=3)
        assert s is not None
        np.testing.assert_allclose(s.center_px, [5.0, 0.0, 10.0])
        assert abs(s.radius_physical - 10.0) < 1e-6

    def test_region_sphere_with_anisotropic_scale(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        viewer.add_shapes(
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]])],
            shape_type="line",
            name="Exclusion line",
        )
        widget._refresh_image_layers()
        widget._region_combos["exclude"].setCurrentText("Exclusion line")
        widget._scale_x.setValue(0.5)
        widget._scale_y.setValue(1.0)
        widget._scale_z.setValue(2.0)
        s = widget._get_region_sphere("exclude", ndim=3)
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
            name="Exclusion line",
        )
        widget._refresh_image_layers()
        widget._region_combos["exclude"].setCurrentText("Exclusion line")
        s = widget._get_region_sphere("exclude", ndim=3)
        assert s is not None
        np.testing.assert_allclose(s.center_px, [10.0, 0.0, 10.0])
        assert abs(s.radius_physical - 10.0) < 1e-6
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestRegionWidgets -v`
Expected: FAIL — attributes `_region_combos`, `_region_show`, `_add_region_buttons`, `_add_region_shapes_layer`, `_get_region_sphere` do not exist yet.

- [ ] **Step 6.3: Refactor the widget**

In `src/napari_ooctyle_analysis/_widget.py`:

(a) Add module-level constant after the existing imports:

```python
from napari_ooctyle_analysis import _regions as regions

REGION_DESCRIPTORS = [
    # key,           label,           edge_color
    ("oocyte",       "Oocyte",        "cyan"),
    ("perinuclear",  "Perinuclear",   "magenta"),
    ("exclude",      "Exclude",       "yellow"),
]
```

(b) In `__init__`, replace `self._shapes_combo = QComboBox()` with:

```python
self._region_combos: dict[str, QComboBox] = {
    key: QComboBox() for key, _, _ in REGION_DESCRIPTORS
}
```

(c) Replace the `# --- Exclude Region ---` block in `_build_segmentation_tab` (the entire `excl_group` block) with:

```python
regions_group = QGroupBox("Regions")
regions_layout = QVBoxLayout()
regions_group.setLayout(regions_layout)

regions_layout.addWidget(QLabel(
    "Draw a line on each region's Shapes layer.\n"
    "Line center = sphere center, half its physical length = radius."
))

self._region_show: dict[str, QCheckBox] = {}
self._add_region_buttons: dict[str, QPushButton] = {}
for key, label, color in REGION_DESCRIPTORS:
    row = QHBoxLayout()
    row.addWidget(QLabel(f"{label}:"))
    row.addWidget(self._region_combos[key])
    btn = QPushButton("New")
    btn.setToolTip(f"Add a new Shapes layer for the {label.lower()} region")
    btn.clicked.connect(
        lambda checked=False, k=key: self._add_region_shapes_layer(k)
    )
    self._add_region_buttons[key] = btn
    row.addWidget(btn)
    show = QCheckBox("show")
    show.setChecked(True)
    self._region_show[key] = show
    row.addWidget(show)
    regions_layout.addLayout(row)

scale_row = QHBoxLayout()
scale_row.addWidget(QLabel("Scale (um/px):"))
for axis_label, attr in [("Z:", "_scale_z"), ("Y:", "_scale_y"), ("X:", "_scale_x")]:
    scale_row.addWidget(QLabel(axis_label))
    sb = QDoubleSpinBox()
    sb.setRange(0.001, 1000.0)
    sb.setDecimals(3)
    sb.setValue(1.0)
    setattr(self, attr, sb)
    scale_row.addWidget(sb)
regions_layout.addLayout(scale_row)

self._region_status = QLabel("")
self._region_status.setWordWrap(True)
regions_layout.addWidget(self._region_status)

layout.addWidget(regions_group)
```

(d) Replace `_add_exclusion_shapes_layer`, `_get_exclusion_sphere`, and `_visualize_exclusion_sphere` with:

```python
def _add_region_shapes_layer(self, key: str):
    label_map = {k: (lbl, col) for k, lbl, col in REGION_DESCRIPTORS}
    label, color = label_map[key]
    layer_name = "Exclusion line" if key == "exclude" else f"{label} line"
    layer = self.viewer.add_shapes(
        name=layer_name, shape_type="line",
        edge_color=color, edge_width=2,
    )
    layer.mode = "add_line"
    self._refresh_image_layers()
    idx = self._region_combos[key].findText(layer.name)
    if idx >= 0:
        self._region_combos[key].setCurrentIndex(idx)

def _get_region_sphere(self, key: str, ndim: int = 3) -> regions.Sphere | None:
    layer_name = self._region_combos[key].currentText()
    if not layer_name:
        return None
    try:
        layer = self.viewer.layers[layer_name]
    except KeyError:
        return None
    if not isinstance(layer, napari.layers.Shapes) or len(layer.data) == 0:
        return None
    line = np.asarray(layer.data[0], dtype=np.float64)
    viewer_point = None
    if line.shape[1] < ndim:
        missing = ndim - line.shape[1]
        viewer_point = np.array(self.viewer.dims.point[:missing], dtype=np.float64)
    return regions.sphere_from_line(
        line, self._get_scale_zyx(), viewer_point=viewer_point, ndim=ndim,
    )

def _visualize_region_sphere(self, key: str, sphere: regions.Sphere):
    label_map = {k: (lbl, col) for k, lbl, col in REGION_DESCRIPTORS}
    label, color = label_map[key]
    surface_name = f"{label} sphere"
    vertices, faces = regions.build_sphere_mesh(sphere)
    for layer in list(self.viewer.layers):
        if layer.name == surface_name:
            self.viewer.layers.remove(layer)
    self.viewer.add_surface(
        (vertices, faces), name=surface_name,
        colormap=color, opacity=0.15,
    )

def _get_all_region_spheres(self, ndim: int = 3) -> dict[str, regions.Sphere | None]:
    return {key: self._get_region_sphere(key, ndim=ndim) for key, _, _ in REGION_DESCRIPTORS}
```

(e) In `_refresh_image_layers`, replace the combo iteration to include the three region combos. Replace:

```python
prev = {
    combo: combo.currentText()
    for combo in [
        self._image_combo, self._shapes_combo,
        self._mask_a_combo, self._mask_b_combo,
    ]
}

self._image_combo.clear()
self._shapes_combo.clear()
self._mask_a_combo.clear()
self._mask_b_combo.clear()

for layer in self.viewer.layers:
    if isinstance(layer, napari.layers.Image):
        self._image_combo.addItem(layer.name)
    elif isinstance(layer, napari.layers.Shapes):
        self._shapes_combo.addItem(layer.name)
    elif isinstance(layer, napari.layers.Labels):
        self._mask_a_combo.addItem(layer.name)
        self._mask_b_combo.addItem(layer.name)
```

with:

```python
all_combos = [
    self._image_combo, self._mask_a_combo, self._mask_b_combo,
    *self._region_combos.values(),
]
prev = {combo: combo.currentText() for combo in all_combos}
for combo in all_combos:
    combo.clear()

for layer in self.viewer.layers:
    if isinstance(layer, napari.layers.Image):
        self._image_combo.addItem(layer.name)
    elif isinstance(layer, napari.layers.Shapes):
        for combo in self._region_combos.values():
            combo.addItem(layer.name)
    elif isinstance(layer, napari.layers.Labels):
        self._mask_a_combo.addItem(layer.name)
        self._mask_b_combo.addItem(layer.name)
```

(f) In `_run_detection`, replace:

```python
exclusion = self._get_exclusion_sphere(ndim=image.ndim)
```

with:

```python
region_spheres = self._get_all_region_spheres(ndim=image.ndim)
```

Until Task 8 lands, the worker still uses the legacy `exclusion=` arg, so pass it derived from the exclude sphere:

```python
self._predict_worker = PredictWorker(
    model=self._model, image=image,
    prob_thresh=self._prob_thresh.value(),
    min_distance=self._min_distance.value(),
    exclude_border=self._exclude_border.isChecked(),
    device=self._device_string(),
    exclusion=(
        (region_spheres["exclude"].center_px,
         region_spheres["exclude"].radius_physical,
         region_spheres["exclude"].scale)
        if region_spheres["exclude"] is not None else None
    ),
)
```

(g) Replace the exclusion-handling block in `_on_detection_finished` (from `exclusion = self._get_exclusion_sphere(...)` through the `if self._show_sphere.isChecked():` branch) with:

```python
region_spheres = self._get_all_region_spheres(ndim=mask.ndim)
exclude = region_spheres["exclude"]
if exclude is not None:
    from napari_ooctyle_analysis._regions import apply_sphere_to_mask
    apply_sphere_to_mask(mask, exclude, mode="zero_inside")
    self._region_status.setText(
        f"Excluded {n_excluded} spots "
        f"(r={exclude.radius_physical:.1f} um at "
        f"{np.array2string(exclude.center_px, precision=1)} px)"
    )
for key in ("oocyte", "perinuclear", "exclude"):
    sphere = region_spheres[key]
    if sphere is not None and self._region_show[key].isChecked():
        self._visualize_region_sphere(key, sphere)
```

(h) Delete the now-unused methods `_get_exclusion_sphere`, `_visualize_exclusion_sphere`, `_add_exclusion_shapes_layer`, and references to `_excl_status` / `_show_sphere`.

- [ ] **Step 6.4: Run the widget tests**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py -v`
Expected: all previous tests pass; new `TestRegionWidgets` tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git commit -m "refactor(_widget): replace exclude-only UI with three-region group"
```

---

## Task 7: Containment validation warning

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 7.1: Write the failing tests**

Append to `src/napari_ooctyle_analysis/_tests/test_widget.py`:

```python
class TestContainmentValidation:
    def _add_line(self, viewer, name, p1, p2):
        viewer.add_shapes(
            [np.array([p1, p2])], shape_type="line", name=name,
        )

    def test_no_warning_when_nested(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        self._add_line(viewer, "Oocyte line",      [50, 0, 0], [50, 0, 100])
        self._add_line(viewer, "Perinuclear line", [50, 50, 30], [50, 50, 70])
        self._add_line(viewer, "Exclusion line",   [50, 50, 45], [50, 50, 55])
        widget._refresh_image_layers()
        for key, name in [("oocyte", "Oocyte line"),
                          ("perinuclear", "Perinuclear line"),
                          ("exclude", "Exclusion line")]:
            widget._region_combos[key].setCurrentText(name)
        widget._update_containment_status()
        assert widget._region_status.text() == ""

    def test_warning_when_exclude_outside_perinuclear(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        self._add_line(viewer, "Oocyte line",      [50, 0, 0], [50, 0, 200])
        self._add_line(viewer, "Perinuclear line", [50, 50, 40], [50, 50, 60])
        self._add_line(viewer, "Exclusion line",   [50, 50, 100], [50, 50, 110])
        widget._refresh_image_layers()
        for key, name in [("oocyte", "Oocyte line"),
                          ("perinuclear", "Perinuclear line"),
                          ("exclude", "Exclusion line")]:
            widget._region_combos[key].setCurrentText(name)
        widget._update_containment_status()
        assert "exclude" in widget._region_status.text().lower()
        assert "perinuclear" in widget._region_status.text().lower()

    def test_warning_when_perinuclear_outside_oocyte(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        self._add_line(viewer, "Oocyte line",      [50, 50, 0], [50, 50, 10])
        self._add_line(viewer, "Perinuclear line", [50, 50, 0], [50, 50, 100])
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["perinuclear"].setCurrentText("Perinuclear line")
        widget._update_containment_status()
        assert "perinuclear" in widget._region_status.text().lower()
        assert "oocyte" in widget._region_status.text().lower()

    def test_missing_region_skips_check(self, make_napari_viewer):
        viewer = make_napari_viewer()
        widget = OoctyleAnalysisWidget(viewer)
        self._add_line(viewer, "Exclusion line", [0, 0, 0], [0, 0, 10])
        widget._refresh_image_layers()
        widget._region_combos["exclude"].setCurrentText("Exclusion line")
        widget._update_containment_status()
        assert widget._region_status.text() == ""
```

- [ ] **Step 7.2: Run the tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestContainmentValidation -v`
Expected: FAIL — `_update_containment_status` does not exist yet.

- [ ] **Step 7.3: Add `_update_containment_status` method**

Add to `OoctyleAnalysisWidget` in `src/napari_ooctyle_analysis/_widget.py`:

```python
def _update_containment_status(self) -> None:
    """Recompute nesting (exclude in perinuclear in oocyte) and update status label."""
    spheres = self._get_all_region_spheres(ndim=3)
    issues: list[str] = []
    if spheres["exclude"] is not None and spheres["perinuclear"] is not None:
        if not regions.contains_sphere(spheres["perinuclear"], spheres["exclude"]):
            issues.append("exclude region is not inside perinuclear region")
    if spheres["perinuclear"] is not None and spheres["oocyte"] is not None:
        if not regions.contains_sphere(spheres["oocyte"], spheres["perinuclear"]):
            issues.append("perinuclear region is not inside oocyte region")
    if spheres["exclude"] is not None and spheres["oocyte"] is not None:
        if not regions.contains_sphere(spheres["oocyte"], spheres["exclude"]):
            issues.append("exclude region is not inside oocyte region")
    if issues:
        self._region_status.setStyleSheet("color: red;")
        self._region_status.setText("Warning: " + "; ".join(issues))
    else:
        self._region_status.setStyleSheet("")
        self._region_status.setText("")
```

Call it at the top of `_run_detection` (right after the image-is-3D check) and at the top of `_run_overlap_analysis`:

```python
self._update_containment_status()
```

- [ ] **Step 7.4: Run the tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestContainmentValidation -v`
Expected: 4 PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git commit -m "feat(_widget): warn when regions are not properly nested"
```

---

## Task 8: Worker — accept region dict and clip to oocyte

**Files:**
- Modify: `src/napari_ooctyle_analysis/_workers.py`
- Modify: `src/napari_ooctyle_analysis/_widget.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 8.1: Write the failing test for clipping**

Append to `src/napari_ooctyle_analysis/_tests/test_widget.py`:

```python
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
            shape_type="line", name="Exclusion line",
        )
        widget._refresh_image_layers()
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["exclude"].setCurrentText("Exclusion line")

        fake_spots = np.array([
            [10.0, 20.0, 25.0],  # inside oocyte, outside exclude
            [10.0, 20.0, 20.0],  # inside exclude
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
            exclude = self.regions.get("exclude")
            if oocyte is not None:
                spots = filter_spots(spots, oocyte, keep="inside")
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if exclude is not None:
                spots = filter_spots(spots, exclude, keep="outside")
                apply_sphere_to_mask(mask, exclude, mode="zero_inside")
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
```

- [ ] **Step 8.2: Run the test to verify it fails**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestOocyteClipping -v`
Expected: FAIL — `PredictWorker.__init__()` rejects unexpected keyword `regions`.

- [ ] **Step 8.3: Update `PredictWorker`**

Replace the entire `PredictWorker` class in `src/napari_ooctyle_analysis/_workers.py` with:

```python
class PredictWorker(QThread):
    """Run spotiflow prediction + Gaussian fitting in a background thread."""

    finished = Signal(object, object, object)
    errored = Signal(str)
    progress = Signal(str, int, int)

    def __init__(
        self,
        model,
        image: np.ndarray,
        prob_thresh: float,
        min_distance: int,
        exclude_border: bool,
        device: str,
        regions: dict | None = None,
        fit_workers: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.image = image
        self.prob_thresh = prob_thresh
        self.min_distance = min_distance
        self.exclude_border = exclude_border
        self.device = device
        self.regions = regions or {}
        self.fit_workers = fit_workers or os.cpu_count() or 1

    def run(self):
        try:
            from napari_ooctyle_analysis._fitting import fit_and_mask_3d
            from napari_ooctyle_analysis._regions import (
                apply_sphere_to_mask,
                filter_spots,
            )

            self.progress.emit("Detecting spots", 0, 0)
            spots, details = self.model.predict(
                self.image,
                prob_thresh=self.prob_thresh,
                min_distance=self.min_distance,
                exclude_border=self.exclude_border,
                device=self.device,
                verbose=False,
                fit_params=False,
            )

            n_before = len(spots)
            oocyte = self.regions.get("oocyte")
            exclude = self.regions.get("exclude")
            if oocyte is not None and len(spots) > 0:
                self.progress.emit("Clipping to oocyte", 0, 0)
                spots = filter_spots(spots, oocyte, keep="inside")
            if exclude is not None and len(spots) > 0:
                self.progress.emit("Filtering excluded region", 0, 0)
                spots = filter_spots(spots, exclude, keep="outside")
            n_excluded = n_before - len(spots)

            img_for_fit = self.image
            if img_for_fit.ndim > 3:
                img_for_fit = img_for_fit[..., 0]

            def _on_fit_progress(current, total):
                self.progress.emit("Fitting spots", current, total)

            self.progress.emit("Fitting spots", 0, len(spots))
            result = fit_and_mask_3d(
                image=img_for_fit,
                spots=spots,
                mask_shape=img_for_fit.shape,
                max_workers=self.fit_workers,
                progress_callback=_on_fit_progress,
            )
            details.fit_params = result.fit_params

            mask = result.mask
            if oocyte is not None:
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if exclude is not None:
                apply_sphere_to_mask(mask, exclude, mode="zero_inside")

            model_meta = {
                "sigma": self.model.config.sigma,
                "grid": tuple(self.model.config.grid) if self.model.config.is_3d else (1, 1),
                "image_shape": self.image.shape,
                "n_excluded": n_excluded,
                "mask": mask,
            }
            self.finished.emit(spots, details, model_meta)
        except Exception as e:
            self.errored.emit(str(e))
```

- [ ] **Step 8.4: Update widget to pass `regions=` and drop the redundant mask clipping**

In `src/napari_ooctyle_analysis/_widget.py`:

(a) In `_run_detection`, change the `PredictWorker(...)` construction to pass `regions=region_spheres`:

```python
self._predict_worker = PredictWorker(
    model=self._model, image=image,
    prob_thresh=self._prob_thresh.value(),
    min_distance=self._min_distance.value(),
    exclude_border=self._exclude_border.isChecked(),
    device=self._device_string(),
    regions=region_spheres,
)
```

(b) In `_on_detection_finished`, remove the `apply_sphere_to_mask(mask, exclude, ...)` call and its import (the worker now does it). The block becomes:

```python
region_spheres = self._get_all_region_spheres(ndim=mask.ndim)
exclude = region_spheres["exclude"]
if exclude is not None:
    self._region_status.setText(
        f"Excluded {n_excluded} spots "
        f"(r={exclude.radius_physical:.1f} um at "
        f"{np.array2string(exclude.center_px, precision=1)} px)"
    )
for key in ("oocyte", "perinuclear", "exclude"):
    sphere = region_spheres[key]
    if sphere is not None and self._region_show[key].isChecked():
        self._visualize_region_sphere(key, sphere)
```

- [ ] **Step 8.5: Run the new test and the full suite**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestOocyteClipping -v`
Expected: PASS.

Run: `pytest src/napari_ooctyle_analysis/ -v`
Expected: all tests still pass.

- [ ] **Step 8.6: Commit**

```bash
git add src/napari_ooctyle_analysis/_workers.py src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git commit -m "feat(_workers): clip mask and spots to oocyte minus exclude"
```

---

## Task 9: Zonal voxel analysis — compute + chart

**Files:**
- Modify: `src/napari_ooctyle_analysis/_analysis.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 9.1: Write the failing test**

Append to `src/napari_ooctyle_analysis/_tests/test_widget.py`:

```python
from napari_ooctyle_analysis._analysis import (
    compute_zonal_voxels,
    create_zonal_figure,
)


class TestComputeZonalVoxels:
    def test_partitions_oocyte_into_peri_and_rest(self):
        shape = (10, 10, 10)
        oocyte = np.ones(shape, dtype=bool)
        peri = np.zeros(shape, dtype=bool); peri[3:7, 3:7, 3:7] = True
        excl = np.zeros(shape, dtype=bool); excl[4:6, 4:6, 4:6] = True
        channel = np.ones(shape, dtype=np.uint8)

        result = compute_zonal_voxels(channel, oocyte, peri, excl)
        # Perinuclear minus exclude: 4^3 - 2^3 = 56
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
```

- [ ] **Step 9.2: Run the tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputeZonalVoxels src/napari_ooctyle_analysis/_tests/test_widget.py::TestZonalFigure -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 9.3: Implement `compute_zonal_voxels` and `create_zonal_figure`**

Append to `src/napari_ooctyle_analysis/_analysis.py`:

```python
def compute_zonal_voxels(
    channel_mask: np.ndarray,
    oocyte_mask: np.ndarray,
    perinuclear_mask: np.ndarray,
    exclude_mask: np.ndarray,
) -> dict:
    """Count channel voxels in (perinuclear - exclude) vs (oocyte - perinuclear)."""
    channel = channel_mask > 0
    peri_zone = perinuclear_mask & ~exclude_mask & oocyte_mask
    rest_zone = oocyte_mask & ~perinuclear_mask

    n_perinuclear = int((channel & peri_zone).sum())
    n_rest_oocyte = int((channel & rest_zone).sum())
    n_total = n_perinuclear + n_rest_oocyte

    return {
        "n_perinuclear": n_perinuclear,
        "n_rest_oocyte": n_rest_oocyte,
        "n_total": n_total,
        "pct_perinuclear": (n_perinuclear / n_total * 100) if n_total > 0 else 0.0,
        "pct_rest_oocyte": (n_rest_oocyte / n_total * 100) if n_total > 0 else 0.0,
    }


def create_zonal_figure(channel_names: list[str], results: list[dict]):
    """Grouped bar chart: per-channel perinuclear vs rest-of-oocyte voxels."""
    from matplotlib.figure import Figure

    n = len(channel_names)
    fig = Figure(figsize=(5.5, 3.5), dpi=100)
    fig.suptitle("Perinuclear vs Rest of Oocyte", fontsize=10, fontweight="bold")

    short_names = [short_label(name) for name in channel_names]
    indices = np.arange(n)
    width = 0.35

    ax1 = fig.add_subplot(121)
    peri_counts = [r["n_perinuclear"] for r in results]
    rest_counts = [r["n_rest_oocyte"] for r in results]
    ax1.bar(indices - width / 2, peri_counts, width=width,
            color="#4C72B0", label="Perinuclear")
    ax1.bar(indices + width / 2, rest_counts, width=width,
            color="#DD8452", label="Rest of oocyte")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Voxels")
    ax1.set_title("Voxel Counts", fontsize=9)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(122)
    peri_pcts = [r["pct_perinuclear"] for r in results]
    rest_pcts = [r["pct_rest_oocyte"] for r in results]
    ax2.bar(indices - width / 2, peri_pcts, width=width,
            color="#4C72B0", label="Perinuclear")
    ax2.bar(indices + width / 2, rest_pcts, width=width,
            color="#DD8452", label="Rest of oocyte")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax2.set_ylabel("% of in-oocyte voxels")
    ax2.set_ylim(0, 110)
    ax2.set_title("Zonal Fraction", fontsize=9)

    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
    return fig
```

- [ ] **Step 9.4: Run the tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputeZonalVoxels src/napari_ooctyle_analysis/_tests/test_widget.py::TestZonalFigure -v`
Expected: PASS.

- [ ] **Step 9.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git commit -m "feat(_analysis): add per-channel zonal voxel count + chart"
```

---

## Task 10: Wire zonal chart into Compute Overlap

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py`
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 10.1: Write the failing test**

Append to `src/napari_ooctyle_analysis/_tests/test_widget.py`:

```python
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

    def test_zonal_chart_added_when_both_regions_drawn(self, make_napari_viewer):
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
            [np.array([[5.0, 5.0, 3.0], [5.0, 5.0, 7.0]])],
            shape_type="line", name="Perinuclear line",
        )
        widget = OoctyleAnalysisWidget(viewer)
        widget._mask_a_combo.setCurrentText("A")
        widget._mask_b_combo.setCurrentText("B")
        widget._region_combos["oocyte"].setCurrentText("Oocyte line")
        widget._region_combos["perinuclear"].setCurrentText("Perinuclear line")
        widget._run_overlap_analysis()
        assert widget._charts_layout.count() == 3
```

- [ ] **Step 10.2: Run the tests to verify they fail**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestZonalChartWiring -v`
Expected: FAIL — second test fails because the zonal chart isn't being added yet.

- [ ] **Step 10.3: Wire zonal chart into `_run_overlap_analysis`**

In `src/napari_ooctyle_analysis/_widget.py`, replace the entire `_run_overlap_analysis` method with:

```python
def _run_overlap_analysis(self):
    self._update_containment_status()
    name_a = self._mask_a_combo.currentText()
    name_b = self._mask_b_combo.currentText()

    if not name_a or not name_b:
        self._overlap_status.setText("Please select two Labels layers.")
        return
    if name_a == name_b:
        self._overlap_status.setText("Please select two different Labels layers.")
        return

    try:
        mask_a = np.asarray(self.viewer.layers[name_a].data)
        mask_b = np.asarray(self.viewer.layers[name_b].data)
    except KeyError as e:
        self._overlap_status.setText(f"Layer not found: {e}")
        return

    if mask_a.shape != mask_b.shape:
        self._overlap_status.setText(
            f"Shape mismatch: {name_a} {mask_a.shape} vs {name_b} {mask_b.shape}"
        )
        return

    result = analysis.compute_overlap(mask_a, mask_b)
    self._add_overlap_chart(name_a, name_b, result)

    if self._show_overlap_layer.isChecked() and result["n_overlap"] > 0:
        layer_name = f"{name_a} & {name_b} Overlap Mask"
        for layer in list(self.viewer.layers):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)
        self.viewer.add_labels(
            result["overlap_mask"], name=layer_name, opacity=0.5,
        )

    self._overlap_status.setText(
        f"{name_a} vs {name_b}: "
        f"{result['n_overlap']:,} overlapping voxels "
        f"({result['pct_a']:.1f}% of A, {result['pct_b']:.1f}% of B)"
    )

    self._maybe_add_zonal_chart(mask_a, mask_b, name_a, name_b)
```

Add a new helper method:

```python
def _maybe_add_zonal_chart(self, mask_a, mask_b, name_a: str, name_b: str) -> None:
    spheres = self._get_all_region_spheres(ndim=mask_a.ndim)
    oocyte = spheres["oocyte"]
    peri = spheres["perinuclear"]
    if oocyte is None or peri is None:
        return
    exclude = spheres["exclude"]
    oocyte_mask = regions.sphere_to_mask(oocyte, mask_a.shape)
    peri_mask = regions.sphere_to_mask(peri, mask_a.shape)
    if exclude is not None:
        excl_mask = regions.sphere_to_mask(exclude, mask_a.shape)
    else:
        excl_mask = np.zeros(mask_a.shape, dtype=bool)

    result_a = analysis.compute_zonal_voxels(mask_a, oocyte_mask, peri_mask, excl_mask)
    result_b = analysis.compute_zonal_voxels(mask_b, oocyte_mask, peri_mask, excl_mask)

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    fig = analysis.create_zonal_figure([name_a, name_b], [result_a, result_b])
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(280)
    count = self._charts_layout.count()
    self._charts_layout.insertWidget(count - 1, canvas)
```

- [ ] **Step 10.4: Run the tests to verify they pass**

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestZonalChartWiring -v`
Expected: PASS.

Run: `pytest src/napari_ooctyle_analysis/_tests/test_widget.py::test_overlap_charts_accumulate src/napari_ooctyle_analysis/_tests/test_widget.py::test_clear_charts -v`
Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_tests/test_widget.py
git commit -m "feat(_widget): append zonal chart when oocyte+perinuclear are drawn"
```

---

## Task 11: Final smoke test

- [ ] **Step 11.1: Run the full test suite with coverage**

Run: `pytest --cov=napari_ooctyle_analysis src/napari_ooctyle_analysis/ -v`
Expected: all tests pass.

- [ ] **Step 11.2: Manual UI verification**

Launch napari with the plugin, draw three nested lines, confirm:
- mesh overlays appear in cyan / magenta / yellow when "show" is checked
- warning appears when geometry is non-nested and clears when fixed
- mask is empty outside oocyte and inside exclude
- a zonal chart appears under the overlap chart in the Analysis tab

If the manual check can't be performed, record that explicitly rather than claiming success.

- [ ] **Step 11.3: Final commit (only if any cleanup needed)**

If steps 11.1–11.2 surfaced no issues, no further commit is needed.

---

## Self-Review

**Spec coverage:** Every spec requirement maps to a task:
- Three regions, sphere from line → Tasks 1, 6
- Each on its own Shapes layer with distinct color → Task 6
- All three optional, graceful fallbacks → Tasks 6, 7, 8, 10
- Containment validation, warn but don't block → Task 7
- Detection clips spots and mask to (oocyte minus exclude) → Task 8
- Existing overlap chart unchanged → Task 10
- New per-channel zonal chart → Tasks 9, 10
- `_regions.py` with Sphere + helpers → Tasks 1-4
- `_segmentation.py` delegates → Task 5

**Placeholders:** None — all step bodies contain concrete code, commands, or both.

**Type consistency:**
- `Sphere(center_px, radius_physical, scale)` used identically in Tasks 1-10.
- `regions: dict[str, Sphere | None]` in Tasks 6, 8 — matches.
- `_get_region_sphere(key, ndim)` / `_get_all_region_spheres(ndim)` — defined Task 6, used in 7, 8, 10.
- `_update_containment_status()` — defined Task 7, used in Task 10.
- `_region_combos[key]`, `_region_show[key]`, `_region_status` — defined Task 6, used consistently.
- `compute_zonal_voxels(channel_mask, oocyte_mask, perinuclear_mask, exclude_mask)` — defined Task 9, called Task 10 with matching positional args.

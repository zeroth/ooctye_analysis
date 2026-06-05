# Auto-Computed Perinuclear Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manually-drawn perinuclear region with one auto-computed from the nucleus (formerly "exclude") and oocyte geometry, with a UI-adjustable fraction.

**Architecture:** The "exclude" region is renamed to "nucleus" throughout (drawn). Perinuclear becomes a *derived* `Sphere` — centered on the nucleus, radius `R_n + frac*(R_o - R_n)` — computed inside `_get_all_region_spheres` instead of read from a Shapes layer. Everything downstream (spot/mask clipping, zonal chart, surface overlay) consumes the same `Sphere` object as before.

**Tech Stack:** Python, NumPy (pure-geometry `_regions.py`), qtpy/napari widget, pytest.

---

## Conventions

- **Run tests with the project venv** (a bare `pytest` is the wrong interpreter):
  `env/Scripts/python.exe -m pytest <args>`
- **Commit with the D: drive fsync workaround:**
  `git -c core.fsync=none commit -m "..."` (and `git -c core.fsync=none add ...`)
- Keep the test suite green at the end of every task.

## File Structure

- `src/napari_ooctyle_analysis/_regions.py` — add pure `compute_perinuclear`.
- `src/napari_ooctyle_analysis/_workers.py` — rename region key `exclude` → `nucleus`.
- `src/napari_ooctyle_analysis/_analysis.py` — rename `compute_zonal_voxels` param.
- `src/napari_ooctyle_analysis/_widget.py` — region descriptors, auto-compute, UI fraction
  spinbox, containment validation, visualization, status text.
- `src/napari_ooctyle_analysis/_tests/test_widget.py` — update existing tests, add new ones.

---

## Task 1: `compute_perinuclear` geometry primitive

**Files:**
- Modify: `src/napari_ooctyle_analysis/_regions.py`
- Test: `src/napari_ooctyle_analysis/_tests/test_widget.py`

- [ ] **Step 1: Write the failing tests**

Add this class near the other geometry tests in `test_widget.py` (after the existing
region-primitive tests). The import for `Sphere`/`compute_perinuclear` comes from `_regions`;
add it to the top-of-file imports if not already present:
`from napari_ooctyle_analysis._regions import Sphere, compute_perinuclear`

```python
class TestComputePerinuclear:
    def _sphere(self, center, radius):
        return Sphere(
            center_px=np.array(center, dtype=np.float64),
            radius_physical=float(radius),
            scale=np.array([1.0, 1.0, 1.0]),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputePerinuclear -v`
Expected: FAIL with `ImportError`/`AttributeError: ... 'compute_perinuclear'`.

- [ ] **Step 3: Implement `compute_perinuclear`**

Add to `_regions.py` (after `contains_sphere`, before `_distance_sq_physical`):

```python
def compute_perinuclear(nucleus: Sphere, oocyte: Sphere, frac: float) -> Sphere:
    """Perinuclear shell sphere: centered on the nucleus, radius a fraction of the gap.

    R_p = R_n + frac * (R_o - R_n), where R_n / R_o are the nucleus / oocyte radii.
    `frac` is expected in [0, 1]; the caller (widget) supplies a validated value.
    """
    r_p = nucleus.radius_physical + frac * (
        oocyte.radius_physical - nucleus.radius_physical
    )
    return Sphere(
        center_px=nucleus.center_px,
        radius_physical=r_p,
        scale=nucleus.scale,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py::TestComputePerinuclear -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_regions.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat(_regions): add compute_perinuclear geometry primitive"
```

---

## Task 2: Rename region descriptors + add perinuclear visualization metadata

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py:37-42` (REGION_DESCRIPTORS)
- Modify: `src/napari_ooctyle_analysis/_widget.py:495-539` (`_add_region_shapes_layer`, `_visualize_region_sphere`)

This task changes the drawn-region list to `oocyte` + `nucleus` only, and introduces a
visualization map covering all three regions (including computed perinuclear). The widget
will not fully work until Task 3 wires the computed sphere — so the suite is run at the end
of Task 3, not here. (Tasks 2 and 3 are committed together.)

- [ ] **Step 1: Replace REGION_DESCRIPTORS and add REGION_VIZ**

Replace lines 37-42:

```python
REGION_DESCRIPTORS = [
    # key,       label,     edge_color   (manually drawn regions only)
    ("oocyte",  "Oocyte",  "cyan"),
    ("nucleus", "Nucleus", "yellow"),
]

# Label + edge color for every region, including the computed perinuclear shell.
REGION_VIZ = {
    "oocyte":      ("Oocyte",      "cyan"),
    "nucleus":     ("Nucleus",     "yellow"),
    "perinuclear": ("Perinuclear", "magenta"),
}
```

- [ ] **Step 2: Drop the exclude special-case layer name in `_add_region_shapes_layer`**

In `_add_region_shapes_layer` (around line 495-498), replace:

```python
        label_map = {k: (lbl, col) for k, lbl, col in REGION_DESCRIPTORS}
        label, color = label_map[key]
        layer_name = "Exclusion line" if key == "exclude" else f"{label} line"
```

with:

```python
        label, color = REGION_VIZ[key]
        layer_name = f"{label} line"
```

- [ ] **Step 3: Use REGION_VIZ in `_visualize_region_sphere`**

In `_visualize_region_sphere` (around line 528-530), replace:

```python
        label_map = {k: (lbl, col) for k, lbl, col in REGION_DESCRIPTORS}
        label, color = label_map[key]
```

with:

```python
        label, color = REGION_VIZ[key]
```

(No standalone test run for this task — continue to Task 3, which makes the widget consistent
again, then run the suite and commit both tasks together.)

---

## Task 3: Auto-compute perinuclear in `_get_all_region_spheres` + visualize all regions

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py:541-542` (`_get_all_region_spheres`)
- Modify: `src/napari_ooctyle_analysis/_widget.py:666-669` (visualize loop)

- [ ] **Step 1: Compute perinuclear in `_get_all_region_spheres`**

Replace the one-line body (lines 541-542):

```python
    def _get_all_region_spheres(self, ndim: int = 3) -> dict:
        return {key: self._get_region_sphere(key, ndim=ndim) for key, _, _ in REGION_DESCRIPTORS}
```

with:

```python
    def _get_all_region_spheres(self, ndim: int = 3) -> dict:
        spheres = {
            key: self._get_region_sphere(key, ndim=ndim)
            for key, _, _ in REGION_DESCRIPTORS
        }
        nucleus = spheres.get("nucleus")
        oocyte = spheres.get("oocyte")
        if nucleus is not None and oocyte is not None:
            frac = self._peri_fraction.value() / 100.0
            spheres["perinuclear"] = regions.compute_perinuclear(nucleus, oocyte, frac)
        else:
            spheres["perinuclear"] = None
        return spheres
```

(`self._peri_fraction` is the spinbox created in Task 4. It is read here, but the widget
constructor builds the UI — including `_peri_fraction` — before any call to this method, so
ordering is safe.)

- [ ] **Step 2: Visualize every returned sphere (incl. computed perinuclear)**

Replace the visualize loop (lines 666-669):

```python
        for key, _, _ in REGION_DESCRIPTORS:
            sphere = region_spheres[key]
            if sphere is not None and self._region_show[key].isChecked():
                self._visualize_region_sphere(key, sphere)
```

with:

```python
        for key, sphere in region_spheres.items():
            show = self._region_show.get(key)
            if sphere is not None and show is not None and show.isChecked():
                self._visualize_region_sphere(key, sphere)
```

(Suite run + commit happens after Task 4 wires the UI controls these methods depend on.)

---

## Task 4: UI — rename rows, add perinuclear fraction spinbox + show toggle

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py:256-289` (Regions group construction)

- [ ] **Step 1: Build drawn-region rows + a dedicated perinuclear row**

Replace the block that builds the per-region rows (lines 256-273):

```python
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
```

with:

```python
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

        # Perinuclear is auto-computed from nucleus + oocyte (not drawn).
        peri_row = QHBoxLayout()
        peri_row.addWidget(QLabel("Perinuclear:"))
        peri_row.addWidget(QLabel("fraction"))
        self._peri_fraction = QDoubleSpinBox()
        self._peri_fraction.setRange(0.0, 100.0)
        self._peri_fraction.setDecimals(1)
        self._peri_fraction.setSingleStep(5.0)
        self._peri_fraction.setValue(40.0)
        self._peri_fraction.setSuffix(" %")
        self._peri_fraction.setToolTip(
            "Perinuclear boundary as a fraction of the nucleus-to-oocyte gap:\n"
            "R_perinuclear = R_nucleus + fraction * (R_oocyte - R_nucleus)"
        )
        peri_row.addWidget(self._peri_fraction)
        peri_show = QCheckBox("show")
        peri_show.setChecked(True)
        self._region_show["perinuclear"] = peri_show
        peri_row.addWidget(peri_show)
        regions_layout.addLayout(peri_row)
```

- [ ] **Step 2: Run the full suite to find every reference still expecting the old model**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py -q`
Expected: Several FAILS in `TestRegionWidgets`, `TestContainmentValidation`,
`TestOocyteClipping`, `TestZonalChartWiring`, and the worker (still using `"exclude"`).
These are fixed in Tasks 5-7. (Do not commit yet.)

---

## Task 5: Rename `exclude` → `nucleus` in worker + detection status

**Files:**
- Modify: `src/napari_ooctyle_analysis/_workers.py:58-65, 86-89`
- Modify: `src/napari_ooctyle_analysis/_widget.py:656-663` (detection-finished status)

- [ ] **Step 1: Rename in the worker `run()`**

In `_workers.py`, replace lines 58-66:

```python
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
```

with:

```python
            n_before = len(spots)
            oocyte = self.regions.get("oocyte")
            nucleus = self.regions.get("nucleus")
            if oocyte is not None and len(spots) > 0:
                self.progress.emit("Clipping to oocyte", 0, 0)
                spots = filter_spots(spots, oocyte, keep="inside")
            if nucleus is not None and len(spots) > 0:
                self.progress.emit("Filtering nucleus region", 0, 0)
                spots = filter_spots(spots, nucleus, keep="outside")
            n_excluded = n_before - len(spots)
```

And replace lines 86-89:

```python
            mask = result.mask
            if oocyte is not None:
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if exclude is not None:
                apply_sphere_to_mask(mask, exclude, mode="zero_inside")
```

with:

```python
            mask = result.mask
            if oocyte is not None:
                apply_sphere_to_mask(mask, oocyte, mode="zero_outside")
            if nucleus is not None:
                apply_sphere_to_mask(mask, nucleus, mode="zero_inside")
```

- [ ] **Step 2: Rename in the detection-finished status text**

In `_widget.py`, replace lines 656-663:

```python
        region_spheres = self._get_all_region_spheres(ndim=mask.ndim)
        exclude = region_spheres["exclude"]
        if exclude is not None:
            self._region_status.setText(
                f"Excluded {n_excluded} spots "
                f"(r={exclude.radius_physical:.1f} um at "
                f"{np.array2string(exclude.center_px, precision=1)} px)"
            )
```

with:

```python
        region_spheres = self._get_all_region_spheres(ndim=mask.ndim)
        nucleus = region_spheres["nucleus"]
        if nucleus is not None:
            self._region_status.setText(
                f"Excluded {n_excluded} spots inside nucleus "
                f"(r={nucleus.radius_physical:.1f} um at "
                f"{np.array2string(nucleus.center_px, precision=1)} px)"
            )
```

- [ ] **Step 3: Update `TestRegionWidgets` and `TestOocyteClipping` in the tests**

In `test_widget.py`, update the `TestRegionWidgets` class header (lines 555-560):

```python
    REGION_KEYS = ("oocyte", "nucleus")
    LAYER_NAMES = {
        "oocyte": "Oocyte line",
        "nucleus": "Nucleus line",
    }
```

In the same class, the three tests that hard-code `"Exclusion line"` / `"exclude"`
(`test_get_region_sphere_from_line`, `test_region_sphere_with_anisotropic_scale`,
`test_region_sphere_2d_line_on_3d`) — replace every `"Exclusion line"` with `"Nucleus line"`
and every `widget._region_combos["exclude"]` / `_get_region_sphere("exclude", ...)` with
`"nucleus"`. (Three occurrences of the layer name, three of the key.)

In `TestOocyteClipping.test_clip_spots_and_mask` (lines 710-745): rename the shapes layer
`"Exclusion line"` → `"Nucleus line"`, the combo key `"exclude"` → `"nucleus"`, and inside
`fake_run` replace `exclude = self.regions.get("exclude")` with
`nucleus = self.regions.get("nucleus")` and the two following `exclude` uses with `nucleus`.

- [ ] **Step 4: Run the affected tests**

Run: `env/Scripts/python.exe -m pytest "src/napari_ooctyle_analysis/_tests/test_widget.py::TestRegionWidgets" "src/napari_ooctyle_analysis/_tests/test_widget.py::TestOocyteClipping" -v`
Expected: PASS.

(Commit happens at the end of Task 7 once the whole suite is green.)

---

## Task 6: Containment validation — nucleus ⊂ oocyte only

**Files:**
- Modify: `src/napari_ooctyle_analysis/_widget.py:544-562` (`_update_containment_status`)
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py` (`TestContainmentValidation`)

- [ ] **Step 1: Replace `_update_containment_status`**

Replace the method body (lines 544-562):

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

with:

```python
    def _update_containment_status(self) -> None:
        """Validate that the nucleus is inside the oocyte (perinuclear is derived)."""
        spheres = self._get_all_region_spheres(ndim=3)
        nucleus = spheres["nucleus"]
        oocyte = spheres["oocyte"]
        issues: list[str] = []
        if nucleus is not None and oocyte is not None:
            if oocyte.radius_physical <= nucleus.radius_physical:
                issues.append("oocyte radius must be larger than nucleus radius")
            elif not regions.contains_sphere(oocyte, nucleus):
                issues.append("nucleus region is not inside oocyte region")
        if issues:
            self._region_status.setStyleSheet("color: red;")
            self._region_status.setText("Warning: " + "; ".join(issues))
        else:
            self._region_status.setStyleSheet("")
            self._region_status.setText("")
```

- [ ] **Step 2: Replace `TestContainmentValidation` tests**

The old tests reference a drawn "Perinuclear line" and an "exclude in perinuclear" check that
no longer exist. Replace the whole class body (lines 642-699) with:

```python
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
        widget._update_containment_status()
        assert widget._region_status.text() == ""
```

- [ ] **Step 3: Run the containment tests**

Run: `env/Scripts/python.exe -m pytest "src/napari_ooctyle_analysis/_tests/test_widget.py::TestContainmentValidation" -v`
Expected: PASS (4 passed).

---

## Task 7: Zonal analysis — rename param, source computed perinuclear, fix wiring tests

**Files:**
- Modify: `src/napari_ooctyle_analysis/_analysis.py:97-105` (`compute_zonal_voxels`)
- Modify: `src/napari_ooctyle_analysis/_widget.py:760-775` (`_maybe_add_zonal_chart`)
- Modify: `src/napari_ooctyle_analysis/_tests/test_widget.py` (`TestZonalChartWiring`)

- [ ] **Step 1: Rename the `compute_zonal_voxels` parameter**

In `_analysis.py`, replace lines 97-106:

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
```

with:

```python
def compute_zonal_voxels(
    channel_mask: np.ndarray,
    oocyte_mask: np.ndarray,
    perinuclear_mask: np.ndarray,
    nucleus_mask: np.ndarray,
) -> dict:
    """Count channel voxels in (perinuclear - nucleus) vs (oocyte - perinuclear)."""
    channel = channel_mask > 0
    peri_zone = perinuclear_mask & ~nucleus_mask & oocyte_mask
    rest_zone = oocyte_mask & ~perinuclear_mask
```

- [ ] **Step 2: Source the nucleus mask in `_maybe_add_zonal_chart`**

In `_widget.py`, replace lines 766-775:

```python
        exclude = spheres["exclude"]
        oocyte_mask = regions.sphere_to_mask(oocyte, mask_a.shape)
        peri_mask = regions.sphere_to_mask(peri, mask_a.shape)
        if exclude is not None:
            excl_mask = regions.sphere_to_mask(exclude, mask_a.shape)
        else:
            excl_mask = np.zeros(mask_a.shape, dtype=bool)

        result_a = analysis.compute_zonal_voxels(mask_a, oocyte_mask, peri_mask, excl_mask)
        result_b = analysis.compute_zonal_voxels(mask_b, oocyte_mask, peri_mask, excl_mask)
```

with:

```python
        nucleus = spheres["nucleus"]
        oocyte_mask = regions.sphere_to_mask(oocyte, mask_a.shape)
        peri_mask = regions.sphere_to_mask(peri, mask_a.shape)
        if nucleus is not None:
            nucleus_mask = regions.sphere_to_mask(nucleus, mask_a.shape)
        else:
            nucleus_mask = np.zeros(mask_a.shape, dtype=bool)

        result_a = analysis.compute_zonal_voxels(mask_a, oocyte_mask, peri_mask, nucleus_mask)
        result_b = analysis.compute_zonal_voxels(mask_b, oocyte_mask, peri_mask, nucleus_mask)
```

- [ ] **Step 3: Update `TestZonalChartWiring.test_zonal_chart_added_when_both_regions_drawn`**

Perinuclear is no longer drawn — the zonal chart now appears when **oocyte + nucleus** are
drawn. Replace that test (lines 820-840) with:

```python
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
```

- [ ] **Step 4: Run the full test suite**

Run: `env/Scripts/python.exe -m pytest src/napari_ooctyle_analysis/_tests/test_widget.py -q`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit Tasks 2-7 together**

```bash
git -c core.fsync=none add src/napari_ooctyle_analysis/_widget.py src/napari_ooctyle_analysis/_workers.py src/napari_ooctyle_analysis/_analysis.py src/napari_ooctyle_analysis/_tests/test_widget.py
git -c core.fsync=none commit -m "feat: auto-compute perinuclear region from nucleus + oocyte

Rename the exclude region to nucleus, derive the perinuclear shell as
R_n + frac*(R_o - R_n) centered on the nucleus, add a UI fraction spinbox,
and remove the manual perinuclear drawing."
```

---

## Self-Review Notes

- **Spec coverage:** geometry primitive (Task 1) ✔; rename exclude→nucleus (Tasks 2,5) ✔;
  computed perinuclear in `_get_all_region_spheres` (Task 3) ✔; UI fraction spinbox + show,
  drop combo/New (Tasks 2,4) ✔; containment collapses to nucleus⊂oocyte + R_o≤R_n warn
  (Task 6) ✔; worker key + status rename (Task 5) ✔; analysis param rename + computed mask
  (Task 7) ✔; tests updated/added (every task) ✔.
- **Ordering note:** Tasks 2-4 leave the widget temporarily inconsistent with the suite by
  design; the suite is only asserted green after Task 7, and a single commit covers Tasks 2-7.
  Task 1 is independently committed and green.
- **Type/name consistency:** `compute_perinuclear(nucleus, oocyte, frac)`, region key
  `"nucleus"`, viz key `"perinuclear"`, `self._peri_fraction`, `REGION_VIZ`, and layer name
  `"Nucleus line"` are used identically across all tasks.
```

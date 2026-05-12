"""Main napari widget — UI layout and event wiring only.

All non-UI logic lives in dedicated modules:
  _workers.py       – background QThread workers
  _segmentation.py  – channel splitting, exclusion sphere, data I/O
  _analysis.py      – overlap computation and chart generation
  _fitting.py       – Gaussian fitting and per-spot mask painting
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QComboBox,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QGroupBox,
    QProgressBar,
    QLineEdit,
    QScrollArea,
)

import napari
import numpy as np

from napari_ooctyle_analysis._workers import PredictWorker, FineTuneWorker
from napari_ooctyle_analysis import _segmentation as seg
from napari_ooctyle_analysis import _analysis as analysis
from napari_ooctyle_analysis import _regions as regions

REGION_DESCRIPTORS = [
    # key,           label,           edge_color
    ("oocyte",       "Oocyte",        "cyan"),
    ("perinuclear",  "Perinuclear",   "magenta"),
    ("exclude",      "Exclude",       "yellow"),
]


class OoctyleAnalysisWidget(QWidget):
    """Primary widget for oocyte analysis in napari.

    Contains two tabs:
      - Segmentation: 3D spot detection via spotiflow
      - Analysis: mask-based overlap analysis with charts
    """

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Pre-create combos that _refresh_image_layers populates,
        # so they exist before _build_segmentation_tab triggers a refresh.
        self._image_combo = QComboBox()
        self._region_combos: dict[str, QComboBox] = {
            key: QComboBox() for key, _, _ in REGION_DESCRIPTORS
        }
        self._mask_a_combo = QComboBox()
        self._mask_b_combo = QComboBox()

        # Tab container
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_segmentation_tab(), "Segmentation")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")

        # Internal state
        self._model = None
        self._predict_worker: PredictWorker | None = None
        self._finetune_worker: FineTuneWorker | None = None

    # ==================================================================
    # Tab 2 – Analysis
    # ==================================================================

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        controls = QGroupBox("Overlap Analysis")
        ctrl_layout = QVBoxLayout()
        controls.setLayout(ctrl_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Channel A:"))
        row.addWidget(self._mask_a_combo)
        refresh_btn = QPushButton("\u21BB")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh layer lists")
        refresh_btn.clicked.connect(self._refresh_image_layers)
        row.addWidget(refresh_btn)
        ctrl_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Channel B:"))
        row.addWidget(self._mask_b_combo)
        ctrl_layout.addLayout(row)

        self._show_overlap_layer = QCheckBox("Show overlap mask layer")
        self._show_overlap_layer.setChecked(True)
        ctrl_layout.addWidget(self._show_overlap_layer)

        btn_row = QHBoxLayout()
        self._overlap_btn = QPushButton("Compute Overlap")
        self._overlap_btn.clicked.connect(self._run_overlap_analysis)
        btn_row.addWidget(self._overlap_btn)

        self._clear_charts_btn = QPushButton("Clear Charts")
        self._clear_charts_btn.clicked.connect(self._clear_overlap_charts)
        btn_row.addWidget(self._clear_charts_btn)
        ctrl_layout.addLayout(btn_row)

        self._overlap_status = QLabel("")
        ctrl_layout.addWidget(self._overlap_status)

        layout.addWidget(controls)

        # Scrollable chart area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._charts_container = QWidget()
        self._charts_layout = QVBoxLayout()
        self._charts_layout.setContentsMargins(0, 0, 0, 0)
        self._charts_container.setLayout(self._charts_layout)
        self._charts_layout.addStretch()
        scroll.setWidget(self._charts_container)
        layout.addWidget(scroll, stretch=1)

        return tab

    # ==================================================================
    # Tab 1 – Segmentation
    # ==================================================================

    def _build_segmentation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # --- Input ---
        img_group = QGroupBox("Input")
        img_layout = QVBoxLayout()
        img_group.setLayout(img_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Image layer:"))
        row.addWidget(self._image_combo)
        self._refresh_btn = QPushButton("\u21BB")
        self._refresh_btn.setFixedWidth(28)
        self._refresh_btn.setToolTip("Refresh layer lists")
        self._refresh_btn.clicked.connect(self._refresh_image_layers)
        row.addWidget(self._refresh_btn)
        img_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Axis order:"))
        self._axis_order = QComboBox()
        self._axis_order.setEditable(True)
        self._axis_order.addItems(["ZCYX", "CZYX", "ZYXC"])
        self._axis_order.setToolTip(
            "Axis order of the selected image.\n"
            "Use Z, Y, X for spatial and C for channel.\n"
            "E.g. CZYX means the first axis is channels."
        )
        row.addWidget(self._axis_order)
        img_layout.addLayout(row)

        self._split_btn = QPushButton("Split Channels")
        self._split_btn.setToolTip(
            "Split a multi-channel image into separate ZYX layers.\n"
            "Each channel becomes its own Image layer for detection."
        )
        self._split_btn.clicked.connect(self._split_channels)
        img_layout.addWidget(self._split_btn)

        self._split_status = QLabel("")
        img_layout.addWidget(self._split_status)

        layout.addWidget(img_group)

        # --- Model ---
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pretrained:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "synth_3d", "smfish_3d", "general",
            "synth_complex", "hybiss", "fluo_live",
        ])
        row.addWidget(self._model_combo)
        model_layout.addLayout(row)

        row = QHBoxLayout()
        self._custom_model_path = QLineEdit()
        self._custom_model_path.setPlaceholderText("Or load custom model folder...")
        self._custom_model_path.setReadOnly(True)
        row.addWidget(self._custom_model_path)
        self._browse_model_btn = QPushButton("Browse")
        self._browse_model_btn.clicked.connect(self._browse_custom_model)
        row.addWidget(self._browse_model_btn)
        model_layout.addLayout(row)

        layout.addWidget(model_group)

        # --- Detection parameters ---
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        param_group.setLayout(param_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Probability threshold:"))
        self._prob_thresh = QDoubleSpinBox()
        self._prob_thresh.setRange(0.0, 1.0)
        self._prob_thresh.setSingleStep(0.05)
        self._prob_thresh.setValue(0.5)
        self._prob_thresh.setDecimals(3)
        row.addWidget(self._prob_thresh)
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min distance (NMS):"))
        self._min_distance = QSpinBox()
        self._min_distance.setRange(1, 50)
        self._min_distance.setValue(2)
        row.addWidget(self._min_distance)
        param_layout.addLayout(row)

        self._exclude_border = QCheckBox("Exclude border spots")
        param_layout.addWidget(self._exclude_border)

        layout.addWidget(param_group)

        # --- Regions ---
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

        # --- Device ---
        device_group = QGroupBox("Device")
        device_layout = QVBoxLayout()
        device_group.setLayout(device_layout)

        self._use_gpu = QCheckBox("Use GPU / CUDA")
        self._use_gpu.setChecked(self._cuda_available())
        self._use_gpu.setEnabled(self._cuda_available())
        if not self._cuda_available():
            self._use_gpu.setToolTip("CUDA is not available on this system")
        device_layout.addWidget(self._use_gpu)

        layout.addWidget(device_group)

        # --- Run ---
        self._detect_btn = QPushButton("Detect Spots")
        self._detect_btn.clicked.connect(self._run_detection)
        layout.addWidget(self._detect_btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        layout.addWidget(self._build_finetune_group())
        layout.addStretch()

        # Layer events
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._refresh_image_layers()

        return tab

    # ==================================================================
    # Fine-tuning UI
    # ==================================================================

    def _build_finetune_group(self) -> QGroupBox:
        group = QGroupBox("Fine-tuning")
        layout = QVBoxLayout()
        group.setLayout(layout)

        for label_text, attr, placeholder in [
            ("Training images dir:", "_train_images_path", "Folder with .tif images"),
            ("Training spots dir:", "_train_spots_path", "Folder with .csv spot coords"),
            ("Validation images dir:", "_val_images_path", "Folder with .tif images"),
            ("Validation spots dir:", "_val_spots_path", "Folder with .csv spot coords"),
            ("Save model to:", "_finetune_save_dir", "Output directory for fine-tuned model"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            le = QLineEdit()
            le.setReadOnly(True)
            le.setPlaceholderText(placeholder)
            setattr(self, attr, le)
            row.addWidget(le)
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda checked=False, w=le: self._browse_dir(w))
            row.addWidget(btn)
            layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs:"))
        self._ft_epochs = QSpinBox()
        self._ft_epochs.setRange(1, 1000)
        self._ft_epochs.setValue(50)
        row.addWidget(self._ft_epochs)

        row.addWidget(QLabel("Batch size:"))
        self._ft_batch_size = QSpinBox()
        self._ft_batch_size.setRange(1, 64)
        self._ft_batch_size.setValue(2)
        row.addWidget(self._ft_batch_size)

        row.addWidget(QLabel("Learning rate:"))
        self._ft_lr = QDoubleSpinBox()
        self._ft_lr.setRange(1e-6, 1e-1)
        self._ft_lr.setDecimals(6)
        self._ft_lr.setSingleStep(1e-4)
        self._ft_lr.setValue(3e-4)
        row.addWidget(self._ft_lr)
        layout.addLayout(row)

        self._finetune_btn = QPushButton("Start Fine-tuning")
        self._finetune_btn.clicked.connect(self._run_finetuning)
        layout.addWidget(self._finetune_btn)

        self._ft_progress_bar = QProgressBar()
        self._ft_progress_bar.setRange(0, 0)
        self._ft_progress_bar.hide()
        layout.addWidget(self._ft_progress_bar)

        self._ft_status_label = QLabel("")
        layout.addWidget(self._ft_status_label)

        return group

    # ==================================================================
    # Layer management
    # ==================================================================

    def _refresh_image_layers(self, event=None):
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

        for combo, text in prev.items():
            idx = combo.findText(text)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _on_layer_change(self, event=None):
        self._refresh_image_layers()

    # ==================================================================
    # Helpers (thin UI glue)
    # ==================================================================

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _device_string(self) -> str:
        return "cuda" if self._use_gpu.isChecked() else "cpu"

    def _browse_custom_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select model folder")
        if path:
            self._custom_model_path.setText(path)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            line_edit.setText(path)

    def _get_selected_image(self) -> np.ndarray | None:
        name = self._image_combo.currentText()
        if not name:
            return None
        try:
            return np.asarray(self.viewer.layers[name].data)
        except KeyError:
            return None

    def _get_scale_zyx(self) -> np.ndarray:
        return np.array([
            self._scale_z.value(),
            self._scale_y.value(),
            self._scale_x.value(),
        ], dtype=np.float64)

    # ==================================================================
    # Channel splitting
    # ==================================================================

    def _split_channels(self):
        image = self._get_selected_image()
        if image is None:
            self._split_status.setText("No image layer selected.")
            return

        axis_order = self._axis_order.currentText().upper().strip()
        try:
            channels, was_multichannel = seg.reorder_to_zyx(image, axis_order)
        except ValueError as e:
            self._split_status.setText(str(e))
            return

        if not was_multichannel:
            self._split_status.setText(
                "No channel axis (C) in axis order — nothing to split."
            )
            return

        for i, ch_data in enumerate(channels):
            self.viewer.add_image(ch_data, name=f"Channel {i}")

        self._split_status.setText(
            f"Split into {len(channels)} channels (ZYX each)."
        )

    # ==================================================================
    # Region spheres
    # ==================================================================

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

    def _get_region_sphere(self, key: str, ndim: int = 3) -> "regions.Sphere | None":
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

    def _visualize_region_sphere(self, key: str, sphere: "regions.Sphere"):
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

    def _get_all_region_spheres(self, ndim: int = 3) -> dict:
        return {key: self._get_region_sphere(key, ndim=ndim) for key, _, _ in REGION_DESCRIPTORS}

    # ==================================================================
    # Detection
    # ==================================================================

    def _load_model(self):
        from spotiflow.model import Spotiflow

        custom_path = self._custom_model_path.text().strip()
        if custom_path:
            self._model = Spotiflow.from_folder(
                custom_path, map_location=self._device_string(),
            )
        else:
            self._model = Spotiflow.from_pretrained(
                self._model_combo.currentText(),
                map_location=self._device_string(),
            )

    def _run_detection(self):
        image = self._get_selected_image()
        if image is None:
            self._status_label.setText("No image layer selected.")
            return
        if image.ndim < 3:
            self._status_label.setText("Please select a 3D image (ZYX).")
            return

        self._detect_btn.setEnabled(False)
        self._status_label.setText("Loading model...")

        try:
            self._load_model()
        except Exception as e:
            self._status_label.setText(f"Model load error: {e}")
            self._detect_btn.setEnabled(True)
            return

        # Create napari progress bar (indeterminate until fitting starts)
        from napari.utils import progress
        self._napari_progress = progress(total=0, desc="Detecting spots")

        region_spheres = self._get_all_region_spheres(ndim=image.ndim)

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
        self._predict_worker.progress.connect(self._on_detection_progress)
        self._predict_worker.finished.connect(self._on_detection_finished)
        self._predict_worker.errored.connect(self._on_detection_error)
        self._predict_worker.start()

    def _on_detection_progress(self, stage: str, current: int, total: int):
        """Update the napari progress bar from worker signals."""
        pbr = getattr(self, "_napari_progress", None)
        if pbr is None:
            return
        pbr.set_description(stage)
        if total > 0:
            pbr.total = total
            pbr.n = current
            pbr.refresh()
        self._status_label.setText(
            f"{stage}... {current}/{total}" if total > 0 else f"{stage}..."
        )

    def _on_detection_finished(self, spots: np.ndarray, details, model_meta: dict):
        from scipy.ndimage import label as ndlabel

        # Close napari progress bar
        pbr = getattr(self, "_napari_progress", None)
        if pbr is not None:
            pbr.close()
            self._napari_progress = None

        self._detect_btn.setEnabled(True)

        n_kept = len(spots)
        n_excluded = model_meta.get("n_excluded", 0)
        n_total = n_kept + n_excluded
        image_name = self._image_combo.currentText()

        mask = model_meta.get("mask")
        if mask is None:
            mask = np.zeros(model_meta["image_shape"], dtype=np.uint8)

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
        else:
            self._region_status.setText("")
        for key, _, _ in REGION_DESCRIPTORS:
            sphere = region_spheres[key]
            if sphere is not None and self._region_show[key].isChecked():
                self._visualize_region_sphere(key, sphere)

        labeled_mask, n_labels = ndlabel(mask)
        self.viewer.add_labels(labeled_mask, name=f"{image_name} mask", opacity=0.4)

        status_parts = [f"Detected {n_total} spots"]
        if n_excluded > 0:
            status_parts.append(f"excluded {n_excluded}")
        status_parts.append(f"kept {n_kept}")
        status_parts.append(f"labels={n_labels}")
        self._status_label.setText(", ".join(status_parts) + ".")

        if n_kept > 0:
            self.viewer.add_points(
                spots, name=f"{image_name} spots",
                size=3, face_color="red", opacity=0.7,
            )

    def _on_detection_error(self, msg: str):
        pbr = getattr(self, "_napari_progress", None)
        if pbr is not None:
            pbr.close()
            self._napari_progress = None
        self._detect_btn.setEnabled(True)
        self._status_label.setText(f"Detection error: {msg}")

    # ==================================================================
    # Overlap analysis
    # ==================================================================

    def _add_overlap_chart(self, name_a: str, name_b: str, result: dict):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        fig = analysis.create_overlap_figure(name_a, name_b, result)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(280)
        count = self._charts_layout.count()
        self._charts_layout.insertWidget(count - 1, canvas)

    def _clear_overlap_charts(self):
        while self._charts_layout.count() > 1:
            item = self._charts_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _run_overlap_analysis(self):
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

    # ==================================================================
    # Fine-tuning
    # ==================================================================

    def _run_finetuning(self):
        train_img_dir = self._train_images_path.text().strip()
        train_spots_dir = self._train_spots_path.text().strip()
        val_img_dir = self._val_images_path.text().strip()
        val_spots_dir = self._val_spots_path.text().strip()
        save_dir = self._finetune_save_dir.text().strip()

        if not all([train_img_dir, train_spots_dir, val_img_dir, val_spots_dir, save_dir]):
            self._ft_status_label.setText("Please fill all data directories.")
            return

        self._finetune_btn.setEnabled(False)
        self._ft_progress_bar.show()
        self._ft_status_label.setText("Loading training data...")

        try:
            train_images, train_spots = seg.load_data_from_dirs(
                train_img_dir, train_spots_dir
            )
            val_images, val_spots = seg.load_data_from_dirs(
                val_img_dir, val_spots_dir
            )
        except Exception as e:
            self._ft_status_label.setText(f"Data load error: {e}")
            self._finetune_btn.setEnabled(True)
            self._ft_progress_bar.hide()
            return

        self._ft_status_label.setText("Loading base model for fine-tuning...")
        try:
            self._load_model()
        except Exception as e:
            self._ft_status_label.setText(f"Model load error: {e}")
            self._finetune_btn.setEnabled(True)
            self._ft_progress_bar.hide()
            return

        versioned_path = seg.generate_save_path(save_dir)

        from spotiflow.model.config import SpotiflowTrainingConfig

        train_config = SpotiflowTrainingConfig(
            num_epochs=self._ft_epochs.value(),
            batch_size=self._ft_batch_size.value(),
            lr=self._ft_lr.value(),
            finetuned_from=self._model_combo.currentText(),
        )

        self._ft_status_label.setText(f"Fine-tuning \u2192 {versioned_path.name} ...")

        self._finetune_worker = FineTuneWorker(
            model=self._model,
            train_images=train_images, train_spots=train_spots,
            val_images=val_images, val_spots=val_spots,
            save_dir=str(versioned_path),
            train_config=train_config,
            device=self._device_string(),
            num_epochs=self._ft_epochs.value(),
        )
        self._finetune_worker.finished.connect(self._on_finetune_finished)
        self._finetune_worker.errored.connect(self._on_finetune_error)
        self._finetune_worker.progress.connect(
            lambda msg: self._ft_status_label.setText(msg)
        )
        self._finetune_worker.start()

    def _on_finetune_finished(self, save_dir: str):
        self._ft_progress_bar.hide()
        self._finetune_btn.setEnabled(True)
        self._ft_status_label.setText(f"Fine-tuning complete! Saved to: {save_dir}")

    def _on_finetune_error(self, msg: str):
        self._ft_progress_bar.hide()
        self._finetune_btn.setEnabled(True)
        self._ft_status_label.setText(f"Fine-tuning error: {msg}")

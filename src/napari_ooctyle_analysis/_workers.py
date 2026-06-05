"""Background QThread workers for long-running operations."""
from __future__ import annotations

import os

import numpy as np
from qtpy.QtCore import QThread, Signal


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
            nucleus = self.regions.get("nucleus")
            if oocyte is not None and len(spots) > 0:
                self.progress.emit("Clipping to oocyte", 0, 0)
                spots = filter_spots(spots, oocyte, keep="inside")
            if nucleus is not None and len(spots) > 0:
                self.progress.emit("Excluding nucleus region", 0, 0)
                spots = filter_spots(spots, nucleus, keep="outside")
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
        except Exception as e:
            self.errored.emit(str(e))


class FineTuneWorker(QThread):
    """Run spotiflow fine-tuning in a background thread."""

    finished = Signal(str)  # save_dir
    errored = Signal(str)
    progress = Signal(str, int, int)  # (stage, current, total)

    def __init__(
        self,
        model,
        train_images,
        train_spots,
        val_images,
        val_spots,
        save_dir: str,
        train_config,
        device: str,
        num_epochs: int,
    ):
        super().__init__()
        self.model = model
        self.train_images = train_images
        self.train_spots = train_spots
        self.val_images = val_images
        self.val_spots = val_spots
        self.save_dir = save_dir
        self.train_config = train_config
        self.device = device
        self.num_epochs = num_epochs

    def run(self):
        try:
            self.progress.emit("Fine-tuning", 0, 0)
            self.model.fit(
                train_images=self.train_images,
                train_spots=self.train_spots,
                val_images=self.val_images,
                val_spots=self.val_spots,
                save_dir=self.save_dir,
                train_config=self.train_config,
                device=self.device,
            )
            self.finished.emit(self.save_dir)
        except Exception as e:
            self.errored.emit(str(e))

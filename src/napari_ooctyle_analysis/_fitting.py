"""Single-pass Gaussian fitting and mask painting for 3D spot data.

Reimplements the core logic from ``spotiflow.utils.fitting`` but writes
each spot's ellipsoidal footprint into a shared mask array during the
same loop, avoiding a second pass over all spots.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from functools import partial
from typing import Union

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)

FWHM_CONSTANT = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ------------------------------------------------------------------
# Gaussian model (same as spotiflow)
# ------------------------------------------------------------------

def _gaussian_3d(zyx, z0, y0, x0, sigma_z, sigma_yx, A, B):
    z, y, x = zyx
    return A * np.exp(
        -(((z - z0) ** 2) / (2 * sigma_z ** 2)
          + ((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigma_yx ** 2))
    ) + B


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# ------------------------------------------------------------------
# Per-spot result with built-in mask painting
# ------------------------------------------------------------------

@dataclass
class SpotFit3D:
    """Result of a single 3D Gaussian fit.

    Knows how to paint its own ellipsoidal footprint into a mask array
    via :meth:`paint_mask`.
    """
    fwhm_z: float
    fwhm_yx: float
    offset_z: float
    offset_y: float
    offset_x: float
    intens_A: float
    intens_B: float
    r_squared: float

    def paint_mask(self, mask: np.ndarray, spot: np.ndarray) -> None:
        """Paint this spot's ellipsoidal region into *mask* (in-place).

        The ellipsoid is centered at ``spot + offset`` with half-max radii
        ``fwhm / 2`` in each axis.  Spots with invalid fits are skipped.

        Args:
            mask: 3D uint8 array to paint into.
            spot: (z, y, x) detected spot coordinate.
        """
        fz, fyx = self.fwhm_z, self.fwhm_yx
        if not (np.isfinite(fz) and np.isfinite(fyx)):
            return
        rz = fz / 2.0
        ryx = fyx / 2.0
        if rz < 0.5 or ryx < 0.5:
            return

        cz = spot[0] + self.offset_z
        cy = spot[1] + self.offset_y
        cx = spot[2] + self.offset_x
        shape = mask.shape

        rz_ceil = int(np.ceil(rz))
        ryx_ceil = int(np.ceil(ryx))

        z0 = max(0, int(np.floor(cz)) - rz_ceil)
        z1 = min(shape[0], int(np.ceil(cz)) + rz_ceil + 1)
        y0 = max(0, int(np.floor(cy)) - ryx_ceil)
        y1 = min(shape[1], int(np.ceil(cy)) + ryx_ceil + 1)
        x0 = max(0, int(np.floor(cx)) - ryx_ceil)
        x1 = min(shape[2], int(np.ceil(cx)) + ryx_ceil + 1)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return

        zz = np.arange(z0, z1, dtype=np.float64) - cz
        yy = np.arange(y0, y1, dtype=np.float64) - cy
        xx = np.arange(x0, x1, dtype=np.float64) - cx
        gz, gy, gx = np.meshgrid(zz, yy, xx, indexing="ij")

        dist_sq = (gz / rz) ** 2 + (gy / ryx) ** 2 + (gx / ryx) ** 2
        mask[z0:z1, y0:y1, x0:x1] |= (dist_sq <= 1.0).astype(np.uint8)


def _fit_single_3d(
    center: np.ndarray,
    image: np.ndarray,
    window: int,
) -> SpotFit3D:
    """Fit a 3D Gaussian to a single spot and return parameters."""
    z_range = np.arange(-window, window + 1)
    y_range = np.arange(-window, window + 1)
    x_range = np.arange(-window, window + 1)
    z, y, x = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    z_idx, y_idx, x_idx = np.mgrid[
        center[0] - window: center[0] + window + 1,
        center[1] - window: center[1] + window + 1,
        center[2] - window: center[2] + window + 1,
    ]
    region = map_coordinates(image, [z_idx, y_idx, x_idx], order=3, mode="reflect")

    try:
        mi, ma = float(np.min(region)), float(np.max(region))
        region_norm = (region - mi) / (ma - mi)

        p0 = (0, 0, 0, 1.5, 2.0, 1, 0)
        lb = (-1e-6, -1e-6, -1e-6, 0.1, 0.1, 0.5, -0.5)
        ub = (1e-6, 1e-6, 1e-6, 10, 10, 1.5, 0.5)

        popt, _ = curve_fit(
            _gaussian_3d,
            (z.ravel(), y.ravel(), x.ravel()),
            region_norm.ravel(),
            p0=p0,
            bounds=(lb, ub),
        )
        pred = _gaussian_3d((z.ravel(), y.ravel(), x.ravel()), *popt)
        r2 = _r_squared(region_norm.ravel(), pred)

    except Exception:
        mi, ma = np.nan, np.nan
        popt = np.full(7, np.nan)
        r2 = 0.0

    return SpotFit3D(
        fwhm_z=FWHM_CONSTANT * popt[3],
        fwhm_yx=FWHM_CONSTANT * popt[4],
        offset_z=popt[0],
        offset_y=popt[1],
        offset_x=popt[2],
        intens_A=(popt[5] + popt[6]) * (ma - mi),
        intens_B=popt[6] * (ma - mi) + mi,
        r_squared=r2,
    )


# ------------------------------------------------------------------
# Aggregated results
# ------------------------------------------------------------------

@dataclass
class SpotFitParams3D:
    """Aggregated fit parameters for all spots (arrays)."""
    fwhm_z: np.ndarray
    fwhm_yx: np.ndarray
    offset_z: np.ndarray
    offset_y: np.ndarray
    offset_x: np.ndarray
    intens_A: np.ndarray
    intens_B: np.ndarray
    r_squared: np.ndarray


@dataclass
class FitAndMaskResult:
    """Combined result of fitting and mask generation."""
    fit_params: SpotFitParams3D
    mask: np.ndarray


# ------------------------------------------------------------------
# Public API: fit + mask in one pass
# ------------------------------------------------------------------

def fit_and_mask_3d(
    image: np.ndarray,
    spots: np.ndarray,
    mask_shape: tuple,
    window: int = 5,
    max_workers: int = 1,
    progress_callback: callable | None = None,
) -> FitAndMaskResult:
    """Fit 3D Gaussians to spots and paint per-spot masks in a single pass.

    For each spot: fit Gaussian → store params → paint ellipsoid into mask.
    No second loop over results.

    Args:
        image: 3D image array (ZYX). Should NOT include a channel dim.
        spots: (N, 3) array of spot coordinates (z, y, x).
        mask_shape: Shape of the output mask (same as image).
        window: Half-size of the fitting window around each spot.
        max_workers: Number of parallel workers for fitting.
        progress_callback: Optional callable(current, total) invoked after
            each spot is processed.

    Returns:
        FitAndMaskResult with aggregated fit_params and the binary mask.
    """
    n = len(spots)
    mask = np.zeros(mask_shape, dtype=np.uint8)

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return FitAndMaskResult(
            fit_params=SpotFitParams3D(
                fwhm_z=empty, fwhm_yx=empty,
                offset_z=empty, offset_y=empty, offset_x=empty,
                intens_A=empty, intens_B=empty, r_squared=empty,
            ),
            mask=mask,
        )

    # Pad image so fitting windows near edges don't go out of bounds
    padded = np.pad(image, window, mode="reflect")
    padded_centers = np.asarray(spots) + window

    keys = [f.name for f in fields(SpotFit3D)]
    arrays = {k: np.empty(n, dtype=np.float64) for k in keys}

    def _consume(i: int, fit: SpotFit3D):
        """Store params and paint mask — called once per spot."""
        for k in keys:
            arrays[k][i] = getattr(fit, k)
        fit.paint_mask(mask, spots[i])
        if progress_callback is not None:
            progress_callback(i + 1, n)

    # Single pass: fit → store → paint for each spot as results arrive
    if max_workers == 1:
        for i, c in enumerate(padded_centers):
            _consume(i, _fit_single_3d(c, padded, window))
    else:
        _fn = partial(_fit_single_3d, image=padded, window=window)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for i, fit in enumerate(pool.map(_fn, padded_centers)):
                _consume(i, fit)

    return FitAndMaskResult(
        fit_params=SpotFitParams3D(**arrays),
        mask=mask,
    )

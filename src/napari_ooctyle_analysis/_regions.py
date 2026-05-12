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

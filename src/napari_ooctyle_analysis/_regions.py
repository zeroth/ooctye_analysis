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

    def __post_init__(self):
        center = np.array(self.center_px, dtype=np.float64, copy=True)
        center.setflags(write=False)
        object.__setattr__(self, "center_px", center)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        scale.setflags(write=False)
        object.__setattr__(self, "scale", scale)


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


def contains_sphere(outer: Sphere, inner: Sphere) -> bool:
    """True iff every point of inner lies inside (or on the surface of) outer.

    Physical-space distance check (anisotropic-scale aware), using the outer
    sphere's scale (both spheres are built from the same image scale).
    """
    delta_px = inner.center_px - outer.center_px
    distance_physical = float(np.linalg.norm(delta_px * outer.scale))
    return distance_physical + inner.radius_physical <= outer.radius_physical + 1e-9

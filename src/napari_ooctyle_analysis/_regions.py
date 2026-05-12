"""Pure geometry primitives for spherical regions of interest.

This module has no Qt and no napari imports - it is pure NumPy so it can be
unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

"""Pure logic for segmentation: channel splitting, exclusion, data loading."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from napari_ooctyle_analysis._regions import (
    Sphere,
    apply_sphere_to_mask,
    build_sphere_mesh,
    filter_spots,
)


def reorder_to_zyx(data: np.ndarray, axis_order: str) -> tuple:
    """Reorder an array from arbitrary axis order to (C, Z, Y, X) or (Z, Y, X).

    Returns (channels_list, was_multichannel) where channels_list is a list
    of 3D ZYX arrays (one per channel).
    """
    axis_order = axis_order.upper().strip()
    expected_ndim = len(axis_order)
    if data.ndim != expected_ndim:
        raise ValueError(
            f"Image has {data.ndim} dimensions but axis order "
            f"'{axis_order}' expects {expected_ndim}."
        )

    has_channel = "C" in axis_order
    for ch in axis_order:
        if ch not in "CZYX":
            raise ValueError(
                f"Unknown axis '{ch}' in axis order. Use C, Z, Y, X."
            )

    if not has_channel:
        src_order = [axis_order.index(a) for a in "ZYX"]
        reordered = np.transpose(data, src_order)
        return [reordered], False

    src_order = [axis_order.index(a) for a in "CZYX"]
    reordered = np.transpose(data, src_order)
    channels = [reordered[c] for c in range(reordered.shape[0])]
    return channels, True


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


def generate_save_path(base_dir: str) -> Path:
    """Generate a versioned save path: YYYY-MM-DD_v<n>.

    Automatically increments <n> if a folder for today already exists.
    """
    base = Path(base_dir)
    today = date.today().isoformat()
    version = 1
    while True:
        name = f"{today}_v{version}"
        candidate = base / name
        if not candidate.exists():
            return candidate
        version += 1


def load_data_from_dirs(images_dir: str, spots_dir: str):
    """Load paired images and spot coordinates from directories.

    Expects matching filenames: image ``foo.tif`` pairs with ``foo.csv``.
    """
    import pandas as pd
    import tifffile

    images_path = Path(images_dir)
    spots_path = Path(spots_dir)

    image_files = sorted(images_path.glob("*.tif")) + sorted(
        images_path.glob("*.tiff")
    )
    if not image_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {images_dir}")

    images = []
    spots = []
    for img_file in image_files:
        csv_file = spots_path / f"{img_file.stem}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Missing spots CSV for {img_file.name}: expected {csv_file}"
            )
        images.append(tifffile.imread(str(img_file)))
        df = pd.read_csv(csv_file)
        spots.append(df.values.astype(np.float32))
    return images, spots


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

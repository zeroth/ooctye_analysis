"""Pure logic and chart generation for overlap analysis."""
from __future__ import annotations

import numpy as np


def compute_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict:
    """Compute voxel-level overlap between two binary masks."""
    bool_a = mask_a > 0
    bool_b = mask_b > 0
    overlap = bool_a & bool_b

    n_a = int(bool_a.sum())
    n_b = int(bool_b.sum())
    n_overlap = int(overlap.sum())

    return {
        "n_a": n_a,
        "n_b": n_b,
        "n_overlap": n_overlap,
        "pct_a": (n_overlap / n_a * 100) if n_a > 0 else 0.0,
        "pct_b": (n_overlap / n_b * 100) if n_b > 0 else 0.0,
        "overlap_mask": overlap.astype(np.uint8),
    }


def short_label(name: str, max_len: int = 15) -> str:
    """Shorten a layer name for chart labels."""
    return name if len(name) <= max_len else name[: max_len - 1] + "\u2026"


def create_overlap_figure(name_a: str, name_b: str, result: dict):
    """Create a matplotlib Figure with overlap bar charts.

    Returns a ``matplotlib.figure.Figure`` (not a canvas — the caller
    wraps it in a Qt widget).
    """
    from matplotlib.figure import Figure

    short_a = short_label(name_a)
    short_b = short_label(name_b)

    fig = Figure(figsize=(5.5, 3.5), dpi=100)
    fig.suptitle(f"{short_a}  vs  {short_b}", fontsize=10, fontweight="bold")

    # --- Left: bar chart of voxel counts ---
    ax1 = fig.add_subplot(121)
    labels = [short_a, short_b, "Overlap"]
    values = [result["n_a"], result["n_b"], result["n_overlap"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax1.bar(range(len(labels)), values, color=colors)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Voxels")
    ax1.set_title("Voxel Counts", fontsize=9)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # --- Right: percentage bars ---
    ax2 = fig.add_subplot(122)
    pct_a = result["pct_a"]
    pct_b = result["pct_b"]
    bar_labels = [f"of {short_a}", f"of {short_b}"]
    pcts = [pct_a, pct_b]
    bar_colors = ["#4C72B0", "#DD8452"]
    bars = ax2.bar(range(2), pcts, color=bar_colors, width=0.5)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha="right")
    ax2.set_ylabel("Overlap %")
    ax2.set_ylim(0, max(110, max(pcts) + 10))
    ax2.set_title("Overlap Fraction", fontsize=9)
    for bar, pct in zip(bars, pcts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
    return fig


def create_intensity_histogram_figure(name_b: str, split: dict):
    """Two side-by-side histograms of B per-spot mean intensity.

    Left = spots overlapping A, right = non-overlapping spots. Shared x-range and
    bins computed from the pooled values so the panels are directly comparable.
    Empty arrays render an empty axes (no crash). Returns a matplotlib Figure.
    """
    from matplotlib.figure import Figure

    overlap = np.asarray(split["overlap"], dtype=np.float64)
    non_overlap = np.asarray(split["non_overlap"], dtype=np.float64)
    short_b = short_label(name_b)

    fig = Figure(figsize=(6.0, 3.2), dpi=100)
    fig.suptitle(f"{short_b} per-spot mean intensity", fontsize=10, fontweight="bold")

    pooled = np.concatenate([overlap, non_overlap]) if (overlap.size + non_overlap.size) else np.array([])
    if pooled.size:
        lo, hi = float(pooled.min()), float(pooled.max())
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 21)
    else:
        bins = 10

    ax1 = fig.add_subplot(121)
    if overlap.size:
        ax1.hist(overlap, bins=bins, color="#4C72B0")
    ax1.set_title(f"Overlapping A (n={split['n_overlap']})", fontsize=9)
    ax1.set_xlabel("Mean intensity", fontsize=8)
    ax1.set_ylabel("Spot count", fontsize=8)

    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    if non_overlap.size:
        ax2.hist(non_overlap, bins=bins, color="#DD8452")
    ax2.set_title(f"Non-overlapping (n={split['n_non_overlap']})", fontsize=9)
    ax2.set_xlabel("Mean intensity", fontsize=8)

    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.3)
    return fig


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


SPOT_PROPERTIES = ("label", "centroid", "area", "intensity_mean")


def compute_spot_regionprops(label_img: np.ndarray, intensity_img: np.ndarray) -> dict:
    """Per-spot region properties measured on ``intensity_img``.

    Wraps ``skimage.measure.regionprops_table`` over a labeled image. Background
    (label 0) is excluded. Returns a dict of equal-length arrays with keys
    ``label``, ``centroid-0/1/2`` (per image dim), ``area``, ``intensity_mean``.
    An all-zero label image yields the same keys mapping to empty arrays.
    """
    from skimage.measure import regionprops_table

    table = regionprops_table(
        label_img.astype(np.int32, copy=False),
        intensity_image=intensity_img,
        properties=SPOT_PROPERTIES,
    )
    return {key: np.asarray(value) for key, value in table.items()}


def split_spot_intensities(
    label_img_b: np.ndarray,
    mask_a: np.ndarray,
    table: dict,
) -> dict:
    """Split B's per-spot ``intensity_mean`` into overlap-with-A vs non-overlap.

    A B spot overlaps A if ANY of its voxels lie in ``mask_a`` (> 0). Returns
    ``{"overlap": ndarray, "non_overlap": ndarray, "n_overlap": int,
    "n_non_overlap": int}``. Labels in ``table`` must match ``label_img_b``.
    """
    labels = np.asarray(table["label"])
    means = np.asarray(table["intensity_mean"], dtype=np.float64)
    overlapping = np.unique(label_img_b[(mask_a > 0) & (label_img_b > 0)])
    is_overlap = np.isin(labels, overlapping)
    overlap = means[is_overlap]
    non_overlap = means[~is_overlap]
    return {
        "overlap": overlap,
        "non_overlap": non_overlap,
        "n_overlap": int(overlap.size),
        "n_non_overlap": int(non_overlap.size),
    }


_SPOT_TABLE_COLUMN_ORDER = (
    "label", "centroid-0", "centroid-1", "centroid-2", "area", "intensity_mean",
)


def spot_table_to_rows(table: dict) -> tuple[list, list]:
    """Convert a regionprops table dict into (header, rows) for CSV writing.

    Columns follow ``_SPOT_TABLE_COLUMN_ORDER``, keeping only keys present in the
    table (any extra keys are appended after, in sorted order). ``rows`` is the
    column-wise zip; an empty table yields the header and an empty row list.
    """
    ordered = [c for c in _SPOT_TABLE_COLUMN_ORDER if c in table]
    extra = sorted(k for k in table if k not in ordered)
    header = ordered + extra
    columns = [np.asarray(table[h]).tolist() for h in header]
    rows = [list(r) for r in zip(*columns)] if columns and len(columns[0]) else []
    return header, rows


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

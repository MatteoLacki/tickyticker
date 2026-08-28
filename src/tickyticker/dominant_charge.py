"""Render the dominant charge state from a charge-count tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def dominant_charge_map(counts: np.ndarray) -> np.ndarray:
    """Return a 2D map of the charge with the largest count in each box.

    ``counts`` has shape ``(charge, 1/K0 bin, m/z bin)``. Empty boxes are 0.
    Equal non-zero counts resolve to the higher charge.
    """
    if counts.ndim != 3 or counts.shape[0] != 3:
        raise ValueError("counts must have shape (3, mobility_bins, mz_bins).")

    max_counts = counts.max(axis=0)
    # Reverse charge order before argmax so ties prefer charge 3, then 2.
    dominant = 3 - np.argmax(counts[::-1], axis=0)
    return np.where(max_counts == 0, 0, dominant).astype(np.uint8)


def write_ascii_map(
    dominant: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    output_path: Path,
) -> None:
    """Write an ASCII map; rows run from high to low inverse ion mobility."""
    if dominant.shape != (mobility_edges.size - 1, mz_edges.size - 1):
        raise ValueError("Map shape does not match the supplied bin edges.")

    lines = [
        "# Dominant charge: . = no evidence; 1, 2, 3 = highest count.",
        f"# m/z bins: [{mz_edges[0]:.0f}, {mz_edges[-1]:.0f}) Da; 1 Da wide.",
        f"# 1/K0 bins: [{mobility_edges[0]:.6f}, {mobility_edges[-1]:.6f}]; high to low rows.",
    ]
    symbols = np.array([".", "1", "2", "3"])
    for row in dominant[::-1]:
        lines.append("".join(symbols[row]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_dominant_charge_map(
    dominant: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    output_path: Path,
) -> None:
    """Save the categorical dominant-charge view as a PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colors
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires the optional dependency. Install with: pip install tickyticker[dev]"
        ) from exc

    colour_map = colors.ListedColormap(["#f5f5f5", "#4c78a8", "#f58518", "#54a24b"])
    normalizer = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colour_map.N)
    figure, axis = plt.subplots(figsize=(16, 3.6), constrained_layout=True)
    image = axis.pcolormesh(
        mz_edges, mobility_edges, dominant, shading="auto", cmap=colour_map, norm=normalizer
    )
    axis.set(xlabel="m/z (1 Da bins)", ylabel="1/K0", title="Dominant charge by isotope-pattern count")
    colour_bar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3], pad=0.01)
    colour_bar.set_ticklabels(["none", "1", "2", "3"])
    colour_bar.set_label("charge")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

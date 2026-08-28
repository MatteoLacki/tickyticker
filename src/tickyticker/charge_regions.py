"""Create charge-resolved precursor-intensity maps from timsTOF MS1 spectra."""

from __future__ import annotations

from time import perf_counter

import argparse
from pathlib import Path

import numpy as np
import opentimspy
from numba import get_num_threads, njit, prange, set_num_threads
from opentimspy import OpenTIMS

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors
    import matplotlib.pyplot as plt
except ImportError:
    colors = None
    plt = None


CHARGES = np.array([1, 2, 3], dtype=np.int64)
_COLUMNS = ("scan", "mz", "intensity")


@njit(cache=True, parallel=True)
def _process_ms1_frame(
    scan: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
    intensities: np.ndarray,
    sampled_scans: np.ndarray,
    scan_mobility_bins: np.ndarray,
    scan_selected: np.ndarray,
    mz_min: float,
    mz_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
    require_decreasing_isotopes: bool,
) -> None:
    """Process one frame in parallel over exclusive charge planes."""
    if scan.size == 0:
        return

    mz_bin_count = intensities.shape[2]
    starts = np.empty(scan.size, dtype=np.int64)
    stops = np.empty(scan.size, dtype=np.int64)
    group_count = 0
    for peak_index in range(scan.size):
        if peak_index == 0 or scan[peak_index] != scan[peak_index - 1]:
            if group_count > 0:
                stops[group_count - 1] = peak_index
            starts[group_count] = peak_index
            group_count += 1
    stops[group_count - 1] = scan.size

    for charge_index in prange(intensities.shape[0]):
        charge = charge_index + 1
        for group_index in range(group_count):
            scan_number = scan[starts[group_index]]
            if not scan_selected[scan_number]:
                continue
            mobility_bin = scan_mobility_bins[scan_number]
            if charge_index == 0:
                sampled_scans[mobility_bin] += 1

            start = starts[group_index]
            stop = stops[group_index]
            scan_mz = mz[start:stop]
            scan_intensity = intensity[start:stop]
            spectrum_max = np.max(scan_intensity)
            if spectrum_max <= 0 or scan_mz.size < 3:
                continue

            for peak_index in range(1, scan_mz.size - 1):
                precursor_intensity = scan_intensity[peak_index]
                if precursor_intensity < min_intensity:
                    continue
                if precursor_intensity < scan_intensity[peak_index - 1] or precursor_intensity <= scan_intensity[peak_index + 1]:
                    continue

                precursor = scan_mz[peak_index]
                mz_bin = int(np.floor(precursor - mz_min))
                if mz_bin < 0 or mz_bin >= mz_bin_count or precursor >= mz_max:
                    continue

                score = 0.0
                matched_isotopes = 0
                previous_intensity = precursor_intensity
                decreasing = True
                for isotope_index in range(1, isotope_count + 1):
                    target = precursor + isotope_index / charge
                    tolerance = target * ppm * 1e-6
                    left = np.searchsorted(scan_mz, target - tolerance, side="left")
                    right = np.searchsorted(scan_mz, target + tolerance, side="right")
                    if right > left:
                        matched_intensity = np.max(scan_intensity[left:right])
                        score += matched_intensity / spectrum_max
                        if matched_intensity >= previous_intensity:
                            decreasing = False
                        previous_intensity = matched_intensity
                        matched_isotopes += 1

                if matched_isotopes == isotope_count and score >= min_score and (not require_decreasing_isotopes or decreasing):
                    intensities[charge_index, mobility_bin, mz_bin] += precursor_intensity


def dominant_charge_map(intensities: np.ndarray) -> np.ndarray:
    """Return the charge with the largest summed intensity per map box."""
    if intensities.ndim != 3 or intensities.shape[0] != 3:
        raise ValueError("intensities must have shape (3, mobility_bins, mz_bins).")
    maxima = intensities.max(axis=0)
    dominant = 3 - np.argmax(intensities[::-1], axis=0)
    return np.where(maxima == 0, 0, dominant).astype(np.uint8)


def _require_matplotlib() -> None:
    if plt is None or colors is None:
        raise RuntimeError("Plotting requires the optional dependency. Install with: pip install tickyticker[dev]")


def _write_ascii_map(dominant: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path) -> None:
    """Write an ASCII dominant-charge map, ordered high to low 1/K0."""
    lines = [
        "# Dominant charge: . = no evidence; 1, 2, 3 = highest summed intensity.",
        f"# m/z bins: [{mz_edges[0]:.0f}, {mz_edges[-1]:.0f}) Da; 1 Da wide.",
        f"# 1/K0 bins: [{mobility_edges[0]:.6f}, {mobility_edges[-1]:.6f}]; high to low rows.",
    ]
    symbols = np.array([".", "1", "2", "3"])
    lines.extend("".join(symbols[row]) for row in dominant[::-1])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_intensity_maps(intensities: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path) -> None:
    """Save per-charge precursor-intensity heatmaps."""
    _require_matplotlib()
    figure, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    for charge, axis in zip(CHARGES, axes, strict=True):
        image = axis.pcolormesh(
            mz_edges, mobility_edges, np.log1p(intensities[charge - 1]), shading="auto", cmap="magma"
        )
        axis.set(ylabel="1/K0", title=f"Charge {charge}: log(1 + summed precursor intensity)")
        figure.colorbar(image, ax=axis, pad=0.01, label="log(1 + intensity)")
    axes[-1].set_xlabel("m/z (1 Da bins)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_dominant_charge_map(dominant: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path) -> None:
    """Save the categorical dominant-charge view as a PNG."""
    _require_matplotlib()
    colour_map = colors.ListedColormap(["#f5f5f5", "#4c78a8", "#f58518", "#54a24b"])
    normalizer = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colour_map.N)
    figure, axis = plt.subplots(figsize=(16, 3.6), constrained_layout=True)
    image = axis.pcolormesh(mz_edges, mobility_edges, dominant, shading="auto", cmap=colour_map, norm=normalizer)
    axis.set(xlabel="m/z (1 Da bins)", ylabel="1/K0", title="Dominant charge by summed precursor intensity")
    colour_bar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3], pad=0.01)
    colour_bar.set_ticklabels(["none", "1", "2", "3"])
    colour_bar.set_label("charge")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyse(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    ppm: float = 10.0,
    isotope_count: int = 3,
    mobility_bins: int = 100,
    mz_min: float = 100.0,
    mz_max: float = 1700.0,
    min_intensity: float = 0.0,
    min_score: float = 0.0,
    require_decreasing_isotopes: bool = False,
    threads: int = 3,
    scans_per_mobility_bin: int = 0,
) -> Path:
    """Create a 3D charge-resolved intensity map and associated plots."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if ppm <= 0 or isotope_count < 1 or mobility_bins < 1 or mz_max <= mz_min:
        raise ValueError("ppm, isotope_count, mobility_bins, and m/z limits must be positive and valid.")
    if threads < 1 or scans_per_mobility_bin < 0:
        raise ValueError("threads must be at least 1 and scans_per_mobility_bin cannot be negative.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    started = perf_counter()
    set_num_threads(min(threads, len(CHARGES)))
    output_dir.mkdir(parents=True, exist_ok=True)
    mz_edges = np.arange(np.floor(mz_min), np.ceil(mz_max) + 1.0, 1.0)
    mz_min, mz_max = float(mz_edges[0]), float(mz_edges[-1])
    if not opentimspy.bruker_bridge_present:
        opentimspy.setup_opensource()
    with OpenTIMS(dataset_path) as dataset:
        ms1_frames = np.asarray(dataset.ms1_frames, dtype=np.uint32)
        if not ms1_frames.size:
            raise RuntimeError("No MS1 frames were found.")
        scan_numbers = np.arange(dataset.min_scan, dataset.max_scan + 1, dtype=np.uint32)
        scan_mobility = dataset.scan_to_inv_ion_mobility(
            scan_numbers, np.full(scan_numbers.size, ms1_frames[0], dtype=np.uint32)
        )
        mobility_min, mobility_max = np.sort(scan_mobility[[0, -1]])
        mobility_edges = np.linspace(mobility_min, mobility_max, mobility_bins + 1)
        scan_mobility_bins = np.searchsorted(mobility_edges, scan_mobility, side="right") - 1
        scan_mobility_bins = np.clip(scan_mobility_bins, 0, mobility_bins - 1).astype(np.int64)
        scan_selected = np.zeros(dataset.max_scan + 1, dtype=np.bool_)
        if scans_per_mobility_bin == 0:
            scan_selected[scan_numbers] = True
        else:
            for mobility_bin in range(mobility_bins):
                members = scan_numbers[scan_mobility_bins == mobility_bin]
                if members.size <= scans_per_mobility_bin:
                    scan_selected[members] = True
                elif scans_per_mobility_bin == 1:
                    scan_selected[members[(members.size - 1) // 2]] = True
                else:
                    positions = np.rint(np.linspace(0, members.size - 1, scans_per_mobility_bin)).astype(np.int64)
                    scan_selected[members[positions]] = True
        scan_mobility_bin_lookup = np.zeros(dataset.max_scan + 1, dtype=np.int64)
        scan_mobility_bin_lookup[scan_numbers] = scan_mobility_bins

        intensities = np.zeros((len(CHARGES), mobility_bins, mz_edges.size - 1), dtype=np.float64)
        sampled_scans = np.zeros(mobility_bins, dtype=np.uint32)
        for frame_number, frame in enumerate(dataset.query_iter(ms1_frames, columns=_COLUMNS), start=1):
            _process_ms1_frame(
                frame["scan"], frame["mz"].astype(np.float64), frame["intensity"].astype(np.float64),
                intensities, sampled_scans, scan_mobility_bin_lookup, scan_selected,
                mz_min, mz_max, ppm, isotope_count, min_intensity, min_score,
                require_decreasing_isotopes,
            )
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} MS1 frames", flush=True)

    runtime_seconds = perf_counter() - started
    npz_path = output_dir / "charge_region_maps.npz"
    np.savez_compressed(
        npz_path,
        intensities=intensities,
        charges=CHARGES,
        mz_edges=mz_edges,
        mobility_edges=mobility_edges,
        sampled_scans_per_mobility_bin=sampled_scans,
        ppm=np.array(ppm),
        isotope_count=np.array(isotope_count),
        min_intensity=np.array(min_intensity),
        min_score=np.array(min_score),
        require_decreasing_isotopes=np.array(require_decreasing_isotopes),
        threads=np.array(get_num_threads()),
        scans_per_mobility_bin=np.array(scans_per_mobility_bin),
        runtime_seconds=np.array(runtime_seconds),
    )
    _plot_intensity_maps(intensities, mz_edges, mobility_edges, output_dir / "charge_region_intensities.png")
    dominant = dominant_charge_map(intensities)
    _write_ascii_map(dominant, mz_edges, mobility_edges, output_dir / "dominant_charge_map.txt")
    _plot_dominant_charge_map(dominant, mz_edges, mobility_edges, output_dir / "dominant_charge_map.png")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a .d dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--isotope-count", type=int, default=3)
    parser.add_argument("--mobility-bins", type=int, default=100)
    parser.add_argument("--mz-min", type=float, default=100.0)
    parser.add_argument("--mz-max", type=float, default=1700.0)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--require-decreasing-isotopes", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--scans-per-mobility-bin", type=int, default=0)
    args = parser.parse_args()
    output = analyse(
        args.dataset, args.output_dir, ppm=args.ppm, isotope_count=args.isotope_count,
        mobility_bins=args.mobility_bins, mz_min=args.mz_min, mz_max=args.mz_max,
        min_intensity=args.min_intensity, min_score=args.min_score,
        require_decreasing_isotopes=args.require_decreasing_isotopes, threads=args.threads,
        scans_per_mobility_bin=args.scans_per_mobility_bin,
    )
    print(f"Saved intensity maps to {output}")
    print(f"Saved plots to {output.parent}")


if __name__ == "__main__":
    main()

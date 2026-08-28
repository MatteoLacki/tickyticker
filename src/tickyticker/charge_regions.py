"""Create charge-resolved precursor-intensity maps from timsTOF MS1 spectra."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import opentimspy
from numba import njit
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
_COLUMNS = ("scan", "mz", "intensity", "inv_ion_mobility")


@njit(cache=True)
def _accumulate_scan_intensities(
    mz: np.ndarray,
    intensity: np.ndarray,
    intensity_map: np.ndarray,
    mobility_bin: int,
    mz_min: float,
    mz_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
    require_decreasing_isotopes: bool,
) -> None:
    """Add accepted local-max precursor intensities directly to the 3D map."""
    if mz.size < 3:
        return

    spectrum_max = np.max(intensity)
    if spectrum_max <= 0:
        return

    mz_bin_count = intensity_map.shape[2]
    for peak_index in range(1, mz.size - 1):
        precursor_intensity = intensity[peak_index]
        if precursor_intensity < min_intensity:
            continue
        if precursor_intensity < intensity[peak_index - 1] or precursor_intensity <= intensity[peak_index + 1]:
            continue

        precursor = mz[peak_index]
        mz_bin = int(np.floor(precursor - mz_min))
        if mz_bin < 0 or mz_bin >= mz_bin_count or precursor >= mz_max:
            continue

        for charge in range(1, 4):
            score = 0.0
            matched_isotopes = 0
            previous_intensity = precursor_intensity
            decreasing = True
            for isotope_index in range(1, isotope_count + 1):
                target = precursor + isotope_index / charge
                tolerance = target * ppm * 1e-6
                left = np.searchsorted(mz, target - tolerance, side="left")
                right = np.searchsorted(mz, target + tolerance, side="right")
                if right > left:
                    matched_intensity = np.max(intensity[left:right])
                    score += matched_intensity / spectrum_max
                    if matched_intensity >= previous_intensity:
                        decreasing = False
                    previous_intensity = matched_intensity
                    matched_isotopes += 1

            if matched_isotopes == isotope_count and score >= min_score and (not require_decreasing_isotopes or decreasing):
                intensity_map[charge - 1, mobility_bin, mz_bin] += precursor_intensity


def _accumulate_frame(
    frame: dict[str, np.ndarray],
    intensity_map: np.ndarray,
    mz_min: float,
    mz_max: float,
    mobility_min: float,
    mobility_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
    require_decreasing_isotopes: bool,
) -> None:
    """Analyse every scan in one frame and add precursor intensities to the map."""
    scan = frame["scan"]
    mz = frame["mz"].astype(np.float64, copy=False)
    intensity = frame["intensity"].astype(np.float64, copy=False)
    mobility = frame["inv_ion_mobility"].astype(np.float64, copy=False)
    if not mz.size:
        return

    # Bruker .d peak data are frame-scan-TOF sorted; m/z is monotonic with TOF.
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    stops = np.r_[starts[1:], scan.size]
    mobility_bin_count = intensity_map.shape[1]
    for start, stop in zip(starts, stops, strict=True):
        mobility_bin = int(
            np.floor((mobility[start] - mobility_min) / (mobility_max - mobility_min) * mobility_bin_count)
        )
        mobility_bin = min(max(mobility_bin, 0), mobility_bin_count - 1)
        _accumulate_scan_intensities(
            mz[start:stop], intensity[start:stop], intensity_map, mobility_bin,
            mz_min, mz_max, ppm, isotope_count, min_intensity, min_score, require_decreasing_isotopes,
        )


def _select_scans(
    frame: dict[str, np.ndarray],
    mobility_min: float,
    mobility_max: float,
    mobility_bins: int,
    scans_per_mobility_bin: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Select evenly spaced scans in every output mobility bin.

    A zero value selects every populated scan. Input peaks are frame-scan-TOF
    sorted, so masking preserves the order required by the detector.
    """
    scan = frame["scan"]
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    scan_mobility = frame["inv_ion_mobility"][starts]
    scan_bins = np.floor(
        (scan_mobility - mobility_min) / (mobility_max - mobility_min) * mobility_bins
    ).astype(np.int64)
    scan_bins = np.clip(scan_bins, 0, mobility_bins - 1)

    if scans_per_mobility_bin == 0:
        selected = np.arange(starts.size)
    else:
        selections = []
        for mobility_bin in range(mobility_bins):
            available = np.flatnonzero(scan_bins == mobility_bin)
            if available.size:
                take = min(scans_per_mobility_bin, available.size)
                selections.append(available[np.linspace(0, available.size - 1, take, dtype=np.int64)])
        selected = np.concatenate(selections) if selections else np.empty(0, dtype=np.int64)
        selected.sort()

    sampled_scans = np.bincount(scan_bins[selected], minlength=mobility_bins).astype(np.uint32)
    if selected.size == starts.size:
        return frame, sampled_scans

    selected_scan_ids = scan[starts[selected]]
    peak_mask = np.isin(scan, selected_scan_ids)
    return {name: values[peak_mask] for name, values in frame.items()}, sampled_scans


def _process_frame_chunk(
    dataset_path: str,
    frame_ids: np.ndarray,
    shape: tuple[int, int, int],
    mz_min: float,
    mz_max: float,
    mobility_min: float,
    mobility_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
    require_decreasing_isotopes: bool,
    scans_per_mobility_bin: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Process one frame chunk into a private intensity map for reduction."""
    if not opentimspy.bruker_bridge_present:
        opentimspy.setup_opensource()

    intensities = np.zeros(shape, dtype=np.float64)
    sampled_scans = np.zeros(shape[1], dtype=np.uint32)
    with OpenTIMS(dataset_path) as dataset:
        for frame in dataset.query_iter(frame_ids, columns=_COLUMNS):
            selected_frame, selected_counts = _select_scans(
                frame, mobility_min, mobility_max, shape[1], scans_per_mobility_bin
            )
            sampled_scans += selected_counts
            _accumulate_frame(
                selected_frame, intensities, mz_min, mz_max, mobility_min, mobility_max,
                ppm, isotope_count, min_intensity, min_score, require_decreasing_isotopes,
            )
    return intensities, sampled_scans, int(frame_ids.size)


def _process_frame_chunks(
    dataset_path: str,
    frame_ids: np.ndarray,
    shape: tuple[int, int, int],
    mz_min: float,
    mz_max: float,
    mobility_min: float,
    mobility_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
    require_decreasing_isotopes: bool,
    workers: int,
    scans_per_mobility_bin: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Process independent MS1-frame chunks and sum private intensity maps."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if scans_per_mobility_bin < 0:
        raise ValueError("scans_per_mobility_bin must be zero or positive.")

    worker_count = min(workers, frame_ids.size)
    chunks = [chunk for chunk in np.array_split(frame_ids, worker_count) if chunk.size]
    arguments: list[tuple[Any, ...]] = [
        (
            dataset_path, chunk, shape, mz_min, mz_max, mobility_min, mobility_max,
            ppm, isotope_count, min_intensity, min_score, require_decreasing_isotopes,
            scans_per_mobility_bin,
        )
        for chunk in chunks
    ]
    if worker_count == 1:
        results = [_process_frame_chunk(*arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process_frame_chunk, *chunk_arguments) for chunk_arguments in arguments]
            results = [future.result() for future in as_completed(futures)]

    intensities = np.zeros(shape, dtype=np.float64)
    sampled_scans = np.zeros(shape[1], dtype=np.uint32)
    for chunk_intensities, chunk_scans, _ in results:
        intensities += chunk_intensities
        sampled_scans += chunk_scans
    return intensities, sampled_scans, worker_count


def dominant_charge_map(intensities: np.ndarray) -> np.ndarray:
    """Return the charge with the largest summed intensity per map box.

    Input shape is ``(charge, 1/K0 bin, m/z bin)``. Empty boxes are 0; ties
    resolve to the higher charge.
    """
    if intensities.ndim != 3 or intensities.shape[0] != 3:
        raise ValueError("intensities must have shape (3, mobility_bins, mz_bins).")
    maxima = intensities.max(axis=0)
    dominant = 3 - np.argmax(intensities[::-1], axis=0)
    return np.where(maxima == 0, 0, dominant).astype(np.uint8)


def _write_ascii_map(
    dominant: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    output_path: Path,
) -> None:
    """Write an ASCII dominant-charge map, ordered high to low 1/K0."""
    if dominant.shape != (mobility_edges.size - 1, mz_edges.size - 1):
        raise ValueError("Map shape does not match the supplied bin edges.")
    lines = [
        "# Dominant charge: . = no evidence; 1, 2, 3 = highest summed intensity.",
        f"# m/z bins: [{mz_edges[0]:.0f}, {mz_edges[-1]:.0f}) Da; 1 Da wide.",
        f"# 1/K0 bins: [{mobility_edges[0]:.6f}, {mobility_edges[-1]:.6f}]; high to low rows.",
    ]
    symbols = np.array([".", "1", "2", "3"])
    lines.extend("".join(symbols[row]) for row in dominant[::-1])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_matplotlib() -> None:
    if plt is None or colors is None:
        raise RuntimeError(
            "Plotting requires the optional dependency. Install with: pip install tickyticker[dev]"
        )


def _plot_intensity_maps(
    intensities: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path
) -> None:
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


def _plot_dominant_charge_map(
    dominant: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path
) -> None:
    """Save the categorical dominant-charge view as a PNG."""
    _require_matplotlib()
    colour_map = colors.ListedColormap(["#f5f5f5", "#4c78a8", "#f58518", "#54a24b"])
    normalizer = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colour_map.N)
    figure, axis = plt.subplots(figsize=(16, 3.6), constrained_layout=True)
    image = axis.pcolormesh(
        mz_edges, mobility_edges, dominant, shading="auto", cmap=colour_map, norm=normalizer
    )
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
    workers: int = 1,
    scans_per_mobility_bin: int = 0,
) -> Path:
    """Create a 3D charge-resolved intensity map and associated plots."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if ppm <= 0 or isotope_count < 1 or mobility_bins < 1 or mz_max <= mz_min:
        raise ValueError("ppm, isotope_count, mobility_bins, and m/z limits must be positive and valid.")
    if workers < 1 or scans_per_mobility_bin < 0:
        raise ValueError("workers must be at least 1 and scans_per_mobility_bin cannot be negative.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mz_edges = np.arange(np.floor(mz_min), np.ceil(mz_max) + 1.0, 1.0)
    mz_min, mz_max = float(mz_edges[0]), float(mz_edges[-1])
    if not opentimspy.bruker_bridge_present:
        opentimspy.setup_opensource()
    with OpenTIMS(dataset_path) as dataset:
        ms1_frames = np.asarray(dataset.ms1_frames, dtype=np.uint32)
        if not ms1_frames.size:
            raise RuntimeError("No MS1 frames were found.")
        scan_bounds = np.array([dataset.min_scan, dataset.max_scan], dtype=np.uint32)
        mobility_min, mobility_max = np.sort(
            dataset.scan_to_inv_ion_mobility(scan_bounds, np.full(2, ms1_frames[0], dtype=np.uint32))
        )

    mobility_edges = np.linspace(mobility_min, mobility_max, mobility_bins + 1)
    shape = (len(CHARGES), mobility_bins, mz_edges.size - 1)
    intensities, sampled_scans, worker_count = _process_frame_chunks(
        str(dataset_path), ms1_frames, shape, mz_min, mz_max, mobility_min, mobility_max,
        ppm, isotope_count, min_intensity, min_score, require_decreasing_isotopes,
        workers, scans_per_mobility_bin,
    )
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
        workers=np.array(worker_count),
        scans_per_mobility_bin=np.array(scans_per_mobility_bin),
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scans-per-mobility-bin", type=int, default=0)
    args = parser.parse_args()
    output = analyse(
        args.dataset, args.output_dir, ppm=args.ppm, isotope_count=args.isotope_count,
        mobility_bins=args.mobility_bins, mz_min=args.mz_min, mz_max=args.mz_max,
        min_intensity=args.min_intensity, min_score=args.min_score,
        require_decreasing_isotopes=args.require_decreasing_isotopes, workers=args.workers,
        scans_per_mobility_bin=args.scans_per_mobility_bin,
    )
    print(f"Saved intensity maps to {output}")
    print(f"Saved plots to {output.parent}")


if __name__ == "__main__":
    main()

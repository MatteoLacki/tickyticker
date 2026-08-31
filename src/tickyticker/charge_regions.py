"""Create charge-resolved precursor-intensity maps from timsTOF MS1 spectra."""

from __future__ import annotations

from time import perf_counter

import argparse
from itertools import chain
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
_COLUMNS = ("scan", "tof", "intensity")
_FINE_MZ_BIN_WIDTH = 1.0 / 12.0
_ISOTOPE_BIN_STEPS = np.array([12, 6, 4], dtype=np.int64)


@njit(cache=True, parallel=True)
def _process_ms1_frame(
    scan: np.ndarray,
    tof: np.ndarray,
    intensity: np.ndarray,
    intensities: np.ndarray,
    all_ms1_intensities: np.ndarray,
    event_histograms: np.ndarray,
    sampled_scans: np.ndarray,
    scan_mobility_bin_lookup: np.ndarray,
    scan_selected: np.ndarray,
    fine_mz_tof_edges: np.ndarray,
    workspaces: np.ndarray,
    touched_bins: np.ndarray,
    isotope_count: int,
    min_intensity: float,
) -> None:
    """Bin and classify one frame in parallel over exclusive mobility slices."""
    if scan.size == 0:
        return

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

    for mobility_bin in prange(intensities.shape[1]):
        spectrum = workspaces[mobility_bin]
        touched = touched_bins[mobility_bin]
        event_histogram = event_histograms[mobility_bin]
        for group_index in range(group_count):
            scan_number = scan[starts[group_index]]
            if not scan_selected[scan_number] or scan_mobility_bin_lookup[scan_number] != mobility_bin:
                continue

            start = starts[group_index]
            stop = stops[group_index]
            touched_count = 0
            for peak_index in range(start, stop):
                raw_intensity = intensity[peak_index]
                if raw_intensity >= 128:
                    event_histogram[127] += 1
                elif raw_intensity == 0:
                    event_histogram[0] += 1
                else:
                    event_histogram[raw_intensity - 1] += 1
                if raw_intensity < min_intensity:
                    continue
                fine_mz_bin = np.searchsorted(fine_mz_tof_edges, tof[peak_index], side="right") - 1
                if fine_mz_bin < 0 or fine_mz_bin >= spectrum.size:
                    continue
                if spectrum[fine_mz_bin] == 0:
                    touched[touched_count] = fine_mz_bin
                    touched_count += 1
                spectrum[fine_mz_bin] += intensity[peak_index]

            sampled_scans[mobility_bin] += 1
            for touched_index in range(touched_count):
                fine_mz_bin = touched[touched_index]
                precursor_intensity = spectrum[fine_mz_bin]
                output_mz_bin = fine_mz_bin // 12
                all_ms1_intensities[mobility_bin, output_mz_bin] += precursor_intensity
                if precursor_intensity < min_intensity:
                    continue
                for charge_index in range(intensities.shape[0] - 1, -1, -1):
                    step = _ISOTOPE_BIN_STEPS[charge_index]
                    previous_intensity = precursor_intensity
                    matches = True
                    for isotope_index in range(1, isotope_count + 1):
                        isotope_bin = fine_mz_bin + isotope_index * step
                        if isotope_bin >= spectrum.size:
                            matches = False
                            break
                        isotope_intensity = spectrum[isotope_bin]
                        if isotope_intensity == 0 or isotope_intensity >= previous_intensity:
                            matches = False
                            break
                        previous_intensity = isotope_intensity
                    if matches:
                        intensities[charge_index, mobility_bin, output_mz_bin] += precursor_intensity
                        break

            for touched_index in range(touched_count):
                spectrum[touched[touched_index]] = 0


@njit(cache=True, parallel=True)
def _best_mz_splits(
    intensities: np.ndarray, mz_start_index: int, mz_stop_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find class-balanced 1+ left and 1+ right m/z splits for each mobility row."""
    mobility_bin_count = intensities.shape[1]
    mz_bin_count = intensities.shape[2]
    if mz_start_index < 0 or mz_stop_index > mz_bin_count or mz_start_index >= mz_stop_index:
        raise ValueError("m/z split limits must select at least one m/z bin.")
    left_splits = np.zeros(mobility_bin_count, dtype=np.int64)
    left_scores = np.zeros(mobility_bin_count, dtype=np.float64)
    right_splits = np.zeros(mobility_bin_count, dtype=np.int64)
    right_scores = np.zeros(mobility_bin_count, dtype=np.float64)
    for mobility_bin in prange(mobility_bin_count):
        total_one = 0.0
        total_other = 0.0
        for mz_bin in range(mz_start_index, mz_stop_index):
            total_one += intensities[0, mobility_bin, mz_bin]
            total_other += intensities[1, mobility_bin, mz_bin] + intensities[2, mobility_bin, mz_bin]

        if total_one == 0.0 or total_other == 0.0:
            continue

        cumulative_one = 0.0
        cumulative_other = 0.0
        best_left_score = 1.0
        best_right_score = 1.0
        best_left_split = mz_start_index
        best_right_split = mz_start_index
        for split in range(mz_start_index + 1, mz_stop_index + 1):
            mz_bin = split - 1
            cumulative_one += intensities[0, mobility_bin, mz_bin]
            cumulative_other += intensities[1, mobility_bin, mz_bin] + intensities[2, mobility_bin, mz_bin]
            left_score = cumulative_one / total_one + (total_other - cumulative_other) / total_other
            right_score = (total_one - cumulative_one) / total_one + cumulative_other / total_other
            if left_score > best_left_score:
                best_left_score = left_score
                best_left_split = split
            if right_score > best_right_score:
                best_right_score = right_score
                best_right_split = split
        left_splits[mobility_bin] = best_left_split
        left_scores[mobility_bin] = best_left_score
        right_splits[mobility_bin] = best_right_split
        right_scores[mobility_bin] = best_right_score
    return left_splits, left_scores, right_splits, right_scores


def _linear_spline_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return a truncated-linear basis, which extrapolates linearly at both ends."""
    design = np.empty((x.size, knots.size), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = x
    for knot_index in range(2, knots.size):
        design[:, knot_index] = np.maximum(0.0, x - knots[knot_index - 1])
    return design


def fit_linear_spline_charge_border(
    intensities: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    point_weights: np.ndarray,
    mz_start_index: int,
    mz_stop_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fit a raw-intensity-weighted spline through class-balanced row splits."""
    left_splits, left_scores, right_splits, right_scores = _best_mz_splits(
        intensities, mz_start_index, mz_stop_index
    )
    one_left = left_scores.sum() >= right_scores.sum()
    split_indices = left_splits if one_left else right_splits
    scores = left_scores if one_left else right_scores
    mobility_centers = (mobility_edges[:-1] + mobility_edges[1:]) / 2.0
    split_mz = mz_edges[split_indices]
    if point_weights.shape != scores.shape:
        raise ValueError("point_weights must contain one total-intensity weight per mobility bin.")
    one_evidence = intensities[0, :, mz_start_index:mz_stop_index].sum(axis=1)
    other_evidence = intensities[1:, :, mz_start_index:mz_stop_index].sum(axis=(0, 2))
    valid = (scores > 0) & (point_weights > 0) & (one_evidence > 0) & (other_evidence > 0)
    valid_count = np.count_nonzero(valid)
    if valid_count < 3:
        raise RuntimeError("At least three mobility bins with both 1+ and 2+/3+ evidence are required.")

    knot_count = max(2, int(np.ceil(valid_count / 10.0)))
    knots = np.linspace(mobility_centers[valid].min(), mobility_centers[valid].max(), knot_count)
    design = _linear_spline_design(mobility_centers[valid], knots)
    sqrt_weights = np.sqrt(point_weights[valid])
    coefficients = np.linalg.lstsq(design * sqrt_weights[:, None], split_mz[valid] * sqrt_weights, rcond=None)[0]
    fitted_mz = _linear_spline_design(mobility_centers, knots) @ coefficients
    return split_indices, split_mz, scores, valid, knots, coefficients, fitted_mz, one_left


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




def _plot_event_intensity_distribution(
    histogram: np.ndarray, min_intensity: float, output_path: Path
) -> None:
    """Plot selected raw MS1 event counts, retaining a final >=128 intensity bin."""
    _require_matplotlib()
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.bar(np.arange(1, 128), histogram[:127], width=0.9, color="#4c78a8", label="exact intensity")
    axis.bar(128, histogram[127], width=0.9, color="#f58518", label="intensity ≥128")
    axis.axvline(min_intensity, color="#e45756", linewidth=1.5, label=f"minimum intensity = {min_intensity:g}")
    axis.set(
        xlabel="raw event intensity (final bin is ≥128)", ylabel="event count",
        title="Selected raw MS1 event-intensity distribution", yscale="log", xlim=(0, 132)
    )
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

def _plot_charge_border(
    all_ms1_intensities: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    split_mz: np.ndarray,
    valid_points: np.ndarray,
    fitted_mz: np.ndarray,
    border_mz_left: float,
    border_mz_right: float,
    output_path: Path,
) -> None:
    """Plot per-row split points and their fitted linear spline over all MS1 intensity."""
    _require_matplotlib()
    mobility_centers = (mobility_edges[:-1] + mobility_edges[1:]) / 2.0
    figure, axis = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    image = axis.pcolormesh(
        mz_edges, mobility_edges, np.log1p(all_ms1_intensities), shading="auto", cmap="magma"
    )
    axis.scatter(
        split_mz[valid_points], mobility_centers[valid_points], s=10, c="white", edgecolors="black",
        linewidths=0.25, label="best row split (both charge classes present)"
    )
    axis.plot(fitted_mz, mobility_centers, color="#00d4ff", linewidth=2.0, label="linear spline border")
    axis.axvline(border_mz_left, color="white", linestyle="--", linewidth=1.0, label="border m/z limits")
    axis.axvline(border_mz_right, color="white", linestyle="--", linewidth=1.0)
    axis.set(xlabel="m/z (1 Da bins)", ylabel="1/K0", title="Charge-1 versus charge-2/3 linear-spline border")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, pad=0.01, label="log(1 + all MS1 intensity)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyse(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    isotope_count: int = 3,
    mobility_bins: int = 100,
    mz_min: float = 100.0,
    mz_max: float = 1700.0,
    min_intensity: float = 30.0,
    border_mz_left: float = 350.0,
    border_mz_right: float = 1200.0,
    frame_stride: int = 1,
    threads: int = 3,
    scans_per_mobility_bin: int = 0,
) -> Path:
    """Create charge-resolved maps and a linear-spline charge-1 boundary."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if isotope_count < 1 or mobility_bins < 1 or mz_max <= mz_min:
        raise ValueError("isotope_count, mobility_bins, and m/z limits must be positive and valid.")
    if min_intensity < 0 or frame_stride < 1 or threads < 1 or scans_per_mobility_bin < 0:
        raise ValueError("min_intensity must be non-negative, frame_stride and threads at least 1, and scan sampling non-negative.")
    if not mz_min <= border_mz_left < border_mz_right <= mz_max:
        raise ValueError("border m/z limits must be ordered and lie within the analysis m/z range.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    started = perf_counter()
    set_num_threads(min(threads, mobility_bins))
    output_dir.mkdir(parents=True, exist_ok=True)
    mz_edges = np.arange(np.floor(mz_min), np.ceil(mz_max) + 1.0, 1.0)
    mz_min, mz_max = float(mz_edges[0]), float(mz_edges[-1])
    mz_centers = (mz_edges[:-1] + mz_edges[1:]) / 2.0
    border_mz_mask = (mz_centers >= border_mz_left) & (mz_centers <= border_mz_right)
    border_mz_start_index = int(np.flatnonzero(border_mz_mask)[0])
    border_mz_stop_index = int(np.flatnonzero(border_mz_mask)[-1]) + 1
    fine_bins_per_dalton = int(round(1.0 / _FINE_MZ_BIN_WIDTH))
    fine_mz_bin_count = (mz_edges.size - 1) * fine_bins_per_dalton
    if not opentimspy.bruker_bridge_present:
        opentimspy.setup_opensource()
    with OpenTIMS(dataset_path) as dataset:
        ms1_frames = np.asarray(dataset.ms1_frames, dtype=np.uint32)[::frame_stride]
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

        frames = dataset.query_iter(ms1_frames, columns=_COLUMNS)
        first_frame = next(frames)
        fine_mz_edges = mz_min + np.arange(fine_mz_bin_count + 1) * _FINE_MZ_BIN_WIDTH
        first_frame_numbers = np.full(fine_mz_edges.size, ms1_frames[0], dtype=np.uint32)
        fine_mz_tof_edges = dataset.mz_to_tof_frame_sorted(fine_mz_edges, first_frame_numbers)

        intensities = np.zeros((len(CHARGES), mobility_bins, mz_edges.size - 1), dtype=np.float64)
        all_ms1_intensities = np.zeros((mobility_bins, mz_edges.size - 1), dtype=np.float64)
        event_histograms = np.zeros((mobility_bins, 128), dtype=np.uint64)
        sampled_scans = np.zeros(mobility_bins, dtype=np.uint32)
        workspaces = np.zeros((mobility_bins, fine_mz_bin_count), dtype=np.float64)
        touched_bins = np.empty((mobility_bins, fine_mz_bin_count), dtype=np.int64)
        for frame_number, frame in enumerate(chain((first_frame,), frames), start=1):
            _process_ms1_frame(
                frame["scan"], frame["tof"], frame["intensity"], intensities, all_ms1_intensities, event_histograms,
                sampled_scans,
                scan_mobility_bin_lookup, scan_selected, fine_mz_tof_edges, workspaces,
                touched_bins, isotope_count, min_intensity,
            )
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} MS1 frames", flush=True)

    raw_event_intensity_histogram = event_histograms.sum(axis=0, dtype=np.uint64)
    border_fit_weights = all_ms1_intensities[:, border_mz_mask].sum(axis=1)
    split_indices, split_mz, split_scores, border_valid_points, spline_knots, spline_coefficients, fitted_mz, one_left = fit_linear_spline_charge_border(
        intensities, mz_edges, mobility_edges, border_fit_weights, border_mz_start_index, border_mz_stop_index
    )
    one_mask = mz_centers[None, :] < fitted_mz[:, None]
    if not one_left:
        one_mask = ~one_mask
    one_mask &= border_mz_mask[None, :]
    non_one_ms1_intensity = all_ms1_intensities[(~one_mask) & border_mz_mask[None, :]].sum()
    runtime_seconds = perf_counter() - started
    npz_path = output_dir / "charge_region_maps.npz"
    np.savez_compressed(
        npz_path,
        intensities=intensities,
        all_ms1_intensities=all_ms1_intensities,
        raw_event_intensity_histogram=raw_event_intensity_histogram,
        raw_event_intensity_histogram_last_bin_lower_bound=np.array(128, dtype=np.uint16),
        one_charge_mask=one_mask,
        non_one_ms1_intensity=np.array(non_one_ms1_intensity),
        border_mz_left=np.array(border_mz_left),
        border_mz_right=np.array(border_mz_right),
        border_considered_mz_bins=border_mz_mask,
        border_split_indices=split_indices,
        border_split_mz=split_mz,
        border_split_scores=split_scores,
        border_valid_points=border_valid_points,
        border_fit_weights=border_fit_weights,
        border_spline_knots=spline_knots,
        border_spline_coefficients=spline_coefficients,
        border_fitted_mz=fitted_mz,
        border_one_charge_on_low_mz_side=np.array(one_left),
        charges=CHARGES,
        mz_edges=mz_edges,
        mobility_edges=mobility_edges,
        sampled_scans_per_mobility_bin=sampled_scans,
        fine_mz_bin_width=np.array(_FINE_MZ_BIN_WIDTH),
        isotope_count=np.array(isotope_count),
        min_intensity=np.array(min_intensity),
        charge_assignment=np.array("exclusive_highest_charge_first"),
        require_decreasing_isotopes=np.array(True),
        threads=np.array(get_num_threads()),
        scans_per_mobility_bin=np.array(scans_per_mobility_bin),
        frame_stride=np.array(frame_stride),
        visited_ms1_frames=np.array(ms1_frames.size),
        runtime_seconds=np.array(runtime_seconds),
    )
    _plot_intensity_maps(intensities, mz_edges, mobility_edges, output_dir / "charge_region_intensities.png")
    dominant = dominant_charge_map(intensities)
    _plot_dominant_charge_map(dominant, mz_edges, mobility_edges, output_dir / "dominant_charge_map.png")
    _plot_event_intensity_distribution(
        raw_event_intensity_histogram, min_intensity, output_dir / "raw_event_intensity_distribution.png"
    )
    _plot_charge_border(
        all_ms1_intensities, mz_edges, mobility_edges, split_mz, border_valid_points, fitted_mz,
        border_mz_left, border_mz_right, output_dir / "charge_border.png"
    )
    return npz_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a .d dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isotope-count", type=int, default=3, help="Required following isotope peaks.")
    parser.add_argument("--mobility-bins", type=int, default=100)
    parser.add_argument("--mz-min", type=float, default=100.0)
    parser.add_argument("--mz-max", type=float, default=1700.0)
    parser.add_argument("--min-intensity", type=float, default=30.0, help="Ignore raw events below this intensity.")
    parser.add_argument("--border-mz-left", type=float, default=350.0, help="Left m/z limit for border fitting and aggregation.")
    parser.add_argument("--border-mz-right", type=float, default=1200.0, help="Right m/z limit for border fitting and aggregation.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Visit every K-th MS1 frame (default: 1).")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--scans-per-mobility-bin", type=int, default=0)
    args = parser.parse_args()
    output = analyse(
        args.dataset, args.output_dir, isotope_count=args.isotope_count,
        mobility_bins=args.mobility_bins, mz_min=args.mz_min, mz_max=args.mz_max,
        min_intensity=args.min_intensity, border_mz_left=args.border_mz_left, border_mz_right=args.border_mz_right,
        frame_stride=args.frame_stride, threads=args.threads,
        scans_per_mobility_bin=args.scans_per_mobility_bin,
    )
    print(f"Saved intensity maps to {output}")
    print(f"Saved plots to {output.parent}")


if __name__ == "__main__":
    main()

"""Create charge-resolved precursor-intensity maps from timsTOF MS1 spectra."""

from __future__ import annotations

from time import perf_counter

import argparse
import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable

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


@dataclass(frozen=True, slots=True)
class ChargeRegionResult:
    """In-memory numerical result of one charge-region analysis."""

    intensities: np.ndarray
    all_ms1_intensities: np.ndarray
    raw_event_intensity_histogram: np.ndarray
    one_charge_mask: np.ndarray
    non_one_ms1_intensity: float
    border_mz_mask: np.ndarray
    polar_origin: np.ndarray
    line_one: np.ndarray
    line_two: np.ndarray
    polar_boundary_radius: float
    polar_boundary: np.ndarray
    one_charge_is_inner: bool
    charges: np.ndarray
    mz_edges: np.ndarray
    mobility_edges: np.ndarray
    sampled_scans_per_mobility_bin: np.ndarray
    line_data: dict[str, object]
    visited_ms1_frames: int
    runtime_seconds: float
    effective_threads: int


@dataclass(frozen=True, slots=True)
class LineTicResult:
    """In-memory thresholded TIC totals on both sides of a fitted line."""

    tic_below_line: int
    tic_above_line: int
    line_intercept: float
    line_slope: float
    mz_min: float
    mz_max: float
    min_intensity: float
    frame_stride: int
    visited_ms1_frames: int
    runtime_seconds: float
    effective_threads: int


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
    fine_bins_per_output_mz_bin: int,
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
                output_mz_bin = fine_mz_bin // fine_bins_per_output_mz_bin
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


def _robust_line_fit(mz: np.ndarray, mobility: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fit mobility = intercept + slope*mz by weighted Huber IRLS."""
    design = np.column_stack((np.ones(mz.size), mz))
    current_weights = weights.copy()
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(30):
        previous = coefficients.copy()
        sqrt_weights = np.sqrt(current_weights)
        coefficients = np.linalg.lstsq(
            design * sqrt_weights[:, None], mobility * sqrt_weights, rcond=None
        )[0]
        residual = mobility - design @ coefficients
        median = np.median(residual)
        scale = max(1.4826 * np.median(np.abs(residual - median)), 1e-8)
        scaled = np.abs(residual) / (1.345 * scale)
        huber_weights = np.minimum(1.0, 1.0 / np.maximum(scaled, 1.0))
        current_weights = weights * huber_weights
        if np.max(np.abs(coefficients - previous)) < 1e-10:
            break
    return coefficients


def _best_polar_radius(
    radius: np.ndarray, one_charge: np.ndarray
) -> tuple[float, float, float, float]:
    """Return class-balanced inner-1+ and outer-1+ radial thresholds and scores."""
    order = np.argsort(radius)
    radius = radius[order]
    one_charge = one_charge[order]
    total_one = float(one_charge.sum())
    total_other = float(one_charge.size - one_charge.sum())
    if total_one == 0.0 or total_other == 0.0 or radius.size < 2:
        return np.nan, 0.0, np.nan, 0.0
    one_left = np.cumsum(one_charge)
    other_left = np.cumsum(~one_charge)
    inner_scores = one_left[:-1] / total_one + (total_other - other_left[:-1]) / total_other
    outer_scores = other_left[:-1] / total_other + (total_one - one_left[:-1]) / total_one
    inner_index = int(np.argmax(inner_scores))
    outer_index = int(np.argmax(outer_scores))
    inner_radius = (radius[inner_index] + radius[inner_index + 1]) / 2.0
    outer_radius = (radius[outer_index] + radius[outer_index + 1]) / 2.0
    return inner_radius, float(inner_scores[inner_index]), outer_radius, float(outer_scores[outer_index])


def fit_polar_charge_border(
    intensities: np.ndarray,
    all_ms1_intensities: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    border_mz_left: float,
    border_mz_right: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool, np.ndarray]:
    """Fit robust 1+/2+ axes then one class-balanced polar radial boundary."""
    dominant = dominant_charge_map(intensities)
    mz_centers = (mz_edges[:-1] + mz_edges[1:]) / 2.0
    mobility_centers = (mobility_edges[:-1] + mobility_edges[1:]) / 2.0
    considered = (mz_centers >= border_mz_left) & (mz_centers <= border_mz_right)
    interior = (mz_centers >= border_mz_left + 3.0) & (mz_centers <= border_mz_right - 3.0)
    grid_mz, grid_mobility = np.meshgrid(mz_centers, mobility_centers)
    point_weights = np.maximum(all_ms1_intensities, 1.0)

    line_coefficients = []
    for charge in (1, 2):
        line_mask = (dominant == charge) & interior[None, :]
        if np.count_nonzero(line_mask) < 3:
            raise RuntimeError(f"At least three uncensored dominant {charge}+ cells are required for polar fitting.")
        line_coefficients.append(
            _robust_line_fit(grid_mz[line_mask], grid_mobility[line_mask], point_weights[line_mask])
        )
    line_one, line_two = line_coefficients
    slope_difference = line_one[1] - line_two[1]
    if abs(slope_difference) < 1e-10:
        raise RuntimeError("Robust 1+ and 2+ axes are effectively parallel; polar origin is undefined.")
    origin_mz = (line_two[0] - line_one[0]) / slope_difference
    origin_mobility = line_one[0] + line_one[1] * origin_mz

    cloud_mask = ((dominant == 1) | (dominant == 2)) & considered[None, :]
    cloud_mz = grid_mz[cloud_mask]
    cloud_mobility = grid_mobility[cloud_mask]
    cloud_one = dominant[cloud_mask] == 1
    radius = np.hypot(cloud_mz - origin_mz, cloud_mobility - origin_mobility)
    inner_radius, inner_score, outer_radius, outer_score = _best_polar_radius(radius, cloud_one)
    if not np.isfinite(inner_radius) or not np.isfinite(outer_radius):
        raise RuntimeError("Both dominant 1+ and 2+ cells are required for the polar split.")
    one_inner = inner_score >= outer_score
    boundary_radius = inner_radius if one_inner else outer_radius

    grid_radius = np.hypot(grid_mz - origin_mz, grid_mobility - origin_mobility)
    one_mask = grid_radius <= boundary_radius if one_inner else grid_radius >= boundary_radius
    one_mask &= considered[None, :]
    boundary_angles = np.linspace(-np.pi, np.pi, 721)
    boundary = np.stack((
        origin_mz + boundary_radius * np.cos(boundary_angles),
        origin_mobility + boundary_radius * np.sin(boundary_angles),
    ))
    origin = np.array([origin_mz, origin_mobility])
    return origin, line_one, line_two, boundary_radius, boundary, one_inner, one_mask



def fit_alpha_separator_line(
    intensities: np.ndarray,
    all_ms1_intensities: np.ndarray,
    mz_edges: np.ndarray,
    mobility_edges: np.ndarray,
    border_mz_left: float,
    border_mz_right: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit the original-coordinate line from robust 1+/2+ axes and alpha overlap."""
    dominant = dominant_charge_map(intensities)
    mz_centers = (mz_edges[:-1] + mz_edges[1:]) / 2.0
    mobility_centers = (mobility_edges[:-1] + mobility_edges[1:]) / 2.0
    interior = (mz_centers >= border_mz_left + 3.0) & (mz_centers <= border_mz_right - 3.0)
    considered = (mz_centers >= border_mz_left) & (mz_centers <= border_mz_right)
    grid_mz, grid_mobility = np.meshgrid(mz_centers, mobility_centers)
    weights = np.maximum(all_ms1_intensities, 1.0)
    lines = []
    for charge in (1, 2):
        mask = (dominant == charge) & interior[None, :]
        if np.count_nonzero(mask) < 3:
            raise RuntimeError(f"At least three uncensored dominant {charge}+ cells are required for line fitting.")
        lines.append(_robust_line_fit(grid_mz[mask], grid_mobility[mask], weights[mask]))
    line_one, line_two = lines
    slope_difference = line_one[1] - line_two[1]
    if abs(slope_difference) < 1e-10:
        raise RuntimeError("Robust 1+ and 2+ axes are effectively parallel; line origin is undefined.")
    origin_mz = (line_two[0] - line_one[0]) / slope_difference
    origin_mobility = line_one[0] + line_one[1] * origin_mz
    cloud = ((dominant == 1) | (dominant == 2)) & considered[None, :]
    alpha = np.arctan2(grid_mobility[cloud] - origin_mobility, grid_mz[cloud] - origin_mz)
    one = dominant[cloud] == 1
    if not np.any(one) or np.all(one):
        raise RuntimeError("Both dominant 1+ and 2+ cells are required for alpha line fitting.")
    alpha_one = np.quantile(alpha[one], (0.05, 0.95))
    alpha_two = np.quantile(alpha[~one], (0.05, 0.95))
    lower, upper = max(alpha_one[0], alpha_two[0]), min(alpha_one[1], alpha_two[1])
    # A gap is an even cleaner separation than an overlap; its midpoint remains the border.
    separator_alpha = (lower + upper) / 2.0
    slope = float(np.tan(separator_alpha))
    if not np.isfinite(slope):
        raise RuntimeError("Alpha separator is vertical and cannot be represented as mobility = intercept + slope*mz.")
    intercept = float(origin_mobility - slope * origin_mz)
    return intercept, slope, float(separator_alpha), np.array((origin_mz, origin_mobility)), np.stack((line_one, line_two))


@njit(cache=True, parallel=True)
def _sum_line_sides_frame(
    scan: np.ndarray,
    tof: np.ndarray,
    intensity: np.ndarray,
    scan_mobility: np.ndarray,
    fine_mz_tof_edges: np.ndarray,
    fine_mz_centers: np.ndarray,
    intercept: float,
    slope: float,
    min_intensity: float,
    partial_below_tic: np.ndarray,
    partial_above_tic: np.ndarray,
) -> None:
    """Sum filtered raw intensity on both line sides, parallel over scans."""
    if scan.size == 0:
        return
    starts = np.full(scan_mobility.size, -1, dtype=np.int64)
    stops = np.full(scan_mobility.size, -1, dtype=np.int64)
    previous_scan = scan[0]
    starts[previous_scan] = 0
    for peak_index in range(1, scan.size):
        current_scan = scan[peak_index]
        if current_scan != previous_scan:
            stops[previous_scan] = peak_index
            starts[current_scan] = peak_index
            previous_scan = current_scan
    stops[previous_scan] = scan.size
    for scan_number in prange(scan_mobility.size):
        start = starts[scan_number]
        if start < 0:
            continue
        below_total = np.uint64(0)
        above_total = np.uint64(0)
        for peak_index in range(start, stops[scan_number]):
            if intensity[peak_index] < min_intensity:
                continue
            fine_bin = (
                np.searchsorted(
                    fine_mz_tof_edges, tof[peak_index], side="right"
                )
                - 1
            )
            if 0 <= fine_bin < fine_mz_centers.size:
                if (
                    scan_mobility[scan_number]
                    < intercept + slope * fine_mz_centers[fine_bin]
                ):
                    below_total += np.uint64(intensity[peak_index])
                else:
                    above_total += np.uint64(intensity[peak_index])
        partial_below_tic[scan_number] += below_total
        partial_above_tic[scan_number] += above_total


def analyse_line_tic(
    dataset_path: Path | str,
    *,
    intercept: float,
    slope: float,
    mz_min: float,
    mz_max: float,
    min_intensity: float = 30.0,
    threads: int = 3,
    frame_stride: int = 1,
    progress: Callable[[str], None] | None = None,
) -> LineTicResult:
    """Return thresholded MS1 TIC below and on/above a fitted line."""
    dataset_path = Path(dataset_path)
    if threads < 1 or frame_stride < 1:
        raise ValueError("threads and frame_stride must be at least 1.")
    if not np.isfinite(mz_min) or not np.isfinite(mz_max) or mz_min >= mz_max:
        raise ValueError("m/z limits must be finite and ordered.")
    if min_intensity < 0 or not np.isfinite(min_intensity):
        raise ValueError("Minimum intensity must be finite and nonnegative.")
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("Line intercept and slope must be finite.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    set_num_threads(threads)
    started = perf_counter()
    if not opentimspy.bruker_bridge_present:
        opentimspy.setup_opensource()
    with OpenTIMS(dataset_path) as dataset:
        ms1_frames = np.asarray(dataset.ms1_frames, dtype=np.uint32)[
            ::frame_stride
        ]
        if not ms1_frames.size:
            raise RuntimeError("No MS1 frames were found.")
        scan_numbers = np.arange(
            dataset.min_scan, dataset.max_scan + 1, dtype=np.uint32
        )
        scan_mobility_values = dataset.scan_to_inv_ion_mobility(
            scan_numbers,
            np.full(scan_numbers.size, ms1_frames[0], dtype=np.uint32),
        )
        scan_mobility = np.zeros(dataset.max_scan + 1, dtype=np.float64)
        scan_mobility[scan_numbers] = scan_mobility_values
        fine_bin_count = int(
            round((mz_max - mz_min) / _FINE_MZ_BIN_WIDTH)
        )
        fine_edges = (
            mz_min
            + np.arange(fine_bin_count + 1) * _FINE_MZ_BIN_WIDTH
        )
        fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2.0
        fine_tof_edges = dataset.mz_to_tof_frame_sorted(
            fine_edges,
            np.full(fine_edges.size, ms1_frames[0], dtype=np.uint32),
        )
        partial_below_tic = np.zeros(scan_mobility.size, dtype=np.uint64)
        partial_above_tic = np.zeros(scan_mobility.size, dtype=np.uint64)
        for frame_number, frame in enumerate(
            dataset.query_iter(ms1_frames, columns=_COLUMNS), start=1
        ):
            _sum_line_sides_frame(
                frame["scan"],
                frame["tof"],
                frame["intensity"],
                scan_mobility,
                fine_tof_edges,
                fine_centers,
                intercept,
                slope,
                min_intensity,
                partial_below_tic,
                partial_above_tic,
            )
            if progress is not None and frame_number % 100 == 0:
                progress(f"Processed {frame_number} MS1 frames")

    result = LineTicResult(
        tic_below_line=int(partial_below_tic.sum(dtype=np.uint64)),
        tic_above_line=int(partial_above_tic.sum(dtype=np.uint64)),
        line_intercept=float(intercept),
        line_slope=float(slope),
        mz_min=float(mz_min),
        mz_max=float(mz_max),
        min_intensity=float(min_intensity),
        frame_stride=frame_stride,
        visited_ms1_frames=int(ms1_frames.size),
        runtime_seconds=perf_counter() - started,
        effective_threads=int(get_num_threads()),
    )
    if progress is not None:
        progress("TIC analysis complete")
    return result


def sum_below_line(
    dataset_path: Path | str,
    line_json_path: Path | str,
    output_path: Path | str,
    *,
    min_intensity: float = 0.0,
    threads: int = 3,
    frame_stride: int = 1,
) -> Path:
    """Save TIC on both sides of a JSON-defined line; retain the legacy name."""
    dataset_path = Path(dataset_path)
    line_json_path = Path(line_json_path)
    output_path = Path(output_path)
    model = json.loads(line_json_path.read_text())
    line = model["line"]
    mz_range = model["analysis_mz_range"]
    result = analyse_line_tic(
        dataset_path,
        intercept=float(line["intercept"]),
        slope=float(line["slope"]),
        mz_min=float(mz_range["min"]),
        mz_max=float(mz_range["max"]),
        min_intensity=min_intensity,
        threads=threads,
        frame_stride=frame_stride,
    )
    payload = {
        "source_line_json": str(line_json_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "line": {
            "intercept": result.line_intercept,
            "slope": result.line_slope,
        },
        "tic_below_line": result.tic_below_line,
        "tic_above_line": result.tic_above_line,
        "event_intensity_threshold": result.min_intensity,
        "analysis_mz_range": {"min": result.mz_min, "max": result.mz_max},
        "frame_stride": result.frame_stride,
        "visited_ms1_frames": result.visited_ms1_frames,
        "threads": result.effective_threads,
        "runtime_seconds": result.runtime_seconds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path

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


def _plot_intensity_maps(
    intensities: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, mz_bin_width: float, output_path: Path
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
    axes[-1].set_xlabel(f"m/z ({mz_bin_width:g} Da bins)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_dominant_charge_map(
    dominant: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, mz_bin_width: float, output_path: Path
) -> None:
    """Save the categorical dominant-charge view as a PNG."""
    _require_matplotlib()
    colour_map = colors.ListedColormap(["#f5f5f5", "#4c78a8", "#f58518", "#54a24b"])
    normalizer = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colour_map.N)
    figure, axis = plt.subplots(figsize=(16, 3.6), constrained_layout=True)
    image = axis.pcolormesh(mz_edges, mobility_edges, dominant, shading="auto", cmap=colour_map, norm=normalizer)
    axis.set(xlabel=f"m/z ({mz_bin_width:g} Da bins)", ylabel="1/K0", title="Dominant charge by summed precursor intensity")
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
    boundary: np.ndarray,
    mz_bin_width: float,
    output_path: Path,
) -> None:
    """Plot the polar charge boundary over all MS1 intensity."""
    _require_matplotlib()
    figure, axis = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    image = axis.pcolormesh(mz_edges, mobility_edges, np.log1p(all_ms1_intensities), shading="auto", cmap="magma")
    axis.plot(boundary[0], boundary[1], color="#00d4ff", linewidth=2.0, label="global polar radial border")
    axis.set(xlabel=f"m/z ({mz_bin_width:g} Da bins)", ylabel="1/K0", title="Charge-1 versus charge-2 polar radial border")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, pad=0.01, label="log(1 + all MS1 intensity)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyse(
    dataset_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    isotope_count: int = 3,
    mobility_bins: int = 100,
    mz_min: float = 100.0,
    mz_max: float = 1700.0,
    mz_bin_width: float = 10.0,
    min_intensity: float = 30.0,
    border_mz_left: float = 350.0,
    border_mz_right: float = 1200.0,
    frame_stride: int = 1,
    threads: int = 3,
    scans_per_mobility_bin: int = 0,
    progress: Callable[[str], None] | None = None,
) -> ChargeRegionResult:
    """Analyse a dataset and optionally persist the complete CLI artifacts.

    When ``output_dir`` is omitted, this is a pure in-memory API: no NPZ, JSON,
    or plot files are written. Callers receive every numerical result directly.
    """
    dataset_path = Path(dataset_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if isotope_count < 1 or mobility_bins < 1 or mz_max <= mz_min or mz_bin_width <= 0:
        raise ValueError("isotope_count, mobility_bins, m/z limits, and m/z bin width must be positive and valid.")
    fine_bins_per_output_mz_bin = int(round(mz_bin_width / _FINE_MZ_BIN_WIDTH))
    if not np.isclose(fine_bins_per_output_mz_bin * _FINE_MZ_BIN_WIDTH, mz_bin_width):
        raise ValueError("mz_bin_width must be an integer multiple of 1/12 Da.")
    if min_intensity < 0 or frame_stride < 1 or threads < 1 or scans_per_mobility_bin < 0:
        raise ValueError("min_intensity must be non-negative, frame_stride and threads at least 1, and scan sampling non-negative.")
    if not mz_min <= border_mz_left < border_mz_right <= mz_max:
        raise ValueError("border m/z limits must be ordered and lie within the analysis m/z range.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    started = perf_counter()
    set_num_threads(min(threads, mobility_bins))
    if resolved_output_dir is not None:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    mz_min, mz_max = float(np.floor(mz_min)), float(np.ceil(mz_max))
    mz_edges = np.arange(mz_min, mz_max + mz_bin_width * 0.5, mz_bin_width)
    if mz_edges[-1] < mz_max:
        mz_edges = np.append(mz_edges, mz_max)
    mz_centers = (mz_edges[:-1] + mz_edges[1:]) / 2.0
    border_mz_mask = (mz_centers >= border_mz_left) & (mz_centers <= border_mz_right)
    border_mz_start_index = int(np.flatnonzero(border_mz_mask)[0])
    border_mz_stop_index = int(np.flatnonzero(border_mz_mask)[-1]) + 1
    fine_mz_bin_count = int(round((mz_max - mz_min) / _FINE_MZ_BIN_WIDTH))
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
                touched_bins, isotope_count, min_intensity, fine_bins_per_output_mz_bin,
            )
            if progress is not None and frame_number % 100 == 0:
                progress(f"Processed {frame_number} MS1 frames")

    raw_event_intensity_histogram = event_histograms.sum(axis=0, dtype=np.uint64)
    if progress is not None:
        progress("Fitting charge border")
    polar_origin, line_one, line_two, polar_boundary_radius, polar_boundary, one_inner, one_mask = fit_polar_charge_border(
        intensities, all_ms1_intensities, mz_edges, mobility_edges, border_mz_left, border_mz_right
    )
    non_one_ms1_intensity = all_ms1_intensities[(~one_mask) & border_mz_mask[None, :]].sum()
    line_intercept, line_slope, line_alpha, line_origin, line_axes = fit_alpha_separator_line(
        intensities, all_ms1_intensities, mz_edges, mobility_edges, border_mz_left, border_mz_right
    )
    line_data: dict[str, object] = {
        "model": "robust_1_2_axes_alpha_separator",
        "line": {"intercept": line_intercept, "slope": line_slope},
        "separator_alpha_radians": line_alpha,
        "origin": {"mz": float(line_origin[0]), "inv_ion_mobility": float(line_origin[1])},
        "robust_axis_coefficients": {"charge_1": {"intercept": float(line_axes[0, 0]), "slope": float(line_axes[0, 1])}, "charge_2": {"intercept": float(line_axes[1, 0]), "slope": float(line_axes[1, 1])}},
        "analysis_mz_range": {"min": mz_min, "max": mz_max},
        "fit_mz_range": {"min": border_mz_left, "max": border_mz_right},
        "min_intensity": min_intensity,
        "visited_ms1_frames": int(ms1_frames.size),
    }
    runtime_seconds = perf_counter() - started
    result = ChargeRegionResult(
        intensities=intensities,
        all_ms1_intensities=all_ms1_intensities,
        raw_event_intensity_histogram=raw_event_intensity_histogram,
        one_charge_mask=one_mask,
        non_one_ms1_intensity=float(non_one_ms1_intensity),
        border_mz_mask=border_mz_mask,
        polar_origin=polar_origin,
        line_one=line_one,
        line_two=line_two,
        polar_boundary_radius=float(polar_boundary_radius),
        polar_boundary=polar_boundary,
        one_charge_is_inner=bool(one_inner),
        charges=CHARGES.copy(),
        mz_edges=mz_edges,
        mobility_edges=mobility_edges,
        sampled_scans_per_mobility_bin=sampled_scans,
        line_data=line_data,
        visited_ms1_frames=int(ms1_frames.size),
        runtime_seconds=runtime_seconds,
        effective_threads=int(get_num_threads()),
    )
    if resolved_output_dir is None:
        if progress is not None:
            progress("Analysis complete")
        return result

    line_json_path = resolved_output_dir / "charge_border_line.json"
    line_json_path.write_text(json.dumps(line_data, indent=2) + "\n")
    npz_path = resolved_output_dir / "charge_region_maps.npz"
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
        border_evidence_source=np.array("dominant_charge_map_1_vs_2"),
        border_model=np.array("robust_axes_global_radial"),
        border_polar_origin=polar_origin,
        border_line_one_coefficients=line_one,
        border_line_two_coefficients=line_two,
        alpha_separator_line_intercept=np.array(line_intercept),
        alpha_separator_line_slope=np.array(line_slope),
        alpha_separator_line_alpha=np.array(line_alpha),
        border_polar_radius=np.array(polar_boundary_radius),
        border_polar_boundary=polar_boundary,
        border_one_charge_is_inner=np.array(one_inner),
        charges=CHARGES,
        mz_edges=mz_edges,
        mobility_edges=mobility_edges,
        sampled_scans_per_mobility_bin=sampled_scans,
        fine_mz_bin_width=np.array(_FINE_MZ_BIN_WIDTH),
        mz_bin_width=np.array(mz_bin_width),
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
    _plot_intensity_maps(
        intensities,
        mz_edges,
        mobility_edges,
        mz_bin_width,
        resolved_output_dir / "charge_region_intensities.png",
    )
    dominant = dominant_charge_map(intensities)
    _plot_dominant_charge_map(
        dominant,
        mz_edges,
        mobility_edges,
        mz_bin_width,
        resolved_output_dir / "dominant_charge_map.png",
    )
    _plot_event_intensity_distribution(
        raw_event_intensity_histogram,
        min_intensity,
        resolved_output_dir / "raw_event_intensity_distribution.png",
    )
    _plot_charge_border(
        all_ms1_intensities,
        mz_edges,
        mobility_edges,
        polar_boundary,
        mz_bin_width,
        resolved_output_dir / "charge_border.png",
    )
    if progress is not None:
        progress("Analysis complete")
    return result

def tic_main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum filtered raw MS1 TIC on both sides of a fitted line."
    )
    parser.add_argument("dataset", type=Path, help="Path to a .d dataset directory.")
    parser.add_argument("--line-json", type=Path, required=True, help="charge_border_line.json from charge-regions.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    output = sum_below_line(
        args.dataset,
        args.line_json,
        args.output,
        min_intensity=args.min_intensity,
        threads=args.threads,
        frame_stride=args.frame_stride,
    )
    print(f"Saved below/above-line TIC to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a .d dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isotope-count", type=int, default=3, help="Required following isotope peaks.")
    parser.add_argument("--mobility-bins", type=int, default=100)
    parser.add_argument("--mz-min", type=float, default=100.0)
    parser.add_argument("--mz-max", type=float, default=1700.0)
    parser.add_argument("--mz-bin-width", type=float, default=10.0, help="Final map m/z bin width in Da (default: 10).")
    parser.add_argument("--min-intensity", type=float, default=30.0, help="Ignore raw events below this intensity.")
    parser.add_argument("--border-mz-left", type=float, default=350.0, help="Left m/z limit for border fitting and aggregation.")
    parser.add_argument("--border-mz-right", type=float, default=1200.0, help="Right m/z limit for border fitting and aggregation.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Visit every K-th MS1 frame (default: 1).")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--scans-per-mobility-bin", type=int, default=0)
    args = parser.parse_args()
    analyse(
        args.dataset, args.output_dir, isotope_count=args.isotope_count,
        mobility_bins=args.mobility_bins, mz_min=args.mz_min, mz_max=args.mz_max, mz_bin_width=args.mz_bin_width,
        min_intensity=args.min_intensity, border_mz_left=args.border_mz_left, border_mz_right=args.border_mz_right,
        frame_stride=args.frame_stride, threads=args.threads,
        scans_per_mobility_bin=args.scans_per_mobility_bin,
        progress=print,
    )
    print(f"Saved intensity maps to {args.output_dir / 'charge_region_maps.npz'}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()

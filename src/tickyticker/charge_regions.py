"""Create charge-state evidence maps from timsTOF MS1 spectra.

The public entry point is :func:`analyse`. The ``charge-regions`` command is a
thin wrapper around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import opentimspy
from numba import njit
from opentimspy import OpenTIMS

from .dominant_charge import dominant_charge_map, plot_dominant_charge_map, write_ascii_map


CHARGES = np.array([1, 2, 3], dtype=np.int64)
_COLUMNS = ("scan", "mz", "intensity", "inv_ion_mobility")


@njit(cache=True)
def _detect_charge_candidates(
    mz: np.ndarray,
    intensity: np.ndarray,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precursor m/z, charge, and normalized isotope-continuation score."""
    result_mz = np.empty(mz.size * 3, dtype=np.float64)
    result_charge = np.empty(mz.size * 3, dtype=np.int64)
    result_score = np.empty(mz.size * 3, dtype=np.float64)
    result_count = 0

    if mz.size < 3:
        return result_mz[:0], result_charge[:0], result_score[:0]

    spectrum_max = np.max(intensity)
    if spectrum_max <= 0:
        return result_mz[:0], result_charge[:0], result_score[:0]

    for peak_index in range(1, mz.size - 1):
        if intensity[peak_index] < min_intensity:
            continue
        if intensity[peak_index] < intensity[peak_index - 1] or intensity[peak_index] <= intensity[peak_index + 1]:
            continue

        precursor = mz[peak_index]
        for charge in range(1, 4):
            score = 0.0
            matched_isotopes = 0
            for isotope_index in range(1, isotope_count + 1):
                target = precursor + isotope_index / charge
                tolerance = target * ppm * 1e-6
                left = np.searchsorted(mz, target - tolerance, side="left")
                right = np.searchsorted(mz, target + tolerance, side="right")
                if right > left:
                    score += np.max(intensity[left:right]) / spectrum_max
                    matched_isotopes += 1

            if matched_isotopes == isotope_count and score >= min_score:
                result_mz[result_count] = precursor
                result_charge[result_count] = charge
                result_score[result_count] = score
                result_count += 1

    return result_mz[:result_count], result_charge[:result_count], result_score[:result_count]


def _ms1_frames(dataset: OpenTIMS) -> Iterator[dict[str, np.ndarray]]:
    yield from dataset.query_iter(dataset.ms1_frames, columns=_COLUMNS)


def _accumulate_frame(
    frame: dict[str, np.ndarray],
    counts: np.ndarray,
    score_sums: np.ndarray,
    mz_min: float,
    mz_max: float,
    mobility_min: float,
    mobility_max: float,
    ppm: float,
    isotope_count: int,
    min_intensity: float,
    min_score: float,
) -> None:
    """Analyse every scan in one frame and add results to per-charge maps."""
    scan = frame["scan"]
    mz = frame["mz"].astype(np.float64, copy=False)
    intensity = frame["intensity"].astype(np.float64, copy=False)
    mobility = frame["inv_ion_mobility"].astype(np.float64, copy=False)
    if not mz.size:
        return

    order = np.lexsort((mz, scan))
    scan, mz, intensity, mobility = scan[order], mz[order], intensity[order], mobility[order]
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    stops = np.r_[starts[1:], scan.size]
    mobility_bin_count, mz_bin_count = counts.shape[1:]

    for start, stop in zip(starts, stops, strict=True):
        candidate_mz, charge, score = _detect_charge_candidates(
            mz[start:stop], intensity[start:stop], ppm, isotope_count, min_intensity, min_score
        )
        if not candidate_mz.size:
            continue

        scan_mobility = float(np.median(mobility[start:stop]))
        mobility_bin = int(
            np.floor((scan_mobility - mobility_min) / (mobility_max - mobility_min) * mobility_bin_count)
        )
        mobility_bin = min(max(mobility_bin, 0), mobility_bin_count - 1)
        mz_bin = np.floor(candidate_mz - mz_min).astype(np.int64)
        valid = (mz_bin >= 0) & (mz_bin < mz_bin_count) & (candidate_mz < mz_max)
        for index in np.flatnonzero(valid):
            charge_index = charge[index] - 1
            counts[charge_index, mobility_bin, mz_bin[index]] += 1
            score_sums[charge_index, mobility_bin, mz_bin[index]] += score[index]


def analyse(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    ppm: float = 10.0,
    isotope_count: int = 3,
    mobility_bins: int = 10,
    mz_min: float = 100.0,
    mz_max: float = 1700.0,
    min_intensity: float = 0.0,
    min_score: float = 0.0,
) -> Path:
    """Create charge maps and return the path to the resulting `.npz` file."""
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if ppm <= 0 or isotope_count < 1 or mobility_bins < 1 or mz_max <= mz_min:
        raise ValueError("ppm, isotope_count, mobility_bins, and m/z limits must be positive and valid.")
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
        reference_frames = np.full(2, ms1_frames[0], dtype=np.uint32)
        mobility_min, mobility_max = np.sort(
            dataset.scan_to_inv_ion_mobility(scan_bounds, reference_frames)
        )
        mobility_edges = np.linspace(mobility_min, mobility_max, mobility_bins + 1)
        shape = (len(CHARGES), mobility_bins, mz_edges.size - 1)
        counts = np.zeros(shape, dtype=np.uint32)
        score_sums = np.zeros(shape, dtype=np.float64)

        for frame_number, frame in enumerate(_ms1_frames(dataset), start=1):
            _accumulate_frame(
                frame, counts, score_sums, mz_min, mz_max, mobility_min, mobility_max,
                ppm, isotope_count, min_intensity, min_score,
            )
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} MS1 frames", flush=True)

    npz_path = output_dir / "charge_region_maps.npz"
    np.savez_compressed(
        npz_path,
        counts=counts,
        score_sums=score_sums,
        charges=CHARGES,
        mz_edges=mz_edges,
        mobility_edges=mobility_edges,
        ppm=np.array(ppm),
        isotope_count=np.array(isotope_count),
        min_intensity=np.array(min_intensity),
        min_score=np.array(min_score),
    )
    _plot_maps(counts, mz_edges, mobility_edges, output_dir / "charge_region_counts.png")
    dominant = dominant_charge_map(counts)
    write_ascii_map(dominant, mz_edges, mobility_edges, output_dir / "dominant_charge_map.txt")
    plot_dominant_charge_map(dominant, mz_edges, mobility_edges, output_dir / "dominant_charge_map.png")
    return npz_path


def _plot_maps(
    counts: np.ndarray, mz_edges: np.ndarray, mobility_edges: np.ndarray, output_path: Path
) -> None:
    """Save separate, readable charge-count heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires the optional dependency. Install with: pip install tickyticker[dev]"
        ) from exc

    figure, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    for charge, axis in zip(CHARGES, axes, strict=True):
        image = axis.pcolormesh(
            mz_edges, mobility_edges, np.log1p(counts[charge - 1]), shading="auto", cmap="magma"
        )
        axis.set(ylabel="1/K0", title=f"Charge {charge}: log(1 + isotope-pattern count)")
        figure.colorbar(image, ax=axis, pad=0.01, label="log(1 + count)")
    axes[-1].set_xlabel("m/z (1 Da bins)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a .d dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--isotope-count", type=int, default=3)
    parser.add_argument("--mobility-bins", type=int, default=10)
    parser.add_argument("--mz-min", type=float, default=100.0)
    parser.add_argument("--mz-max", type=float, default=1700.0)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    output = analyse(
        args.dataset,
        args.output_dir,
        ppm=args.ppm,
        isotope_count=args.isotope_count,
        mobility_bins=args.mobility_bins,
        mz_min=args.mz_min,
        mz_max=args.mz_max,
        min_intensity=args.min_intensity,
        min_score=args.min_score,
    )
    print(f"Saved charge-count maps to {output}")
    print(f"Saved plot to {output.parent / 'charge_region_counts.png'}")


if __name__ == "__main__":
    main()

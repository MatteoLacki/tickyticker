# tickyticker

`tickyticker` maps isotope-spacing evidence for charge states 1–3 in Bruker
timsTOF MS1 data. It reads `.d` directories frame-by-frame with OpenTIMS,
bins each scan at 1/12 Da using a first-MS1-frame TOF-to-m/z lookup, and
aggregates exclusive charge assignments into one 3D precursor-intensity map.
The final map width is configurable with `--mz-bin-width` and defaults to 10 Da.

## Install

```bash
# Core analysis only
python -m pip install -e .

# Core analysis plus PNG plot generation
python -m pip install -e '.[dev]'
```

## Run

```bash
charge-regions data/G260811_092_Slot2-1_1_24990.d --output-dir charge_regions_092
below-line-tic data/G260811_092_Slot2-1_1_24990.d --line-json charge_regions_092/charge_border_line.json --output charge_regions_092/below_line_tic.json
```

Raw events below `--min-intensity` are excluded before m/z binning (default:
30). Each candidate requires three following, nonzero isotope bins with strictly
decreasing intensities. The charge-border fit and non-1+ aggregate use only
`--border-mz-left` through `--border-mz-right` (defaults: 350–1200 m/z).
Use `--frame-stride K` (default: 1) to visit every K-th MS1 frame. Charge 3 (4-bin spacing) is tested first, followed by
charge 2 (6 bins) and charge 1 (12 bins), so an accepted precursor contributes
its raw intensity to exactly one charge map. Use `charge-regions --help` to
view all parameters.

Each run writes:

- `charge_region_maps.npz`: one `(charge, 1/K0 bin, m/z bin)` intensity tensor, with m/z and 1/K0 bin edges;
- `charge_region_intensities.png`: a three-panel charge-resolved intensity heatmap;
- `dominant_charge_map.png`: the charge with the highest summed intensity per box, as a categorical plot;
- `charge_border.png`: a robust polar charge border: dominant 1+/2+ axes, their intersection, and one class-balanced global radial split over all MS1 intensity;
- `raw_event_intensity_distribution.png`: log-scale distribution of all visited raw events, with the selected minimum-intensity threshold;
- `all_ms1_intensities`, `raw_event_intensity_histogram` (128 `uint64` bins; last is ≥128), `one_charge_mask`, border data, and `non_one_ms1_intensity` in the `.npz` result;
- `sampled_scans_per_mobility_bin`: exposure for normalizing intensity maps produced with scan subsampling.

Raw files under `data/` are input-only and are never modified.

## Parallelism and scan subsampling

Use `--threads` to set Numba CPU parallelism (default: 3). Each MS1 frame is
split into independent 1/K0-bin tasks. A task owns its output mobility slice,
so it bins and adds scan intensities directly without locks or float atomics.
Scan-to-1/K0-bin assignments are evaluated once with OpenTIMS for the first
MS1 frame and reused thereafter. Use `--scans-per-mobility-bin N` to select
`N` evenly spaced scans in each populated 1/K0 bin of every MS1 frame. `0`
(the default) processes all scans. When subsampling, divide each mobility row
by `sampled_scans_per_mobility_bin` before comparing intensities between runs.

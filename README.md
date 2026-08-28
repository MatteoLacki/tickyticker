# tickyticker

`tickyticker` maps isotope-spacing evidence for charge states 1–3 in Bruker
timsTOF MS1 data. It reads `.d` directories frame-by-frame with OpenTIMS,
detects local m/z maxima within each scan, and aggregates successful isotope
continuations into one 3D charge-resolved precursor-intensity map.

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
```

The default search tolerance is 10 ppm and three subsequent isotope positions
are required. Use `charge-regions --help` to view all parameters.

Each run writes:

- `charge_region_maps.npz`: one `(charge, 1/K0 bin, m/z bin)` intensity tensor, with m/z and 1/K0 bin edges;
- `charge_region_intensities.png`: a three-panel charge-resolved intensity heatmap;
- `dominant_charge_map.png` and `dominant_charge_map.txt`: the charge with the highest summed intensity per box, as a categorical plot and ASCII map.
- `sampled_scans_per_mobility_bin`: exposure for normalizing intensity maps produced with scan subsampling.

Raw files under `data/` are input-only and are never modified.

## Parallelism and scan subsampling

Use `--threads` to set Numba CPU parallelism (default: 3; at most three threads are useful). Every MS1 frame is processed by one Numba parallel function with one task per charge. Each task owns its complete disjoint `(charge, 1/K0 bin, m/z bin)` plane and writes directly to the final intensity tensor. Scan-to-1/K0-bin assignments are evaluated once with OpenTIMS for the first MS1 frame and reused thereafter. Use `--scans-per-mobility-bin N` to
select `N` evenly spaced scans in each populated 1/K0 bin of every MS1 frame.
`0` (the default) processes all scans. When subsampling, divide each mobility
row by `sampled_scans_per_mobility_bin` before comparing intensities between
runs.

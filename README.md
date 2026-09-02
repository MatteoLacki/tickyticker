# tickyticker

`tickyticker` identifies charge-resolved isotope evidence in Bruker timsTOF
MS1 data and maps it in m/z × inverse-ion-mobility (1/K0) space.

It processes `.d` directories one MS1 frame at a time with OpenTIMS, NumPy,
and Numba. Raw data are never modified.

## Installation

```bash
python -m pip install -e .
# Add PNG plot support:
python -m pip install -e ".[dev]"
```

## Charge-region analysis

```bash
charge-regions data/G260811_092_Slot2-1_1_24990.d \
  --output-dir results/hela_092
```

The command bins each scan onto a 1/12 Da intermediate grid, detects strictly
decreasing isotope continuations for charges 1, 2, and 3, then adds precursor
intensity to an exclusive 3D tensor:

```text
(charge, 1/K0 bin, m/z bin)
```

Charge 3 is tested first, followed by charge 2 and charge 1. The final m/z
bin width defaults to 10 Da (`--mz-bin-width`); the default map has 100 equal
1/K0 bins. Events below `--min-intensity` (default: 30) do not enter the map
or charge evidence.

The command also fits a separator line from robust dominant 1+ and 2+ axes in
the configurable border m/z interval (default: 350–1200). Its intercept and
slope are stored in `charge_border_line.json`.

Main options:

```bash
charge-regions --help
```

Useful controls include `--threads` (default: 3), `--frame-stride`,
`--scans-per-mobility-bin`, `--min-intensity`, `--border-mz-left`, and
`--border-mz-right`.

### Python API

Call `analyse()` without an output directory to receive the complete numerical
result in memory without writing NPZ, JSON, or PNG files:

```python
from tickyticker.charge_regions import analyse

result = analyse("data/sample.d", progress=print)
print(result.intensities.shape)
print(result.line_data["line"])
```

The returned `ChargeRegionResult` also contains the raw-event histogram,
coordinate edges, sampled-scan counts, fitted-border arrays, runtime, and
effective thread count. Passing `output_dir` retains the archival CLI behavior
and writes the complete standard artifact set in addition to returning the
result.

## Below-line TIC pass

Use the line JSON from a completed charge-region run to revisit the raw MS1
events and sum all event intensity below the fitted line:

```bash
below-line-tic data/G260811_092_Slot2-1_1_24990.d \
  --line-json results/hela_092/charge_border_line.json \
  --output results/hela_092/below_line_tic.json \
  --threads 3
```

This second pass is Numba-parallel over scans and uses the first-frame
calibrated TOF-to-m/z lookup. `below_line_tic.json` records the total TIC,
line intercept and slope, m/z range, frame stride, thread count, visited MS1
frames, and runtime.

## Outputs

`charge-regions` writes these files into the output directory:

- `charge_region_maps.npz` — charge-resolved intensity tensor and coordinate edges.
- `charge_region_intensities.png` — one heatmap per charge.
- `dominant_charge_map.png` — charge with the greatest intensity in each map box.
- `charge_border.png` — fitted charge-border view over total MS1 intensity.
- `charge_border_line.json` — fitted line parameters for `below-line-tic`.
- `raw_event_intensity_distribution.png` — log-scale raw-event intensity histogram.

The `.npz` contains `intensities`, `all_ms1_intensities`, `charges`,
`mz_edges`, `mobility_edges`, and `sampled_scans_per_mobility_bin`, together
with fitting metadata and exact run parameters.

## License

MIT. See [LICENSE](LICENSE).

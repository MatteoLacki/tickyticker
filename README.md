# tickyticker

`tickyticker` maps isotope-spacing evidence for charge states 1–3 in Bruker
timsTOF MS1 data. It reads `.d` directories frame-by-frame with OpenTIMS,
detects local m/z maxima within each scan, and aggregates successful isotope
continuations into m/z × inverse-ion-mobility (1/K0) count maps.

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

- `charge_region_maps.npz`: separate count and normalized-score maps for
  charges 1, 2, and 3, with m/z and 1/K0 bin edges;
- `charge_region_counts.png`: a three-panel charge-count heatmap;
- `dominant_charge_map.png` and `dominant_charge_map.txt`: the charge with the highest count per box, as a categorical plot and ASCII map.

Raw files under `data/` are input-only and are never modified.

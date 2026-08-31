# Project guide

This repository analyses Bruker timsTOF `.d` raw files to identify isotope
charge-resolved precursor intensity in m/z × inverse-ion-mobility (1/K0) space.

## Data and outputs

- Raw datasets under `data/` are read-only inputs.
- `acquisition_summary.csv` contains metadata extracted from `analysis.tdf`.
- Give each new analysis run a distinct output directory; do not overwrite a
  completed result without explicit instruction.

## Package workflow

The installable package lives in `src/tickyticker/`. Runtime dependencies are
defined only in `pyproject.toml`.

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/charge-regions DATASET.d --output-dir OUTPUT_DIRECTORY
```

`charge-regions` iterates MS1 frames with OpenTIMS, applies a first-MS1-frame
TOF-to-m/z lookup to bin every scan at 1/12 Da, then tests strictly decreasing,
nonzero isotope envelopes for charges 3, 2, and 1 at 4-, 6-, and 12-bin
spacings. Each accepted precursor is assigned exclusively to the first matching
charge and added to its 1 Da m/z × 100 equal-width 1/K0-bin intensity map.
Preserve configurable isotope count, minimum raw-event intensity (default 30), charge-border m/z limits (defaults 350–1200), and MS1 frame stride (default 1); sub-threshold events do not enter binned maps or charge evidence, but are counted in the raw-event histogram.

## Implementation conventions

- Use OpenTIMS, NumPy, and Numba for raw-data processing.
- Parallelize each MS1 frame with Numba over its three exclusive charge planes; do not add a separate process-worker layer. Determine scan-to-1/K0-bin assignments once from the first MS1 frame with OpenTIMS, then reuse that lookup for every frame.
- Process frames one at a time; never load all raw data into memory.
- Bruker `.d` peak arrays are frame-scan-TOF sorted; rely on that ordering rather than re-sorting frames.
- Use calibrated `mz` and `inv_ion_mobility` coordinates in outputs.
- Preserve independent intensity maps for all charge states unless explicitly asked to
  collapse them.
- Save numerical results as `.npz`, plus PNG intensity, dominant-charge, and fitted linear-spline charge-border views.
- Record runtime and exact parameters for every full dataset run.

## Validation

```bash
.venv/bin/python -m py_compile src/tickyticker/charge_regions.py
```

After a full run, confirm that the `.npz` includes `intensities`, `sampled_scans_per_mobility_bin`,
`charges`, `mz_edges`, and `mobility_edges`, and that the PNG plot exists.

# OSA Analyzer

Desktop application and Python package for optical spectrum analyzer (OSA)
measurements of fiber refractive-index (RI) sensors, developed for SMF–NCF–SMF
multimode-interference sensors measured on a Yokogawa AQ6370.

* **Load** Yokogawa TXT exports (wavelength axis from `WL.txt`) or Excel/CSV
  tables; repeated measurements at the same RI are kept, `-210 dBm` no-signal
  samples become missing values, optional reference normalisation.
* **Track** the sensing dip/peak inside a wavelength window, with sub-sample
  wavelength estimation, prominence and width, and a warning when the
  resonance leaves the window.
* **Fit** wavelength or power against RI (linear to cubic, one or two RI
  regions): sensitivity ± standard error, 95 % CI, adjusted R², detection
  limit from the fit and from repeat scatter.
* **Learn** RI from spectra (linear, random forest, gradient boosting,
  XGBoost, CatBoost, ANN, 1-D CNN) with group-aware splits, cross-validation,
  and augmentation applied to training rows only.
* **Export** journal-ready figures (white background, size in mm, 600 dpi) and
  Excel results. Each result file gets a `.provenance.json` with software
  versions, all settings and SHA-256 hashes of the input files.
* **Reproduce** any analysis without the GUI from the command line.

## Install and run

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e .[all]             # or: pip install -e .   (core only)
python -m osa_analyzer            # GUI
```

`[boost]` adds XGBoost and CatBoost, `[deep]` adds PyTorch (ANN/CNN-1D). Models
whose library is missing are greyed out in the GUI. For the CPU-only PyTorch
build: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Typical workflow (GUI)

1. **Data**: *Add TXT files…* and select the spectra **plus `WL.txt`**. The
   wavelength file is detected automatically; without it, set the sweep range
   (your AQ6370 files are 1520–1580 nm, 501 points). RI comes from the file
   name: `1.46_rep2.txt` → 1.46. Files without an RI (air, reference) are
   loaded for plotting only.
2. **Spectra**: inspect; export a figure at single/double-column size.
3. **Resonance**: choose dip or peak, drag the shaded window over the
   resonance, *Detect*. Fix any "at window edge" warnings before fitting.
4. **Fitting**: wavelength- or power-vs-RI; one region or low/high regions.
5. **Machine learning**: build a dataset (7 resonance features, or the raw
   spectrum in the window), choose the model and the split, train. Save the
   model, predict new TXT spectra, export results.

Sessions (`File → Save session`) store settings and file paths, not data.

## Command line

```bash
# resonance table, fit results, figures and provenance into results/
python -m osa_analyzer analyze data/*.txt --window 1530 1545 --smoothing 5 \
       --regions 1.30:1.39 1.46:1.64 --out results/

# ML table in the same layout as the published datasets
python -m osa_analyzer dataset data/*.txt --window 1530 1545 --out dataset.xlsx

# leakage-safe training with 5-fold CV, leave-RI-out split
python -m osa_analyzer train dataset.xlsx --model XGBoost --split ri --cv 5 --out ml/
```

`python -m osa_analyzer <command> --help` lists every option.

The analysis core does not depend on Qt and can be used from scripts or Jupyter:

```python
from osa_analyzer.core import io, resonance, fitting
spectra, report = io.load_files(paths)
det = resonance.detect(spectra.labeled(),
                       resonance.DetectionParams(window_min=1530, window_max=1545, smoothing=5))
res = det.labeled()
fit = fitting.fit_polynomial([r.ri for r in res], [r.wavelength for r in res], degree=1)
print(fitting.format_fit(fit))
```

## Architecture: Model-View-Presenter

```
osa_analyzer/
  core/          MODEL: numpy/scipy/sklearn only, no Qt
    spectrum.py    Spectrum, SpectrumSet
    io.py          TXT / Excel / CSV readers, RI from names, axis handling
    processing.py  smoothing, reference normalisation
    resonance.py   windowed dip/peak tracking, sub-sample refinement
    fitting.py     polynomial sensitivity fits, CI, detection limit
    features.py    ML tables (engineered / wide)
    augmentation.py  seeded training-set augmentation
    ml.py          models, group-aware evaluation, model files
    plotting.py    journal-style matplotlib figures
    report.py      provenance records
    session.py     .osasession files
    instrument.py  Yokogawa AQ6370 TCP client (experimental)
  presenters/    PRESENTER: all application logic, no Qt widgets
  views/         VIEW: passive PyQt5 widgets (signals out, setters in)
  app.py         composition root
  cli.py         command line
```

Views never decide anything. They emit signals ("detect clicked") and expose
setters ("show this table"). Presenters own the logic and talk to the core.
Because presenters import no Qt widgets, the whole workflow is tested
headless with fake views (`tests/test_presenters.py`). A test also enforces
that `core/` and `presenters/` never import PyQt.

## Changes from v7 (`legacy/OSA_v7.py`)

| Area | v7 | v8 |
|---|---|---|
| Repeated measurements | spectra keyed by RI; `1.46_repeat` overwrote `1.46` | every spectrum kept with its own label |
| TXT wavelength axis | default 1520–1620 nm, but the lab's sweeps are 1520–1580 nm (`WL.txt`); a wrong range scales every nm/RIU sensitivity by 1.67 | `WL.txt` auto-detected; default 1520–1580 nm |
| RI from file name | integer fallback (`AC_Lap17` → RI 17) | decimals in 1.0–2.5 only, optional regex |
| Resonance for fitting | first peak anywhere in the spectrum, peaks preferred over dips | tracked inside a window, edge warning |
| Wavelength resolution | nearest sample (0.12 nm) | parabolic sub-sample estimate (optional) |
| ML augmentation | applied before the train/test split (copies of test spectra in training) | applied to training rows only; copies stay with their source |
| ANN/CNN early stopping | monitored the test set | validation carved from training rows |
| Imputation | medians from the whole dataset | fitted on training rows inside the pipeline |
| Evaluation | one random 80/20 split | group-aware split, optional leave-RI-out, k-fold CV with mean ± sd |
| Power augmentation | dBm × (1 ± 3 %) | additive offset in dB |
| Wavelength-shift augmentation | `np.roll` (wraps spectrum ends) | interpolation with edge hold |
| Publication figure | dark background, white text | white journal style, size in mm |
| Interactive vs publication tabs | two tabs with duplicated settings | one settings panel, interactive + preview |
| Model files | raw pickle | joblib bundle with feature spec, versions, metrics; load warning |
| Reproducibility | none | provenance JSON on every export, seeds everywhere, CLI |
| Live OSA | connect only; stale receive buffer on reconnect | buffer reset; trace download into the workspace |
| Code | one 6,334-line file | MVP package, 33 tests |

**If any number in a paper or thesis came from v7's ML tab with augmentation
enabled, or from TXT files loaded with the 1520–1620 nm default, recompute it
with v8.**

## Tests

```bash
pip install -e .[test]
pytest
```

The GUI test runs real widgets offscreen (`QT_QPA_PLATFORM=offscreen` is set
by `tests/conftest.py`).

## Citing

See `CITATION.cff`. After the first GitHub release is archived on Zenodo, add
the DOI there and in the paper's data/code availability statement.

## Releasing on GitHub and Zenodo

1. Fill the `TODO` fields in `CITATION.cff`, `.zenodo.json`, `pyproject.toml`
   and `LICENSE`. In `.zenodo.json`, delete the `related_identifiers` block
   until the paper has a DOI (Zenodo rejects placeholder identifiers).
2. Push to GitHub. On zenodo.org → *GitHub*, switch the repository on.
3. Create a GitHub release tagged `v8.0.0`. Zenodo archives it and mints a DOI.

## License

MIT (see `LICENSE`).

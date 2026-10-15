# Changelog

## 8.0.0 (2026-10-15)

Rewrite of OSA Analyzer v7 as a Model-View-Presenter package. See the
"Changes from v7" table in README.md for the scientific fixes.

### Added
- Analysis core usable without Qt; command line `analyze`, `dataset`, `train`.
- Windowed resonance tracking with parabolic sub-sample wavelength, prominence,
  width at half prominence and window-edge warning.
- Sensitivity 95 % CI, adjusted R², detection limit (fit and repeats), error
  bars for repeated measurements.
- Group-aware train/test split, leave-RI-out split, k-fold cross-validation,
  Gradient Boosting model, cancellable background training.
- Reference normalisation (transmission in dB).
- Provenance JSON for every export; seeds for every random step.
- Journal-style figure export with sizes in mm.
- Live OSA: trace download into the workspace (experimental).
- 33 automated tests including an offscreen GUI test.

### Fixed
- Repeated measurements at one RI no longer overwrite each other.
- TXT wavelength default now matches the lab sweep (1520–1580 nm); `WL.txt`
  is used automatically.
- Augmentation, early stopping and imputation no longer see test data.
- Integer file-name numbers are no longer read as RI.
- Wavelength-shift augmentation no longer wraps spectrum ends.

### Removed
- Separate "Publication Plot" tab (merged into Spectra).
- Plot presets (sessions cover them).
- Duplicate "Export dataset" button.
- Constant features `mean_wavelength` and `wavelength_range` (identical for
  every spectrum).
- Multiplicative power-scaling augmentation (replaced by dB offset).

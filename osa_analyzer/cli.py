"""Command line: the same analysis as the GUI, scriptable and reproducible.

    python -m osa_analyzer                         # start the GUI
    python -m osa_analyzer analyze data/*.txt --window 1530 1545 --out results/
    python -m osa_analyzer dataset data/*.txt --window 1530 1545 --out ds.csv
    python -m osa_analyzer train ds.csv --model "Random Forest" --cv 5 --out ml/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import APP_NAME, __version__


def _spectra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("files", nargs="+", help="TXT files (include WL.txt) or one Excel/CSV")
    p.add_argument("--sheet", help="Excel sheet name")
    p.add_argument("--wl-file", help="wavelength axis file for TXT data")
    p.add_argument("--start", type=float, default=1520.0, help="TXT axis start, nm")
    p.add_argument("--stop", type=float, default=1580.0, help="TXT axis stop, nm")
    p.add_argument("--ri-pattern", help="regex whose first group is the RI")
    p.add_argument("--reference", help="reference spectrum to subtract (dB)")
    p.add_argument("--mode", choices=("dip", "peak"), default="dip")
    p.add_argument("--window", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--select", choices=("extreme", "prominent"), default="extreme")
    p.add_argument("--smoothing", type=int, default=5)
    p.add_argument("--refine", choices=("parabolic", "none"), default="parabolic")
    p.add_argument("--prominence", type=float, default=1.0)


def _load(args):
    from .core import io
    from .core.processing import normalize_to_reference
    from .core.resonance import DetectionParams
    from .core.spectrum import SpectrumSet

    axis = io.AxisSpec(args.start, args.stop, args.wl_file)
    spectra, rep = io.load_files(args.files, axis, args.sheet, args.ri_pattern)
    print(rep.text())
    if args.reference:
        ref = io.load_single_spectrum(args.reference, axis)
        spectra = SpectrumSet(normalize_to_reference(s, ref) for s in spectra)
        print(f"Normalised to reference {Path(args.reference).name}")
    params = DetectionParams(mode=args.mode, window_min=args.window[0] if args.window else None,
                             window_max=args.window[1] if args.window else None,
                             select=args.select, smoothing=args.smoothing,
                             prominence=args.prominence, refine=args.refine)
    return spectra, params, axis


def cmd_analyze(args) -> int:
    from .core import fitting, plotting, report, resonance

    spectra, params, axis = _load(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = resonance.detect(spectra, params)
    det.to_dataframe().to_csv(out / "resonances.csv", index=False)
    edge = [r.label for r in det.resonances if r.at_edge]
    if edge:
        print(f"WARNING: resonance on the window edge for {', '.join(edge)}")
    res = det.labeled()
    regions = [fitting.Region("All RI")]
    if args.regions:
        regions = []
        for i, spec in enumerate(args.regions):
            lo, hi = (float(v) for v in spec.split(":"))
            regions.append(fitting.Region(f"Region {i + 1}", lo, hi))
    y = [r.wavelength if args.quantity == "wavelength" else r.power for r in res]
    fits, problems = fitting.fit_regions([r.ri for r in res], y, args.degree, regions,
                                         args.quantity, [r.label for r in res])
    for f in fits:
        print(fitting.format_fit(f) + "\n")
    for p in problems:
        print(f"not fitted: {p}")

    import pandas as pd
    if fits:
        with pd.ExcelWriter(out / "fit_results.xlsx", engine="openpyxl") as xw:
            for f in fits:
                tag = f.region.name.replace(" ", "_")[:20]
                f.summary_frame().to_excel(xw, sheet_name=f"Summary_{tag}"[:31], index=False)
                f.data_frame().to_excel(xw, sheet_name=f"Data_{tag}"[:31], index=False)
    plotting.apply_style()
    fig = plotting.new_figure(plotting.FIGURE_SIZES_MM["Single column (85 × 65 mm)"])
    plotting.draw_spectra(fig, [s for s in spectra if s.is_labeled] or list(spectra), {},
                          det.resonances, (params.window_min, params.window_max))
    plotting.save_figure(fig, out / "spectra.png", 600)
    if fits:
        fig = plotting.new_figure((85 * max(1, len(fits)) ** 0.5, 65))
        plotting.draw_fits(fig, fits)
        plotting.save_figure(fig, out / "fit.png", 600)
    rep = report.build_report(
        "cli_analyze", {"argv": sys.argv[1:], "detection": params.to_dict(),
                        "axis": axis.describe(), "degree": args.degree,
                        "quantity": args.quantity, "regions": args.regions},
        [s.source for s in spectra],
        {f.region.describe(): dict(f.summary_rows()) for f in fits})
    report.write_json(out / "provenance.json", rep)
    print(f"Wrote {out.resolve()}")
    return 0


def cmd_dataset(args) -> int:
    from .core import features

    spectra, params, _ = _load(args)
    spec = features.FeatureSpec(args.kind, params, normalized=bool(args.reference))
    df, skipped = features.build_features(spectra, spec)
    out = Path(args.out)
    (df.to_excel(out, index=False) if out.suffix.lower() == ".xlsx" else
     df.to_csv(out, index=False))
    print(f"{len(df)} rows x {len(features.feature_columns(df))} features -> {out}")
    if skipped:
        print(f"skipped (no resonance): {', '.join(skipped)}")
    return 0


def cmd_train(args) -> int:
    from .core import features, io, ml, plotting, report
    from .core.augmentation import AugmentConfig

    df, notes = features.clean_external(io.read_table(args.dataset, args.sheet))
    for n in notes:
        print(n)
    params = {}
    for item in args.param or []:
        key, _, value = item.partition("=")
        params[key] = float(value) if "." in value else int(value)
    cfg = ml.EvalConfig(model=args.model, params=params, test_size=args.test_size,
                        split=args.split, cv_folds=args.cv, scale=not args.no_scale,
                        seed=args.seed, augment=AugmentConfig() if args.augment else None)
    result = ml.train_and_evaluate(df, cfg, progress=lambda m, f=None: print(f"  {m}"))
    for k, v in result.metric_rows():
        print(f"{k:34s} {v}")
    for n in result.notes:
        print(f"note: {n}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    if result.cv is not None:
        result.cv.to_csv(out / "cross_validation.csv", index=False)
    if result.importance is not None:
        result.importance.to_csv(out / "feature_importance.csv", index=False)
    env = report.environment()
    ml.save_bundle(out / "model.joblib", result, None, env)
    plotting.apply_style()
    fig = plotting.new_figure((120, 100))
    plotting.draw_ml(fig, result)
    plotting.save_figure(fig, out / "predictions.png", 600)
    rep = report.build_report("cli_train", {"argv": sys.argv[1:], "evaluation": cfg.to_dict(),
                                            "features": result.feature_names},
                              [args.dataset], {"metrics": result.metrics,
                                               "cv": None if result.cv is None
                                               else result.cv.to_dict("records")})
    report.write_json(out / "provenance.json", rep)
    print(f"Wrote {out.resolve()}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="osa_analyzer", description=f"{APP_NAME} {__version__}")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("gui", help="start the graphical interface (default)")

    a = sub.add_parser("analyze", help="track resonances and fit sensitivity")
    _spectra_args(a)
    a.add_argument("--quantity", choices=("wavelength", "power"), default="wavelength")
    a.add_argument("--degree", type=int, choices=(1, 2, 3), default=1)
    a.add_argument("--regions", nargs="*", metavar="LO:HI",
                   help="fit separate RI regions, e.g. 1.30:1.39 1.46:1.64")
    a.add_argument("--out", required=True, help="output folder")
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("dataset", help="build an ML table from spectra")
    _spectra_args(d)
    d.add_argument("--kind", choices=("engineered", "wide"), default="engineered")
    d.add_argument("--out", required=True, help="output .csv or .xlsx")
    d.set_defaults(func=cmd_dataset)

    t = sub.add_parser("train", help="train and evaluate an RI regression model")
    t.add_argument("dataset", help=".csv/.xlsx with ri_value and feature columns")
    t.add_argument("--sheet")
    t.add_argument("--model", default="Random Forest")
    t.add_argument("--param", action="append", metavar="KEY=VALUE",
                   help="hyperparameter, repeatable (e.g. --param n_estimators=300)")
    t.add_argument("--split", choices=("sample", "ri"), default="sample")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--cv", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-scale", action="store_true")
    t.add_argument("--augment", action="store_true", help="default augmentation on "
                                                        "training rows")
    t.add_argument("--out", required=True, help="output folder")
    t.set_defaults(func=cmd_train)
    return p


def main(argv=None) -> int:
    # Windows consoles (cp1252) cannot print every symbol used in reports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parser().parse_args(argv)
    if args.command in (None, "gui"):
        from .app import main as gui_main
        return gui_main()
    return args.func(args)

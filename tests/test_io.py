import numpy as np
import pandas as pd
import pytest

from osa_analyzer.core import io


@pytest.mark.parametrize("name,expected", [
    ("1.46_rep2", 1.46), ("1.46_repeat", 1.46), ("RI 1.3", 1.3), ("1.48_1", 1.48),
    ("AC_Lap17", None), ("Air condition", None), ("Reference", None), ("133", None),
    ("sample_1.333", 1.333),
])
def test_parse_ri(name, expected):
    assert io.parse_ri(name) == expected


def test_parse_ri_custom_pattern():
    assert io.parse_ri("probe1.5mm_RI_1.333", r"RI_(\d\.\d+)") == 1.333


def test_txt_folder_keeps_repeats_and_uses_wl_file(txt_folder):
    spectra, report = io.load_files(sorted(txt_folder.glob("*.txt")))
    s = spectra.summary()
    assert s["n_spectra"] == 13                 # 12 labelled + air, WL.txt not a spectrum
    assert s["n_labeled"] == 12
    assert s["n_repeated_ri"] == 6              # v7 kept only one spectrum per RI
    assert "WL.txt" in report.axis
    assert s["wavelength_range"] == (1520.0, 1580.0)
    assert spectra["1.30"].n_sentinel == 1
    assert np.isnan(spectra["1.30"].power[0])
    assert not spectra["Air condition"].is_labeled


def test_txt_without_axis_file_uses_range(txt_folder):
    files = [p for p in txt_folder.glob("*.txt") if p.name != "WL.txt"]
    spectra, report = io.load_files(files, io.AxisSpec(1520, 1580))
    first = next(iter(spectra))
    assert first.step_nm == pytest.approx(0.12)


def test_txt_length_mismatch_is_skipped(txt_folder):
    (txt_folder / "1.50.txt").write_text("\t".join(["-40"] * 400))
    spectra, report = io.load_files(sorted(txt_folder.glob("*.txt")))
    assert "1.50" not in spectra
    assert any("1.50.txt" in s for s in report.skipped)


def test_table_with_wl_header_and_repeat_columns(tmp_path):
    wl = np.linspace(1520, 1580, 501)
    df = pd.DataFrame({"WL": wl, "1.48_1": -50 + 0 * wl, "1.48_2": -51 + 0 * wl,
                       "Unnamed: 3": np.nan, "1.52_1": -52 + 0 * wl})
    path = tmp_path / "rep.xlsx"
    df.to_excel(path, index=False, sheet_name="Repibility")
    spectra, report = io.load_files([path], sheet="Repibility")
    assert spectra.labels == ["1.48_1", "1.48_2", "1.52_1"]
    assert spectra.ri_values() == [1.48, 1.52]
    assert not report.skipped


def test_mixed_selection_rejected(tmp_path, txt_folder):
    xlsx = tmp_path / "x.xlsx"
    pd.DataFrame({"Wavelength": [1, 2, 3]}).to_excel(xlsx, index=False)
    with pytest.raises(ValueError):
        io.load_files([xlsx, txt_folder / "1.30.txt"])

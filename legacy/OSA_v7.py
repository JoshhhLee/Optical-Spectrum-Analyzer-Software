"""
OSA Analyzer Pro v6.1 - ENHANCED FITTING + ML EDITION (BUGFIXED)

Features:
- PyQtGraph (Interactive) + Matplotlib (Publication)
- Multi-TXT batch loading
- Peak/Dip Detection (NO FWHM)
- ENHANCED FITTING: Single/Dual Region, Power/Wavelength vs RI
- Auto-import from Peak Detection
- Scientific plot styling
- Machine Learning with Data Augmentation
- Hyperparameter Tuning

Author: Momow & Kirito
Date: February 13, 2026 (Bugfix release)
"""

import sys
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QDoubleSpinBox,
    QCheckBox, QComboBox, QScrollArea, QGroupBox, QSpinBox,
    QTabWidget, QSplitter, QFrame, QProgressBar, QStatusBar,
    QMessageBox, QSlider, QRadioButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QSizePolicy, QProgressDialog, QInputDialog, QHeaderView,
    QFormLayout, QToolButton
)

from PyQt5.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPalette, QColor

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Tambahkan ini setelah baris "import matplotlib.pyplot as plt"
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,     # untuk export
    'savefig.bbox': 'tight'
})

# PyQtGraph imports
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, InfiniteLine, TextItem, ScatterPlotItem

# Scipy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Set PyQtGraph config
pg.setConfigOptions(antialias=True, background='#1E1E1E', foreground='w')


# ==================== DATA STRUCTURES ====================

@dataclass
class ValidationResult:
    valid: bool
    issues: List[str]
    warnings: List[str]
    info: Dict


class OSAData:
    """Data model for OSA spectrum measurements"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.source_file: Optional[str] = None
        self.file_type: Optional[str] = None
        self.sheet_name: Optional[str] = None
        self.metadata: Dict = {}
    
    def is_loaded(self) -> bool:
        return self.df is not None and not self.df.empty
    
    def get_wavelength(self) -> np.ndarray:
        if not self.is_loaded():
            return np.array([])
        return self.df['Wavelength'].values
    
    def get_power(self, ri_value: float) -> np.ndarray:
        if not self.is_loaded():
            return np.array([])
        
        possible_names = [
            f'RI {ri_value:.2f}',
            f'RI {ri_value}',
            f'{ri_value:.2f}',
            f'{ri_value}'
        ]
        
        for col_name in possible_names:
            if col_name in self.df.columns:
                return self.df[col_name].values
        
        raise ValueError(f"RI value {ri_value} not found in data")
    
    def get_ri_values(self) -> List[float]:
        if not self.is_loaded():
            return []
        
        ri_cols = [col for col in self.df.columns if col.startswith('RI')]
        ri_values = []
        
        for col in ri_cols:
            ri_val = self.extract_ri_from_column(col)
            if ri_val is not None:
                ri_values.append(ri_val)
        
        return sorted(ri_values)
    
    def get_spectrum(self, ri_value: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_wavelength(), self.get_power(ri_value)
    
    @staticmethod
    def extract_ri_from_column(col_name: str) -> Optional[float]:
        match = re.search(r'(\d+\.?\d*)', col_name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def get_info_dict(self) -> Dict:
        if not self.is_loaded():
            return {}
        
        ri_values = self.get_ri_values()
        wl = self.get_wavelength()
        
        return {
            'source_file': self.source_file,
            'file_type': self.file_type,
            'sheet_name': self.sheet_name,
            'num_wavelengths': len(wl),
            'wavelength_range': (wl.min(), wl.max()) if len(wl) > 0 else (0, 0),
            'num_ri_values': len(ri_values),
            'ri_range': (min(ri_values), max(ri_values)) if ri_values else (0, 0),
            'ri_values': ri_values,
            'shape': self.df.shape
        }

# ==================== DATA PARSERS ====================

class ExcelParser:
    @staticmethod
    def get_sheet_names(file_path: str) -> List[str]:
        try:
            xl = pd.ExcelFile(file_path)
            return xl.sheet_names
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")
    
    @staticmethod
    def load_sheet(file_path: str, sheet_name: str) -> OSAData:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()
            
            if 'Wavelength' not in df.columns:
                raise ValueError("Missing 'Wavelength' column")
            
            osa_data = OSAData()
            osa_data.df = df
            osa_data.source_file = file_path
            osa_data.file_type = 'excel'
            osa_data.sheet_name = sheet_name
            osa_data.metadata['raw_shape'] = df.shape
            
            return osa_data
        except Exception as e:
            raise ValueError(f"Failed to load sheet '{sheet_name}': {e}")


class TXTParser:
    # Defaults matched to Yokogawa AQ6370D in the user's typical sweep.
    # Anyone using a different OSA / sweep just changes these via the UI.
    DEFAULT_WL_START = 1520.0
    DEFAULT_WL_END = 1620.0
    
    @staticmethod
    def generate_wavelength_array(start: float = None,
                                   end: float = None,
                                   num_points: int = 501) -> np.ndarray:
        if start is None:
            start = TXTParser.DEFAULT_WL_START
        if end is None:
            end = TXTParser.DEFAULT_WL_END
        return np.linspace(start, end, num_points)
    
    @staticmethod
    def extract_ri_from_filename(filename: str) -> Optional[float]:
        basename = Path(filename).stem
        # Accept both decimal (1.333) and integer (133) RI representations
        match = re.search(r'(\d+\.\d+)', basename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        match = re.search(r'(\d+)', basename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _parse_txt_content(content: str) -> List[float]:
        """Parse TXT content into a list of floats.
        Handles tab, whitespace, comma, and newline separated values."""
        # Try tab first (Yokogawa AQ6370D default), then any whitespace, then comma
        if '\t' in content:
            tokens = content.split('\t')
        elif ',' in content:
            tokens = content.split(',')
        else:
            tokens = content.split()
        return [float(v) for v in tokens if v.strip()]
    
    @staticmethod
    def load_txt_file(file_path: str,
                      ri_value: Optional[float] = None,
                      wavelength_range: Optional[Tuple[float, float]] = None) -> OSAData:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            power_values = TXTParser._parse_txt_content(content)
            
            if len(power_values) < 2:
                raise ValueError(f"Insufficient data points: {len(power_values)} (need at least 2)")
            
            # Auto-detect point count; generate wavelength array of matching length
            if wavelength_range is None:
                wavelength_range = (TXTParser.DEFAULT_WL_START, TXTParser.DEFAULT_WL_END)
            wavelength = TXTParser.generate_wavelength_array(
                start=wavelength_range[0],
                end=wavelength_range[1],
                num_points=len(power_values)
            )
            
            if ri_value is None:
                ri_value = TXTParser.extract_ri_from_filename(file_path)
                if ri_value is None:
                    raise ValueError("Could not extract RI value from filename")
            
            df = pd.DataFrame({
                'Wavelength': wavelength,
                f'RI {ri_value}': power_values
            })
            
            osa_data = OSAData()
            osa_data.df = df
            osa_data.source_file = file_path
            osa_data.file_type = 'txt'
            osa_data.metadata['ri_value'] = ri_value
            osa_data.metadata['num_points'] = len(power_values)
            osa_data.metadata['wavelength_range'] = wavelength_range
            
            return osa_data
        except Exception as e:
            raise ValueError(f"Failed to load TXT file: {e}")
    
    @staticmethod
    def load_multiple_txt_files(file_paths: List[str],
                                 wavelength_range: Optional[Tuple[float, float]] = None) -> OSAData:
        if not file_paths:
            raise ValueError("No files provided")
        
        if wavelength_range is None:
            wavelength_range = (TXTParser.DEFAULT_WL_START, TXTParser.DEFAULT_WL_END)
        
        # Pass 1: read all files, determine reference point count from first valid file
        loaded_files = []  # (file_path, ri_value, power_values)
        reference_num_points = None
        skipped = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                
                power_values = TXTParser._parse_txt_content(content)
                
                if len(power_values) < 2:
                    skipped.append(f"{Path(file_path).name}: insufficient points")
                    continue
                
                if reference_num_points is None:
                    reference_num_points = len(power_values)
                elif len(power_values) != reference_num_points:
                    skipped.append(
                        f"{Path(file_path).name}: {len(power_values)} points, "
                        f"expected {reference_num_points}"
                    )
                    continue
                
                ri_value = TXTParser.extract_ri_from_filename(file_path)
                if ri_value is None:
                    skipped.append(f"{Path(file_path).name}: could not extract RI")
                    continue
                
                loaded_files.append((file_path, ri_value, power_values))
            except Exception as e:
                skipped.append(f"{Path(file_path).name}: {e}")
                continue
        
        if skipped:
            for msg in skipped:
                print(f"⚠️ Skipped {msg}")
        
        if not loaded_files:
            raise ValueError("No valid TXT files could be loaded")
        
        # Build merged dataframe with wavelength axis matching detected point count
        wavelength = TXTParser.generate_wavelength_array(
            start=wavelength_range[0],
            end=wavelength_range[1],
            num_points=reference_num_points
        )
        merged_data = {'Wavelength': wavelength}
        ri_values_loaded = []
        files_loaded = []
        for fp, ri_value, power_values in loaded_files:
            merged_data[f'RI {ri_value}'] = power_values
            ri_values_loaded.append(ri_value)
            files_loaded.append(Path(fp).name)
        
        df = pd.DataFrame(merged_data)
        
        osa_data = OSAData()
        osa_data.df = df
        osa_data.source_file = f"{len(loaded_files)} TXT files"
        osa_data.file_type = 'txt_multi'
        osa_data.metadata['num_files'] = len(loaded_files)
        osa_data.metadata['num_points'] = reference_num_points
        osa_data.metadata['wavelength_range'] = wavelength_range
        osa_data.metadata['ri_values'] = sorted(ri_values_loaded)
        osa_data.metadata['files_loaded'] = files_loaded
        if skipped:
            osa_data.metadata['skipped'] = skipped
        
        return osa_data


class DataValidator:
    INVALID_POWER_VALUE = -210.0
    
    @staticmethod
    def validate(osa_data: OSAData) -> ValidationResult:
        issues = []
        warnings = []
        info = {}
        
        if not osa_data.is_loaded():
            return ValidationResult(valid=False, issues=["No data loaded"], warnings=[], info={})
        
        df = osa_data.df
        num_rows, num_cols = df.shape
        info['num_rows'] = num_rows
        info['num_cols'] = num_cols
        
        if 'Wavelength' in df.columns:
            wl = df['Wavelength'].values
            info['wavelength_range'] = (wl.min(), wl.max())
        else:
            issues.append("Missing 'Wavelength' column")
        
        ri_values = osa_data.get_ri_values()
        info['ri_values'] = ri_values
        info['num_ri_values'] = len(ri_values)
        
        if ri_values:
            info['ri_range'] = (min(ri_values), max(ri_values))
        else:
            issues.append("No RI columns found")
        
        invalid_count = (df == DataValidator.INVALID_POWER_VALUE).sum().sum()
        total_values = df.shape[0] * (df.shape[1] - 1)
        invalid_percentage = (invalid_count / total_values) * 100 if total_values > 0 else 0
        
        info['invalid_count'] = int(invalid_count)
        info['invalid_percentage'] = invalid_percentage
        
        if invalid_percentage > 0:
            warnings.append(f"{invalid_percentage:.1f}% invalid measurements")
        
        valid = len(issues) == 0
        
        return ValidationResult(valid=valid, issues=issues, warnings=warnings, info=info)


class DataLoader:
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext in ['.txt']:
            return 'txt'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    @staticmethod
    def load_file(file_path: str, sheet_name: Optional[str] = None,
                  wavelength_range: Optional[Tuple[float, float]] = None) -> OSAData:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = DataLoader.detect_file_type(file_path)
        
        if file_type == 'excel':
            if sheet_name is None:
                sheet_names = ExcelParser.get_sheet_names(file_path)
                if not sheet_names:
                    raise ValueError("No sheets found in Excel file")
                sheet_name = sheet_names[0]
            return ExcelParser.load_sheet(file_path, sheet_name)
        elif file_type == 'txt':
            return TXTParser.load_txt_file(file_path, wavelength_range=wavelength_range)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def load_multiple_files(file_paths: List[str],
                            wavelength_range: Optional[Tuple[float, float]] = None) -> OSAData:
        return TXTParser.load_multiple_txt_files(file_paths, wavelength_range=wavelength_range)
    
    @staticmethod
    def load_and_validate(file_path: str, sheet_name: Optional[str] = None,
                          wavelength_range: Optional[Tuple[float, float]] = None) -> Tuple[OSAData, ValidationResult]:
        osa_data = DataLoader.load_file(file_path, sheet_name, wavelength_range)
        validation = DataValidator.validate(osa_data)
        return osa_data, validation
    
    @staticmethod
    def load_and_validate_multiple(file_paths: List[str],
                                    wavelength_range: Optional[Tuple[float, float]] = None) -> Tuple[OSAData, ValidationResult]:
        osa_data = DataLoader.load_multiple_files(file_paths, wavelength_range)
        validation = DataValidator.validate(osa_data)
        return osa_data, validation


class DataLoadThread(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    
    def __init__(self, file_paths, sheet_name=None, multi_mode=False,
                 wavelength_range=None):
        super().__init__()
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.sheet_name = sheet_name
        self.multi_mode = multi_mode
        self.wavelength_range = wavelength_range
    
    def run(self):
        try:
            if self.multi_mode and len(self.file_paths) > 1:
                osa_data, validation = DataLoader.load_and_validate_multiple(
                    self.file_paths, wavelength_range=self.wavelength_range
                )
            else:
                osa_data, validation = DataLoader.load_and_validate(
                    self.file_paths[0], self.sheet_name,
                    wavelength_range=self.wavelength_range
                )
            
            self.finished.emit(osa_data, validation)
        except Exception as e:
            self.error.emit(str(e))

# ==================== PEAK/DIP DETECTOR ====================

class PeakDipDetector:
    """Peak and Dip detection for OSA spectra"""
    
    def __init__(self):
        self.peaks_data = {}
        self.dips_data = {}
    
    @staticmethod
    def find_peaks_in_spectrum(wavelength, power, mode='peak', height=None, distance=10, prominence=1.0):
        """Find peaks or dips in spectrum"""
        results = {'peaks': [], 'dips': []}
        
        mask = (~np.isnan(power)) & (power > -210.0)
        wl = wavelength[mask]
        pwr = power[mask]
        
        if len(wl) == 0:
            return results
        
        if mode in ['peak', 'both']:
            try:
                peak_indices, peak_props = find_peaks(
                    pwr,
                    height=height if height is not None else -np.inf,
                    distance=distance,
                    prominence=prominence
                )
                
                for i, idx in enumerate(peak_indices):
                    results['peaks'].append({
                        'index': idx,
                        'wavelength': float(wl[idx]),
                        'power': float(pwr[idx]),
                        'prominence': float(peak_props['prominences'][i]) if 'prominences' in peak_props else 0.0
                    })
            except Exception as e:
                # Silently pass; better to use logging in production
                pass
        
        if mode in ['dip', 'both']:
            try:
                dip_indices, dip_props = find_peaks(
                    -pwr,
                    height=-height if height is not None else -np.inf,
                    distance=distance,
                    prominence=prominence
                )
                
                for i, idx in enumerate(dip_indices):
                    results['dips'].append({
                        'index': idx,
                        'wavelength': float(wl[idx]),
                        'power': float(pwr[idx]),
                        'prominence': float(dip_props['prominences'][i]) if 'prominences' in dip_props else 0.0
                    })
            except Exception as e:
                pass
        
        return results
    
    def detect_all_ri(self, osa_data, ri_values, mode='both', **kwargs):
        """Detect peaks/dips for all RI values"""
        self.peaks_data = {}
        self.dips_data = {}
        
        for ri_value in ri_values:
            try:
                wavelength, power = osa_data.get_spectrum(ri_value)
                results = self.find_peaks_in_spectrum(wavelength, power, mode=mode, **kwargs)
                
                if results['peaks']:
                    self.peaks_data[ri_value] = results['peaks']
                if results['dips']:
                    self.dips_data[ri_value] = results['dips']
            except Exception as e:
                # Silent fail
                continue
        
        return self.peaks_data, self.dips_data
    
    def get_summary_stats(self):
        """Get summary statistics"""
        stats = {
            'num_ri_peaks': len(self.peaks_data),
            'num_ri_dips': len(self.dips_data),
            'total_peaks': sum(len(v) for v in self.peaks_data.values()),
            'total_dips': sum(len(v) for v in self.dips_data.values()),
        }
        
        all_peak_wl = [peak['wavelength'] for peaks in self.peaks_data.values() for peak in peaks]
        if all_peak_wl:
            stats['avg_peak_wavelength'] = np.mean(all_peak_wl)
            stats['std_peak_wavelength'] = np.std(all_peak_wl)
        
        all_dip_wl = [dip['wavelength'] for dips in self.dips_data.values() for dip in dips]
        if all_dip_wl:
            stats['avg_dip_wavelength'] = np.mean(all_dip_wl)
            stats['std_dip_wavelength'] = np.std(all_dip_wl)
        
        return stats
    
    def export_to_dataframe(self, mode='both'):
        """Export results to pandas DataFrame"""
        rows = []
        
        if mode in ['peak', 'both']:
            for ri_value, peaks in self.peaks_data.items():
                for peak in peaks:
                    rows.append({
                        'Type': 'Peak',
                        'RI': ri_value,
                        'Wavelength (nm)': peak['wavelength'],
                        'Power (dBm)': peak['power'],
                        'Prominence': peak['prominence']
                    })
        
        if mode in ['dip', 'both']:
            for ri_value, dips in self.dips_data.items():
                for dip in dips:
                    rows.append({
                        'Type': 'Dip',
                        'RI': ri_value,
                        'Wavelength (nm)': dip['wavelength'],
                        'Power (dBm)': dip['power'],
                        'Prominence': dip['prominence']
                    })
        
        if not rows:
            return pd.DataFrame(columns=['Type', 'RI', 'Wavelength (nm)', 'Power (dBm)', 'Prominence'])
        
        return pd.DataFrame(rows)

# ==================== ENHANCED FITTING ENGINE ====================

class EnhancedFittingEngine:
    """Enhanced curve fitting with Single/Dual region support
    - Auto-import from peak detection
    - Power vs RI and Wavelength vs RI
    - Linear (degree=1), Quadratic (degree=2), Cubic (degree=3) fitting
    - Sensitivity uncertainty (slope ± stderr)
    - Scientific plot styling
    """
    
    DEGREE_LABELS = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic'}
    
    def __init__(self):
        self.ri_data = None
        self.power_data = None
        self.wavelength_data = None
        
        self.single_fit = None
        self.dual_fit = None
    
    def import_from_peak_detection(self, ri_vals, power_vals, wavelength_vals):
        """Import data directly from peak detection"""
        self.ri_data = np.array(ri_vals)
        self.power_data = np.array(power_vals)
        self.wavelength_data = np.array(wavelength_vals)
    
    def _get_y_data(self, mode):
        """Return (y_data, ylabel, unit) for the requested mode."""
        if mode == "wavelength":
            return self.wavelength_data, "Wavelength (nm)", "nm/RIU"
        else:
            return self.power_data, "Power (dBm)", "dBm/RIU"
    
    @staticmethod
    def _format_polynomial_equation(coeffs):
        """Format polynomial coefficients (numpy order: highest power first) as a readable equation."""
        degree = len(coeffs) - 1
        parts = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power == 0:
                term = f"{c:+.4f}"
            elif power == 1:
                term = f"{c:+.4f}·RI"
            else:
                # Use scientific notation for high-degree terms (they're usually small)
                term = f"{c:+.4e}·RI^{power}"
            parts.append(term)
        eq = "Y = " + " ".join(parts)
        # Strip leading "+" for clean look
        eq = eq.replace("= +", "= ")
        return eq
    
    def _compute_polynomial_fit(self, x, y, degree, mode, region_name, ylabel, unit):
        """Compute polynomial fit with uncertainty. Pure function — does NOT mutate self.
        
        Returns a result dict containing coefficients, errors, R², RMSE, sensitivity,
        and a callable poly1d for evaluation.
        """
        n = len(x)
        # np.polyfit requires N > deg+2 for the covariance estimate it returns;
        # otherwise the covariance is degenerate and numpy emits a warning.
        if n > degree + 2:
            coeffs, cov = np.polyfit(x, y, degree, cov=True)
            errors = np.sqrt(np.diag(cov))
        else:
            coeffs = np.polyfit(x, y, degree)
            cov = None
            errors = np.full(degree + 1, np.nan)
        
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        
        # Goodness of fit
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        residuals = y - y_pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        # Sensitivity = dy/dRI at the midpoint of the fitting range.
        # For linear: this is exactly the slope (constant).
        # For polynomial: it's the local sensitivity at RI_center.
        deriv = np.polyder(poly)
        ri_center = float(np.mean(x))
        sensitivity = float(deriv(ri_center))
        
        # Sensitivity uncertainty:
        #   Linear: just the standard error of the slope coefficient.
        #   Polynomial: propagate via gradient of S w.r.t. each coefficient.
        if degree == 1:
            sensitivity_err = float(errors[0]) if not np.isnan(errors[0]) else float('nan')
        else:
            if cov is not None:
                # S = sum_{k=0..deg-1} (deg-k) * coeffs[k] * ri_center^(deg-k-1)
                # ∂S/∂coeffs[k] = (deg-k) * ri_center^(deg-k-1)  for k < deg
                # ∂S/∂coeffs[deg] = 0
                grad = np.zeros(degree + 1)
                for k in range(degree):
                    power = degree - k - 1
                    grad[k] = (degree - k) * (ri_center ** power)
                sensitivity_err = float(np.sqrt(grad @ cov @ grad))
            else:
                sensitivity_err = float('nan')
        
        return {
            'mode': mode,
            'region': region_name,
            'ylabel': ylabel,
            'unit': unit,
            'ri': x,
            'y': y,
            'y_fit': y_pred,
            'residuals': residuals,
            'coeffs': coeffs,             # numpy order: highest power → lowest
            'errors': errors,             # one std dev per coefficient
            'cov': cov,
            'degree': degree,
            'poly': poly,                 # callable: poly(ri_values)
            # Legacy fields preserved for backward compatibility:
            'm': float(coeffs[-2]) if degree >= 1 else 0.0,  # linear coeff
            'c': float(coeffs[-1]),                          # intercept
            'r2': float(r2),
            'rmse': rmse,
            'equation': EnhancedFittingEngine._format_polynomial_equation(coeffs),
            'sensitivity': sensitivity,
            'sensitivity_err': sensitivity_err,
            'ri_center': ri_center,
        }
    
    def fit_single_region(self, mode="wavelength", region="all",
                         low_range=(1.30, 1.39), high_range=(1.46, 1.64),
                         degree=1):
        """Single region fit (linear or polynomial). Stores result in self.single_fit."""
        if self.ri_data is None or len(self.ri_data) == 0:
            raise ValueError("No data loaded. Import from peak detection first!")
        
        y_data, ylabel, unit = self._get_y_data(mode)
        
        if region == "low":
            mask = (self.ri_data >= low_range[0]) & (self.ri_data <= low_range[1])
            region_name = f"Low {low_range[0]:.2f}-{low_range[1]:.2f}"
        elif region == "high":
            mask = (self.ri_data >= high_range[0]) & (self.ri_data <= high_range[1])
            region_name = f"High {high_range[0]:.2f}-{high_range[1]:.2f}"
        else:
            mask = np.ones(len(self.ri_data), dtype=bool)
            region_name = "All RI"
        
        ri_filtered = self.ri_data[mask]
        y_filtered = y_data[mask]
        
        if len(ri_filtered) < degree + 1:
            raise ValueError(
                f"Insufficient data points ({len(ri_filtered)}) for degree-{degree} fit "
                f"in region {region_name}. Need at least {degree + 1} points."
            )
        
        result = self._compute_polynomial_fit(
            ri_filtered, y_filtered, degree, mode, region_name, ylabel, unit
        )
        self.single_fit = result
        return result
    
    def fit_dual_regions(self, mode="wavelength",
                        low_range=(1.30, 1.39), high_range=(1.46, 1.64),
                        degree=1):
        """Dual region fit (Low + High). Does NOT touch self.single_fit anymore."""
        if self.ri_data is None or len(self.ri_data) == 0:
            raise ValueError("No data loaded. Import from peak detection first!")
        
        y_data, ylabel, unit = self._get_y_data(mode)
        
        # Low region
        low_fit = None
        mask_low = (self.ri_data >= low_range[0]) & (self.ri_data <= low_range[1])
        ri_low = self.ri_data[mask_low]
        y_low = y_data[mask_low]
        if len(ri_low) >= degree + 1:
            try:
                low_fit = self._compute_polynomial_fit(
                    ri_low, y_low, degree, mode,
                    f"Low {low_range[0]:.2f}-{low_range[1]:.2f}",
                    ylabel, unit
                )
            except Exception:
                low_fit = None
        
        # High region
        high_fit = None
        mask_high = (self.ri_data >= high_range[0]) & (self.ri_data <= high_range[1])
        ri_high = self.ri_data[mask_high]
        y_high = y_data[mask_high]
        if len(ri_high) >= degree + 1:
            try:
                high_fit = self._compute_polynomial_fit(
                    ri_high, y_high, degree, mode,
                    f"High {high_range[0]:.2f}-{high_range[1]:.2f}",
                    ylabel, unit
                )
            except Exception:
                high_fit = None
        
        self.dual_fit = {
            'mode': mode,
            'low': low_fit,
            'high': high_fit
        }
        return self.dual_fit
    
    def _plot_fit_on_axes(self, ax, fit):
        """Internal helper: render a fit dict onto a given axes. Stateless."""
        ax.clear()
        ax.scatter(fit['ri'], fit['y'], marker='x', color='k', s=35,
                  label='Experimental data', zorder=3)
        
        ri_smooth = np.linspace(fit['ri'].min(), fit['ri'].max(), 200)
        y_smooth = fit['poly'](ri_smooth)
        deg_label = EnhancedFittingEngine.DEGREE_LABELS.get(fit['degree'], f"Degree {fit['degree']}")
        ax.plot(ri_smooth, y_smooth, color='r', linewidth=1.8,
               label=f"{deg_label} fit", zorder=2)
        
        ax.set_xlabel('Refractive Index (RIU)', fontsize=12, fontweight='bold')
        ax.set_ylabel(fit['ylabel'], fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.margins(x=0.05, y=0.05)
        
        eq_text = f"{fit['equation']}\n"
        eq_text += f"R² = {fit['r2']:.4f}\n"
        eq_text += f"RMSE = {fit['rmse']:.4f}\n"
        if not np.isnan(fit['sensitivity_err']):
            eq_text += f"S = {fit['sensitivity']:.4f} ± {fit['sensitivity_err']:.4f} {fit['unit']}"
        else:
            eq_text += f"S = {fit['sensitivity']:.4f} {fit['unit']}"
        if fit['degree'] > 1:
            eq_text += f"\n(local S at RI={fit['ri_center']:.3f})"
        
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
               fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax.legend(loc='best', fontsize=11)
    
    def plot_single_fit(self, ax):
        """Plot single fit with scientific styling"""
        if self.single_fit is None:
            ax.text(0.5, 0.5, 'No fit data available\n\nPeak Analysis first',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_axis_off()
            return
        
        self._plot_fit_on_axes(ax, self.single_fit)
        ax.set_title(f"{self.single_fit['ylabel']} vs RI - Region {self.single_fit['region']}",
                    fontsize=12, fontweight='bold')
    
    def plot_dual_fit(self, fig):
        """Plot dual fit (2 subplots side-by-side). No longer mutates single_fit."""
        if self.dual_fit is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No dual fit data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        
        fig.clear()
        
        has_low = self.dual_fit['low'] is not None
        has_high = self.dual_fit['high'] is not None
        
        if not has_low and not has_high:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available for both regions',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        
        if has_low:
            ax_low = fig.add_subplot(1, 2, 1)
            self._plot_fit_on_axes(ax_low, self.dual_fit['low'])
            ax_low.set_title("Low RI Region", fontsize=12, fontweight='bold', color="#000000")
        
        if has_high:
            ax_high = fig.add_subplot(1, 2, 2)
            self._plot_fit_on_axes(ax_high, self.dual_fit['high'])
            ax_high.set_title("High RI Region", fontsize=12, fontweight='bold', color="#000000")
        
        fig.tight_layout()
    
    @staticmethod
    def _build_fit_summary_df(fit):
        """Build a flat summary DataFrame for any fit (linear or polynomial)."""
        rows = [
            ('Mode', fit['mode']),
            ('Region', fit['region']),
            ('Degree', fit['degree']),
            ('Degree_Type', EnhancedFittingEngine.DEGREE_LABELS.get(fit['degree'], 'Polynomial')),
            ('Equation', fit['equation']),
            ('R²', fit['r2']),
            ('RMSE', fit['rmse']),
            ('Sensitivity', fit['sensitivity']),
            ('Sensitivity_Error', fit['sensitivity_err']),
            ('Sensitivity_Unit', fit['unit']),
            ('Sensitivity_Eval_RI', fit['ri_center']),
            ('N_points', len(fit['ri'])),
        ]
        degree = fit['degree']
        for i, (c, err) in enumerate(zip(fit['coeffs'], fit['errors'])):
            power = degree - i
            if power == 0:
                name = 'Intercept'
            elif power == 1:
                name = 'Coeff_RI (slope)'
            else:
                name = f'Coeff_RI^{power}'
            rows.append((name, float(c)))
            rows.append((f'{name}_Error', float(err)))
        return pd.DataFrame(rows, columns=['Parameter', 'Value'])
    
    def export_fit_results(self, filepath):
        """Export fit results to Excel with fallback engine"""
        try:
            import openpyxl
            engine = 'openpyxl'
        except ImportError:
            try:
                import xlsxwriter
                engine = 'xlsxwriter'
            except ImportError:
                QMessageBox.critical(None, "Export Error",
                                    "No Excel writer found. Please install openpyxl or xlsxwriter.")
                return
        
        with pd.ExcelWriter(filepath, engine=engine) as writer:
            if self.single_fit is not None:
                fit = self.single_fit
                df_data = pd.DataFrame({
                    'RI': fit['ri'],
                    'Y': fit['y'],
                    'Y_fit': fit['y_fit'],
                    'Residual': fit['residuals']
                })
                df_data.to_excel(writer, sheet_name='Single_Fit_Data', index=False)
                
                df_summary = EnhancedFittingEngine._build_fit_summary_df(fit)
                df_summary.to_excel(writer, sheet_name='Single_Fit_Summary', index=False)
            
            if self.dual_fit is not None:
                for region_key in ['low', 'high']:
                    fit = self.dual_fit[region_key]
                    if fit is None:
                        continue
                    
                    df_data = pd.DataFrame({
                        'RI': fit['ri'],
                        'Y': fit['y'],
                        'Y_fit': fit['y_fit'],
                        'Residual': fit['residuals']
                    })
                    data_sheet = f"Dual_Fit_{region_key.capitalize()}_Data"
                    df_data.to_excel(writer, sheet_name=data_sheet[:31], index=False)
                    
                    df_summary = EnhancedFittingEngine._build_fit_summary_df(fit)
                    summary_sheet = f"Dual_Fit_{region_key.capitalize()}_Summary"
                    df_summary.to_excel(writer, sheet_name=summary_sheet[:31], index=False)

# ==================== MACHINE LEARNING CLASSES ====================

class FeatureExtractor:
    """Extract features from OSA spectra for ML"""
    
    @staticmethod
    def extract_peak_features(wavelength, power):
        """Extract peak-related features"""
        features = {}
        
        mask = (~np.isnan(power)) & (power > -210.0)
        wl = wavelength[mask]
        pwr = power[mask]
        
        if len(wl) == 0:
            return None
        
        peak_idx, _ = find_peaks(pwr, distance=10, prominence=1.0)
        
        if len(peak_idx) > 0:
            features['peak_wavelength'] = float(wl[peak_idx[0]])
            features['peak_power'] = float(pwr[peak_idx[0]])
            features['num_peaks'] = len(peak_idx)
        else:
            features['peak_wavelength'] = float(wl[np.argmax(pwr)])
            features['peak_power'] = float(np.max(pwr))
            features['num_peaks'] = 0
        
        features['mean_power'] = float(np.mean(pwr))
        features['std_power'] = float(np.std(pwr))
        features['min_power'] = float(np.min(pwr))
        features['max_power'] = float(np.max(pwr))
        features['power_range'] = features['max_power'] - features['min_power']
        features['mean_wavelength'] = float(np.mean(wl))
        features['wavelength_range'] = float(wl.max() - wl.min())
        
        return features


class MLDatasetBuilder:
    """Build ML dataset from OSA data"""
    
    @staticmethod
    def build_from_peak_data(ri_values, power_values, wavelength_values):
        data = []
        for i in range(len(ri_values)):
            row = {
                'ri_value': ri_values[i],
                'peak_wavelength': wavelength_values[i],
                'peak_power': power_values[i]
            }
            data.append(row)
        return pd.DataFrame(data)
    
    @staticmethod
    def build_from_osa_data(osa_data, ri_values):
        """Build the ~10-feature engineered dataset from peak detection.
        Each row is one spectrum reduced to peak position, mean/std/range power, etc."""
        data = []
        for ri_value in ri_values:
            try:
                wavelength, power = osa_data.get_spectrum(ri_value)
                features = FeatureExtractor.extract_peak_features(wavelength, power)
                if features:
                    features['ri_value'] = ri_value
                    data.append(features)
            except Exception:
                continue
        
        if not data:
            raise ValueError("No features could be extracted")
        
        df = pd.DataFrame(data)
        cols = [col for col in df.columns if col != 'ri_value'] + ['ri_value']
        return df[cols]
    
    @staticmethod
    def build_wide_from_osa_data(osa_data, ri_values):
        """Build a wide-format ML dataset from OSA spectra.
        
        Each row is one full spectrum: target column `ri_value`, plus one
        `WL_<wavelength>` column per wavelength sample. This is the format
        commonly used in spectroscopy ML pipelines (e.g. Yokogawa AQ6370D
        CSV exports). Pairs naturally with tree-based regressors which can
        handle the high feature dimensionality.
        
        Yokogawa's -210.0 sentinel values are preserved as-is (they survive
        through training because MLTrainer.prepare_data treats them as missing).
        """
        if not ri_values:
            raise ValueError("No RI values selected")
        
        # Use the first available spectrum's wavelength axis as the canonical x-axis
        first_wl = None
        for ri in ri_values:
            try:
                wl, _ = osa_data.get_spectrum(ri)
                first_wl = wl
                break
            except Exception:
                continue
        if first_wl is None:
            raise ValueError("No valid spectra found for selected RI values")
        
        # Column names: WL_<wavelength>, formatted to 2 decimals (matches user CSV convention)
        wl_cols = [f"WL_{w:.2f}" for w in first_wl]
        n_cols = len(wl_cols)
        
        # Collect rows into a single 2D numpy array, then wrap in DataFrame.
        # This avoids the pandas "highly fragmented DataFrame" performance warning
        # that appears when building large frames row-by-row.
        ri_collected = []
        rows = []
        skipped = 0
        for ri in ri_values:
            try:
                wl, power = osa_data.get_spectrum(ri)
                if len(power) != n_cols:
                    skipped += 1
                    continue
                ri_collected.append(float(ri))
                rows.append(np.asarray(power, dtype=float))
            except Exception:
                skipped += 1
                continue
        
        if not rows:
            raise ValueError("No spectra could be converted into rows")
        
        spectrum_matrix = np.vstack(rows)  # shape: (n_samples, n_wavelengths)
        df = pd.DataFrame(spectrum_matrix, columns=wl_cols)
        df.insert(0, 'ri_value', ri_collected)
        # ri_value already first; column order matches user's CSV layout
        return df


class DataAugmenter:
    """Data augmentation for OSA spectra.
    
    Each technique adds n_augment independent copies of the original df.
    combined_augmentation applies techniques independently (no chaining):
       result_size = N_original + sum(N_original * n_augment_i)
    instead of the previous geometric-growth bug where techniques chained.
    
    Works with both dataset formats:
      • Engineered features: columns like peak_wavelength, peak_power, mean_power…
      • Wide spectrum:      columns like WL_1520.00, WL_1520.12, … (one per wavelength)
    Format is auto-detected from column names.
    """
    
    # ---- Format detection ----
    
    @staticmethod
    def _is_wide_format(df):
        """Wide format has columns named WL_<float> (or wl_<float>)."""
        return any(str(c).startswith(('WL_', 'wl_')) for c in df.columns)
    
    @staticmethod
    def _get_wide_columns(df):
        """Return the WL_* columns sorted by their embedded wavelength value."""
        wl_cols = [c for c in df.columns if str(c).startswith(('WL_', 'wl_'))]
        try:
            wl_cols.sort(key=lambda c: float(str(c).split('_')[1]))
        except (IndexError, ValueError):
            pass  # leave as-is if names don't parse
        return wl_cols
    
    @staticmethod
    def _get_power_feature_cols(df, target_col='ri_value'):
        """Columns to apply power-scaling style augmentation to.
        Wide format → all WL_* columns. Engineered → 'power'-named cols (legacy)."""
        if DataAugmenter._is_wide_format(df):
            return DataAugmenter._get_wide_columns(df)
        return [c for c in df.columns
                if 'power' in str(c).lower() and c != target_col]
    
    @staticmethod
    def _get_noise_feature_cols(df, target_col='ri_value'):
        """All numeric, non-target columns get noise."""
        return [c for c in df.columns
                if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    
    # ---- Private helpers: each returns ONE augmented copy (not the original) ----
    
    @staticmethod
    def _gen_noise_copy(df, target_col, noise_level):
        """Add Gaussian noise. Noise std = noise_level × column std.
        Vectorized: one matrix op instead of per-column assignment (which
        would fragment the underlying DataFrame block manager)."""
        feature_cols = DataAugmenter._get_noise_feature_cols(df, target_col)
        df_aug = df.copy()
        if not feature_cols:
            return df_aug
        feature_vals = df[feature_cols].to_numpy(dtype=float)
        stds = np.nan_to_num(np.std(feature_vals, axis=0))
        noise = np.random.normal(0, 1.0, feature_vals.shape) * (noise_level * stds)
        df_aug[feature_cols] = feature_vals + noise
        return df_aug
    
    @staticmethod
    def _gen_shift_copy(df, target_col, shift_range):
        """Wavelength shift.
        
        Wide format: roll each row's spectrum by ±shift_range nm (computed in column-step units).
                     Edge values wrap around — fine for small shifts.
        Engineered:  shift every wavelength-named column by a constant ± shift_range nm.
        """
        df_aug = df.copy()
        
        if DataAugmenter._is_wide_format(df):
            wl_cols = DataAugmenter._get_wide_columns(df)
            if len(wl_cols) < 2:
                return df_aug
            try:
                w0 = float(str(wl_cols[0]).split('_')[1])
                w1 = float(str(wl_cols[1]).split('_')[1])
                step = abs(w1 - w0)
            except (IndexError, ValueError):
                return df_aug
            if step <= 0:
                return df_aug
            shift_nm = np.random.uniform(-shift_range, shift_range)
            shift_steps = int(round(shift_nm / step))
            if shift_steps == 0:
                return df_aug
            # Roll each row's spectrum
            arr = df_aug[wl_cols].to_numpy()
            df_aug[wl_cols] = np.roll(arr, shift_steps, axis=1)
            return df_aug
        
        # Engineered format
        wavelength_cols = [c for c in df.columns
                          if 'wavelength' in str(c).lower() and c != target_col]
        if not wavelength_cols:
            return df_aug
        shift = np.random.uniform(-shift_range, shift_range)
        for col in wavelength_cols:
            df_aug[col] = df[col] + shift
        return df_aug
    
    @staticmethod
    def _gen_scale_copy(df, target_col, scale_range):
        """Multiplicative power scaling. Vectorized to avoid block fragmentation."""
        df_aug = df.copy()
        power_cols = DataAugmenter._get_power_feature_cols(df, target_col)
        if not power_cols:
            return df_aug
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        feature_vals = df[power_cols].to_numpy(dtype=float)
        df_aug[power_cols] = feature_vals * scale
        return df_aug
    
    # ---- Public API: each returns original + n_augment copies ----
    
    @staticmethod
    def add_gaussian_noise(df, target_col='ri_value', noise_level=0.01, n_augment=1):
        copies = [DataAugmenter._gen_noise_copy(df, target_col, noise_level)
                  for _ in range(n_augment)]
        return pd.concat([df] + copies, ignore_index=True)
    
    @staticmethod
    def wavelength_shift(df, target_col='ri_value', shift_range=0.5, n_augment=1):
        copies = [DataAugmenter._gen_shift_copy(df, target_col, shift_range)
                  for _ in range(n_augment)]
        return pd.concat([df] + copies, ignore_index=True)
    
    @staticmethod
    def power_scaling(df, target_col='ri_value', scale_range=0.05, n_augment=1):
        copies = [DataAugmenter._gen_scale_copy(df, target_col, scale_range)
                  for _ in range(n_augment)]
        return pd.concat([df] + copies, ignore_index=True)
    
    @staticmethod
    def combined_augmentation(df, target_col='ri_value', config=None, add_labels=False):
        """Apply multiple techniques INDEPENDENTLY to the original df.
        
        If add_labels=True, an 'augmentation' string column is added describing
        which technique created each row (e.g. 'original', 'noise_0.010_copy1',
        'wl_shift_0.30nm_copy1'). This column is non-numeric so prepare_data
        drops it during training; it survives in the saved CSV.
        """
        if config is None:
            config = {
                'gaussian_noise':   {'enabled': True, 'noise_level': 0.01, 'n_augment': 1},
                'wavelength_shift': {'enabled': True, 'shift_range': 0.3,  'n_augment': 1},
                'power_scaling':    {'enabled': True, 'scale_range': 0.03, 'n_augment': 1}
            }
        
        parts = [df.copy()]
        # Track labels in parallel and assign them ALL at the end. Assigning the
        # 'augmentation' column to each part inside the loop fragments the
        # underlying block manager when the frame has many columns (e.g. 1000+
        # wavelength features), which triggers a pandas PerformanceWarning.
        labels = ['original'] * len(df) if add_labels else None
        
        if config.get('gaussian_noise', {}).get('enabled'):
            cfg = config['gaussian_noise']
            for i in range(cfg.get('n_augment', 1)):
                d = DataAugmenter._gen_noise_copy(df, target_col, cfg.get('noise_level', 0.01))
                parts.append(d)
                if add_labels:
                    labels.extend([f"noise_{cfg.get('noise_level', 0.01):.3f}_c{i+1}"] * len(d))
        
        if config.get('wavelength_shift', {}).get('enabled'):
            cfg = config['wavelength_shift']
            for i in range(cfg.get('n_augment', 1)):
                d = DataAugmenter._gen_shift_copy(df, target_col, cfg.get('shift_range', 0.3))
                parts.append(d)
                if add_labels:
                    labels.extend([f"wl_shift_{cfg.get('shift_range', 0.3):.2f}nm_c{i+1}"] * len(d))
        
        if config.get('power_scaling', {}).get('enabled'):
            cfg = config['power_scaling']
            for i in range(cfg.get('n_augment', 1)):
                d = DataAugmenter._gen_scale_copy(df, target_col, cfg.get('scale_range', 0.03))
                parts.append(d)
                if add_labels:
                    labels.extend([f"scale_{cfg.get('scale_range', 0.03):.3f}_c{i+1}"] * len(d))
        
        result = pd.concat(parts, ignore_index=True)
        if add_labels:
            # Defragment after concat (which merges many independent block managers)
            # before single column assignment, otherwise pandas warns.
            result = result.copy()
            result['augmentation'] = labels
        return result
    
    @staticmethod
    def predict_size(n_original, config):
        """Predict the total size after augmentation for the live UI preview."""
        total = n_original
        for key in ('gaussian_noise', 'wavelength_shift', 'power_scaling'):
            if config.get(key, {}).get('enabled'):
                total += n_original * config[key].get('n_augment', 0)
        return total
    
    @staticmethod
    def get_augmentation_summary(df_original, df_augmented):
        n_orig = len(df_original)
        n_total = len(df_augmented)
        return {
            'original_samples': n_orig,
            'augmented_samples': n_total - n_orig,
            'total_samples': n_total,
            'augmentation_factor': n_total / n_orig if n_orig > 0 else 0,
            'added_samples': n_total - n_orig
        }


class _TorchModels:
    """Lazy-loaded PyTorch model definitions for ANN and CNN-1D regression.
    
    Kept in a single class with lazy imports so the rest of the app — and users
    who never touch deep learning — never pay the torch import cost. torch is
    only imported the first time a DL model is actually built.
    
    Both models are small by design: RI-sensor datasets are typically tens to a
    few hundred samples, so large networks would overfit instantly. The defaults
    are deliberately modest.
    """
    
    _torch = None
    _nn = None
    
    @classmethod
    def _ensure_torch(cls):
        """Import torch on first use. Raises a friendly error if not installed."""
        if cls._torch is None:
            try:
                import torch
                import torch.nn as nn
                cls._torch = torch
                cls._nn = nn
            except ImportError:
                raise ImportError(
                    "PyTorch is not installed in this environment.\n\n"
                    "Install the CPU build with:\n"
                    "  pip install torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cpu"
                )
        return cls._torch, cls._nn
    
    @classmethod
    def build_ann(cls, n_features, hidden_layers=2, neurons=128, dropout=0.2):
        """Multi-layer perceptron for tabular regression.
        
        Architecture: input → [Linear→ReLU→Dropout] × hidden_layers → Linear(1)
        Works on any feature count (engineered ~10 or wide ~500).
        """
        torch, nn = cls._ensure_torch()
        
        layers = []
        in_dim = n_features
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        
        return nn.Sequential(*layers)
    
    @classmethod
    def build_cnn1d(cls, n_features, conv_filters=32, kernel_size=7,
                    fc_neurons=64, dropout=0.2):
        """1D CNN for raw-spectrum regression.
        
        Treats the spectrum as a 1-channel signal of length n_features.
        Architecture:
          Conv1d → ReLU → MaxPool → Conv1d → ReLU → AdaptiveMaxPool
          → Flatten → Linear → ReLU → Dropout → Linear(1)
        
        Convolution learns local spectral patterns (dip shape, edges) which is
        exactly the structure present in an RI-shifting resonance dip — this is
        why CNN-1D often beats tree models on raw spectra.
        """
        torch, nn = cls._ensure_torch()
        
        # Guard: kernel can't exceed the signal length
        k = min(kernel_size, max(3, n_features // 4))
        # Ensure odd kernel for symmetric padding
        if k % 2 == 0:
            k += 1
        pad = k // 2
        
        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=k, padding=pad)
                self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2,
                                       kernel_size=k, padding=pad)
                self.pool = nn.MaxPool1d(2)
                self.adaptive = nn.AdaptiveMaxPool1d(8)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(conv_filters * 2 * 8, fc_neurons)
                self.fc2 = nn.Linear(fc_neurons, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # x arrives as (batch, features); add channel dim → (batch, 1, features)
                x = x.unsqueeze(1)
                x = self.pool(self.relu(self.conv1(x)))
                x = self.relu(self.conv2(x))
                x = self.adaptive(x)
                x = x.flatten(1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return CNN1D()


class _TorchRegressor:
    """Adapter that gives a trained torch model an sklearn-like .predict(X).
    
    This lets the rest of the app (predict_dataframe, worst_predictions, the
    metrics table) treat ANN/CNN-1D models exactly like RandomForest etc.
    without special-casing. Note: pickling a torch model requires torch to be
    importable at load time, which is fine since you can only have trained one
    in the first place if torch was installed.
    """
    def __init__(self, torch_model, torch_module):
        self._model = torch_model
        self._torch = torch_module
    
    def predict(self, X):
        import numpy as _np
        self._model.eval()
        arr = _np.asarray(X, dtype=_np.float32)
        with self._torch.no_grad():
            t = self._torch.from_numpy(arr)
            out = self._model(t).numpy().flatten()
        return out
    
    def __getstate__(self):
        # For pickling: store the torch model's state via torch.save into bytes
        import io
        buf = io.BytesIO()
        self._torch.save(self._model, buf)
        return {'model_bytes': buf.getvalue()}
    
    def __setstate__(self, state):
        import io
        import torch as _torch
        self._torch = _torch
        buf = io.BytesIO(state['model_bytes'])
        # weights_only=False needed because we saved the whole module
        self._model = _torch.load(buf, weights_only=False)


class MLTrainer:
    """ML model training and evaluation"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.metrics = None
    
    def prepare_data(self, df, target_col='ri_value', test_size=0.2, scale=True, random_state=42):
        """Prepare X/y splits for training.
        
        Robust to several real-world inputs:
          • Non-numeric feature columns (e.g. 'augmentation' labels, categorical metadata)
            are dropped silently — only numeric features survive.
          • Yokogawa -210.0 sentinel cells are treated as missing.
          • Any remaining NaN cells are filled with the column median.
        These three behaviors mean the same prepare_data path works whether the
        dataset came from peak detection, wide-spectrum builder, or an external CSV.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = pd.to_numeric(df[target_col], errors='coerce')
        if y.isna().any():
            mask = ~y.isna()
            df = df.loc[mask]
            y = y.loc[mask]
        
        X = df.drop(columns=[target_col])
        # Keep only numeric features (drops augmentation labels, string metadata, etc.)
        X = X.select_dtypes(include=[np.number]).copy()
        if X.shape[1] == 0:
            raise ValueError("No numeric feature columns found after dropping target.")
        
        # Yokogawa sentinel → missing → median imputation
        if (X == -210.0).any().any():
            X = X.replace(-210.0, np.nan)
        if X.isna().any().any():
            medians = X.median()
            medians = medians.fillna(0.0)
            X = X.fillna(medians)
        
        self.feature_names = X.columns.tolist()
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.feature_names,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.feature_names,
                index=self.X_test.index
            )
    
    def train(self, model_name='Random Forest', **kwargs):
        if self.X_train is None:
            raise ValueError("Call prepare_data() first!")
        
        if model_name == 'Linear Regression':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        
        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestRegressor
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'random_state': 42,
                'n_jobs': -1
            }
            self.model = RandomForestRegressor(**params)
        
        elif model_name == 'XGBoost':
            try:
                from xgboost import XGBRegressor
                params = {
                    'n_estimators': kwargs.get('n_estimators', 100),
                    'learning_rate': kwargs.get('learning_rate', 0.1),
                    'max_depth': kwargs.get('max_depth', 6),
                    'random_state': 42,
                    'n_jobs': -1
                }
                self.model = XGBRegressor(**params)
            except ImportError:
                raise ImportError("XGBoost not installed! Use: pip install xgboost")
        
        elif model_name == 'CatBoost':
            try:
                from catboost import CatBoostRegressor
                params = {
                    'iterations': kwargs.get('iterations', 100),
                    'learning_rate': kwargs.get('learning_rate', 0.03),
                    'depth': kwargs.get('depth', 6),
                    'random_state': 42,
                    'verbose': 0
                }
                self.model = CatBoostRegressor(**params)
            except ImportError:
                raise ImportError("CatBoost not installed! Use: pip install catboost")
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.fit(self.X_train, self.y_train)
        
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        self.metrics = {
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfitting_gap': train_r2 - test_r2,
            'n_samples': len(self.X_train) + len(self.X_test),
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'n_features': len(self.feature_names),
            'y_train': self.y_train.values,
            'y_train_pred': y_train_pred,
            'y_test': self.y_test.values,
            'y_test_pred': y_test_pred
        }
        
        return self.metrics
    
    def train_deep_learning(self, model_name='ANN', epochs=100, batch_size=32,
                            learning_rate=0.001, early_stopping=True,
                            patience=15, progress_callback=None, **kwargs):
        """Train an ANN or CNN-1D model with PyTorch (CPU).
        
        progress_callback, if given, is called once per epoch with a dict:
          {'epoch', 'total_epochs', 'train_loss', 'val_loss', 'best_val_loss'}
        so the UI can draw a live loss curve. Training runs on whatever device
        torch defaults to; for the OSA app that is CPU, which is plenty fast for
        the small datasets involved.
        
        Returns the same metrics dict shape as the sklearn train() path, so the
        rest of the app (results table, worst-predictions, export) works unchanged.
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first!")
        
        torch, nn = _TorchModels._ensure_torch()
        torch.manual_seed(42)
        
        n_features = len(self.feature_names)
        
        # Build the requested architecture
        if model_name == 'ANN':
            model = _TorchModels.build_ann(
                n_features,
                hidden_layers=kwargs.get('hidden_layers', 2),
                neurons=kwargs.get('neurons', 128),
                dropout=kwargs.get('dropout', 0.2),
            )
        elif model_name == 'CNN-1D':
            if n_features < 12:
                raise ValueError(
                    f"CNN-1D needs wide-format spectral data, but the current "
                    f"dataset has only {n_features} features.\n\n"
                    "Switch the dataset source to 'Raw Spectrum (Wide)' so each "
                    "wavelength is its own column, then rebuild the dataset."
                )
            model = _TorchModels.build_cnn1d(
                n_features,
                conv_filters=kwargs.get('conv_filters', 32),
                kernel_size=kwargs.get('kernel_size', 7),
                fc_neurons=kwargs.get('fc_neurons', 64),
                dropout=kwargs.get('dropout', 0.2),
            )
        else:
            raise ValueError(f"Unknown DL model: {model_name}")
        
        # Convert data to tensors. DL works best on scaled inputs; if the user
        # didn't enable scaling, we still standardize internally for stable
        # gradients (and stash that scaler so prediction matches).
        X_tr = np.asarray(self.X_train, dtype=np.float32)
        X_te = np.asarray(self.X_test, dtype=np.float32)
        y_tr = np.asarray(self.y_train, dtype=np.float32).reshape(-1, 1)
        y_te = np.asarray(self.y_test, dtype=np.float32).reshape(-1, 1)
        
        if self.scaler is None:
            # Internal standardization for DL stability — saved so predict() aligns.
            from sklearn.preprocessing import StandardScaler
            internal_scaler = StandardScaler()
            X_tr = internal_scaler.fit_transform(X_tr).astype(np.float32)
            X_te = internal_scaler.transform(X_te).astype(np.float32)
            self.scaler = internal_scaler
        
        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_te_t = torch.from_numpy(X_te)
        y_te_t = torch.from_numpy(y_te)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        n = X_tr_t.shape[0]
        batch_size = min(batch_size, n)  # tiny datasets: don't exceed sample count
        
        best_val = float('inf')
        best_state = None
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            model.train()
            # Shuffle each epoch
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_te_t)
                val_loss = criterion(val_pred, y_te_t).item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Track best + early stopping
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if progress_callback is not None:
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val,
                })
            
            if early_stopping and epochs_no_improve >= patience:
                # Restore the best weights seen so far before stopping
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        else:
            # Loop finished without early-stop: still restore best weights
            if best_state is not None:
                model.load_state_dict(best_state)
        
        # Wrap the torch model so .predict() works like an sklearn model
        self.model = _TorchRegressor(model, torch)
        
        # Compute the standard metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_tr_t).numpy().flatten()
            y_test_pred = model(X_te_t).numpy().flatten()
        
        y_train_true = y_tr.flatten()
        y_test_true = y_te.flatten()
        
        train_r2 = r2_score(y_train_true, y_train_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        
        self.metrics = {
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfitting_gap': train_r2 - test_r2,
            'n_samples': len(y_train_true) + len(y_test_true),
            'n_train': len(y_train_true),
            'n_test': len(y_test_true),
            'n_features': n_features,
            'y_train': y_train_true,
            'y_train_pred': y_train_pred,
            'y_test': y_test_true,
            'y_test_pred': y_test_pred,
            'epochs_run': len(history['train_loss']),
            'loss_history': history,
        }
        return self.metrics
    
    def get_feature_importance(self):
        if self.model is None:
            return None
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return df
        return None
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Train a model first!")
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names
            )
        return self.model.predict(X)
    
    def save_model(self, file_path):
        """Save the trained model + scaler + feature names + metrics to a pickle file.
        
        Pickled bundle is self-contained: a session loading this file gets back the
        full trained state including the exact feature column ordering the model expects.
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'format_version': 1,
            }, f)
    
    def load_model(self, file_path):
        """Restore a previously saved model bundle. Replaces current trainer state.
        
        Returns the metrics dict from when the model was originally trained. Raises
        ValueError if the file is unreadable, corrupted, or missing required keys.
        """
        import pickle
        try:
            with open(file_path, 'rb') as f:
                bundle = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not read model file: {e}")
        
        required = {'model', 'feature_names'}
        if not required.issubset(bundle.keys()):
            raise ValueError("Not a valid saved model (missing required fields).")
        
        self.model = bundle['model']
        self.scaler = bundle.get('scaler')  # may be None if not scaled at train time
        self.feature_names = bundle['feature_names']
        self.metrics = bundle.get('metrics', {})
        return self.metrics
    
    def predict_dataframe(self, df):
        """Predict on a DataFrame, automatically aligning columns to the trained model.
        
        Returns (predictions, info) where info describes any column adjustments made.
        Raises ValueError if no model is loaded or critical features are missing.
        
        Handles three real-world prediction issues:
          1. Extra columns the model wasn't trained on → silently dropped (logged in info)
          2. Missing columns the model expects → ValueError with a clear list
          3. Yokogawa -210.0 sentinels and NaN → imputed with column median (or 0 if all-NaN)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train a model or load one first.")
        if not self.feature_names:
            raise ValueError("Model has no feature names — cannot align prediction input.")
        
        info = {'dropped_extra': [], 'imputed_cells': 0, 'sentinel_cells': 0}
        
        # Drop any obvious non-numeric / non-feature columns (e.g. ri_value, augmentation labels)
        candidates = df.select_dtypes(include=[np.number]).copy()
        info['dropped_extra'] = [c for c in df.columns if c not in candidates.columns]
        
        missing = [c for c in self.feature_names if c not in candidates.columns]
        if missing:
            sample = ', '.join(missing[:5])
            if len(missing) > 5:
                sample += f', … (+{len(missing) - 5} more)'
            raise ValueError(
                f"Input is missing {len(missing)} feature(s) the model needs: {sample}\n\n"
                "Make sure the new data has the same wavelength columns / feature names "
                "as the training set."
            )
        
        # Align: keep only training features, in the original order
        X = candidates[self.feature_names].copy()
        
        # Clean sentinels and NaN the same way the training cleaner did
        n_sentinel = int((X == -210.0).sum().sum())
        if n_sentinel > 0:
            X = X.replace(-210.0, np.nan)
            info['sentinel_cells'] = n_sentinel
        n_nan = int(X.isna().sum().sum())
        if n_nan > 0:
            medians = X.median().fillna(0.0)
            X = X.fillna(medians)
            info['imputed_cells'] = n_nan
        
        # Apply the saved scaler (if any) before predicting
        if self.scaler is not None:
            X_arr = self.scaler.transform(X)
            X = pd.DataFrame(X_arr, columns=self.feature_names, index=X.index)
        
        predictions = self.model.predict(X)
        return predictions, info
    
    def worst_predictions(self, n=10):
        """Return the n training/test samples with the largest absolute error.
        
        Useful for spotting mislabeled samples or genuine outliers. Pulls from the
        metrics dict captured at training time, so the train/test split is reproducible
        within a single session.
        """
        if not self.metrics:
            raise ValueError("No metrics available. Train a model first.")
        
        rows = []
        for split in ('train', 'test'):
            y_true = self.metrics.get(f'y_{split}')
            y_pred = self.metrics.get(f'y_{split}_pred')
            if y_true is None or y_pred is None:
                continue
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            for i in range(len(y_true_arr)):
                rows.append({
                    'split': split,
                    'index': int(i),
                    'true_ri': float(y_true_arr[i]),
                    'predicted_ri': float(y_pred_arr[i]),
                    'abs_error': float(abs(y_true_arr[i] - y_pred_arr[i])),
                })
        
        if not rows:
            raise ValueError("No prediction data found in metrics.")
        
        df = pd.DataFrame(rows).sort_values('abs_error', ascending=False).head(n)
        df.reset_index(drop=True, inplace=True)
        return df

# ==================== PYQTGRAPH WIDGETS ====================

class FastPlotWidget(PlotWidget):
    """PyQtGraph-based fast interactive plot - FIXED X-AXIS (nm only)"""
    
    def __init__(self, parent=None):
        super().__init__(parent, background='#1E1E1E')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        
        self.getAxis('bottom').enableAutoSIPrefix(False)
        self.getAxis('left').enableAutoSIPrefix(False)
        
        self.setLabel('bottom', 'Wavelength (nm)')
        self.setLabel('left', 'Power (dBm)')
        
        self.getAxis('bottom').setTextPen('#FFF')
        self.getAxis('bottom').setPen('#FFF')
        self.getAxis('left').setTextPen('#FFF')
        self.getAxis('left').setPen('#FFF')
        
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange()
        
        self.vLine = InfiniteLine(angle=90, movable=False, pen=mkPen('#F57C00', width=1, style=Qt.DashLine))
        self.hLine = InfiniteLine(angle=0, movable=False, pen=mkPen('#F57C00', width=1, style=Qt.DashLine))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        
        self.cursorLabel = TextItem(anchor=(0, 1), color='#FFF', fill='#2D2D2D')
        self.addItem(self.cursorLabel)
        self.cursorLabel.setPos(0, 0)
        
        self.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        self.plot_items = []
    
    def on_mouse_moved(self, pos):
        if self.sceneBoundingRect().contains(pos):
            mousePoint = self.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.cursorLabel.setText(f"λ={mousePoint.x():.2f} nm, P={mousePoint.y():.2f} dBm")
            self.cursorLabel.setPos(mousePoint.x(), mousePoint.y())
    
    def plot_spectra(self, osa_data, ri_values, settings):
        for item in self.plot_items:
            self.removeItem(item)
        self.plot_items.clear()
        
        xmin = settings['xmin']
        xmax = settings['xmax']
        ymin = settings['ymin']
        ymax = settings['ymax']
        auto_yaxis = settings['auto_yaxis']
        show_grid = settings['show_grid']
        show_legend = settings['show_legend']
        linewidth = settings['linewidth']
        colors = settings.get('colors', None)
        
        if colors is None:
            colors = []
            for i in range(len(ri_values)):
                t = i / max(1, len(ri_values) - 1)
                r = int(255 * (0.267 + 0.533 * t))
                g = int(255 * (0.005 + 0.874 * t - 0.879 * t**2))
                b = int(255 * (0.329 - 0.334 * t))
                colors.append((r, g, b))
        else:
            colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        
        all_y = []
        
        for i, ri_value in enumerate(ri_values):
            try:
                wavelength, power = osa_data.get_spectrum(ri_value)
                mask = (wavelength >= xmin) & (wavelength <= xmax)
                wl_filtered = wavelength[mask]
                p_filtered = power[mask]
                
                p_filtered = np.where(p_filtered == -210.0, np.nan, p_filtered)
                
                valid_p = p_filtered[~np.isnan(p_filtered)]
                if len(valid_p) > 0:
                    all_y.extend(valid_p)
                
                pen = mkPen(color=colors[i], width=linewidth)
                plot_item = self.plot(wl_filtered, p_filtered, pen=pen, name=f'RI {ri_value:.2f}')
                self.plot_items.append(plot_item)
            except Exception:
                continue
        
        self.setXRange(xmin, xmax, padding=0)
        
        if auto_yaxis and len(all_y) > 0:
            y_min, y_max = np.min(all_y), np.max(all_y)
            y_range = y_max - y_min
            padding = y_range * 0.05
            self.setYRange(y_min - padding, y_max + padding, padding=0)
        else:
            if ymin is not None and ymax is not None:
                self.setYRange(ymin, ymax, padding=0)
        
        self.showGrid(x=show_grid, y=show_grid, alpha=0.3)
        
        if show_legend and len(ri_values) <= 15:
            self.addLegend(offset=(-10, 10))
        elif hasattr(self.plotItem, 'legend') and self.plotItem.legend is not None:
            self.plotItem.legend.scene().removeItem(self.plotItem.legend)
            self.plotItem.legend = None
    
    def clear_plot(self):
        for item in self.plot_items:
            self.removeItem(item)
        self.plot_items.clear()
        self.clear()


class PeakDipPlotWidget(PlotWidget):
    """PyQtGraph widget for peak/dip detection - FIXED X-AXIS (nm only)"""
    
    def __init__(self, parent=None):
        super().__init__(parent, background='#1E1E1E')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        
        self.getAxis('bottom').enableAutoSIPrefix(False)
        self.getAxis('left').enableAutoSIPrefix(False)
        
        self.setLabel('bottom', 'Wavelength (nm)')
        self.setLabel('left', 'Power (dBm)')
        
        self.getAxis('bottom').setTextPen('#FFF')
        self.getAxis('bottom').setPen('#FFF')
        self.getAxis('left').setTextPen('#FFF')
        self.getAxis('left').setPen('#FFF')
        
        self.setTitle('Peak/Dip Detection', color='#F57C00', size='13pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange()
        
        self.plot_items = []
        self.peak_markers = []
        self.dip_markers = []
    
    def plot_with_peaks_dips(self, osa_data, ri_values, peaks_data, dips_data, settings):
        self.clear_all()
        
        xmin = settings.get('xmin', 1520)
        xmax = settings.get('xmax', 1580)
        ymin = settings.get('ymin', -65)
        ymax = settings.get('ymax', -35)
        auto_yaxis = settings.get('auto_yaxis', True)
        colors = settings.get('colors', None)
        linewidth = settings.get('linewidth', 1.5)
        show_grid = settings.get('show_grid', True)
        
        if colors is None:
            colors = []
            for i in range(len(ri_values)):
                t = i / max(1, len(ri_values) - 1)
                r = int(255 * (0.267 + 0.533 * t))
                g = int(255 * (0.005 + 0.874 * t - 0.879 * t**2))
                b = int(255 * (0.329 - 0.334 * t))
                colors.append((r, g, b))
        else:
            colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        
        all_y = []
        
        for i, ri_value in enumerate(ri_values):
            try:
                wavelength, power = osa_data.get_spectrum(ri_value)
                mask = (wavelength >= xmin) & (wavelength <= xmax)
                wl_filtered = wavelength[mask]
                p_filtered = power[mask]
                
                p_filtered = np.where(p_filtered == -210.0, np.nan, p_filtered)
                
                valid_p = p_filtered[~np.isnan(p_filtered)]
                if len(valid_p) > 0:
                    all_y.extend(valid_p)
                
                pen = mkPen(color=colors[i], width=linewidth)
                plot_item = self.plot(wl_filtered, p_filtered, pen=pen, name=f'RI {ri_value:.2f}')
                self.plot_items.append(plot_item)
            except Exception:
                continue
        
        for ri_value in ri_values:
            if ri_value in peaks_data:
                peaks = peaks_data[ri_value]
                wl_peaks = [p['wavelength'] for p in peaks if xmin <= p['wavelength'] <= xmax]
                pwr_peaks = [p['power'] for p in peaks if xmin <= p['wavelength'] <= xmax]
                if wl_peaks:
                    scatter = ScatterPlotItem(
                        x=wl_peaks, y=pwr_peaks,
                        symbol='o', size=12,
                        pen=mkPen('r', width=2), brush='r'
                    )
                    self.addItem(scatter)
                    self.peak_markers.append(scatter)
        
        for ri_value in ri_values:
            if ri_value in dips_data:
                dips = dips_data[ri_value]
                wl_dips = [d['wavelength'] for d in dips if xmin <= d['wavelength'] <= xmax]
                pwr_dips = [d['power'] for d in dips if xmin <= d['wavelength'] <= xmax]
                if wl_dips:
                    scatter = ScatterPlotItem(
                        x=wl_dips, y=pwr_dips,
                        symbol='t', size=12,
                        pen=mkPen('b', width=2), brush='b'
                    )
                    self.addItem(scatter)
                    self.dip_markers.append(scatter)
        
        self.setXRange(xmin, xmax, padding=0)
        
        if auto_yaxis and len(all_y) > 0:
            y_min, y_max = np.min(all_y), np.max(all_y)
            y_range = y_max - y_min
            padding_val = y_range * 0.05
            self.setYRange(y_min - padding_val, y_max + padding_val, padding=0)
        else:
            if ymin is not None and ymax is not None:
                self.setYRange(ymin, ymax, padding=0)
        
        self.showGrid(x=show_grid, y=show_grid, alpha=0.3)
    
    def clear_all(self):
        for item in self.plot_items:
            self.removeItem(item)
        self.plot_items.clear()
        for marker in self.peak_markers:
            self.removeItem(marker)
        self.peak_markers.clear()
        for marker in self.dip_markers:
            self.removeItem(marker)
        self.dip_markers.clear()
        self.clear()

# ==================== MATPLOTLIB CANVASES ====================

class PublicationCanvas(FigureCanvas):
    """Matplotlib canvas for publication plots"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2D2D2D')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setup_publication_style()
        self.fig.tight_layout()
    
    def setup_publication_style(self):
        self.axes.set_facecolor('#1E1E1E')
        self.axes.tick_params(colors='white', which='both', labelsize=10)
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.spines['right'].set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        self.axes.xaxis.label.set_fontsize(12)
        self.axes.yaxis.label.set_fontsize(12)
        self.axes.title.set_fontsize(14)
    
    def plot_publication_quality(self, osa_data, ri_values, settings):
        self.axes.clear()
        self.setup_publication_style()
        
        xmin = settings['xmin']
        xmax = settings['xmax']
        ymin = settings['ymin']
        ymax = settings['ymax']
        auto_yaxis = settings['auto_yaxis']
        show_grid = settings['show_grid']
        show_legend = settings['show_legend']
        linewidth = settings['linewidth']
        colors = settings.get('colors', plt.cm.viridis(np.linspace(0, 1, len(ri_values))))
        
        all_power_values = []
        
        for i, ri_value in enumerate(ri_values):
            try:
                wavelength, power = osa_data.get_spectrum(ri_value)
                mask = (wavelength >= xmin) & (wavelength <= xmax)
                wavelength_filtered = wavelength[mask]
                power_filtered = power[mask]
                power_filtered = np.where(power_filtered == -210.0, np.nan, power_filtered)
                
                valid_power = power_filtered[~np.isnan(power_filtered)]
                if len(valid_power) > 0:
                    all_power_values.extend(valid_power)
                
                self.axes.plot(wavelength_filtered, power_filtered,
                             label=f'RI {ri_value:.2f}',
                             color=colors[i],
                             linewidth=linewidth,
                             alpha=0.85,
                             solid_capstyle='round')
            except Exception:
                continue
        
        self.axes.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold', color='white')
        self.axes.set_ylabel('Power (dBm)', fontsize=12, fontweight='bold', color='white')
        self.axes.set_title('OSA Spectrum - Publication Quality', fontsize=14, fontweight='bold', color='white', pad=15)
        
        self.axes.set_xlim(xmin, xmax)
        
        if auto_yaxis and len(all_power_values) > 0:
            y_data_min = np.min(all_power_values)
            y_data_max = np.max(all_power_values)
            y_range = y_data_max - y_data_min
            y_padding = y_range * 0.05
            self.axes.set_ylim(y_data_min - y_padding, y_data_max + y_padding)
        else:
            if ymin is not None and ymax is not None:
                self.axes.set_ylim(ymin, ymax)
        
        if show_grid:
            self.axes.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)
        
        if show_legend and len(ri_values) <= 15:
            ncol = 2 if len(ri_values) > 8 else 1
            self.axes.legend(loc='best', facecolor='#2D2D2D', edgecolor='white',
                           labelcolor='white', fontsize=9, ncol=ncol,
                           framealpha=0.95, shadow=True, fancybox=True)
        
        self.axes.minorticks_on()
        self.axes.tick_params(which='minor', length=3, color='gray')
        
        self.fig.tight_layout()
        self.draw()
    
    def clear_plot(self):
        self.axes.clear()
        self.setup_publication_style()
        self.axes.text(0.5, 0.5, 'No data plotted\n\nGenerate Plot',
                      ha='center', va='center', color='gray', fontsize=12,
                      transform=self.axes.transAxes)
        self.fig.tight_layout()
        self.draw()


class FittingCanvas(FigureCanvas):
    """Matplotlib canvas for fitting plots with scientific styling"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#FFFFFF")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setup_style()
        self.fig.tight_layout()
    
    def setup_style(self):
        self.ax.set_facecolor("#FFFFFF")
        self.ax.tick_params(colors='black', which='both', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.title.set_color('black')
    
    def clear_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#FFFFFF')
        ax.text(0.5, 0.5, 'No fit data\n\nPeak Analysis → Import to Fitting',
               ha='center', va='center', color='gray', fontsize=11,
               transform=ax.transAxes)
        ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()


class DualFittingCanvas(FigureCanvas):
    """Matplotlib canvas for dual fitting plots (2 subplots side-by-side)"""
    
    def __init__(self, parent=None, width=10, height=5, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#FFFFFF")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(600, 300)
        self.fig.tight_layout()
    
    def clear_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#FFFFFF")
        ax.text(0.5, 0.5, 'No dual fit data\n\nPeak Analysis → Import to Fitting',
               ha='center', va='center', color='gray', fontsize=11,
               transform=ax.transAxes)
        ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()

# ==================== MODERN THEME ====================

class SessionManager:
    """Save and restore the app's working state across runs.
    
    A session captures everything the user has configured but not the raw
    spectrum data itself — instead it remembers the source file paths and
    re-loads them. This keeps session files tiny (~5 KB JSON vs many MB)
    and avoids storing duplicate data.
    
    Format: JSON, .osasession extension. Version-tagged so we can evolve
    the schema without breaking old files (best-effort restore on mismatch).
    """
    
    FORMAT_VERSION = 1
    EXTENSION = ".osasession"
    
    @staticmethod
    def collect_widget_state(widget_map):
        """Read every widget's value into a JSON-serializable dict.
        
        widget_map is {key: widget}. Each widget is introspected based on
        its Qt class. Anything we don't know how to read is silently skipped
        so the session file stays robust to UI changes.
        """
        state = {}
        for key, w in widget_map.items():
            if w is None:
                continue
            try:
                cls = w.__class__.__name__
                if cls in ('QSpinBox', 'QDoubleSpinBox'):
                    state[key] = w.value()
                elif cls == 'QCheckBox':
                    state[key] = w.isChecked()
                elif cls == 'QComboBox':
                    state[key] = w.currentText()
                elif cls == 'QRadioButton':
                    state[key] = w.isChecked()
                elif cls == 'QLineEdit':
                    state[key] = w.text()
            except Exception:
                continue
        return state
    
    @staticmethod
    def apply_widget_state(widget_map, state):
        """Restore widget values from a state dict. Missing keys are skipped
        (so older session files won't break newer UI with extra widgets)."""
        for key, value in state.items():
            w = widget_map.get(key)
            if w is None:
                continue
            try:
                cls = w.__class__.__name__
                if cls in ('QSpinBox', 'QDoubleSpinBox'):
                    w.setValue(value)
                elif cls == 'QCheckBox':
                    w.setChecked(bool(value))
                elif cls == 'QComboBox':
                    idx = w.findText(str(value))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif cls == 'QRadioButton':
                    w.setChecked(bool(value))
                elif cls == 'QLineEdit':
                    w.setText(str(value))
            except Exception:
                continue
    
    @staticmethod
    def save(file_path, session_dict):
        """Write a session dict to disk. Adds version + timestamp metadata."""
        import json, datetime
        payload = {
            'format_version': SessionManager.FORMAT_VERSION,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            **session_dict
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    
    @staticmethod
    def load(file_path):
        """Read a session file. Returns the dict; raises ValueError on bad format."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if 'format_version' not in payload:
            raise ValueError("Not a valid session file (missing format_version).")
        # Future versions could migrate here. For now: warn but try to load.
        return payload


class YokogawaOSAConnection:
    """Thin TCP/SCPI client for Yokogawa AQ6370-series Optical Spectrum Analyzers.
    
    Uses a raw socket rather than pyvisa to keep dependencies minimal and the
    failure mode visible — every byte sent and received is explicit, which
    makes the first round of integration debugging much easier.
    
    The AQ6370 series uses a small custom handshake before SCPI starts working:
    on connect the OSA sends a welcome banner, then expects 'OPEN "<user>"',
    then either a password line or " " (a single space) if the account has
    no password, then responds with "ready" before any SCPI command works.
    All this is documented in the Yokogawa programming manual; we implement
    the minimum subset needed to authenticate and send SCPI queries.
    
    All methods are synchronous; the calling code is responsible for running
    long operations (sweep waits) on a background thread so the UI stays
    responsive.
    """
    
    DEFAULT_PORT = 10001
    DEFAULT_USER = "anonymous"
    DEFAULT_TIMEOUT = 5.0  # seconds for any single recv() call
    
    # Line terminator used in SCPI commands and expected in responses.
    # The AQ6370 uses LF after sending; we send CRLF to be safe.
    _SEND_EOL = b"\r\n"
    _RECV_EOL = b"\n"
    _BUF_SIZE = 8192
    
    def __init__(self):
        import socket
        self._socket_module = socket
        self.sock = None
        self.connected = False
        self.host = None
        self.port = None
        self.idn = None        # *IDN? response once we have it
        self.last_error = None
    
    def connect(self, host, port=None, username=None, password="",
                timeout=None):
        """Open a socket, perform the AQ6370 handshake, then send *IDN?.
        
        Returns the *IDN? response string on success.
        Raises ConnectionError (with a human-readable message) on any failure.
        """
        port = port if port is not None else self.DEFAULT_PORT
        username = username if username is not None else self.DEFAULT_USER
        timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        
        # Always start clean — if a previous connect failed mid-handshake,
        # we don't want leftover state confusing the next attempt.
        self.disconnect()
        
        try:
            self.sock = self._socket_module.socket(
                self._socket_module.AF_INET,
                self._socket_module.SOCK_STREAM
            )
            self.sock.settimeout(timeout)
            self.sock.connect((host, int(port)))
        except OSError as e:
            self.sock = None
            self.last_error = f"Could not reach {host}:{port} — {e}"
            raise ConnectionError(self.last_error)
        
        try:
            # Step 1: read the welcome banner (the OSA sends "AQ6370 ..." then "\r\n")
            banner = self._read_line()
            
            # Step 2: send the OPEN handshake with the username
            # AQ6370 expects:  OPEN "anonymous"\r\n
            # then it asks for a password
            self._send_raw(f'OPEN "{username}"')
            challenge = self._read_line()  # usually "AUTHENTICATE CRAM-MD5..." or similar
            
            # Step 3: send the password (single space if empty — the OSA's quirk)
            # We send the password line as plain text. On firmware that requires
            # MD5 auth this will fail with a specific error message; we surface
            # that to the user so they can switch to a configured account.
            pw_line = password if password else " "
            self._send_raw(pw_line)
            auth_response = self._read_line()
            
            # Step 4: confirm authentication. AQ6370 replies "ready" when OK
            # and "OPENED FAILED" (or similar) on a bad password.
            if "ready" not in auth_response.lower():
                # Some firmware sends an extra prompt line; read once more
                try:
                    extra = self._read_line()
                    if "ready" not in extra.lower():
                        raise ConnectionError(
                            f"Authentication failed. OSA replied: "
                            f"{repr(auth_response)} / {repr(extra)}"
                        )
                except Exception:
                    raise ConnectionError(
                        f"Authentication failed. OSA replied: {repr(auth_response)}"
                    )
            
            self.host = host
            self.port = int(port)
            self.connected = True
            
            # Step 5: ask the OSA who it is. If this fails, we're not really connected.
            try:
                self.idn = self.query("*IDN?")
            except Exception as e:
                self.idn = None
                # Connection itself is OK; *IDN? failing is unusual but not fatal
                # for raw connectivity testing.
            
            return self.idn or "(connected — no *IDN? response)"
        except ConnectionError:
            self.disconnect()
            raise
        except OSError as e:
            self.disconnect()
            self.last_error = f"Network error during handshake: {e}"
            raise ConnectionError(self.last_error)
        except Exception as e:
            self.disconnect()
            self.last_error = f"Unexpected handshake error: {e}"
            raise ConnectionError(self.last_error)
    
    def disconnect(self):
        """Close the socket if open. Safe to call multiple times."""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.connected = False
        self.idn = None
    
    def query(self, command, timeout=None):
        """Send a SCPI query and return the response string (newline-stripped).
        
        Raises ConnectionError if the socket isn't open.
        """
        if not self.connected or self.sock is None:
            raise ConnectionError("Not connected to OSA.")
        if timeout is not None:
            self.sock.settimeout(timeout)
        self._send_raw(command)
        return self._read_line()
    
    def write(self, command):
        """Send a SCPI command with no response expected."""
        if not self.connected or self.sock is None:
            raise ConnectionError("Not connected to OSA.")
        self._send_raw(command)
    
    # ---- low-level helpers ----
    
    def _send_raw(self, text):
        """Send a single command line. Always terminated with CRLF."""
        if isinstance(text, str):
            data = text.encode('ascii', errors='replace')
        else:
            data = bytes(text)
        if not data.endswith(self._SEND_EOL):
            data = data + self._SEND_EOL
        self.sock.sendall(data)
    
    def _read_line(self):
        """Read until LF or socket timeout. Returns the line as a stripped string.
        
        Accumulates bytes in a buffer because socket.recv() may return either
        a partial line or several lines at once. We keep reading until we have
        at least one complete line.
        """
        if not hasattr(self, '_recv_buffer'):
            self._recv_buffer = b""
        
        while self._RECV_EOL not in self._recv_buffer:
            chunk = self.sock.recv(self._BUF_SIZE)
            if not chunk:
                # Connection closed mid-read
                raise ConnectionError("OSA closed the connection unexpectedly.")
            self._recv_buffer += chunk
        
        line, _, remainder = self._recv_buffer.partition(self._RECV_EOL)
        self._recv_buffer = remainder
        return line.decode('ascii', errors='replace').strip()


class ModernTheme:
    DARK = {
        'bg_main': '#1E1E1E',
        'bg_panel': '#2D2D2D',
        'bg_widget': '#3C3C3C',
        'bg_hover': '#4A4A4A',
        'primary': '#1976D2',
        'primary_hover': '#1565C0',
        'primary_light': '#42A5F5',
        'success': '#388E3C',
        'warning': '#F57C00',
        'danger': '#D32F2F',
        'text_primary': '#FFFFFF',
        'text_secondary': '#B0B0B0',
        'border': '#555555',
    }
    
    @staticmethod
    def apply_dark_theme(app):
        app.setStyle('Fusion')
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ModernTheme.DARK['bg_main']))
        palette.setColor(QPalette.WindowText, QColor(ModernTheme.DARK['text_primary']))
        palette.setColor(QPalette.Base, QColor(ModernTheme.DARK['bg_widget']))
        palette.setColor(QPalette.AlternateBase, QColor(ModernTheme.DARK['bg_panel']))
        palette.setColor(QPalette.Text, QColor(ModernTheme.DARK['text_primary']))
        palette.setColor(QPalette.Button, QColor(ModernTheme.DARK['bg_panel']))
        palette.setColor(QPalette.ButtonText, QColor(ModernTheme.DARK['text_primary']))
        palette.setColor(QPalette.Highlight, QColor(ModernTheme.DARK['primary']))
        palette.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
        app.setPalette(palette)
    
    @staticmethod
    def get_stylesheet():
        return f"""
            QMainWindow {{
                background-color: {ModernTheme.DARK['bg_main']};
            }}
            
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {ModernTheme.DARK['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding: 8px;
                background-color: {ModernTheme.DARK['bg_panel']};
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {ModernTheme.DARK['primary_light']};
            }}
            
            QPushButton {{
                background-color: {ModernTheme.DARK['bg_widget']};
                color: {ModernTheme.DARK['text_primary']};
                border: 1px solid {ModernTheme.DARK['border']};
                border-radius: 6px;
                padding: 8px 12px;
                min-height: 28px;
                font-size: 10pt;
                text-align: center;
            }}
            
            QPushButton:hover {{
                background-color: {ModernTheme.DARK['bg_hover']};
                border: 1px solid {ModernTheme.DARK['primary']};
            }}
            
            QPushButton:disabled {{
                background-color: {ModernTheme.DARK['bg_panel']};
                color: #666666;
            }}
            
            QPushButton#primaryButton {{
                background-color: {ModernTheme.DARK['primary']};
                color: white;
                font-weight: bold;
                min-height: 40px;
                font-size: 11pt;
                border: none;
            }}
            
            QPushButton#primaryButton:hover {{
                background-color: {ModernTheme.DARK['primary_hover']};
            }}
            
            QPushButton#successButton {{
                background-color: {ModernTheme.DARK['success']};
                color: white;
                font-weight: bold;
                border: none;
            }}
            
            QPushButton#successButton:hover {{
                background-color: #2E7D32;
            }}
            
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {ModernTheme.DARK['bg_widget']};
                color: {ModernTheme.DARK['text_primary']};
                border: 1px solid {ModernTheme.DARK['border']};
                border-radius: 4px;
                padding: 6px;
                min-height: 24px;
            }}
            
            QCheckBox, QRadioButton {{
                color: {ModernTheme.DARK['text_primary']};
                spacing: 8px;
                padding: 4px;
            }}
            
            QLabel {{
                color: {ModernTheme.DARK['text_primary']};
            }}
            
            QTabWidget::pane {{
                border: 1px solid {ModernTheme.DARK['border']};
                background-color: {ModernTheme.DARK['bg_panel']};
            }}
            
            QTabBar::tab {{
                background-color: {ModernTheme.DARK['bg_widget']};
                color: {ModernTheme.DARK['text_secondary']};
                border: 1px solid {ModernTheme.DARK['border']};
                padding: 10px 16px;
                margin-right: 2px;
                min-width: 80px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {ModernTheme.DARK['primary']};
                color: white;
                font-weight: bold;
            }}
            
            QScrollBar:vertical {{
                background-color: {ModernTheme.DARK['bg_panel']};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {ModernTheme.DARK['bg_hover']};
                border-radius: 5px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {ModernTheme.DARK['primary']};
            }}
            
            QScrollBar:horizontal {{
                background-color: {ModernTheme.DARK['bg_panel']};
                height: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {ModernTheme.DARK['bg_hover']};
                border-radius: 5px;
                min-width: 20px;
            }}
            
            QTextEdit {{
                background-color: {ModernTheme.DARK['bg_widget']};
                color: {ModernTheme.DARK['text_primary']};
                border: 1px solid {ModernTheme.DARK['border']};
                border-radius: 4px;
                padding: 8px;
            }}
            
            QTableWidget {{
                background-color: {ModernTheme.DARK['bg_widget']};
                color: {ModernTheme.DARK['text_primary']};
                border: 1px solid {ModernTheme.DARK['border']};
                gridline-color: {ModernTheme.DARK['border']};
            }}
            
            QTableWidget::item {{
                padding: 5px;
            }}
            
            QTableWidget::item:selected {{
                background-color: {ModernTheme.DARK['primary']};
            }}
            
            QHeaderView::section {{
                background-color: {ModernTheme.DARK['bg_panel']};
                color: {ModernTheme.DARK['text_primary']};
                padding: 5px;
                border: 1px solid {ModernTheme.DARK['border']};
                font-weight: bold;
            }}
        """

# ==================== MAIN APPLICATION ====================

class OSAAnalyzerPro(QMainWindow):
    """OSA Analyzer Pro v6.1 - ENHANCED FITTING + ML EDITION (BUGFIXED)"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSA Analyzer v6.1")
        self.setMinimumSize(1000, 600)
        self.resize(1500, 900)
        
        self.settings = QSettings("OSA", "AnalyzerPro")
        
        self.current_files = []
        self.current_data: Optional[OSAData] = None
        self.sheet_names = []
        self.ri_checkboxes = {}
        
        self.peak_dip_detector = PeakDipDetector()
        self.fitting_engine = EnhancedFittingEngine()
        self.ml_trainer = MLTrainer()
        # Live OSA connection — created on first connect attempt.
        # Kept as a single instance so reconnect/disconnect cycles reuse it.
        self.osa_connection = YokogawaOSAConnection()
        self.ml_dataset = None
        
        self.last_ri_vals = []
        self.last_power_vals = []
        self.last_wavelength_vals = []
        
        self.init_ui()
        self.init_menu_bar()
        self.init_status_bar()
        
        self.setStyleSheet(ModernTheme.get_stylesheet())
        
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
    
    def closeEvent(self, event):
        self.settings.setValue('geometry', self.saveGeometry())
        # Close any active OSA connection so the instrument is free for
        # the next controller (Yokogawa Viewer, etc.) without needing a reboot.
        try:
            if hasattr(self, 'osa_connection') and self.osa_connection.connected:
                self.osa_connection.disconnect()
        except Exception:
            pass
        event.accept()
    
    # ========== Drag and Drop ==========
    def dragEnterEvent(self, event):
        """Accept drops if they contain file URLs of supported types."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid_exts = ('.xlsx', '.xls', '.txt', '.osasession')
            if any(url.toLocalFile().lower().endswith(valid_exts) for url in urls):
                event.acceptProposedAction()
                return
        event.ignore()
    
    def dropEvent(self, event):
        """Route dropped files based on extension.
        
        - One .osasession → load session
        - One .xlsx/.xls → load as Excel (with sheet picker if multi-sheet)
        - One or more .txt → load as TXT (single or multi-file flow)
        - Mixed types → take only the first kind we recognize, warn about the rest
        """
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if not paths:
            return
        
        session_files = [p for p in paths if p.lower().endswith('.osasession')]
        excel_files = [p for p in paths if p.lower().endswith(('.xlsx', '.xls'))]
        txt_files = [p for p in paths if p.lower().endswith('.txt')]
        
        if session_files:
            if len(session_files) > 1:
                QMessageBox.information(self, "Multiple sessions dropped",
                                       "Only one session can be loaded at a time. "
                                       f"Loading: {Path(session_files[0]).name}")
            self._load_session_file(session_files[0])
            return
        
        # Switch to the data tab so the user sees the result of the drop
        self.tabs.setCurrentIndex(0)
        
        if excel_files:
            if len(excel_files) > 1:
                QMessageBox.information(self, "Multiple Excel files",
                                       "Loading the first one. To batch-load several files, "
                                       "use TXT format instead.")
            self.current_files = [excel_files[0]]
            self.file_edit.setText(excel_files[0])
            self._populate_sheet_combo(excel_files[0])
            event.acceptProposedAction()
            return
        
        if txt_files:
            self.current_files = txt_files
            label = (f"{len(txt_files)} TXT files" if len(txt_files) > 1
                    else txt_files[0])
            self.file_edit.setText(label)
            self.sheet_combo.clear()
            event.acceptProposedAction()
            return
        
        QMessageBox.warning(self, "Unsupported file type",
                          "Drop .xlsx, .xls, .txt, or .osasession files.")
    
    def _populate_sheet_combo(self, excel_path):
        """Fill the sheet combo with dimensions per sheet — used by both
        the browse flow and drag-and-drop."""
        self.sheet_combo.clear()
        try:
            sheet_names = ExcelParser.get_sheet_names(excel_path)
            # Read each sheet's shape so the dropdown is informative
            for name in sheet_names:
                try:
                    df = pd.read_excel(excel_path, sheet_name=name, nrows=0)
                    n_cols = len(df.columns)
                    df_full = pd.read_excel(excel_path, sheet_name=name, usecols=[0])
                    n_rows = len(df_full)
                    label = f"{name}  ({n_rows} rows × {n_cols} cols)"
                    self.sheet_combo.addItem(label, name)  # store real name as userData
                except Exception:
                    self.sheet_combo.addItem(name, name)
        except Exception as e:
            QMessageBox.warning(self, "Sheet listing failed",
                              f"Could not read sheets from this file:\n{e}")
    
    # ========== Session Save / Load ==========
    def _build_widget_map(self):
        """Map every persistable widget to a stable key. The keys form the
        session file's schema — keep them consistent across versions, or
        loaders for old sessions won't find them."""
        return {
            # Data tab
            'txt_wl_start':       getattr(self, 'txt_wl_start_spin', None),
            'txt_wl_end':         getattr(self, 'txt_wl_end_spin', None),
            # Plot tabs
            'fast_xmin':          getattr(self, 'fast_xmin_spin', None),
            'fast_xmax':          getattr(self, 'fast_xmax_spin', None),
            'pub_xmin':           getattr(self, 'pub_xmin_spin', None),
            'pub_xmax':           getattr(self, 'pub_xmax_spin', None),
            # Peak detection
            'peak_mode_peak':     getattr(self, 'peak_mode_radio', None),
            'peak_mode_dip':      getattr(self, 'dip_mode_radio', None),
            'peak_mode_both':     getattr(self, 'both_mode_radio', None),
            'peak_height':        getattr(self, 'peak_height_spin', None),
            'peak_distance':      getattr(self, 'peak_distance_spin', None),
            'peak_prominence':    getattr(self, 'peak_prominence_spin', None),
            # Fitting
            'fit_mode':           getattr(self, 'fit_mode_combo', None),
            'fit_degree':         getattr(self, 'fit_degree_combo', None),
            'fit_low_min':        getattr(self, 'fit_low_min_spin', None),
            'fit_low_max':        getattr(self, 'fit_low_max_spin', None),
            'fit_high_min':       getattr(self, 'fit_high_min_spin', None),
            'fit_high_max':       getattr(self, 'fit_high_max_spin', None),
            # ML
            'ml_source':          getattr(self, 'ml_source_combo', None),
            'ml_model':           getattr(self, 'ml_model_combo', None),
            'ml_test_size':       getattr(self, 'ml_test_size_spin', None),
            'ml_scale':           getattr(self, 'ml_scale_cb', None),
            'ml_augment':         getattr(self, 'ml_augment_cb', None),
            'ml_aug_noise':       getattr(self, 'ml_aug_noise_cb', None),
            'ml_aug_noise_level': getattr(self, 'ml_aug_noise_level', None),
            'ml_aug_noise_n':     getattr(self, 'ml_aug_noise_n', None),
            'ml_aug_shift':       getattr(self, 'ml_aug_shift_cb', None),
            'ml_aug_shift_range': getattr(self, 'ml_aug_shift_range', None),
            'ml_aug_shift_n':     getattr(self, 'ml_aug_shift_n', None),
            'ml_aug_scale':       getattr(self, 'ml_aug_scale_cb', None),
            'ml_aug_scale_range': getattr(self, 'ml_aug_scale_range', None),
            'ml_aug_scale_n':     getattr(self, 'ml_aug_scale_n', None),
        }
    
    def save_session(self):
        """Save the current state to a .osasession JSON file."""
        from pathlib import Path
        default_name = "osa_session" + SessionManager.EXTENSION
        if self.current_files:
            stem = Path(self.current_files[0]).stem
            default_name = f"{stem}_session{SessionManager.EXTENSION}"
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", default_name,
            f"OSA Session (*{SessionManager.EXTENSION});;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(SessionManager.EXTENSION):
            path += SessionManager.EXTENSION
        
        # Capture state: file paths + active sheet + RI selection + all widget values
        try:
            selected_ri = self.get_selected_ri_values() if self.current_data else []
        except Exception:
            selected_ri = []
        
        sheet_name = None
        if hasattr(self, 'sheet_combo') and self.sheet_combo.count() > 0:
            data = self.sheet_combo.currentData()
            sheet_name = data if data else self.sheet_combo.currentText()
        
        session_data = {
            'files': self.current_files or [],
            'sheet_name': sheet_name,
            'selected_ri': selected_ri,
            'widgets': SessionManager.collect_widget_state(self._build_widget_map()),
            'has_data': self.current_data is not None and self.current_data.is_loaded(),
        }
        
        try:
            SessionManager.save(path, session_data)
            self.status_bar.showMessage(f"✅ Session saved: {Path(path).name}")
            QMessageBox.information(
                self, "Session saved",
                f"Saved to:\n{Path(path).name}\n\n"
                f"Files referenced: {len(session_data['files'])}\n"
                f"RI values selected: {len(selected_ri)}\n"
                f"Widget settings: {len(session_data['widgets'])}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save session:\n{e}")
    
    def load_session(self):
        """Open a .osasession file via dialog and restore state from it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            f"OSA Session (*{SessionManager.EXTENSION});;All Files (*)"
        )
        if not path:
            return
        self._load_session_file(path)
    
    def _load_session_file(self, path):
        """Restore state from a .osasession file. Used by both the menu and drag-drop."""
        from pathlib import Path
        try:
            payload = SessionManager.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed",
                               f"Could not read session file:\n{e}")
            return
        
        # Apply widget settings FIRST — that way they're already correct when we
        # trigger the data reload (e.g. txt_wl_start_spin influences how TXT files
        # are interpreted).
        widget_state = payload.get('widgets', {})
        SessionManager.apply_widget_state(self._build_widget_map(), widget_state)
        
        # Check which referenced files still exist
        files = payload.get('files', [])
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            msg = "These files referenced in the session are missing:\n\n"
            msg += "\n".join(f"  • {Path(f).name}" for f in missing[:5])
            if len(missing) > 5:
                msg += f"\n  … and {len(missing) - 5} more"
            msg += "\n\nLoad the session anyway (without data)?"
            reply = QMessageBox.question(self, "Files Missing", msg,
                                        QMessageBox.Yes | QMessageBox.Cancel)
            if reply != QMessageBox.Yes:
                return
            files = [f for f in files if Path(f).exists()]
        
        # Stash the selection so on_data_loaded can apply it after data arrives
        self._pending_ri_selection = payload.get('selected_ri', [])
        
        if files and payload.get('has_data'):
            self.current_files = files
            # Update Data tab UI to reflect what we're loading
            if len(files) == 1:
                self.file_edit.setText(files[0])
                if files[0].lower().endswith(('.xlsx', '.xls')):
                    self._populate_sheet_combo(files[0])
                    # Restore the chosen sheet
                    sheet_name = payload.get('sheet_name')
                    if sheet_name:
                        idx = self.sheet_combo.findData(sheet_name)
                        if idx < 0:
                            idx = self.sheet_combo.findText(sheet_name)
                        if idx >= 0:
                            self.sheet_combo.setCurrentIndex(idx)
            else:
                self.file_edit.setText(f"{len(files)} TXT files")
            
            self.load_data()  # asynchronous; selection applied in on_data_loaded
            self.status_bar.showMessage(
                f"✅ Session loaded, re-loading {len(files)} file(s)..."
            )
        else:
            self.status_bar.showMessage(
                "✅ Session settings restored (no data files referenced)"
            )
        
        self.tabs.setCurrentIndex(0)
    
    def _apply_pending_ri_selection(self):
        """After data is loaded, restore the RI checkbox selection from the
        pending session (if any). Called from on_data_loaded."""
        if not hasattr(self, '_pending_ri_selection') or not self._pending_ri_selection:
            return
        try:
            wanted = set(float(x) for x in self._pending_ri_selection)
        except Exception:
            wanted = set()
        for ri, cb in getattr(self, 'ri_checkboxes', {}).items():
            try:
                cb.setChecked(float(ri) in wanted)
            except Exception:
                continue
        self._pending_ri_selection = []
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.addTab(self.create_live_osa_tab(),          "0 · Live OSA")
        self.tabs.addTab(self.create_data_load_tab(),         "1 · Data")
        self.tabs.addTab(self.create_interactive_tab(),       "2 · Interactive Plot")
        self.tabs.addTab(self.create_publication_tab(),       "3 · Publication Plot")
        self.tabs.addTab(self.create_peak_detection_panel(),  "4 · Peak / Dip Detection")
        self.tabs.addTab(self.create_enhanced_fitting_panel(),"5 · Fitting")
        self.tabs.addTab(self.create_ml_tab(),                "6 · Machine Learning")
        # Default to Data tab on launch (not Live OSA) so users who don't have
        # an instrument connected aren't greeted by a connect dialog.
        self.tabs.setCurrentIndex(1)
        
        main_layout.addWidget(self.tabs)
    
    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open data...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_single_file)
        
        file_menu.addSeparator()
        save_session_action = file_menu.addAction("Save session...")
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.setToolTip(
            "Save the current state (loaded files, RI selection, all tab settings) "
            "so you can resume later."
        )
        save_session_action.triggered.connect(self.save_session)
        
        load_session_action = file_menu.addAction("Load session...")
        load_session_action.setShortcut("Ctrl+Shift+O")
        load_session_action.setToolTip(
            "Re-open a previously saved session. "
            "Re-loads data files and restores all settings."
        )
        load_session_action.triggered.connect(self.load_session)
        
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
        # Enable accepting drag-and-drop on the window itself
        self.setAcceptDrops(True)
    
    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def show_about(self):
        QMessageBox.about(self, "About OSA Analyzer Pro",
                         "<h3>OSA Analyzer Pro v6.1</h3>"
                         "<p><b>Enhanced Fitting + Machine Learning Edition</b></p>"
                         "<p>Bugfix release: Feb 13, 2026</p>"
                         "<p>Features:</p><ul>"
                         "<li>Interactive & Publication Plotting</li>"
                         "<li>Peak/Dip Detection</li>"
                         "<li>Enhanced Curve Fitting (linear only)</li>"
                         "<li>Machine Learning with Data Augmentation</li>"
                         "</ul><p>Authors: Momow & Kirito</p>")
    
    # ========== Data Load Tab ==========
    # ========== Live OSA Tab (Pass 1: connection + *IDN? only) ==========
    def create_live_osa_tab(self):
        """Build the Live OSA tab. Pass 1 scope: connect and confirm identity.
        
        Sweep configuration and actual trace acquisition come in later passes.
        For now, the whole point is to prove the network handshake works
        against a real instrument before we build anything on top of it.
        """
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainlayout = QHBoxLayout(tab)
        mainlayout.setContentsMargins(8, 8, 8, 8)
        
        # ---- Left: connection panel ----
        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        left.setMaximumWidth(360)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        
        # CONNECTION group
        conn_group = QGroupBox("CONNECTION")
        conn_layout = QFormLayout(conn_group)
        conn_layout.setSpacing(6)
        
        self.osa_ip_edit = QLineEdit("192.168.1.10")
        self.osa_ip_edit.setPlaceholderText("e.g. 192.168.1.10")
        self.osa_ip_edit.setToolTip(
            "IP address shown on the OSA: SYSTEM → NETWORK → IP ADDRESS.\n"
            "For direct cable: pick an address in the same subnet as your laptop\n"
            "(e.g. OSA = 192.168.1.10, laptop = 192.168.1.11)."
        )
        conn_layout.addRow("IP address:", self.osa_ip_edit)
        
        self.osa_port_spin = QSpinBox()
        self.osa_port_spin.setRange(1, 65535)
        self.osa_port_spin.setValue(YokogawaOSAConnection.DEFAULT_PORT)
        self.osa_port_spin.setToolTip(
            "TCP port for SCPI remote. Default 10001 on AQ6370-series.\n"
            "Shown as REMOTE PORT NO. in SYSTEM → NETWORK."
        )
        conn_layout.addRow("Port:", self.osa_port_spin)
        
        self.osa_user_edit = QLineEdit(YokogawaOSAConnection.DEFAULT_USER)
        self.osa_user_edit.setToolTip(
            "Username from the OSA's user-account setup.\n"
            "Default account is 'anonymous'; yours may differ."
        )
        conn_layout.addRow("Username:", self.osa_user_edit)
        
        self.osa_pass_edit = QLineEdit("")
        self.osa_pass_edit.setEchoMode(QLineEdit.Password)
        self.osa_pass_edit.setToolTip("Leave blank if the account has no password.")
        conn_layout.addRow("Password:", self.osa_pass_edit)
        
        self.osa_timeout_spin = QDoubleSpinBox()
        self.osa_timeout_spin.setRange(1.0, 60.0)
        self.osa_timeout_spin.setValue(YokogawaOSAConnection.DEFAULT_TIMEOUT)
        self.osa_timeout_spin.setSuffix(" s")
        self.osa_timeout_spin.setDecimals(1)
        self.osa_timeout_spin.setToolTip(
            "How long to wait for each network response before giving up.\n"
            "5 s is fine for a healthy direct connection. Increase if you see\n"
            "timeouts on a slower network."
        )
        conn_layout.addRow("Timeout:", self.osa_timeout_spin)
        left_layout.addWidget(conn_group)
        
        # Connect / Disconnect buttons
        btn_row = QHBoxLayout()
        self.osa_connect_btn = QPushButton("Connect")
        self.osa_connect_btn.setObjectName("primaryButton")
        self.osa_connect_btn.clicked.connect(self.osa_connect)
        self.osa_disconnect_btn = QPushButton("Disconnect")
        self.osa_disconnect_btn.clicked.connect(self.osa_disconnect)
        self.osa_disconnect_btn.setEnabled(False)
        btn_row.addWidget(self.osa_connect_btn)
        btn_row.addWidget(self.osa_disconnect_btn)
        left_layout.addLayout(btn_row)
        
        # Status indicator — coloured dot + label
        status_row = QHBoxLayout()
        self.osa_status_dot = QLabel("●")
        self.osa_status_dot.setStyleSheet("color: #888; font-size: 14pt;")
        self.osa_status_label = QLabel("Disconnected")
        self.osa_status_label.setStyleSheet("font-size: 10pt;")
        status_row.addWidget(self.osa_status_dot)
        status_row.addWidget(self.osa_status_label, stretch=1)
        left_layout.addLayout(status_row)
        
        # Instrument info readout (filled after a successful connection)
        info_group = QGroupBox("INSTRUMENT")
        info_layout = QFormLayout(info_group)
        self.osa_info_idn = QLabel("—")
        self.osa_info_idn.setWordWrap(True)
        self.osa_info_idn.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addRow("*IDN?:", self.osa_info_idn)
        self.osa_info_host = QLabel("—")
        info_layout.addRow("Host:", self.osa_info_host)
        left_layout.addWidget(info_group)
        
        # Manual SCPI test field — extremely useful when iterating on Pass 2+
        scpi_group = QGroupBox("SEND COMMAND (advanced)")
        scpi_layout = QVBoxLayout(scpi_group)
        scpi_layout.addWidget(QLabel(
            "Send a raw SCPI command and see the response. Examples:\n"
            "  *IDN?    — identity\n"
            "  :SENS:WAV:STAR?  — sweep start wavelength"
        ))
        scpi_row = QHBoxLayout()
        self.osa_scpi_edit = QLineEdit()
        self.osa_scpi_edit.setPlaceholderText("*IDN?")
        self.osa_scpi_edit.returnPressed.connect(self.osa_send_scpi)
        scpi_row.addWidget(self.osa_scpi_edit, stretch=1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.osa_send_scpi)
        scpi_row.addWidget(send_btn)
        scpi_layout.addLayout(scpi_row)
        left_layout.addWidget(scpi_group)
        
        left_layout.addStretch()
        
        # ---- Right: log + future trace plot area ----
        right = QFrame()
        right.setFrameShape(QFrame.StyledPanel)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        title = QLabel("Live OSA — direct instrument control (Pass 1: connection only)")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['success']}; padding: 4px 0;")
        right_layout.addWidget(title)
        
        # Log area: every send/recv ends up here. Critical for iterating on the
        # SCPI protocol since the actual bytes we exchange are visible.
        self.osa_log = QTextEdit()
        self.osa_log.setReadOnly(True)
        self.osa_log.setFont(QFont("Consolas", 9))
        self.osa_log.setStyleSheet(
            f"background-color: {ModernTheme.DARK['bg_widget']}; "
            f"color: {ModernTheme.DARK['text_primary']};"
        )
        self.osa_log.setPlaceholderText(
            "Communication log will appear here. Each line shows what was sent "
            "to the OSA and what it sent back.\n\n"
            "Try this first:\n"
            "  1. Set the IP / port to match your OSA's network settings\n"
            "  2. Click Connect\n"
            "  3. If it works, *IDN? response should appear above and in this log\n\n"
            "If anything goes wrong, the exact error message will appear here too — "
            "share it and we'll figure out what to adjust."
        )
        right_layout.addWidget(self.osa_log)
        
        # Quick action buttons that the user might want once connected
        action_row = QHBoxLayout()
        clear_log_btn = QPushButton("Clear log")
        clear_log_btn.clicked.connect(lambda: self.osa_log.clear())
        action_row.addWidget(clear_log_btn)
        action_row.addStretch()
        right_layout.addLayout(action_row)
        
        # Note about future passes
        note = QLabel(
            "<i>This tab will grow in later passes to include: sweep "
            "configuration, single/repeat acquisition, live trace plotting, "
            "and one-click 'capture as RI x.xx' for the dataset.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: #888; font-size: 9pt; padding: 4px;")
        right_layout.addWidget(note)
        
        mainlayout.addWidget(left)
        mainlayout.addWidget(right, stretch=1)
        return tab
    
    def _osa_log(self, text, level='info'):
        """Append a line to the Live OSA log with timestamp + level prefix."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        colors = {
            'info':  ModernTheme.DARK['text_primary'],
            'send':  '#42A5F5',
            'recv':  '#9CCC65',
            'error': '#EF5350',
            'ok':    ModernTheme.DARK['success'],
        }
        color = colors.get(level, ModernTheme.DARK['text_primary'])
        prefix = {
            'info': '·', 'send': '>', 'recv': '<', 'error': '✗', 'ok': '✓'
        }.get(level, '·')
        # Append with HTML formatting so different message types are visually distinct.
        # Plain-text alternative would work too but the colour cue is helpful when
        # scanning a long log.
        self.osa_log.append(
            f'<span style="color:#666">{ts}</span> '
            f'<span style="color:{color}">{prefix} {text}</span>'
        )
    
    def _set_osa_status(self, connected, message=None):
        """Update the coloured dot + label that indicates connection state."""
        if connected:
            self.osa_status_dot.setStyleSheet("color: #66BB6A; font-size: 14pt;")
            self.osa_status_label.setText(message or "Connected")
        else:
            self.osa_status_dot.setStyleSheet("color: #888; font-size: 14pt;")
            self.osa_status_label.setText(message or "Disconnected")
        self.osa_connect_btn.setEnabled(not connected)
        self.osa_disconnect_btn.setEnabled(connected)
    
    def osa_connect(self):
        """Attempt to connect to the OSA using the values in the form."""
        host = self.osa_ip_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "No IP", "Enter the OSA's IP address first.")
            return
        port = self.osa_port_spin.value()
        username = self.osa_user_edit.text().strip() or YokogawaOSAConnection.DEFAULT_USER
        password = self.osa_pass_edit.text()
        timeout = self.osa_timeout_spin.value()
        
        self._osa_log(f"Connecting to {host}:{port} as '{username}'...", 'info')
        self._set_osa_status(False, "Connecting...")
        QApplication.processEvents()
        
        try:
            idn = self.osa_connection.connect(host, port, username, password, timeout)
        except ConnectionError as e:
            self._osa_log(str(e), 'error')
            self._set_osa_status(False, "Connection failed")
            QMessageBox.critical(
                self, "Connection failed",
                f"Could not connect to {host}:{port}.\n\n{e}\n\n"
                "Check the OSA log for details. Common causes:\n"
                "  • Wrong IP / port\n"
                "  • Cable not plugged in or wrong subnet\n"
                "  • Authentication failure (username/password mismatch)\n"
                "  • The OSA is already connected to another controller"
            )
            return
        except Exception as e:
            self._osa_log(f"Unexpected: {e}", 'error')
            self._set_osa_status(False, "Connection failed")
            QMessageBox.critical(self, "Unexpected error", str(e))
            return
        
        self._osa_log(f"Connected. *IDN? = {idn}", 'ok')
        self.osa_info_idn.setText(idn)
        self.osa_info_host.setText(f"{host}:{port}")
        self._set_osa_status(True, f"Connected to {host}")
        self.status_bar.showMessage(f"✅ OSA connected: {idn}")
    
    def osa_disconnect(self):
        """Close the OSA connection if open."""
        if self.osa_connection.connected:
            self._osa_log("Disconnecting...", 'info')
            self.osa_connection.disconnect()
            self._osa_log("Disconnected.", 'ok')
        self.osa_info_idn.setText("—")
        self.osa_info_host.setText("—")
        self._set_osa_status(False, "Disconnected")
        self.status_bar.showMessage("OSA disconnected")
    
    def osa_send_scpi(self):
        """Send the SCPI command from the input field and log the response."""
        if not self.osa_connection.connected:
            QMessageBox.warning(self, "Not connected", "Connect to the OSA first.")
            return
        cmd = self.osa_scpi_edit.text().strip()
        if not cmd:
            return
        
        self._osa_log(cmd, 'send')
        try:
            # A SCPI query ends with '?'; commands don't expect a response.
            if cmd.endswith('?'):
                response = self.osa_connection.query(cmd)
                self._osa_log(response, 'recv')
            else:
                self.osa_connection.write(cmd)
                self._osa_log("(command sent, no response expected)", 'info')
        except ConnectionError as e:
            self._osa_log(f"Connection lost: {e}", 'error')
            self._set_osa_status(False, "Connection lost")
        except Exception as e:
            self._osa_log(f"Error: {e}", 'error')
        
        self.osa_scpi_edit.clear()
    
    def create_data_load_tab(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        left_panel = self.create_data_load_panel()
        right_panel = self.create_data_info_panel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)
        return tab
    
    def create_data_load_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(500)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(12)
        
        # DATA SOURCE
        data_group = QGroupBox("DATA SOURCE")
        data_layout = QVBoxLayout(data_group)
        data_layout.addWidget(QLabel("Files:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select files...")
        self.file_edit.setReadOnly(True)
        data_layout.addWidget(self.file_edit)
        
        btn_layout = QHBoxLayout()
        browse_single_btn = QPushButton("Browse Excel…")
        browse_single_btn.setToolTip("Load a single .xlsx or .xls file. Choose the sheet below.")
        browse_single_btn.clicked.connect(self.browse_single_file)
        browse_multi_btn = QPushButton("Browse TXT files…")
        browse_multi_btn.setToolTip(
            "Load one or more raw OSA .txt files (one RI per file).\n"
            "RI value is read from the filename (e.g. sample_1.33.txt → RI 1.33)."
        )
        browse_multi_btn.clicked.connect(self.browse_multiple_txt)
        browse_multi_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {ModernTheme.DARK['success']}; color: white; font-weight: bold; }}
            QPushButton:hover {{ background-color: #2E7D32; }}
        """)
        btn_layout.addWidget(browse_single_btn)
        btn_layout.addWidget(browse_multi_btn)
        data_layout.addLayout(btn_layout)
        
        data_layout.addSpacing(8)
        data_layout.addWidget(QLabel("Sheet (Excel):"))
        self.sheet_combo = QComboBox()
        self.sheet_combo.setEnabled(False)
        data_layout.addWidget(self.sheet_combo)
        
        data_layout.addSpacing(8)
        self.load_status = QLabel("No data loaded")
        self.load_status.setWordWrap(True)
        data_layout.addWidget(self.load_status)
        layout.addWidget(data_group)
        
        # TXT WAVELENGTH RANGE (only used for TXT files; Excel files carry their own wavelength)
        txt_wl_group = QGroupBox("TXT WAVELENGTH RANGE")
        txt_wl_group.setToolTip("Used when loading TXT files only. "
                                "Excel files carry their own wavelength column and ignore this.")
        txt_wl_layout = QVBoxLayout(txt_wl_group)
        
        txt_wl_info = QLabel(
            "OSA TXT files don't store wavelength — set the sweep range your instrument used. "
            "Number of points is auto-detected from the file."
        )
        txt_wl_info.setWordWrap(True)
        txt_wl_info.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        txt_wl_layout.addWidget(txt_wl_info)
        
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Start:"))
        self.txt_wl_start_spin = QDoubleSpinBox()
        self.txt_wl_start_spin.setRange(800.0, 2500.0)
        self.txt_wl_start_spin.setValue(1520.0)
        self.txt_wl_start_spin.setSuffix(" nm")
        self.txt_wl_start_spin.setDecimals(2)
        self.txt_wl_start_spin.setSingleStep(1.0)
        wl_row.addWidget(self.txt_wl_start_spin)
        
        wl_row.addWidget(QLabel("End:"))
        self.txt_wl_end_spin = QDoubleSpinBox()
        self.txt_wl_end_spin.setRange(800.0, 2500.0)
        self.txt_wl_end_spin.setValue(1620.0)
        self.txt_wl_end_spin.setSuffix(" nm")
        self.txt_wl_end_spin.setDecimals(2)
        self.txt_wl_end_spin.setSingleStep(1.0)
        wl_row.addWidget(self.txt_wl_end_spin)
        txt_wl_layout.addLayout(wl_row)
        
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        preset_c_btn = QPushButton("C-band\n1520–1580")
        preset_c_btn.setToolTip("ITU C-band: 1520–1580 nm")
        preset_c_btn.clicked.connect(lambda: self._set_txt_wl_range(1520.0, 1580.0))
        preset_cl_btn = QPushButton("AQ6370D\n1520–1620")
        preset_cl_btn.setToolTip("Yokogawa AQ6370D typical sweep: 1520–1620 nm")
        preset_cl_btn.clicked.connect(lambda: self._set_txt_wl_range(1520.0, 1620.0))
        preset_full_btn = QPushButton("Full\n1250–1650")
        preset_full_btn.setToolTip("Full O+E+S+C+L band: 1250–1650 nm")
        preset_full_btn.clicked.connect(lambda: self._set_txt_wl_range(1250.0, 1650.0))
        preset_row.addWidget(preset_c_btn)
        preset_row.addWidget(preset_cl_btn)
        preset_row.addWidget(preset_full_btn)
        txt_wl_layout.addLayout(preset_row)
        
        layout.addWidget(txt_wl_group)
        
        # DATA INFO
        info_group = QGroupBox("DATA INFO")
        info_layout = QVBoxLayout(info_group)
        self.info_samples = QLabel("Samples: —")
        self.info_wavelengths = QLabel("Wavelength points: —")
        self.info_range = QLabel("Wavelength range: —")
        info_layout.addWidget(self.info_samples)
        info_layout.addWidget(self.info_wavelengths)
        info_layout.addWidget(self.info_range)
        layout.addWidget(info_group)
        
        # RI SELECTION
        self.ri_group = QGroupBox("SELECT RI VALUES")
        ri_layout = QVBoxLayout(self.ri_group)
        self.ri_scroll = QScrollArea()
        self.ri_scroll.setWidgetResizable(True)
        self.ri_scroll.setMinimumHeight(150)
        self.ri_scroll_widget = QWidget()
        self.ri_scroll_layout = QVBoxLayout(self.ri_scroll_widget)
        self.ri_scroll_layout.addWidget(QLabel("Load data first..."))
        self.ri_scroll.setWidget(self.ri_scroll_widget)
        ri_layout.addWidget(self.ri_scroll)
        
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self.select_all_ri)
        select_none_btn = QPushButton("None")
        select_none_btn.clicked.connect(self.select_none_ri)
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(select_none_btn)
        ri_layout.addLayout(select_layout)
        layout.addWidget(self.ri_group)
        
        layout.addStretch()
        
        self.load_btn = QPushButton("LOAD DATA")
        self.load_btn.setObjectName("primaryButton")
        self.load_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_btn)
        
        scroll.setWidget(scroll_widget)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        return panel
    
    def create_data_info_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("DATA INFORMATION")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['primary_light']};")
        layout.addWidget(title)
        
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_info_text.setPlaceholderText(
            "No data loaded.\n\n"
            "• Excel/XLSX files: use 'Browse Excel…' and pick the sheet.\n"
            "• TXT files: use 'Browse TXT files…' (one file per RI value).\n"
            "  RI is parsed from the filename, e.g. 'sample_1.33.txt' → RI 1.33."
        )
        layout.addWidget(self.data_info_text)
        
        info_panel = QFrame()
        info_panel.setStyleSheet(f"""
            QFrame {{ background-color: {ModernTheme.DARK['bg_widget']}; border: 2px solid {ModernTheme.DARK['primary']}; border-radius: 8px; padding: 12px; }}
        """)
        info_panel.setMaximumHeight(290)
        info_layout = QVBoxLayout(info_panel)
        info_label = QLabel(
            "<b style='color:#42A5F5;'>QUICK START</b><br><br>"
            "<b>1.</b> Drag files onto the window, or use Browse Excel / Browse TXT.<br>"
            "<b>2.</b> Set the wavelength range to match your OSA sweep (TXT only).<br>"
            "<b>3.</b> Pick which RI values to include, then click <b>LOAD DATA</b>.<br>"
            "<b>4.</b> Go to tab <b>2/3</b> to view spectra.<br>"
            "<b>5.</b> Tab <b>4</b> detects peaks/dips and feeds them to fitting.<br>"
            "<b>6.</b> Tab <b>5</b> fits sensitivity (linear or polynomial, with uncertainty).<br>"
            "<b>7.</b> Tab <b>6</b> trains an ML model on extracted features.<br><br>"
            "<i style='color:#aaa;'>Tip: AQ6370D sweeps default to 1520–1620 nm. "
            "Use <b>File → Save Session</b> (Ctrl+S) to resume later.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {ModernTheme.DARK['text_primary']}; font-size: 10pt;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_panel)
        return panel
    
    # ========== Data Loading Helpers ==========
    def browse_single_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Data Files (*.xlsx *.xls *.txt);;All Files (*)"
        )
        if file_path:
            self.file_edit.setText(Path(file_path).name)
            self.current_files = [file_path]
            try:
                if file_path.endswith(('.xlsx', '.xls')):
                    self._populate_sheet_combo(file_path)
                    self.sheet_combo.setEnabled(True)
                else:
                    self.sheet_combo.clear()
                    self.sheet_combo.setEnabled(False)
                self.status_bar.showMessage(f"File: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not read: {e}")
    
    def browse_multiple_txt(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple TXT Files", "", "TXT Files (*.txt);;All Files (*)"
        )
        if file_paths:
            self.current_files = file_paths
            num_files = len(file_paths)
            filenames = [Path(f).name for f in file_paths]
            display_text = f"{num_files} files: {', '.join(filenames[:3])}"
            if num_files > 3:
                display_text = f"{display_text}, ... (+{num_files-3} more)"
            self.file_edit.setText(display_text)
            self.sheet_combo.setEnabled(False)
            self.status_bar.showMessage(f"{num_files} TXT files selected")
            QMessageBox.information(self, "Multi-TXT Selected", f"{num_files} TXT files selected!\n\nClick 'LOAD DATA' to merge.")
    
    def load_data(self):
        if not self.current_files:
            QMessageBox.warning(self, "No File", "Select files first!")
            return
        
        sheet_name = None
        multi_mode = False
        if len(self.current_files) == 1 and self.current_files[0].endswith(('.xlsx', '.xls')):
            # The combo's display text now includes dimensions ("Sheet1  (501 rows × 6 cols)");
            # the real sheet name lives in the item's userData. Fall back to text if missing.
            data = self.sheet_combo.currentData()
            sheet_name = data if data else self.sheet_combo.currentText()
        elif len(self.current_files) > 1:
            multi_mode = True
        
        # Wavelength range only matters for TXT files, but always validate it
        wl_start = self.txt_wl_start_spin.value()
        wl_end = self.txt_wl_end_spin.value()
        if wl_start >= wl_end:
            QMessageBox.warning(self, "Invalid Wavelength Range",
                              f"TXT wavelength Start ({wl_start} nm) must be less than End ({wl_end} nm).")
            return
        wavelength_range = (wl_start, wl_end)
        
        self.load_status.setText("Loading...")
        self.load_btn.setEnabled(False)
        QApplication.processEvents()
        
        self.load_thread = DataLoadThread(
            self.current_files, sheet_name, multi_mode,
            wavelength_range=wavelength_range
        )
        self.load_thread.finished.connect(self.on_data_loaded)
        self.load_thread.error.connect(self.on_load_error)
        self.load_thread.start()
    
    def on_data_loaded(self, osa_data, validation):
        self.current_data = osa_data
        if validation.valid:
            self.load_status.setText("Loaded!")
        else:
            self.load_status.setText(f"Issues: {', '.join(validation.issues)}")
        
        info = osa_data.get_info_dict()
        self.info_samples.setText(f"Samples: {info['num_ri_values']}")
        self.info_wavelengths.setText(f"Points: {info['num_wavelengths']}")
        wl_min, wl_max = info['wavelength_range']
        self.info_range.setText(f"Range: {wl_min:.2f}-{wl_max:.2f} nm")
        
        self.fast_xmin_spin.setValue(wl_min)
        self.fast_xmax_spin.setValue(wl_max)
        self.pub_xmin_spin.setValue(wl_min)
        self.pub_xmax_spin.setValue(wl_max)
        
        self.populate_ri_checkboxes(osa_data.get_ri_values())
        
        # If a session restore is in flight, apply its RI selection now that
        # the checkboxes have been populated.
        self._apply_pending_ri_selection()
        
        info_text = f"✅ Data Loaded!\n\n"
        if osa_data.file_type == 'txt_multi':
            info_text += f"Mode: MULTI-TXT ({osa_data.metadata.get('num_files', 0)} files)\n\n"
        else:
            info_text += f"File: {Path(info['source_file']).name}\n"
            info_text += f"Type: {info['file_type'].upper()}\n\n"
        info_text += f"Samples: {info['num_ri_values']}\n"
        info_text += f"Points: {info['num_wavelengths']}\n"
        info_text += f"Range: {wl_min:.2f} - {wl_max:.2f} nm\n"
        info_text += f"RI: {info['ri_range'][0]:.2f} - {info['ri_range'][1]:.2f}\n\nReady for plotting!"
        self.data_info_text.setText(info_text)
        
        self.load_btn.setEnabled(True)
        self.status_bar.showMessage(f"Loaded {info['num_ri_values']} samples")
    
    def on_load_error(self, error_msg):
        self.load_status.setText("Failed!")
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Load failed:\n{error_msg}")
    
    def populate_ri_checkboxes(self, ri_values):
        for i in reversed(range(self.ri_scroll_layout.count())):
            widget = self.ri_scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.ri_checkboxes = {}
        for ri_value in ri_values:
            cb = QCheckBox(f"RI {ri_value:.2f}")
            cb.setChecked(True)
            self.ri_checkboxes[ri_value] = cb
            self.ri_scroll_layout.addWidget(cb)
        self.ri_scroll_layout.addStretch()
    
    def select_all_ri(self):
        for cb in self.ri_checkboxes.values():
            cb.setChecked(True)
    
    def select_none_ri(self):
        for cb in self.ri_checkboxes.values():
            cb.setChecked(False)
    
    def _set_txt_wl_range(self, start_nm, end_nm):
        """Quick preset for TXT wavelength range buttons."""
        self.txt_wl_start_spin.setValue(start_nm)
        self.txt_wl_end_spin.setValue(end_nm)
    
    def _clear_layout(self, layout):
        """Recursively clear all widgets AND sub-layouts from a layout.
        
        Necessary because the original cleanup in update_manual_param_widgets
        only deleted direct widget children; sub-layouts (which held the
        actual label+spinbox pairs) survived and caused labels to overlap
        when switching ML models.
        """
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    self._clear_layout(sub)
    
    def get_selected_ri(self):
        return [ri for ri, cb in self.ri_checkboxes.items() if cb.isChecked()]
    
    def get_selected_ri_values(self):
        return self.get_selected_ri()
    
    # ========== Plotting Helpers ==========
    def get_color_palette(self, num_colors, plot_type):
        """Modern colormap retrieval (no deprecation)"""
        if plot_type == 'pyqtgraph':
            palette_name = self.fast_palette_combo.currentText().lower()
        else:
            palette_name = self.pub_palette_combo.currentText().lower()
        
        cmap_mapping = {
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'cool': 'cool',
            'warm': 'hot',
            'rainbow': 'rainbow',
            'turbo': 'turbo'
        }
        cmap_name = cmap_mapping.get(palette_name, 'viridis')
        # Use modern method
        cmap = plt.colormaps[cmap_name]
        return cmap(np.linspace(0, 1, num_colors))
    
    # ========== Interactive Plot Tab ==========
    def create_interactive_tab(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        title = QLabel("Interactive Spectrum — pan, zoom, hover for values")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['success']}; padding: 4px 0;")
        right_layout.addWidget(title)
        self.fast_plot = FastPlotWidget(self)
        right_layout.addWidget(self.fast_plot)
        
        left_panel = self.create_plot_settings_panel('pyqtgraph')
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1100])
        main_layout.addWidget(splitter)
        return tab
    
    def create_publication_tab(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        title = QLabel("Publication Figure — export as PNG / PDF / SVG")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['primary_light']}; padding: 4px 0;")
        right_layout.addWidget(title)
        
        self.pub_canvas = PublicationCanvas(self, width=10, height=7, dpi=100)
        right_layout.addWidget(self.pub_canvas)
        self.pub_toolbar = NavigationToolbar(self.pub_canvas, self)
        self.pub_toolbar.setStyleSheet(f"background-color: {ModernTheme.DARK['bg_panel']};")
        right_layout.addWidget(self.pub_toolbar)
        
        export_panel = QFrame()
        export_panel.setStyleSheet(f"""
            QFrame {{ background-color: {ModernTheme.DARK['bg_widget']}; border: 1px solid {ModernTheme.DARK['border']}; border-radius: 6px; padding: 12px; }}
        """)
        export_panel.setMaximumHeight(90)
        export_layout = QVBoxLayout(export_panel)
        export_label = QLabel("Export")
        export_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        export_layout.addWidget(export_label)
        
        export_btn_layout = QHBoxLayout()
        export_png_btn = QPushButton("PNG")
        export_png_btn.setMinimumHeight(40)
        export_png_btn.clicked.connect(lambda: self.export_publication('png'))
        export_pdf_btn = QPushButton("PDF")
        export_pdf_btn.setMinimumHeight(40)
        export_pdf_btn.clicked.connect(lambda: self.export_publication('pdf'))
        export_svg_btn = QPushButton("SVG")
        export_svg_btn.setMinimumHeight(40)
        export_svg_btn.clicked.connect(lambda: self.export_publication('svg'))
        export_btn_layout.addWidget(export_png_btn)
        export_btn_layout.addWidget(export_pdf_btn)
        export_btn_layout.addWidget(export_svg_btn)
        export_layout.addLayout(export_btn_layout)
        right_layout.addWidget(export_panel)
        
        left_panel = self.create_plot_settings_panel('matplotlib')
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1100])
        main_layout.addWidget(splitter)
        return tab
    
    def create_plot_settings_panel(self, plot_type):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(500)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(12)
        
        # AXIS RANGE
        axis_group = QGroupBox("AXIS RANGE")
        axis_layout = QVBoxLayout(axis_group)
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X min:"))
        if plot_type == 'pyqtgraph':
            self.fast_xmin_spin = QDoubleSpinBox()
            self.fast_xmin_spin.setRange(800, 2500)
            self.fast_xmin_spin.setValue(1520)
            self.fast_xmin_spin.setSuffix(" nm")
            x_layout.addWidget(self.fast_xmin_spin)
        else:
            self.pub_xmin_spin = QDoubleSpinBox()
            self.pub_xmin_spin.setRange(800, 2500)
            self.pub_xmin_spin.setValue(1520)
            self.pub_xmin_spin.setSuffix(" nm")
            x_layout.addWidget(self.pub_xmin_spin)
        
        x_layout.addWidget(QLabel("X max:"))
        if plot_type == 'pyqtgraph':
            self.fast_xmax_spin = QDoubleSpinBox()
            self.fast_xmax_spin.setRange(800, 2500)
            self.fast_xmax_spin.setValue(1620)
            self.fast_xmax_spin.setSuffix(" nm")
            x_layout.addWidget(self.fast_xmax_spin)
        else:
            self.pub_xmax_spin = QDoubleSpinBox()
            self.pub_xmax_spin.setRange(800, 2500)
            self.pub_xmax_spin.setValue(1620)
            self.pub_xmax_spin.setSuffix(" nm")
            x_layout.addWidget(self.pub_xmax_spin)
        axis_layout.addLayout(x_layout)
        
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y min:"))
        if plot_type == 'pyqtgraph':
            self.fast_ymin_spin = QDoubleSpinBox()
            self.fast_ymin_spin.setRange(-100, 0)
            self.fast_ymin_spin.setValue(-65)
            self.fast_ymin_spin.setSuffix(" dBm")
            y_layout.addWidget(self.fast_ymin_spin)
        else:
            self.pub_ymin_spin = QDoubleSpinBox()
            self.pub_ymin_spin.setRange(-100, 0)
            self.pub_ymin_spin.setValue(-65)
            self.pub_ymin_spin.setSuffix(" dBm")
            y_layout.addWidget(self.pub_ymin_spin)
        
        y_layout.addWidget(QLabel("Y max:"))
        if plot_type == 'pyqtgraph':
            self.fast_ymax_spin = QDoubleSpinBox()
            self.fast_ymax_spin.setRange(-100, 0)
            self.fast_ymax_spin.setValue(-35)
            self.fast_ymax_spin.setSuffix(" dBm")
            y_layout.addWidget(self.fast_ymax_spin)
        else:
            self.pub_ymax_spin = QDoubleSpinBox()
            self.pub_ymax_spin.setRange(-100, 0)
            self.pub_ymax_spin.setValue(-35)
            self.pub_ymax_spin.setSuffix(" dBm")
            y_layout.addWidget(self.pub_ymax_spin)
        axis_layout.addLayout(y_layout)
        
        if plot_type == 'pyqtgraph':
            self.fast_auto_y_cb = QCheckBox("Auto Y-axis")
            self.fast_auto_y_cb.setChecked(True)
            axis_layout.addWidget(self.fast_auto_y_cb)
        else:
            self.pub_auto_y_cb = QCheckBox("Auto Y-axis")
            self.pub_auto_y_cb.setChecked(True)
            axis_layout.addWidget(self.pub_auto_y_cb)
        layout.addWidget(axis_group)
        
        # APPEARANCE
        appearance_group = QGroupBox("APPEARANCE")
        appearance_layout = QVBoxLayout(appearance_group)
        if plot_type == 'pyqtgraph':
            self.fast_grid_cb = QCheckBox("Show Grid")
            self.fast_grid_cb.setChecked(True)
            appearance_layout.addWidget(self.fast_grid_cb)
            self.fast_legend_cb = QCheckBox("Show Legend")
            self.fast_legend_cb.setChecked(True)
            appearance_layout.addWidget(self.fast_legend_cb)
        else:
            self.pub_grid_cb = QCheckBox("Show Grid")
            self.pub_grid_cb.setChecked(True)
            appearance_layout.addWidget(self.pub_grid_cb)
            self.pub_legend_cb = QCheckBox("Show Legend")
            self.pub_legend_cb.setChecked(True)
            appearance_layout.addWidget(self.pub_legend_cb)
        
        lw_layout = QHBoxLayout()
        lw_layout.addWidget(QLabel("Line Width:"))
        if plot_type == 'pyqtgraph':
            self.fast_lw_spin = QDoubleSpinBox()
            self.fast_lw_spin.setRange(0.5, 5.0)
            self.fast_lw_spin.setValue(1.5)
            self.fast_lw_spin.setSingleStep(0.5)
            lw_layout.addWidget(self.fast_lw_spin)
        else:
            self.pub_lw_spin = QDoubleSpinBox()
            self.pub_lw_spin.setRange(0.5, 5.0)
            self.pub_lw_spin.setValue(1.5)
            self.pub_lw_spin.setSingleStep(0.5)
            lw_layout.addWidget(self.pub_lw_spin)
        appearance_layout.addLayout(lw_layout)
        
        palette_layout = QHBoxLayout()
        palette_layout.addWidget(QLabel("Palette:"))
        if plot_type == 'pyqtgraph':
            self.fast_palette_combo = QComboBox()
            self.fast_palette_combo.addItems(['Viridis', 'Plasma', 'Inferno', 'Cool', 'Warm', 'Rainbow', 'Turbo'])
            palette_layout.addWidget(self.fast_palette_combo)
        else:
            self.pub_palette_combo = QComboBox()
            self.pub_palette_combo.addItems(['Viridis', 'Plasma', 'Inferno', 'Cool', 'Warm', 'Rainbow', 'Turbo'])
            palette_layout.addWidget(self.pub_palette_combo)
        appearance_layout.addLayout(palette_layout)
        layout.addWidget(appearance_group)
        
        # QUICK ZOOM
        zoom_group = QGroupBox("QUICK ZOOM")
        zoom_layout = QVBoxLayout(zoom_group)
        zoom_btn_layout = QHBoxLayout()
        full_btn = QPushButton("Full")
        full_btn.clicked.connect(lambda: self.quick_zoom('full', plot_type))
        peak_btn = QPushButton("Peak ±3nm")
        peak_btn.clicked.connect(lambda: self.quick_zoom('peak', plot_type))
        narrow_btn = QPushButton("Narrow ±1nm")
        narrow_btn.clicked.connect(lambda: self.quick_zoom('narrow', plot_type))
        zoom_btn_layout.addWidget(full_btn)
        zoom_btn_layout.addWidget(peak_btn)
        zoom_btn_layout.addWidget(narrow_btn)
        zoom_layout.addLayout(zoom_btn_layout)
        layout.addWidget(zoom_group)
        
        # PRESETS
        preset_group = QGroupBox("PRESETS")
        preset_layout = QVBoxLayout(preset_group)
        preset_btn_layout = QHBoxLayout()
        save_preset_btn = QPushButton("Save")
        save_preset_btn.clicked.connect(lambda: self.save_plot_preset(plot_type))
        load_preset_btn = QPushButton("Load")
        load_preset_btn.clicked.connect(lambda: self.load_plot_preset(plot_type))
        preset_btn_layout.addWidget(save_preset_btn)
        preset_btn_layout.addWidget(load_preset_btn)
        preset_layout.addLayout(preset_btn_layout)
        layout.addWidget(preset_group)
        
        layout.addStretch()
        
        if plot_type == 'pyqtgraph':
            plot_btn = QPushButton("GENERATE PLOT")
            plot_btn.setObjectName("primaryButton")
            plot_btn.clicked.connect(self.plot_fast)
            layout.addWidget(plot_btn)
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda: self.fast_plot.clear_plot())
            layout.addWidget(clear_btn)
        else:
            plot_btn = QPushButton("GENERATE")
            plot_btn.setObjectName("primaryButton")
            plot_btn.clicked.connect(self.plot_publication)
            layout.addWidget(plot_btn)
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda: self.pub_canvas.clear_plot())
            layout.addWidget(clear_btn)
        
        scroll.setWidget(scroll_widget)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        return panel
    
    def plot_fast(self):
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Load data first!")
            return
        selected_ri = self.get_selected_ri()
        if not selected_ri:
            QMessageBox.warning(self, "No Selection", "Select RI values!")
            return
        settings = {
            'xmin': self.fast_xmin_spin.value(),
            'xmax': self.fast_xmax_spin.value(),
            'ymin': self.fast_ymin_spin.value(),
            'ymax': self.fast_ymax_spin.value(),
            'auto_yaxis': self.fast_auto_y_cb.isChecked(),
            'show_grid': self.fast_grid_cb.isChecked(),
            'show_legend': self.fast_legend_cb.isChecked(),
            'show_markers': False,
            'linewidth': self.fast_lw_spin.value(),
            'colors': self.get_color_palette(len(selected_ri), 'pyqtgraph')
        }
        try:
            self.fast_plot.plot_spectra(self.current_data, selected_ri, settings)
            self.status_bar.showMessage(f"✅ Plotted {len(selected_ri)} spectra")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plot failed:\n{e}")
    
    def plot_publication(self):
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Load data first!")
            return
        selected_ri = self.get_selected_ri()
        if not selected_ri:
            QMessageBox.warning(self, "No Selection", "Select RI values!")
            return
        settings = {
            'xmin': self.pub_xmin_spin.value(),
            'xmax': self.pub_xmax_spin.value(),
            'ymin': self.pub_ymin_spin.value(),
            'ymax': self.pub_ymax_spin.value(),
            'auto_yaxis': self.pub_auto_y_cb.isChecked(),
            'show_grid': self.pub_grid_cb.isChecked(),
            'show_legend': self.pub_legend_cb.isChecked(),
            'show_markers': False,
            'linewidth': self.pub_lw_spin.value(),
            'colors': self.get_color_palette(len(selected_ri), 'matplotlib')
        }
        try:
            self.pub_canvas.plot_publication_quality(self.current_data, selected_ri, settings)
            self.status_bar.showMessage(f"✅ Generated {len(selected_ri)} spectra")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plot failed:\n{e}")
    
    def export_publication(self, format):
        extensions = {'png': 'PNG (*.png)', 'pdf': 'PDF (*.pdf)', 'svg': 'SVG (*.svg)'}
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", f"spectrum.{format}", extensions[format])
        if file_path:
            try:
                dpi = 300 if format == 'png' else 100
                self.pub_canvas.fig.savefig(file_path, dpi=dpi, facecolor='#2D2D2D', bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Exported:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    def save_plot_preset(self, plot_type):
        preset_name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and preset_name:
            if plot_type == 'pyqtgraph':
                settings = {
                    'xmin': self.fast_xmin_spin.value(),
                    'xmax': self.fast_xmax_spin.value(),
                    'ymin': self.fast_ymin_spin.value(),
                    'ymax': self.fast_ymax_spin.value(),
                    'auto_y': self.fast_auto_y_cb.isChecked(),
                    'grid': self.fast_grid_cb.isChecked(),
                    'legend': self.fast_legend_cb.isChecked(),
                    'linewidth': self.fast_lw_spin.value(),
                    'palette': self.fast_palette_combo.currentIndex()
                }
            else:
                settings = {
                    'xmin': self.pub_xmin_spin.value(),
                    'xmax': self.pub_xmax_spin.value(),
                    'ymin': self.pub_ymin_spin.value(),
                    'ymax': self.pub_ymax_spin.value(),
                    'auto_y': self.pub_auto_y_cb.isChecked(),
                    'grid': self.pub_grid_cb.isChecked(),
                    'legend': self.pub_legend_cb.isChecked(),
                    'linewidth': self.pub_lw_spin.value(),
                    'palette': self.pub_palette_combo.currentIndex()
                }
            self.settings.setValue(f"preset/{plot_type}/{preset_name}", settings)
            QMessageBox.information(self, "Saved", f"✅ '{preset_name}' saved!")
    
    def load_plot_preset(self, plot_type):
        self.settings.beginGroup(f"preset/{plot_type}")
        preset_names = self.settings.childKeys()
        self.settings.endGroup()
        if not preset_names:
            QMessageBox.information(self, "No Presets", "No saved presets.")
            return
        preset_name, ok = QInputDialog.getItem(self, "Load Preset", "Select:", preset_names, 0, False)
        if ok and preset_name:
            settings = self.settings.value(f"preset/{plot_type}/{preset_name}")
            if plot_type == 'pyqtgraph':
                self.fast_xmin_spin.setValue(settings['xmin'])
                self.fast_xmax_spin.setValue(settings['xmax'])
                self.fast_ymin_spin.setValue(settings['ymin'])
                self.fast_ymax_spin.setValue(settings['ymax'])
                self.fast_auto_y_cb.setChecked(settings['auto_y'])
                self.fast_grid_cb.setChecked(settings['grid'])
                self.fast_legend_cb.setChecked(settings['legend'])
                self.fast_lw_spin.setValue(settings['linewidth'])
                self.fast_palette_combo.setCurrentIndex(settings['palette'])
            else:
                self.pub_xmin_spin.setValue(settings['xmin'])
                self.pub_xmax_spin.setValue(settings['xmax'])
                self.pub_ymin_spin.setValue(settings['ymin'])
                self.pub_ymax_spin.setValue(settings['ymax'])
                self.pub_auto_y_cb.setChecked(settings['auto_y'])
                self.pub_grid_cb.setChecked(settings['grid'])
                self.pub_legend_cb.setChecked(settings['legend'])
                self.pub_lw_spin.setValue(settings['linewidth'])
                self.pub_palette_combo.setCurrentIndex(settings['palette'])
            QMessageBox.information(self, "Loaded", f"✅ '{preset_name}' loaded!")
    
    def quick_zoom(self, preset_type, plot_mode):
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Load data first!")
            return
        wl = self.current_data.get_wavelength()
        wl_min, wl_max = wl.min(), wl.max()
        wl_center = (wl_min + wl_max) / 2
        if preset_type == 'full':
            xmin, xmax = wl_min, wl_max
        elif preset_type == 'peak':
            xmin, xmax = wl_center - 3, wl_center + 3
        elif preset_type == 'narrow':
            xmin, xmax = wl_center - 1, wl_center + 1
        else:
            return
        if plot_mode == 'pyqtgraph':
            self.fast_xmin_spin.setValue(xmin)
            self.fast_xmax_spin.setValue(xmax)
        else:
            self.pub_xmin_spin.setValue(xmin)
            self.pub_xmax_spin.setValue(xmax)
    
    # ========== Peak Detection Tab ==========
    def create_peak_detection_panel(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainlayout = QHBoxLayout(tab)
        mainlayout.setContentsMargins(8, 8, 8, 8)
        
        leftpanel = self.create_peak_settings_panel()
        rightpanel = QFrame()
        rightpanel.setFrameShape(QFrame.StyledPanel)
        rightpanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rightlayout = QVBoxLayout(rightpanel)
        title = QLabel("Peak / Dip Detection — feeds detected points into Fitting")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['warning']}; padding: 4px 0;")
        rightlayout.addWidget(title)
        
        self.peakplot = PeakDipPlotWidget(self)
        rightlayout.addWidget(self.peakplot)
        
        tablepanel = QGroupBox("DETECTION RESULTS")
        tablelayout = QVBoxLayout(tablepanel)
        self.peak_table = QTableWidget()
        self.peak_table.setColumnCount(5)
        self.peak_table.setHorizontalHeaderLabels(["Type", "RI", "Wavelength (nm)", "Power (dBm)", "Prominence"])
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setMinimumHeight(200)
        self.peak_table.setMaximumHeight(300)
        tablelayout.addWidget(self.peak_table)
        
        exportbtnlayout = QHBoxLayout()
        exportcsvbtn = QPushButton("Export CSV")
        exportcsvbtn.clicked.connect(lambda: self.export_peak_data('csv'))
        exportbtnlayout.addWidget(exportcsvbtn)
        tablelayout.addLayout(exportbtnlayout)
        rightlayout.addWidget(tablepanel)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(leftpanel)
        splitter.addWidget(rightpanel)
        splitter.setSizes([350, 1100])
        mainlayout.addWidget(splitter)
        return tab
    
    def create_peak_settings_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(500)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollwidget = QWidget()
        layout = QVBoxLayout(scrollwidget)
        layout.setSpacing(12)
        
        modegroup = QGroupBox("DETECTION MODE")
        modelayout = QVBoxLayout(modegroup)
        self.peak_mode_radio = QRadioButton("Peak only")
        self.dip_mode_radio = QRadioButton("Dip only")
        self.both_mode_radio = QRadioButton("Both (Peak + Dip)")
        self.both_mode_radio.setChecked(True)
        modelayout.addWidget(self.peak_mode_radio)
        modelayout.addWidget(self.dip_mode_radio)
        modelayout.addWidget(self.both_mode_radio)
        layout.addWidget(modegroup)
        
        paramgroup = QGroupBox("DETECTION PARAMETERS")
        paramlayout = QVBoxLayout(paramgroup)
        heightlayout = QHBoxLayout()
        height_label = QLabel("Power threshold:")
        height_label.setToolTip(
            "Peak mode: peaks must reach AT LEAST this power.\n"
            "Dip mode: dips must reach AT MOST this power (go below this value).\n"
            "Both mode: applied symmetrically to peaks and dips.\n"
            "Units: dBm"
        )
        heightlayout.addWidget(height_label)
        self.peak_height_spin = QDoubleSpinBox()
        self.peak_height_spin.setRange(-100, 0)
        self.peak_height_spin.setValue(-60)
        self.peak_height_spin.setSuffix(" dBm")
        heightlayout.addWidget(self.peak_height_spin)
        paramlayout.addLayout(heightlayout)
        
        distancelayout = QHBoxLayout()
        distancelayout.addWidget(QLabel("Distance:"))
        self.peak_distance_spin = QSpinBox()
        self.peak_distance_spin.setRange(1, 100)
        self.peak_distance_spin.setValue(10)
        distancelayout.addWidget(self.peak_distance_spin)
        paramlayout.addLayout(distancelayout)
        
        prominencelayout = QHBoxLayout()
        prominencelayout.addWidget(QLabel("Prominence:"))
        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(0.1, 10.0)
        self.peak_prominence_spin.setValue(1.0)
        prominencelayout.addWidget(self.peak_prominence_spin)
        paramlayout.addLayout(prominencelayout)
        layout.addWidget(paramgroup)
        
        detectbtn = QPushButton("DETECT")
        detectbtn.setObjectName("primaryButton")
        detectbtn.clicked.connect(self.detect_peaks_dips)
        layout.addWidget(detectbtn)
        
        statsgroup = QGroupBox("STATISTICS")
        statslayout = QVBoxLayout(statsgroup)
        self.peak_stats_label = QLabel("Run detection first...")
        self.peak_stats_label.setWordWrap(True)
        self.peak_stats_label.setStyleSheet("font-size: 9pt; color: #888;")
        statslayout.addWidget(self.peak_stats_label)
        layout.addWidget(statsgroup)
        
        layout.addStretch()
        scroll.setWidget(scrollwidget)
        panellayout = QVBoxLayout(panel)
        panellayout.setContentsMargins(0, 0, 0, 0)
        panellayout.addWidget(scroll)
        return panel
    
    def detect_peaks_dips(self):
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Load data first!")
            return
        selected_ri = self.get_selected_ri()
        if not selected_ri:
            QMessageBox.warning(self, "No Selection", "Select RI values!")
            return
        if self.peak_mode_radio.isChecked():
            mode = 'peak'
        elif self.dip_mode_radio.isChecked():
            mode = 'dip'
        else:
            mode = 'both'
        
        height = self.peak_height_spin.value()
        distance = self.peak_distance_spin.value()
        prominence = self.peak_prominence_spin.value()
        
        try:
            peaks_data, dips_data = self.peak_dip_detector.detect_all_ri(
                self.current_data, selected_ri,
                mode=mode, height=height, distance=distance, prominence=prominence
            )
            self.store_peak_data_for_fitting(selected_ri, peaks_data, dips_data)
            
            settings = {
                'xmin': self.fast_xmin_spin.value(),
                'xmax': self.fast_xmax_spin.value(),
                'ymin': self.fast_ymin_spin.value(),
                'ymax': self.fast_ymax_spin.value(),
                'auto_yaxis': self.fast_auto_y_cb.isChecked(),
                'show_grid': self.fast_grid_cb.isChecked(),
                'linewidth': self.fast_lw_spin.value()
            }
            self.peakplot.plot_with_peaks_dips(self.current_data, selected_ri, peaks_data, dips_data, settings)
            self.update_peak_table(mode)
            
            stats = self.peak_dip_detector.get_summary_stats()
            stats_text = f"✅ Peaks: {stats.get('total_peaks', 0)} ({stats.get('num_ri_peaks', 0)} RI)\n"
            stats_text += f"✅ Dips: {stats.get('total_dips', 0)} ({stats.get('num_ri_dips', 0)} RI)\n"
            if 'avg_peak_wavelength' in stats:
                stats_text += f"\nAvg Peak: {stats['avg_peak_wavelength']:.2f} nm"
            if 'avg_dip_wavelength' in stats:
                stats_text += f"\nAvg Dip: {stats['avg_dip_wavelength']:.2f} nm"
            self.peak_stats_label.setText(stats_text)
            self.status_bar.showMessage(f"✅ Detected {stats.get('total_peaks', 0)} peaks, {stats.get('total_dips', 0)} dips")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed:\n{e}")
    
    def store_peak_data_for_fitting(self, ri_values, peaks_data, dips_data):
        self.last_ri_vals = []
        self.last_power_vals = []
        self.last_wavelength_vals = []
        for ri in ri_values:
            if ri in peaks_data and peaks_data[ri]:
                peak = peaks_data[ri][0]
                self.last_ri_vals.append(ri)
                self.last_wavelength_vals.append(peak['wavelength'])
                self.last_power_vals.append(peak['power'])
            elif ri in dips_data and dips_data[ri]:
                dip = dips_data[ri][0]
                self.last_ri_vals.append(ri)
                self.last_wavelength_vals.append(dip['wavelength'])
                self.last_power_vals.append(dip['power'])
    
    def import_peak_data_to_fitting(self):
        if not self.last_ri_vals:
            QMessageBox.warning(self, "No Peak Data", "Run Peak Analysis first!\n\nGo to 'Peak and Dip Detection' tab → Click DETECT")
            return
        try:
            self.fitting_engine.import_from_peak_detection(
                self.last_ri_vals,
                self.last_power_vals,
                self.last_wavelength_vals
            )
            num_points = len(self.last_ri_vals)
            ri_range = f"{min(self.last_ri_vals):.2f} - {max(self.last_ri_vals):.2f}"
            self.fit_import_status.setText(f"✅ Imported {num_points} points\nRI range: {ri_range}")
            self.status_bar.showMessage(f"✅ Imported {num_points} peak data points for fitting")
            QMessageBox.information(self, "Success", f"✅ Imported {num_points} data points\nRI range: {ri_range}\n\nNow you can perform Single or Dual fitting!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Import failed:\n{e}")
    
    def update_peak_table(self, mode):
        df = self.peak_dip_detector.export_to_dataframe(mode)
        self.peak_table.setRowCount(len(df))
        for i, row in df.iterrows():
            self.peak_table.setItem(i, 0, QTableWidgetItem(str(row['Type'])))
            self.peak_table.setItem(i, 1, QTableWidgetItem(f"{row['RI']:.2f}"))
            self.peak_table.setItem(i, 2, QTableWidgetItem(f"{row['Wavelength (nm)']:.3f}"))
            self.peak_table.setItem(i, 3, QTableWidgetItem(f"{row['Power (dBm)']:.2f}"))
            self.peak_table.setItem(i, 4, QTableWidgetItem(f"{row['Prominence']:.2f}"))
        self.peak_table.resizeColumnsToContents()
    
    def export_peak_data(self, format):
        if not self.peak_dip_detector.peaks_data and not self.peak_dip_detector.dips_data:
            QMessageBox.warning(self, "No Data", "Run detection first!")
            return
        if self.peak_mode_radio.isChecked():
            mode = 'peak'
        elif self.dip_mode_radio.isChecked():
            mode = 'dip'
        else:
            mode = 'both'
        df = self.peak_dip_detector.export_to_dataframe(mode)
        if df.empty:
            QMessageBox.warning(self, "No Data", "No peaks/dips detected!")
            return
        if format == 'csv':
            file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "peak_dip_results.csv", "CSV (*.csv)")
            if file_path:
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Exported:\n{file_path}")
        elif format == 'excel':
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "peak_dip_results.xlsx", "Excel (*.xlsx)")
            if file_path:
                df.to_excel(file_path, index=False, sheet_name='PeakDipResults')
                QMessageBox.information(self, "Success", f"Exported:\n{file_path}")
    
    # ========== Fitting Tab ==========
    def create_enhanced_fitting_panel(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainlayout = QHBoxLayout(tab)
        mainlayout.setContentsMargins(8, 8, 8, 8)
        
        leftpanel = self.create_fitting_settings_panel()
        rightpanel = QFrame()
        rightpanel.setFrameShape(QFrame.StyledPanel)
        rightpanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rightlayout = QVBoxLayout(rightpanel)
        title = QLabel("Sensitivity Fitting — linear or polynomial, with uncertainty")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['primary_light']}; padding: 4px 0;")
        rightlayout.addWidget(title)
        
        self.fit_canvas_tabs = QTabWidget()
        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)
        self.single_fit_canvas = FittingCanvas(self, width=10, height=6, dpi=100)
        single_layout.addWidget(self.single_fit_canvas)
        self.fit_canvas_tabs.addTab(single_widget, "Single Fit")
        
        dual_widget = QWidget()
        dual_layout = QVBoxLayout(dual_widget)
        self.dual_fit_canvas = DualFittingCanvas(self, width=10, height=5, dpi=100)
        dual_layout.addWidget(self.dual_fit_canvas)
        self.fit_canvas_tabs.addTab(dual_widget, "Dual Fit")
        rightlayout.addWidget(self.fit_canvas_tabs)
        
        resultspanel = QGroupBox("FIT RESULTS")
        resultslayout = QVBoxLayout(resultspanel)
        resultslayout.setContentsMargins(8, 8, 8, 8)
        self.fit_results_text = QTextEdit()
        self.fit_results_text.setReadOnly(True)
        self.fit_results_text.setMaximumHeight(200)
        self.fit_results_text.setPlaceholderText("Fitting results will appear here...")
        resultslayout.addWidget(self.fit_results_text)
        
        exportbtnlayout = QHBoxLayout()
        export_fit_btn = QPushButton("Export Results")
        export_fit_btn.clicked.connect(self.export_fitting_results)
        exportbtnlayout.addWidget(export_fit_btn)
        resultslayout.addLayout(exportbtnlayout)
        rightlayout.addWidget(resultspanel)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(leftpanel)
        splitter.addWidget(rightpanel)
        splitter.setSizes([350, 1100])
        mainlayout.addWidget(splitter)
        return tab
    
    def create_fitting_settings_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(500)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollwidget = QWidget()
        layout = QVBoxLayout(scrollwidget)
        layout.setSpacing(12)
        
        # IMPORT DATA
        importgroup = QGroupBox("IMPORT DATA")
        importlayout = QVBoxLayout(importgroup)
        importbtn = QPushButton("Import from Peak Detection")
        importbtn.setObjectName("successButton")
        importbtn.clicked.connect(self.import_peak_data_to_fitting)
        importlayout.addWidget(importbtn)
        self.fit_import_status = QLabel("No data imported yet")
        self.fit_import_status.setWordWrap(True)
        self.fit_import_status.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        importlayout.addWidget(self.fit_import_status)
        layout.addWidget(importgroup)
        
        # FITTING MODE
        modegroup = QGroupBox("FITTING MODE")
        modelayout = QVBoxLayout(modegroup)
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["Wavelength vs RI", "Power vs RI"])
        modelayout.addWidget(QLabel("Y-axis:"))
        modelayout.addWidget(self.fit_mode_combo)
        
        modelayout.addSpacing(6)
        modelayout.addWidget(QLabel("Polynomial degree:"))
        self.fit_degree_combo = QComboBox()
        self.fit_degree_combo.addItems([
            "1 — Linear (Y = m·RI + c)",
            "2 — Quadratic (Y = a·RI² + b·RI + c)",
            "3 — Cubic (Y = a·RI³ + b·RI² + c·RI + d)"
        ])
        self.fit_degree_combo.setToolTip(
            "Linear is best for sensor sensitivity reporting.\n"
            "Use quadratic/cubic only if the data shows clear nonlinearity."
        )
        modelayout.addWidget(self.fit_degree_combo)
        layout.addWidget(modegroup)
        
        # REGION RANGES
        rangegroup = QGroupBox("REGION RANGES")
        rangelayout = QVBoxLayout(rangegroup)
        infolabel = QLabel("Adjust Low/High RI ranges for Dual Fit:")
        infolabel.setWordWrap(True)
        infolabel.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        rangelayout.addWidget(infolabel)
        rangelayout.addSpacing(8)
        
        lowlabel = QLabel("Low Region:")
        lowlabel.setStyleSheet("font-weight: bold; color: #42A5F5;")
        rangelayout.addWidget(lowlabel)
        lowlayout = QHBoxLayout()
        lowlayout.addWidget(QLabel("Min:"))
        self.fit_low_min_spin = QDoubleSpinBox()
        self.fit_low_min_spin.setRange(1.20, 2.00)
        self.fit_low_min_spin.setDecimals(3)
        self.fit_low_min_spin.setSingleStep(0.01)
        self.fit_low_min_spin.setValue(1.300)
        lowlayout.addWidget(self.fit_low_min_spin)
        lowlayout.addWidget(QLabel(" - Max:"))
        self.fit_low_max_spin = QDoubleSpinBox()
        self.fit_low_max_spin.setRange(1.20, 2.00)
        self.fit_low_max_spin.setDecimals(3)
        self.fit_low_max_spin.setSingleStep(0.01)
        self.fit_low_max_spin.setValue(1.390)
        lowlayout.addWidget(self.fit_low_max_spin)
        rangelayout.addLayout(lowlayout)
        rangelayout.addSpacing(8)
        
        highlabel = QLabel("High Region:")
        highlabel.setStyleSheet("font-weight: bold; color: #F44336;")
        rangelayout.addWidget(highlabel)
        highlayout = QHBoxLayout()
        highlayout.addWidget(QLabel("Min:"))
        self.fit_high_min_spin = QDoubleSpinBox()
        self.fit_high_min_spin.setRange(1.20, 2.00)
        self.fit_high_min_spin.setDecimals(3)
        self.fit_high_min_spin.setSingleStep(0.01)
        self.fit_high_min_spin.setValue(1.460)
        highlayout.addWidget(self.fit_high_min_spin)
        highlayout.addWidget(QLabel(" - Max:"))
        self.fit_high_max_spin = QDoubleSpinBox()
        self.fit_high_max_spin.setRange(1.20, 2.00)
        self.fit_high_max_spin.setDecimals(3)
        self.fit_high_max_spin.setSingleStep(0.01)
        self.fit_high_max_spin.setValue(1.640)
        highlayout.addWidget(self.fit_high_max_spin)
        rangelayout.addLayout(highlayout)
        layout.addWidget(rangegroup)
        
        layout.addStretch()
        
        # FITTING BUTTONS
        fitbtngroup = QGroupBox("PERFORM FITTING")
        fitbtnlayout = QVBoxLayout(fitbtngroup)
        singlefitbtn = QPushButton("Single Fit")
        singlefitbtn.setObjectName("primaryButton")
        singlefitbtn.setMinimumHeight(45)
        singlefitbtn.clicked.connect(lambda: self.perform_fitting('single'))
        fitbtnlayout.addWidget(singlefitbtn)
        dualfitbtn = QPushButton("Dual Fit (Split)")
        dualfitbtn.setObjectName("primaryButton")
        dualfitbtn.setMinimumHeight(45)
        dualfitbtn.clicked.connect(lambda: self.perform_fitting('dual'))
        fitbtnlayout.addWidget(dualfitbtn)
        layout.addWidget(fitbtngroup)
        
        layout.addStretch()
        scroll.setWidget(scrollwidget)
        panellayout = QVBoxLayout(panel)
        panellayout.setContentsMargins(0, 0, 0, 0)
        panellayout.addWidget(scroll)
        return panel
    
    def perform_fitting(self, fit_type='single'):
        if self.fitting_engine.ri_data is None:
            QMessageBox.warning(self, "No Data", "Import peak data first!\n\nClick 'Import from Peak Detection'")
            return
        
        # Read fitting mode + degree from UI
        mode_text = self.fit_mode_combo.currentText()
        mode = "wavelength" if "Wavelength" in mode_text else "power"
        degree = self.fit_degree_combo.currentIndex() + 1  # combo idx 0→degree 1, 1→2, 2→3
        
        low_range = (self.fit_low_min_spin.value(), self.fit_low_max_spin.value())
        high_range = (self.fit_high_min_spin.value(), self.fit_high_max_spin.value())
        
        # Range validation — only nag when the validation actually matters for the chosen fit type
        if fit_type == 'dual':
            if low_range[0] >= low_range[1]:
                QMessageBox.warning(self, "Invalid Range",
                                  f"Low Region: Min ({low_range[0]:.3f}) must be less than Max ({low_range[1]:.3f}).")
                return
            if high_range[0] >= high_range[1]:
                QMessageBox.warning(self, "Invalid Range",
                                  f"High Region: Min ({high_range[0]:.3f}) must be less than Max ({high_range[1]:.3f}).")
                return
            if low_range[1] > high_range[0]:
                # Not a hard error — just a warning, since some sensors have overlapping regions
                reply = QMessageBox.question(
                    self, "Overlapping Regions",
                    f"Low region max ({low_range[1]:.3f}) is greater than High region min ({high_range[0]:.3f}).\n"
                    "Regions overlap — continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        
        try:
            if fit_type == 'single':
                fit_result = self.fitting_engine.fit_single_region(
                    mode=mode, region="all",
                    low_range=low_range, high_range=high_range,
                    degree=degree
                )
                if fit_result:
                    self.single_fit_canvas.ax.clear()
                    self.fitting_engine.plot_single_fit(self.single_fit_canvas.ax)
                    self.single_fit_canvas.fig.tight_layout()
                    self.single_fit_canvas.draw()
                    
                    self.fit_results_text.setText(
                        self._build_fit_results_text(fit_result, "SINGLE FIT RESULTS")
                    )
                    self.fit_canvas_tabs.setCurrentIndex(0)
                    self.status_bar.showMessage(
                        f"✅ Single fit (deg {degree}): R²={fit_result['r2']:.4f}, "
                        f"S={fit_result['sensitivity']:.2f} {fit_result['unit']}"
                    )
            
            elif fit_type == 'dual':
                dual_result = self.fitting_engine.fit_dual_regions(
                    mode=mode, low_range=low_range, high_range=high_range,
                    degree=degree
                )
                if dual_result:
                    self.dual_fit_canvas.fig.clear()
                    self.fitting_engine.plot_dual_fit(self.dual_fit_canvas.fig)
                    self.dual_fit_canvas.draw()
                    
                    sections = [f"DUAL FIT RESULTS",
                               f"",
                               f"Mode: {mode.capitalize()} vs RI",
                               f"Degree: {degree} ({EnhancedFittingEngine.DEGREE_LABELS.get(degree, 'Polynomial')})",
                               ""]
                    
                    if dual_result['low'] is not None:
                        sections.append("─── LOW REGION ───")
                        sections.append(self._build_fit_results_text(dual_result['low'], header=None))
                        sections.append("")
                    else:
                        sections.append("LOW REGION: No data (insufficient points or out of range)")
                        sections.append("")
                    
                    if dual_result['high'] is not None:
                        sections.append("─── HIGH REGION ───")
                        sections.append(self._build_fit_results_text(dual_result['high'], header=None))
                    else:
                        sections.append("HIGH REGION: No data (insufficient points or out of range)")
                    
                    self.fit_results_text.setText("\n".join(sections))
                    self.fit_canvas_tabs.setCurrentIndex(1)
                    self.status_bar.showMessage(f"✅ Dual fitting completed (degree {degree})")
        except ValueError as e:
            QMessageBox.warning(self, "Fit Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fitting failed:\n{e}")
            import traceback
            traceback.print_exc()
    
    def _build_fit_results_text(self, fit, header="FIT RESULTS"):
        """Build a multi-line results text block for any fit dict (linear or polynomial)."""
        lines = []
        if header:
            lines.append(header)
            lines.append("")
        
        deg = fit['degree']
        deg_label = EnhancedFittingEngine.DEGREE_LABELS.get(deg, f"Degree {deg}")
        lines.append(f"Region: {fit['region']}")
        lines.append(f"Degree: {deg} ({deg_label})")
        lines.append(f"Equation: {fit['equation']}")
        lines.append("")
        lines.append(f"R²    = {fit['r2']:.6f}")
        lines.append(f"RMSE  = {fit['rmse']:.4f}")
        lines.append(f"Data points: {len(fit['ri'])}")
        lines.append("")
        lines.append("Coefficients (with std error):")
        for i, (c, err) in enumerate(zip(fit['coeffs'], fit['errors'])):
            power = deg - i
            if power == 0:
                name = "Intercept   "
            elif power == 1:
                name = "Slope (RI¹) "
            else:
                name = f"Coeff (RI^{power})"
            if not np.isnan(err):
                lines.append(f"  {name}: {c: .6g} ± {err:.4g}")
            else:
                lines.append(f"  {name}: {c: .6g}  (uncertainty N/A)")
        lines.append("")
        if not np.isnan(fit['sensitivity_err']):
            lines.append(f"Sensitivity: {fit['sensitivity']:.4f} ± {fit['sensitivity_err']:.4f} {fit['unit']}")
        else:
            lines.append(f"Sensitivity: {fit['sensitivity']:.4f} {fit['unit']}")
        if deg > 1:
            lines.append(f"  (local dy/dRI evaluated at RI = {fit['ri_center']:.3f})")
        
        return "\n".join(lines)
    
    def export_fitting_results(self):
        if self.fitting_engine.single_fit is None and self.fitting_engine.dual_fit is None:
            QMessageBox.warning(self, "No Results", "Perform fitting first!")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Fitting Results", "fitting_results.xlsx", "Excel (*.xlsx)")
        if filepath:
            try:
                self.fitting_engine.export_fit_results(filepath)
                QMessageBox.information(self, "Success", f"Exported:\n{filepath}")
                self.status_bar.showMessage(f"✅ Exported: {Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    # ========== Machine Learning Tab ==========
    def create_ml_tab(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainlayout = QHBoxLayout(tab)
        mainlayout.setContentsMargins(8, 8, 8, 8)
        
        leftpanel = self.create_ml_settings_panel()
        rightpanel = QFrame()
        rightpanel.setFrameShape(QFrame.StyledPanel)
        rightpanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rightlayout = QVBoxLayout(rightpanel)
        title = QLabel("Machine Learning — predict RI from spectral features")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {ModernTheme.DARK['primary_light']}; padding: 4px 0;")
        rightlayout.addWidget(title)
        
        self.ml_plot_tabs = QTabWidget()
        pred_widget = QWidget()
        pred_layout = QVBoxLayout(pred_widget)
        self.ml_plot_canvas = PublicationCanvas(self, width=10, height=8, dpi=100)
        pred_layout.addWidget(self.ml_plot_canvas)
        self.ml_plot_tabs.addTab(pred_widget, "Predictions")
        
        importance_widget = QWidget()
        importance_layout = QVBoxLayout(importance_widget)
        self.ml_importance_canvas = PublicationCanvas(self, width=10, height=6, dpi=100)
        importance_layout.addWidget(self.ml_importance_canvas)
        self.ml_plot_tabs.addTab(importance_widget, "Feature Importance")
        rightlayout.addWidget(self.ml_plot_tabs)
        
        metrics_label = QLabel("TRAINING METRICS")
        metrics_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        rightlayout.addWidget(metrics_label)
        self.ml_metrics_table = QTableWidget()
        self.ml_metrics_table.setColumnCount(2)
        self.ml_metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.ml_metrics_table.horizontalHeader().setStretchLastSection(True)
        self.ml_metrics_table.setMaximumHeight(200)
        self.ml_metrics_table.setAlternatingRowColors(True)
        rightlayout.addWidget(self.ml_metrics_table)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(leftpanel)
        splitter.addWidget(rightpanel)
        splitter.setSizes([350, 1100])
        mainlayout.addWidget(splitter)
        return tab
    
    def create_ml_settings_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(500)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollwidget = QWidget()
        layout = QVBoxLayout(scrollwidget)
        layout.setSpacing(12)
        
        # DATASET SOURCE
        datagroup = QGroupBox("DATASET SOURCE")
        datalayout = QVBoxLayout(datagroup)
        datalayout.addWidget(QLabel("Source:"))
        self.ml_source_combo = QComboBox()
        self.ml_source_combo.addItems([
            "Peak Detection (Features)",
            "Raw Spectrum (Wide)",
            "External File (.csv/.xlsx)",
        ])
        self.ml_source_combo.setToolTip(
            "Peak Detection (Features): build a compact ~10-feature dataset from "
            "the spectra loaded in tab 1 (peak position, power, mean/std/range, etc.). "
            "Fast and interpretable.\n\n"
            "Raw Spectrum (Wide): build a wide-format dataset where each row is "
            "one full spectrum (one column per wavelength sample, plus ri_value). "
            "Works well with tree-based ML (Random Forest, XGBoost, CatBoost).\n\n"
            "External File: load any CSV/XLSX with an RI target column. Both "
            "engineered and wide formats supported; non-numeric columns are "
            "dropped automatically. Yokogawa '-210.0' sentinel values are "
            "treated as missing and imputed."
        )
        datalayout.addWidget(self.ml_source_combo)
        
        # Row of action buttons: build dataset / save dataset
        buttonrow = QHBoxLayout()
        loaddatasetbtn = QPushButton("Load / Build Dataset")
        loaddatasetbtn.setToolTip("Generate the ML dataset from the selected source.")
        loaddatasetbtn.clicked.connect(self.ml_load_dataset)
        buttonrow.addWidget(loaddatasetbtn)
        
        savedatasetbtn = QPushButton("Save Dataset…")
        savedatasetbtn.setToolTip(
            "Save the current dataset (after augmentation if applied) to CSV or XLSX.\n"
            "Useful for reusing the same dataset later or sharing it."
        )
        savedatasetbtn.clicked.connect(self.ml_save_dataset)
        buttonrow.addWidget(savedatasetbtn)
        datalayout.addLayout(buttonrow)
        
        self.ml_data_status = QLabel("No dataset loaded")
        self.ml_data_status.setWordWrap(True)
        self.ml_data_status.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        datalayout.addWidget(self.ml_data_status)
        layout.addWidget(datagroup)
        
        # MODEL SELECTION
        modelgroup = QGroupBox("MODEL SELECTION")
        modellayout = QVBoxLayout(modelgroup)
        modellayout.addWidget(QLabel("Choose model:"))
        self.ml_model_combo = QComboBox()
        self.ml_model_combo.addItems(["Linear Regression", "Random Forest", "XGBoost", "CatBoost",
                                       "ANN", "CNN-1D"])
        modellayout.addWidget(self.ml_model_combo)
        self.ml_model_info = QLabel("Fast linear model")
        self.ml_model_info.setWordWrap(True)
        self.ml_model_info.setStyleSheet("font-size: 9pt; color: #aaa; font-style: italic;")
        modellayout.addWidget(self.ml_model_info)
        self.ml_model_combo.currentTextChanged.connect(self.update_ml_model_info)
        layout.addWidget(modelgroup)
        
        # DATA AUGMENTATION
        augmentgroup = QGroupBox("DATA AUGMENTATION")
        augmentlayout = QVBoxLayout(augmentgroup)
        
        self.ml_augment_cb = QCheckBox("Enable augmentation")
        self.ml_augment_cb.setChecked(False)
        self.ml_augment_cb.setToolTip(
            "Generate synthetic training samples by perturbing the original data.\n"
            "Helps small datasets, but won't fix fundamentally bad features."
        )
        augmentlayout.addWidget(self.ml_augment_cb)
        
        # Container that's only shown when augmentation is enabled
        self.ml_aug_config_widget = QWidget()
        aug_cfg_layout = QVBoxLayout(self.ml_aug_config_widget)
        aug_cfg_layout.setContentsMargins(8, 4, 4, 4)
        aug_cfg_layout.setSpacing(6)
        
        # --- Gaussian noise ---
        noise_box = QFrame()
        noise_box.setStyleSheet(f"QFrame {{ border-left: 2px solid {ModernTheme.DARK['primary_light']}; padding-left: 6px; }}")
        noise_lay = QVBoxLayout(noise_box)
        noise_lay.setContentsMargins(6, 2, 2, 2)
        noise_lay.setSpacing(2)
        self.ml_aug_noise_cb = QCheckBox("Gaussian noise")
        self.ml_aug_noise_cb.setChecked(True)
        self.ml_aug_noise_cb.setToolTip("Add zero-mean Gaussian noise scaled to each feature's std.")
        noise_lay.addWidget(self.ml_aug_noise_cb)
        noise_form = QFormLayout()
        noise_form.setContentsMargins(16, 0, 0, 0)
        noise_form.setSpacing(4)
        self.ml_aug_noise_level = QDoubleSpinBox()
        self.ml_aug_noise_level.setRange(0.001, 1.0)
        self.ml_aug_noise_level.setValue(0.01)
        self.ml_aug_noise_level.setSingleStep(0.005)
        self.ml_aug_noise_level.setDecimals(3)
        self.ml_aug_noise_level.setToolTip("Noise std as a fraction of each feature's std (e.g. 0.01 = 1%).")
        noise_form.addRow("noise level:", self.ml_aug_noise_level)
        self.ml_aug_noise_n = QSpinBox()
        self.ml_aug_noise_n.setRange(1, 10)
        self.ml_aug_noise_n.setValue(1)
        self.ml_aug_noise_n.setToolTip("Number of independent noisy copies to generate.")
        noise_form.addRow("copies:", self.ml_aug_noise_n)
        noise_lay.addLayout(noise_form)
        aug_cfg_layout.addWidget(noise_box)
        
        # --- Wavelength shift ---
        shift_box = QFrame()
        shift_box.setStyleSheet(f"QFrame {{ border-left: 2px solid {ModernTheme.DARK['success']}; padding-left: 6px; }}")
        shift_lay = QVBoxLayout(shift_box)
        shift_lay.setContentsMargins(6, 2, 2, 2)
        shift_lay.setSpacing(2)
        self.ml_aug_shift_cb = QCheckBox("Wavelength shift")
        self.ml_aug_shift_cb.setChecked(True)
        self.ml_aug_shift_cb.setToolTip("Add a constant nm offset to wavelength features (simulates calibration drift).")
        shift_lay.addWidget(self.ml_aug_shift_cb)
        shift_form = QFormLayout()
        shift_form.setContentsMargins(16, 0, 0, 0)
        shift_form.setSpacing(4)
        self.ml_aug_shift_range = QDoubleSpinBox()
        self.ml_aug_shift_range.setRange(0.01, 5.0)
        self.ml_aug_shift_range.setValue(0.3)
        self.ml_aug_shift_range.setSingleStep(0.1)
        self.ml_aug_shift_range.setSuffix(" nm")
        self.ml_aug_shift_range.setDecimals(2)
        self.ml_aug_shift_range.setToolTip("Max ± shift in nm (uniform random per copy).")
        shift_form.addRow("shift range:", self.ml_aug_shift_range)
        self.ml_aug_shift_n = QSpinBox()
        self.ml_aug_shift_n.setRange(1, 10)
        self.ml_aug_shift_n.setValue(1)
        shift_form.addRow("copies:", self.ml_aug_shift_n)
        shift_lay.addLayout(shift_form)
        aug_cfg_layout.addWidget(shift_box)
        
        # --- Power scaling ---
        scale_box = QFrame()
        scale_box.setStyleSheet(f"QFrame {{ border-left: 2px solid {ModernTheme.DARK['warning']}; padding-left: 6px; }}")
        scale_lay = QVBoxLayout(scale_box)
        scale_lay.setContentsMargins(6, 2, 2, 2)
        scale_lay.setSpacing(2)
        self.ml_aug_scale_cb = QCheckBox("Power scaling")
        self.ml_aug_scale_cb.setChecked(True)
        self.ml_aug_scale_cb.setToolTip("Multiply power features by (1 ± scale_range) (simulates gain variation).")
        scale_lay.addWidget(self.ml_aug_scale_cb)
        scale_form = QFormLayout()
        scale_form.setContentsMargins(16, 0, 0, 0)
        scale_form.setSpacing(4)
        self.ml_aug_scale_range = QDoubleSpinBox()
        self.ml_aug_scale_range.setRange(0.001, 0.5)
        self.ml_aug_scale_range.setValue(0.03)
        self.ml_aug_scale_range.setSingleStep(0.01)
        self.ml_aug_scale_range.setDecimals(3)
        self.ml_aug_scale_range.setToolTip("Max ± fractional scale (e.g. 0.03 = ±3%).")
        scale_form.addRow("scale range:", self.ml_aug_scale_range)
        self.ml_aug_scale_n = QSpinBox()
        self.ml_aug_scale_n.setRange(1, 10)
        self.ml_aug_scale_n.setValue(1)
        scale_form.addRow("copies:", self.ml_aug_scale_n)
        scale_lay.addLayout(scale_form)
        aug_cfg_layout.addWidget(scale_box)
        
        # Live preview of resulting dataset size
        self.ml_aug_preview = QLabel("Preview: load dataset first")
        self.ml_aug_preview.setStyleSheet("font-size: 9pt; color: #aaa; padding: 4px;")
        self.ml_aug_preview.setWordWrap(True)
        aug_cfg_layout.addWidget(self.ml_aug_preview)
        
        # Action button: materialize augmentation into the current dataset
        self.ml_aug_apply_btn = QPushButton("Build Augmented Dataset")
        self.ml_aug_apply_btn.setToolTip(
            "Apply the augmentation settings above and REPLACE the current dataset "
            "with the augmented version. The new dataset gets an 'augmentation' label "
            "column tagging each row (e.g. 'original', 'noise_0.010_c1') — that column "
            "is visible in the CSV export but ignored during training."
        )
        self.ml_aug_apply_btn.clicked.connect(self.ml_apply_augmentation)
        aug_cfg_layout.addWidget(self.ml_aug_apply_btn)
        
        augmentlayout.addWidget(self.ml_aug_config_widget)
        self.ml_aug_config_widget.setVisible(False)
        
        # Wire up: show/hide config + live preview updates
        self.ml_augment_cb.toggled.connect(self.ml_aug_config_widget.setVisible)
        self.ml_augment_cb.toggled.connect(self._update_aug_preview)
        for w in (self.ml_aug_noise_cb, self.ml_aug_shift_cb, self.ml_aug_scale_cb):
            w.toggled.connect(self._update_aug_preview)
        for w in (self.ml_aug_noise_n, self.ml_aug_shift_n, self.ml_aug_scale_n):
            w.valueChanged.connect(self._update_aug_preview)
        
        self.ml_aug_status = QLabel("")
        self.ml_aug_status.setWordWrap(True)
        self.ml_aug_status.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        augmentlayout.addWidget(self.ml_aug_status)
        layout.addWidget(augmentgroup)
        
        # HYPERPARAMETERS
        manualparamgroup = QGroupBox("HYPERPARAMETERS")
        manualparamlayout = QVBoxLayout(manualparamgroup)
        self.ml_param_hint = QLabel("Parameters will appear after model selection")
        self.ml_param_hint.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        manualparamlayout.addWidget(self.ml_param_hint)
        
        # Use QFormLayout — handles label+field pairs natively and removeRow()
        # properly destroys BOTH the label and the field widget on each row.
        # The old QVBoxLayout-of-QHBoxLayouts approach orphaned the labels on
        # model switch, causing them to stack on top of each other.
        self.ml_manual_param_container = QWidget()
        self.ml_manual_param_form = QFormLayout(self.ml_manual_param_container)
        self.ml_manual_param_form.setContentsMargins(5, 5, 5, 5)
        self.ml_manual_param_form.setSpacing(8)
        self.ml_manual_param_form.setLabelAlignment(Qt.AlignLeft)
        self.ml_manual_param_widgets = {}
        manualparamlayout.addWidget(self.ml_manual_param_container)
        self.ml_model_combo.currentTextChanged.connect(self.update_manual_param_widgets)
        layout.addWidget(manualparamgroup)
        
        # TRAINING SETTINGS
        traingroup = QGroupBox("TRAINING SETTINGS")
        trainlayout = QVBoxLayout(traingroup)
        testlayout = QHBoxLayout()
        testlayout.addWidget(QLabel("Test size:"))
        self.ml_test_size_spin = QDoubleSpinBox()
        self.ml_test_size_spin.setRange(0.1, 0.5)
        self.ml_test_size_spin.setValue(0.2)
        self.ml_test_size_spin.setSingleStep(0.05)
        testlayout.addWidget(self.ml_test_size_spin)
        trainlayout.addLayout(testlayout)
        self.ml_scale_cb = QCheckBox("Feature Scaling (StandardScaler)")
        self.ml_scale_cb.setChecked(True)
        trainlayout.addWidget(self.ml_scale_cb)
        layout.addWidget(traingroup)
        
        layout.addStretch()
        
        self.ml_train_btn = QPushButton("TRAIN MODEL")
        self.ml_train_btn.setObjectName("primaryButton")
        self.ml_train_btn.setMinimumHeight(45)
        self.ml_train_btn.clicked.connect(self.ml_train_model)
        layout.addWidget(self.ml_train_btn)
        
        # MODEL I/O — save the trained model to disk, or load one back
        model_io_group = QGroupBox("TRAINED MODEL")
        model_io_layout = QVBoxLayout(model_io_group)
        
        model_io_row = QHBoxLayout()
        save_model_btn = QPushButton("Save Model…")
        save_model_btn.setToolTip(
            "Save the trained model (with its scaler and feature names) to a .pkl file.\n"
            "Load it later to predict on new spectra without retraining."
        )
        save_model_btn.clicked.connect(self.ml_save_model)
        load_model_btn = QPushButton("Load Model…")
        load_model_btn.setToolTip(
            "Load a previously saved model. After loading, use Predict on New Spectrum "
            "to apply it to fresh measurements."
        )
        load_model_btn.clicked.connect(self.ml_load_model)
        model_io_row.addWidget(save_model_btn)
        model_io_row.addWidget(load_model_btn)
        model_io_layout.addLayout(model_io_row)
        
        # PREDICTION — apply the (possibly just-loaded) model to a new spectrum
        predict_btn = QPushButton("Predict RI on New Spectrum…")
        predict_btn.setToolTip(
            "Load a new OSA file (TXT/Excel/CSV) and predict its RI value using "
            "the currently loaded model. Works on any data the model can align to its features."
        )
        predict_btn.clicked.connect(self.ml_predict_new_spectrum)
        model_io_layout.addWidget(predict_btn)
        
        # Status line for the prediction result
        self.ml_predict_result = QLabel("")
        self.ml_predict_result.setWordWrap(True)
        self.ml_predict_result.setStyleSheet(
            "font-size: 10pt; padding: 6px; "
            f"color: {ModernTheme.DARK['text_primary']};"
        )
        model_io_layout.addWidget(self.ml_predict_result)
        
        # DIAGNOSTICS — surface the worst-fit samples
        worst_btn = QPushButton("Show Worst Predictions")
        worst_btn.setToolTip(
            "List the training/test samples where the model's error is largest. "
            "Useful for spotting bad augmentation copies, mislabeled samples, or "
            "transitional RI regions the model struggles with."
        )
        worst_btn.clicked.connect(self.ml_show_worst_predictions)
        model_io_layout.addWidget(worst_btn)
        
        layout.addWidget(model_io_group)
        
        exportlayout = QHBoxLayout()
        exportdatabtn = QPushButton("Export Dataset (CSV)")
        exportdatabtn.setToolTip("Save the feature dataset (before augmentation) to CSV.")
        exportdatabtn.clicked.connect(self.ml_export_dataset)
        exportresultsbtn = QPushButton("Export Results (XLSX)")
        exportresultsbtn.setToolTip("Save training metrics and feature importance to Excel.")
        exportresultsbtn.clicked.connect(self.ml_export_results)
        exportlayout.addWidget(exportdatabtn)
        exportlayout.addWidget(exportresultsbtn)
        layout.addLayout(exportlayout)
        
        scroll.setWidget(scrollwidget)
        panellayout = QVBoxLayout(panel)
        panellayout.setContentsMargins(0, 0, 0, 0)
        panellayout.addWidget(scroll)
        return panel
    
    def _build_augmentation_config(self):
        """Read augmentation settings from the UI into the dict combined_augmentation expects."""
        return {
            'gaussian_noise': {
                'enabled':     self.ml_aug_noise_cb.isChecked(),
                'noise_level': self.ml_aug_noise_level.value(),
                'n_augment':   self.ml_aug_noise_n.value(),
            },
            'wavelength_shift': {
                'enabled':     self.ml_aug_shift_cb.isChecked(),
                'shift_range': self.ml_aug_shift_range.value(),
                'n_augment':   self.ml_aug_shift_n.value(),
            },
            'power_scaling': {
                'enabled':     self.ml_aug_scale_cb.isChecked(),
                'scale_range': self.ml_aug_scale_range.value(),
                'n_augment':   self.ml_aug_scale_n.value(),
            },
        }
    
    def _update_aug_preview(self):
        """Update the augmentation size preview label."""
        if not self.ml_augment_cb.isChecked():
            self.ml_aug_preview.setText("(disabled — using original data only)")
            return
        if self.ml_dataset is None:
            self.ml_aug_preview.setText("Preview: load dataset first")
            return
        config = self._build_augmentation_config()
        n_orig = len(self.ml_dataset)
        n_total = DataAugmenter.predict_size(n_orig, config)
        factor = n_total / n_orig if n_orig else 0
        self.ml_aug_preview.setText(
            f"Preview: {n_orig} original → {n_total} samples after augmentation ({factor:.1f}×)"
        )
    
    def ml_apply_augmentation(self):
        """Materialize the augmentation into self.ml_dataset (destructive).
        
        After this runs, the dataset has an 'augmentation' label column and
        N_orig × (1 + sum of n_augments) rows. Training-time augmentation should
        usually be turned off afterwards to avoid layering more on top — we
        offer to do that automatically."""
        if self.ml_dataset is None:
            QMessageBox.warning(self, "No Dataset",
                              "Load or build a dataset first.")
            return
        if not self.ml_augment_cb.isChecked():
            QMessageBox.information(
                self, "Augmentation disabled",
                "Enable 'Augment training data' first, then configure which "
                "techniques to apply."
            )
            return
        
        # If the dataset already has augmentation labels, warn before stacking more
        already_augmented = 'augmentation' in self.ml_dataset.columns
        if already_augmented:
            reply = QMessageBox.question(
                self, "Dataset already augmented",
                "This dataset already has an 'augmentation' column. Applying again "
                "will layer more augmentation on top of the existing one.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return
        
        try:
            config = self._build_augmentation_config()
            # Drop existing label column before re-augmenting so we don't get
            # 'augmentation_x' / 'augmentation_y' merge artifacts
            base = self.ml_dataset.copy()
            if 'augmentation' in base.columns:
                base = base.drop(columns=['augmentation'])
            
            augmented = DataAugmenter.combined_augmentation(
                base, target_col='ri_value', config=config, add_labels=True
            )
            
            n_orig = len(base)
            n_new = len(augmented)
            self.ml_dataset = augmented
            
            # Show the augmentation breakdown
            label_counts = augmented['augmentation'].value_counts().to_dict()
            breakdown = "\n".join(f"  • {label}: {n}" for label, n in label_counts.items())
            self.ml_aug_status.setText(
                f"✅ Augmented dataset built: {n_orig} → {n_new} rows\n{breakdown}"
            )
            self.status_bar.showMessage(
                f"✅ Built augmented dataset: {n_orig} → {n_new} rows"
            )
            self._update_aug_preview()
            
            # Offer to turn off training-time augmentation to avoid double-stacking
            reply = QMessageBox.question(
                self, "Augmentation applied",
                f"Built {n_new} rows ({n_new - n_orig} new from augmentation).\n\n"
                "Disable training-time augmentation to avoid applying it again "
                "on the next 'Train Model' click?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.ml_augment_cb.setChecked(False)
        
        except Exception as e:
            QMessageBox.critical(self, "Augmentation Failed",
                               f"Could not build augmented dataset:\n{e}")
            import traceback; traceback.print_exc()
    
    def update_ml_model_info(self, model_name):
        info_map = {
            'Linear Regression': 'Fast linear model, good baseline',
            'Random Forest': 'Ensemble model, handles non-linearity',
            'XGBoost': 'Gradient boosting, high accuracy (requires xgboost)',
            'CatBoost': 'Advanced boosting, robust (requires catboost)',
            'ANN': 'Neural network (MLP). Works on any dataset. Needs torch.',
            'CNN-1D': 'Conv net for raw spectra — needs Wide-format data. Needs torch.',
        }
        self.ml_model_info.setText(info_map.get(model_name, "Select a model"))
    
    # Hyperparameter specs per model. Each entry is a list of (key, label, factory).
    # The factory builds a fresh widget on demand. Keeping this as data instead of
    # repeated if/elif branches lets us add new models with one entry.
    @staticmethod
    def _ml_param_specs():
        def _spinbox(lo, hi, val, special=None):
            def f():
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setValue(val)
                if special is not None:
                    w.setSpecialValueText(special)
                return w
            return f
        
        def _doublespinbox(lo, hi, val, step=0.01, decimals=3):
            def f():
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setValue(val)
                w.setSingleStep(step)
                w.setDecimals(decimals)
                return w
            return f
        
        return {
            'Linear Regression': [],  # no hyperparameters
            'Random Forest': [
                ('n_estimators',      'n_estimators:',      _spinbox(10, 500, 100)),
                ('max_depth',         'max_depth:',         _spinbox(-1, 50, -1, special="None")),
                ('min_samples_split', 'min_samples_split:', _spinbox(2, 20, 2)),
            ],
            'XGBoost': [
                ('n_estimators',  'n_estimators:',  _spinbox(10, 500, 100)),
                ('learning_rate', 'learning_rate:', _doublespinbox(0.001, 1.0, 0.1)),
                ('max_depth',     'max_depth:',     _spinbox(1, 20, 6)),
            ],
            'CatBoost': [
                ('iterations',    'iterations:',    _spinbox(10, 1000, 100)),
                ('learning_rate', 'learning_rate:', _doublespinbox(0.001, 1.0, 0.03)),
                ('depth',         'depth:',         _spinbox(1, 16, 6)),
            ],
            'ANN': [
                ('hidden_layers', 'hidden layers:',  _spinbox(1, 6, 2)),
                ('neurons',       'neurons/layer:',  _spinbox(8, 512, 128)),
                ('dropout',       'dropout:',        _doublespinbox(0.0, 0.8, 0.2)),
                ('learning_rate', 'learning rate:',  _doublespinbox(0.0001, 0.1, 0.001, step=0.0005, decimals=4)),
                ('epochs',        'epochs:',         _spinbox(10, 1000, 100)),
                ('batch_size',    'batch size:',     _spinbox(4, 256, 32)),
            ],
            'CNN-1D': [
                ('conv_filters',  'conv filters:',   _spinbox(8, 128, 32)),
                ('kernel_size',   'kernel size:',    _spinbox(3, 31, 7)),
                ('fc_neurons',    'FC neurons:',     _spinbox(16, 256, 64)),
                ('dropout',       'dropout:',        _doublespinbox(0.0, 0.8, 0.2)),
                ('learning_rate', 'learning rate:',  _doublespinbox(0.0001, 0.1, 0.001, step=0.0005, decimals=4)),
                ('epochs',        'epochs:',         _spinbox(10, 1000, 100)),
                ('batch_size',    'batch size:',     _spinbox(4, 256, 32)),
            ],
        }
    
    def update_manual_param_widgets(self, model_name):
        """Rebuild the hyperparameter form when the model selection changes.
        
        BUGFIX: previously this used a QVBoxLayout of QHBoxLayouts. The cleanup
        loop only called deleteLater() on items whose .widget() was non-None —
        but nested layouts return None, leaving their child QLabels and QSpinBoxes
        as orphaned widgets that stayed visible and overlapped on each model switch.
        QFormLayout.removeRow() properly destroys both the label AND the field.
        """
        # Properly remove every row (destroys label + field widget per row)
        while self.ml_manual_param_form.rowCount() > 0:
            self.ml_manual_param_form.removeRow(0)
        self.ml_manual_param_widgets.clear()
        
        specs = self._ml_param_specs().get(model_name, [])
        if not specs:
            self.ml_param_hint.setText(f"No tunable hyperparameters for {model_name}.")
            self.ml_param_hint.setVisible(True)
            return
        
        self.ml_param_hint.setVisible(False)
        for key, label, factory in specs:
            widget = factory()
            self.ml_manual_param_form.addRow(label, widget)
            self.ml_manual_param_widgets[key] = widget
    
    @staticmethod
    def _clean_external_dataset(df, target_col='ri_value'):
        """Normalize an externally-loaded dataset for ML training.
        
        Handles three real-world issues seen in user CSVs:
          1. Non-numeric metadata columns (categorical labels like 'Ratio_1-2',
             'original'/'noise_0.3dBm'): sklearn cannot use these directly, so we
             drop them. The user is told which columns were dropped.
          2. Yokogawa AQ6370D '-210.0' sentinel values: these mark wavelength
             samples below the noise floor. Treated as missing, not as real -210 dBm
             readings (which would dominate the model).
          3. Any remaining NaN values: filled with the column median (a stable
             choice for spectra where most values are similar).
        
        Returns (cleaned_df, report) where report explains what was done so we
        can surface it in the UI.
        """
        report = {
            'original_shape': df.shape,
            'dropped_columns': [],
            'sentinel_cells': 0,
            'imputed_cells': 0,
        }
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        
        # Coerce target to numeric (in case it was parsed as string)
        df = df.copy()
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isna().any():
            n_bad = int(df[target_col].isna().sum())
            df = df.dropna(subset=[target_col])
            report['rows_dropped_bad_target'] = n_bad
        
        # Separate features from target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # 1. Drop non-numeric feature columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = X.drop(columns=non_numeric)
            report['dropped_columns'] = non_numeric
        
        # 2. Replace Yokogawa -210 sentinel with NaN
        n_sentinel = int((X == -210.0).sum().sum())
        if n_sentinel > 0:
            X = X.replace(-210.0, np.nan)
            report['sentinel_cells'] = n_sentinel
        
        # 3. Impute any remaining NaN with column median
        n_nan = int(X.isna().sum().sum())
        if n_nan > 0:
            # Use median of each column; if a column is entirely NaN, fall back to 0
            medians = X.median()
            medians = medians.fillna(0.0)
            X = X.fillna(medians)
            report['imputed_cells'] = n_nan
        
        result = X.copy()
        result[target_col] = y.values
        report['final_shape'] = result.shape
        return result, report
    
    @staticmethod
    def _format_clean_report(report):
        """Turn a cleanup report into a user-facing message."""
        lines = []
        orig_rows, orig_cols = report['original_shape']
        final_rows, final_cols = report['final_shape']
        lines.append(f"✅ Loaded {final_rows} samples, {final_cols - 1} features")
        
        if report['dropped_columns']:
            cols = report['dropped_columns']
            if len(cols) <= 3:
                lines.append(f"  • Dropped non-numeric: {', '.join(cols)}")
            else:
                lines.append(f"  • Dropped {len(cols)} non-numeric columns "
                            f"(e.g. {', '.join(cols[:2])}…)")
        
        if report.get('rows_dropped_bad_target', 0):
            lines.append(f"  • Dropped {report['rows_dropped_bad_target']} rows with bad RI value")
        
        if report['sentinel_cells']:
            lines.append(f"  • Replaced {report['sentinel_cells']:,} '-210.0' sentinel cells "
                        f"(Yokogawa no-signal marker)")
        
        if report['imputed_cells']:
            lines.append(f"  • Filled {report['imputed_cells']:,} missing cells with column median")
        
        return "\n".join(lines)
    
    def ml_load_dataset(self):
        source = self.ml_source_combo.currentText()
        try:
            if 'Peak Detection' in source:
                if self.current_data is None or not self.current_data.is_loaded():
                    QMessageBox.warning(self, "No Data", "Load OSA data first!")
                    return
                selected_ri = self.get_selected_ri_values()
                if not selected_ri:
                    QMessageBox.warning(self, "No Selection", "Select RI values first!")
                    return
                progress = QProgressDialog("Extracting features...", None, 0, len(selected_ri), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                
                self.ml_dataset = MLDatasetBuilder.build_from_osa_data(self.current_data, selected_ri)
                progress.close()
                self.ml_data_status.setText(
                    f"✅ Engineered features: {len(self.ml_dataset)} samples × "
                    f"{len(self.ml_dataset.columns)-1} features"
                )
                self._update_aug_preview()
            
            elif 'Raw Spectrum' in source:
                if self.current_data is None or not self.current_data.is_loaded():
                    QMessageBox.warning(self, "No Data", "Load OSA data first (tab 1)!")
                    return
                selected_ri = self.get_selected_ri_values()
                if not selected_ri:
                    QMessageBox.warning(self, "No Selection",
                                       "Select RI values first (tab 1, RI Selection panel)!")
                    return
                progress = QProgressDialog("Building wide-format dataset...", None,
                                          0, len(selected_ri), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                
                self.ml_dataset = MLDatasetBuilder.build_wide_from_osa_data(
                    self.current_data, selected_ri
                )
                progress.close()
                
                # Quick stats for the status panel
                n_sentinel = int((self.ml_dataset.select_dtypes(include=[np.number]) == -210.0).sum().sum())
                msg = (
                    f"✅ Wide spectrum dataset:\n"
                    f"   {len(self.ml_dataset)} samples × "
                    f"{len(self.ml_dataset.columns)-1} wavelength features"
                )
                if n_sentinel > 0:
                    msg += f"\n   ({n_sentinel:,} '-210.0' cells will be imputed at training time)"
                self.ml_data_status.setText(msg)
                self._update_aug_preview()
            
            elif 'External File' in source:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Load ML Dataset", "",
                    "Data Files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)"
                )
                if not file_path:
                    return
                
                if file_path.endswith('.csv'):
                    raw_df = pd.read_csv(file_path)
                else:
                    raw_df = pd.read_excel(file_path)
                
                # Resolve the target column — accept several common names
                ri_col = None
                possible_names = ['ri_value', 'RI', 'ri', 'Ri', 'RI_value', 'refractive_index']
                for col_name in possible_names:
                    if col_name in raw_df.columns:
                        ri_col = col_name
                        break
                if ri_col is None:
                    QMessageBox.warning(
                        self, "Invalid Format",
                        f"Dataset must have one of these target columns:\n"
                        f"  {', '.join(possible_names)}\n\n"
                        f"Found: {', '.join(raw_df.columns.tolist()[:6])}…"
                    )
                    return
                if ri_col != 'ri_value':
                    raw_df = raw_df.rename(columns={ri_col: 'ri_value'})
                
                # Clean: drop non-numeric, handle -210 sentinels, impute NaN
                cleaned, report = self._clean_external_dataset(raw_df, target_col='ri_value')
                
                # If cleanup is significant, show a confirmation dialog so the user
                # knows what we changed and can bail out.
                significant = (
                    report['dropped_columns']
                    or report['sentinel_cells'] > 0
                    or report.get('rows_dropped_bad_target', 0) > 0
                )
                if significant:
                    msg = (
                        f"<b>Dataset loaded — preprocessing applied:</b><br><br>"
                        f"{self._format_clean_report(report).replace(chr(10), '<br>')}<br><br>"
                        "Continue with these settings?"
                    )
                    reply = QMessageBox.question(
                        self, "Dataset Preprocessing",
                        msg, QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes
                    )
                    if reply != QMessageBox.Yes:
                        return
                
                self.ml_dataset = cleaned
                self.ml_data_status.setText(self._format_clean_report(report))
                self._update_aug_preview()
            
            self.status_bar.showMessage(f"✅ Dataset ready: {len(self.ml_dataset)} samples")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")
            import traceback; traceback.print_exc()
    
    def ml_save_dataset(self):
        """Save the current ml_dataset to CSV or XLSX. Format chosen by extension."""
        if self.ml_dataset is None:
            QMessageBox.warning(self, "No Dataset",
                              "Load or build a dataset first.")
            return
        
        default_name = "osa_ml_dataset.csv"
        if 'augmentation' in self.ml_dataset.columns:
            default_name = "osa_ml_dataset_augmented.csv"
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset", default_name,
            "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not path:
            return
        
        try:
            if path.lower().endswith('.xlsx'):
                self.ml_dataset.to_excel(path, index=False)
            else:
                if not path.lower().endswith('.csv'):
                    path += '.csv'
                self.ml_dataset.to_csv(path, index=False)
            
            n_rows, n_cols = self.ml_dataset.shape
            self.status_bar.showMessage(
                f"✅ Saved {n_rows} rows × {n_cols} cols to {Path(path).name}"
            )
            QMessageBox.information(self, "Saved",
                                   f"Dataset saved:\n{Path(path).name}\n\n"
                                   f"{n_rows} rows × {n_cols} columns")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save dataset:\n{e}")
    
    def ml_train_model(self):
        if self.ml_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Load dataset first!")
            return
        
        try:
            model_name = self.ml_model_combo.currentText()
            test_size = self.ml_test_size_spin.value()
            scale = self.ml_scale_cb.isChecked()
            
            params = {}
            for param_name, widget in self.ml_manual_param_widgets.items():
                if isinstance(widget, QSpinBox):
                    value = widget.value()
                    if param_name == 'max_depth' and value == -1:
                        value = None
                    params[param_name] = value
                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
            
            # BUGFIX: Use a local copy so self.ml_dataset stays pristine across retrains.
            # Otherwise repeated training with augmentation enabled would augment an
            # already-augmented dataset, growing it geometrically on each click.
            training_dataset = self.ml_dataset
            
            # Apply augmentation if enabled — config now read from UI
            if self.ml_augment_cb.isChecked():
                config = self._build_augmentation_config()
                training_dataset = DataAugmenter.combined_augmentation(
                    self.ml_dataset, target_col='ri_value', config=config
                )
                summary = DataAugmenter.get_augmentation_summary(self.ml_dataset, training_dataset)
                self.ml_aug_status.setText(
                    f"✅ Augmented: {summary['original_samples']} → {summary['total_samples']} "
                    f"samples ({summary['augmentation_factor']:.1f}×)"
                )
            else:
                self.ml_aug_status.setText("")
            
            self.status_bar.showMessage(f"Preparing data...")
            QApplication.processEvents()
            
            self.ml_trainer.prepare_data(
                training_dataset, target_col='ri_value',
                test_size=test_size, scale=scale
            )
            
            self.status_bar.showMessage(f"Training {model_name}...")
            QApplication.processEvents()
            
            is_deep_learning = model_name in ('ANN', 'CNN-1D')
            if is_deep_learning:
                # DL models run for many epochs — show a progress dialog that
                # updates each epoch with the current loss, and let the user cancel.
                # DL params come from the same widget dict; pull out the training
                # controls (epochs, batch_size, learning_rate) and pass the rest
                # as architecture kwargs.
                epochs = params.pop('epochs', 100)
                batch_size = params.pop('batch_size', 32)
                learning_rate = params.pop('learning_rate', 0.001)
                
                progress = QProgressDialog(
                    f"Training {model_name}...\nEpoch 0 / {epochs}",
                    "Cancel", 0, epochs, self
                )
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                
                cancelled = {'flag': False}
                
                def on_epoch(info):
                    if progress.wasCanceled():
                        cancelled['flag'] = True
                        # Raising here would be caught mid-training; instead we
                        # rely on the trainer finishing the current epoch then
                        # we stop reporting. PyTorch loop checks happen via the
                        # callback, so we set the dialog and let it ride out.
                    progress.setValue(info['epoch'])
                    progress.setLabelText(
                        f"Training {model_name}...\n"
                        f"Epoch {info['epoch']} / {info['total_epochs']}\n"
                        f"Train loss: {info['train_loss']:.5f}  "
                        f"Val loss: {info['val_loss']:.5f}\n"
                        f"Best val: {info['best_val_loss']:.5f}"
                    )
                    QApplication.processEvents()
                
                try:
                    metrics = self.ml_trainer.train_deep_learning(
                        model_name, epochs=epochs, batch_size=batch_size,
                        learning_rate=learning_rate, early_stopping=True,
                        patience=15, progress_callback=on_epoch, **params
                    )
                finally:
                    progress.close()
                
                # Surface how many epochs actually ran (early stopping may cut it short)
                epochs_run = metrics.get('epochs_run', epochs)
                if epochs_run < epochs:
                    self.status_bar.showMessage(
                        f"Early-stopped at epoch {epochs_run} (val loss plateaued)"
                    )
            else:
                metrics = self.ml_trainer.train(model_name, **params)
            
            self.ml_metrics_table.setRowCount(8)
            metrics_data = [
                ("Model", metrics['model_name']),
                ("Train R²", f"{metrics['train_r2']:.6f}"),
                ("Test R²", f"{metrics['test_r2']:.6f}"),
                ("Train RMSE", f"{metrics['train_rmse']:.6f}"),
                ("Test RMSE", f"{metrics['test_rmse']:.6f}"),
                ("Overfitting Gap", f"{metrics['overfitting_gap']:.6f}"),
                ("Train Samples", str(metrics['n_train'])),
                ("Test Samples", str(metrics['n_test']))
            ]
            for i, (metric, value) in enumerate(metrics_data):
                self.ml_metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.ml_metrics_table.setItem(i, 1, QTableWidgetItem(value))
            self.ml_metrics_table.resizeColumnsToContents()
            
            self.plot_ml_prediction(metrics)
            importance_df = self.ml_trainer.get_feature_importance()
            if importance_df is not None:
                self.plot_feature_importance(importance_df)
            
            self.status_bar.showMessage(f"✅ Training complete! Test R² = {metrics['test_r2']:.4f}")
            QMessageBox.information(self, "Success", f"✅ Training Complete!\n\nModel: {model_name}\nTest R² = {metrics['test_r2']:.6f}\nTest RMSE = {metrics['test_rmse']:.6f}\nOverfitting Gap = {metrics['overfitting_gap']:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed:\n{e}")
            import traceback
            traceback.print_exc()
    
    def plot_ml_prediction(self, metrics):
        fig = self.ml_plot_canvas.fig
        fig.clear()
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
                      top=0.92, bottom=0.08, left=0.08, right=0.95)
        
        color_train = '#42A5F5'
        color_test = '#66BB6A'
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(metrics['y_train'], metrics['y_train_pred'],
                    alpha=0.6, s=50, color=color_train, edgecolor='white', linewidth=0.5)
        min_val = min(metrics['y_train'].min(), metrics['y_train_pred'].min())
        max_val = max(metrics['y_train'].max(), metrics['y_train_pred'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Actual RI (RIU)', fontsize=10, fontweight='bold', color='white')
        ax1.set_ylabel('Predicted RI (RIU)', fontsize=10, fontweight='bold', color='white')
        ax1.set_title(f"Training Set (R²={metrics['train_r2']:.4f})",
                      fontsize=11, fontweight='bold', color='white')
        ax1.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax1.legend(loc='best', facecolor='#2D2D2D', edgecolor='white', labelcolor='white', fontsize=8)
        ax1.set_facecolor('#1E1E1E')
        ax1.tick_params(colors='white', labelsize=9)
        for spine in ax1.spines.values():
            spine.set_color('white')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(metrics['y_test'], metrics['y_test_pred'],
                    alpha=0.6, s=50, color=color_test, edgecolor='white', linewidth=0.5)
        min_val = min(metrics['y_test'].min(), metrics['y_test_pred'].min())
        max_val = max(metrics['y_test'].max(), metrics['y_test_pred'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax2.set_xlabel('Actual RI (RIU)', fontsize=10, fontweight='bold', color='white')
        ax2.set_ylabel('Predicted RI (RIU)', fontsize=10, fontweight='bold', color='white')
        ax2.set_title(f"Testing Set (R²={metrics['test_r2']:.4f})",
                      fontsize=11, fontweight='bold', color='white')
        ax2.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax2.legend(loc='best', facecolor='#2D2D2D', edgecolor='white', labelcolor='white', fontsize=8)
        ax2.set_facecolor('#1E1E1E')
        ax2.tick_params(colors='white', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color('white')
        
        ax3 = fig.add_subplot(gs[1, 0])
        train_residuals = metrics['y_train'] - metrics['y_train_pred']
        ax3.scatter(metrics['y_train_pred'], train_residuals,
                    alpha=0.6, s=50, color=color_train, edgecolor='white', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Predicted RI (RIU)', fontsize=10, fontweight='bold', color='white')
        ax3.set_ylabel('Residuals', fontsize=10, fontweight='bold', color='white')
        ax3.set_title(f"Training Residuals (RMSE={metrics['train_rmse']:.4f})",
                      fontsize=11, fontweight='bold', color='white')
        ax3.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax3.set_facecolor('#1E1E1E')
        ax3.tick_params(colors='white', labelsize=9)
        for spine in ax3.spines.values():
            spine.set_color('white')
        
        ax4 = fig.add_subplot(gs[1, 1])
        # For deep-learning models, the 4th panel shows the training loss curve
        # (more informative than test residuals — you can see convergence and
        # whether early stopping kicked in). Tree/linear models keep residuals.
        loss_history = metrics.get('loss_history')
        if loss_history and loss_history.get('train_loss'):
            epochs_x = range(1, len(loss_history['train_loss']) + 1)
            ax4.plot(epochs_x, loss_history['train_loss'],
                     color=color_train, lw=1.5, label='Train loss')
            ax4.plot(epochs_x, loss_history['val_loss'],
                     color=color_test, lw=1.5, label='Val loss')
            ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold', color='white')
            ax4.set_ylabel('MSE loss', fontsize=10, fontweight='bold', color='white')
            ax4.set_title(f"Training Curve ({metrics.get('epochs_run', '?')} epochs)",
                          fontsize=11, fontweight='bold', color='white')
            ax4.legend(loc='best', facecolor='#2D2D2D', edgecolor='white',
                       labelcolor='white', fontsize=8)
            ax4.set_yscale('log')  # loss usually spans orders of magnitude
        else:
            test_residuals = metrics['y_test'] - metrics['y_test_pred']
            ax4.scatter(metrics['y_test_pred'], test_residuals,
                        alpha=0.6, s=50, color=color_test, edgecolor='white', linewidth=0.5)
            ax4.axhline(y=0, color='r', linestyle='--', lw=2)
            ax4.set_xlabel('Predicted RI (RIU)', fontsize=10, fontweight='bold', color='white')
            ax4.set_ylabel('Residuals', fontsize=10, fontweight='bold', color='white')
            ax4.set_title(f"Testing Residuals (RMSE={metrics['test_rmse']:.4f})",
                          fontsize=11, fontweight='bold', color='white')
        ax4.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax4.set_facecolor('#1E1E1E')
        ax4.tick_params(colors='white', labelsize=9)
        for spine in ax4.spines.values():
            spine.set_color('white')
        
        fig.suptitle(f"ML Prediction: {metrics['model_name']}",
                     fontsize=12, fontweight='bold', color='white', y=0.98)
        self.ml_plot_canvas.draw()
    
    def plot_feature_importance(self, importance_df):
        fig = self.ml_importance_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        top_n = min(10, len(importance_df))
        df_plot = importance_df.head(top_n)
        bars = ax.barh(df_plot['feature'], df_plot['importance'],
                       color='#42A5F5', edgecolor='white', linewidth=0.8)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold', color='white')
        ax.set_title(f'Top {top_n} Feature Importance',
                     fontsize=13, fontweight='bold', color='white')
        ax.grid(True, axis='x', alpha=0.3, color='gray', linestyle='--')
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
        fig.tight_layout()
        self.ml_importance_canvas.draw()
    
    def ml_export_dataset(self):
        if self.ml_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Load dataset first!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Dataset", "ml_dataset.csv", "CSV (*.csv)")
        if file_path:
            try:
                self.ml_dataset.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Exported:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    # ========== Model I/O, Prediction, Diagnostics ==========
    
    def ml_save_model(self):
        """Save the trained model to a .pkl file via dialog."""
        if self.ml_trainer.model is None:
            QMessageBox.warning(self, "No Model",
                              "Train a model first before saving.")
            return
        
        default_name = "osa_model.pkl"
        model_name = self.ml_trainer.metrics.get('model_name', 'model') if self.ml_trainer.metrics else 'model'
        default_name = f"osa_{model_name.lower().replace(' ', '_')}.pkl"
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model", default_name,
            "Pickle Model (*.pkl);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith('.pkl'):
            path += '.pkl'
        
        try:
            self.ml_trainer.save_model(path)
            n_features = len(self.ml_trainer.feature_names) if self.ml_trainer.feature_names else 0
            self.status_bar.showMessage(f"✅ Model saved: {Path(path).name}")
            QMessageBox.information(
                self, "Model saved",
                f"Saved to:\n{Path(path).name}\n\n"
                f"Model: {model_name}\n"
                f"Features: {n_features}\n"
                f"Includes: model, scaler, feature names, metrics"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save model:\n{e}")
    
    def ml_load_model(self):
        """Load a saved model from disk, replacing the current trainer state."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trained Model", "",
            "Pickle Model (*.pkl);;All Files (*)"
        )
        if not path:
            return
        
        try:
            metrics = self.ml_trainer.load_model(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed",
                               f"Could not load model:\n{e}")
            return
        
        model_name = metrics.get('model_name', 'unknown') if metrics else 'unknown'
        n_features = len(self.ml_trainer.feature_names) if self.ml_trainer.feature_names else 0
        test_r2 = metrics.get('test_r2') if metrics else None
        
        # If the loaded model came with metrics, surface them in the results table
        # so the user sees the same context they'd see after a fresh train run.
        if metrics:
            try:
                self._populate_ml_metrics_table(metrics)
            except Exception:
                pass
        
        msg_parts = [
            f"Loaded: {Path(path).name}",
            f"Model: {model_name}",
            f"Features: {n_features}",
        ]
        if test_r2 is not None:
            msg_parts.append(f"Original Test R²: {test_r2:.4f}")
        
        self.ml_predict_result.setText("✅ Model loaded — ready to predict")
        self.status_bar.showMessage(f"✅ Loaded model: {model_name} ({n_features} features)")
        QMessageBox.information(self, "Model loaded", "\n".join(msg_parts))
    
    def _populate_ml_metrics_table(self, metrics):
        """Re-fill the ML metrics table from a metrics dict. Used after both
        training and model-loading so the right-hand panel reflects the
        currently-active model."""
        if not hasattr(self, 'ml_metrics_table'):
            return
        rows = [
            ('Model',      metrics.get('model_name', '—')),
            ('Train R²',   f"{metrics.get('train_r2', float('nan')):.4f}"),
            ('Test R²',    f"{metrics.get('test_r2', float('nan')):.4f}"),
            ('Train MAE',  f"{metrics.get('train_mae', float('nan')):.5f}"),
            ('Test MAE',   f"{metrics.get('test_mae', float('nan')):.5f}"),
            ('Train RMSE', f"{metrics.get('train_rmse', float('nan')):.5f}"),
            ('Test RMSE',  f"{metrics.get('test_rmse', float('nan')):.5f}"),
        ]
        self.ml_metrics_table.setRowCount(len(rows))
        self.ml_metrics_table.setColumnCount(2)
        self.ml_metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        for i, (k, v) in enumerate(rows):
            self.ml_metrics_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.ml_metrics_table.setItem(i, 1, QTableWidgetItem(str(v)))
        self.ml_metrics_table.resizeColumnsToContents()
    
    def ml_predict_new_spectrum(self):
        """Load a new spectrum file and predict its RI value using the current model."""
        if self.ml_trainer.model is None:
            QMessageBox.warning(self, "No Model",
                              "Train a model or load one first before predicting.")
            return
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum for Prediction", "",
            "Spectrum Files (*.csv *.xlsx *.xls *.txt);;All Files (*)"
        )
        if not path:
            return
        
        # Build a DataFrame from the input file. Three cases:
        #   - CSV/XLSX: assume already in feature format (one row per spectrum,
        #     columns matching the model's feature names). Predict all rows.
        #   - TXT: a single Yokogawa-style spectrum file. Convert to one row
        #     using the same wavelength range the user set on the Data tab.
        try:
            ext = Path(path).suffix.lower()
            if ext in ('.csv',):
                df = pd.read_csv(path)
            elif ext in ('.xlsx', '.xls'):
                df = pd.read_excel(path)
            elif ext == '.txt':
                # Single-spectrum TXT — build a one-row wide DataFrame
                with open(path, 'r') as f:
                    content = f.read().strip()
                power_values = TXTParser._parse_txt_content(content)
                if len(power_values) < 2:
                    raise ValueError(f"TXT file has only {len(power_values)} data points.")
                wl_start = self.txt_wl_start_spin.value()
                wl_end = self.txt_wl_end_spin.value()
                if wl_start >= wl_end:
                    raise ValueError("TXT wavelength Start must be less than End "
                                    "(check the Data tab settings).")
                wavelength = TXTParser.generate_wavelength_array(
                    start=wl_start, end=wl_end, num_points=len(power_values)
                )
                # Column names mirror MLDatasetBuilder.build_wide_from_osa_data
                wl_cols = [f"WL_{w:.2f}" for w in wavelength]
                df = pd.DataFrame([power_values], columns=wl_cols)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            QMessageBox.critical(self, "Could not read file",
                               f"Failed to load spectrum:\n{e}")
            return
        
        # Drop the target column if it's present (e.g. user passed labeled data)
        for tgt in ('ri_value', 'RI', 'ri', 'refractive_index'):
            if tgt in df.columns:
                df = df.drop(columns=[tgt])
        
        try:
            predictions, info = self.ml_trainer.predict_dataframe(df)
        except ValueError as e:
            QMessageBox.warning(self, "Prediction Failed", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Unexpected error:\n{e}")
            return
        
        # Format output: single value if one row, summary if many
        n = len(predictions)
        lines = [f"<b>File:</b> {Path(path).name}"]
        if n == 1:
            lines.append(f"<b>Predicted RI:</b> {predictions[0]:.5f}")
        else:
            lines.append(f"<b>Rows predicted:</b> {n}")
            lines.append(f"<b>RI range:</b> {predictions.min():.5f} – {predictions.max():.5f}")
            lines.append(f"<b>Mean:</b> {predictions.mean():.5f} "
                        f"(std {predictions.std():.5f})")
        
        notes = []
        if info['dropped_extra']:
            d = info['dropped_extra']
            label = d[0] if len(d) == 1 else f"{len(d)} non-feature columns"
            notes.append(f"Ignored {label}")
        if info['sentinel_cells']:
            notes.append(f"{info['sentinel_cells']:,} '-210.0' cells imputed")
        if info['imputed_cells']:
            notes.append(f"{info['imputed_cells']:,} missing cells imputed")
        if notes:
            lines.append("<i>(" + "; ".join(notes) + ")</i>")
        
        self.ml_predict_result.setText("<br>".join(lines))
        
        # For multi-row predictions, also offer to save them
        if n > 1:
            reply = QMessageBox.question(
                self, "Save predictions?",
                f"Predicted {n} spectra. Save predictions to CSV?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Predictions",
                    f"{Path(path).stem}_predictions.csv",
                    "CSV (*.csv)"
                )
                if save_path:
                    if not save_path.lower().endswith('.csv'):
                        save_path += '.csv'
                    out_df = pd.DataFrame({
                        'row': range(n),
                        'predicted_ri': predictions,
                    })
                    out_df.to_csv(save_path, index=False)
                    self.status_bar.showMessage(
                        f"✅ Predictions saved: {Path(save_path).name}"
                    )
    
    def ml_show_worst_predictions(self):
        """Pop up a table of the worst-fit samples from the last training run."""
        if not self.ml_trainer.metrics:
            QMessageBox.warning(self, "No Results",
                              "Train a model first to see worst-fit samples.")
            return
        
        try:
            worst_df = self.ml_trainer.worst_predictions(n=15)
        except ValueError as e:
            QMessageBox.warning(self, "Diagnostic Unavailable", str(e))
            return
        
        # Build a modal dialog with a QTableWidget and a Save CSV button.
        # Using a dialog (not a new tab) keeps this a transient diagnostic view —
        # power users can dismiss it once they've found the bad samples.
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Worst predictions — top 15 by absolute error")
        dlg.resize(560, 440)
        v = QVBoxLayout(dlg)
        
        hdr = QLabel(
            "These are the samples where the model's prediction differs most from "
            "the true RI. Large errors can indicate: (a) mislabeled samples, "
            "(b) augmented copies that drifted too far, or (c) transitional RI "
            "regions the model handles poorly."
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet("font-size: 10pt; color: #aaa; padding: 4px 0;")
        v.addWidget(hdr)
        
        table = QTableWidget()
        table.setRowCount(len(worst_df))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['split', 'index', 'true RI', 'predicted RI'])
        # Add a 5th column for absolute error (highlighted)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(['split', 'index', 'true RI', 'predicted RI', 'abs error'])
        for i, row in worst_df.iterrows():
            table.setItem(i, 0, QTableWidgetItem(str(row['split'])))
            table.setItem(i, 1, QTableWidgetItem(str(int(row['index']))))
            table.setItem(i, 2, QTableWidgetItem(f"{row['true_ri']:.5f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{row['predicted_ri']:.5f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{row['abs_error']:.5f}"))
        table.resizeColumnsToContents()
        table.setSelectionBehavior(QTableWidget.SelectRows)
        v.addWidget(table)
        
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        def on_save():
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save Worst Predictions",
                "worst_predictions.csv", "CSV (*.csv)"
            )
            if path:
                if not path.lower().endswith('.csv'):
                    path += '.csv'
                worst_df.to_csv(path, index=False)
                self.status_bar.showMessage(
                    f"✅ Worst predictions saved: {Path(path).name}"
                )
        btns.accepted.connect(on_save)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        
        dlg.exec_()
    
    def ml_export_results(self):
        if self.ml_trainer.metrics is None:
            QMessageBox.warning(self, "No Results", "Train a model first!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "ml_results.xlsx", "Excel (*.xlsx)")
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    metrics_df = pd.DataFrame([{
                        'Model': self.ml_trainer.metrics['model_name'],
                        'Train R²': self.ml_trainer.metrics['train_r2'],
                        'Test R²': self.ml_trainer.metrics['test_r2'],
                        'Train MAE': self.ml_trainer.metrics['train_mae'],
                        'Test MAE': self.ml_trainer.metrics['test_mae'],
                        'Train RMSE': self.ml_trainer.metrics['train_rmse'],
                        'Test RMSE': self.ml_trainer.metrics['test_rmse'],
                    }])
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    importance_df = self.ml_trainer.get_feature_importance()
                    if importance_df is not None:
                        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
                QMessageBox.information(self, "Success", f"Exported:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

# ==================== MAIN FUNCTION ====================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OSA Analyzer Pro")
    app.setApplicationVersion("6.1")
    
    ModernTheme.apply_dark_theme(app)
    
    window = OSAAnalyzerPro()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

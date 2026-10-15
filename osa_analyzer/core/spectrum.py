"""Spectrum containers.

A measurement is one :class:`Spectrum`: a wavelength axis, a power trace and
an optional refractive-index label. Repeated measurements at the same RI are
separate spectra with distinct labels (``1.46``, ``1.46_repeat``, ...). v7
keyed spectra by RI, so repeats silently overwrote each other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Iterator

import numpy as np

# Yokogawa AQ6370 writes this value where the signal is below the noise floor.
# It is converted to NaN on load so no later stage can mistake it for data.
SENTINEL_DBM = -210.0


@dataclass(frozen=True, eq=False)
class Spectrum:
    label: str
    wavelength: np.ndarray      # nm, strictly increasing
    power: np.ndarray           # dBm (or dB after reference normalisation)
    ri: float | None = None     # None = unlabelled (air, reference, unknown)
    source: str = ""
    unit: str = "dBm"
    n_sentinel: int = 0

    @classmethod
    def create(cls, label: str, wavelength, power, ri: float | None = None,
               source: str = "", unit: str = "dBm") -> "Spectrum":
        wl = np.asarray(wavelength, dtype=float).ravel()
        p = np.asarray(power, dtype=float).ravel().copy()
        if wl.shape != p.shape:
            raise ValueError(
                f"{label}: wavelength has {wl.size} points but power has {p.size}")
        if wl.size < 3:
            raise ValueError(f"{label}: need at least 3 points, got {wl.size}")
        sentinel = p <= SENTINEL_DBM + 1e-9
        n_sentinel = int(sentinel.sum())
        p[sentinel] = np.nan
        if np.any(np.diff(wl) <= 0):
            order = np.argsort(wl, kind="stable")
            wl, p = wl[order], p[order]
            if np.any(np.diff(wl) == 0):
                raise ValueError(f"{label}: wavelength axis has duplicate values")
        return cls(label, wl, p, None if ri is None else float(ri),
                   source, unit, n_sentinel)

    @property
    def is_labeled(self) -> bool:
        return self.ri is not None

    @property
    def step_nm(self) -> float:
        return float(np.median(np.diff(self.wavelength)))

    def window(self, lo: float | None, hi: float | None):
        """(wavelength, power) restricted to lo <= wl <= hi."""
        mask = np.ones(self.wavelength.size, dtype=bool)
        if lo is not None:
            mask &= self.wavelength >= lo
        if hi is not None:
            mask &= self.wavelength <= hi
        return self.wavelength[mask], self.power[mask]

    def with_power(self, power, unit: str | None = None) -> "Spectrum":
        return replace(self, power=np.asarray(power, dtype=float),
                       unit=self.unit if unit is None else unit)


class SpectrumSet:
    """Ordered collection of spectra with unique labels."""

    def __init__(self, spectra: Iterable[Spectrum] = ()):
        self._items: list[Spectrum] = []
        seen: dict[str, int] = {}
        for s in spectra:
            label = s.label
            if label in seen:
                seen[label] += 1
                label = f"{s.label} #{seen[s.label]}"
                s = replace(s, label=label)
            seen.setdefault(label, 1)
            self._items.append(s)
        self._by_label = {s.label: s for s in self._items}

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Spectrum]:
        return iter(self._items)

    def __getitem__(self, label: str) -> Spectrum:
        return self._by_label[label]

    def __contains__(self, label: str) -> bool:
        return label in self._by_label

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self._items]

    def subset(self, labels: Iterable[str]) -> "SpectrumSet":
        wanted = set(labels)
        return SpectrumSet(s for s in self._items if s.label in wanted)

    def labeled(self) -> list[Spectrum]:
        return [s for s in self._items if s.is_labeled]

    def ri_values(self) -> list[float]:
        return sorted({s.ri for s in self._items if s.ri is not None})

    def same_axis(self) -> bool:
        if len(self._items) < 2:
            return True
        ref = self._items[0].wavelength
        return all(s.wavelength.size == ref.size
                   and np.allclose(s.wavelength, ref, atol=1e-6)
                   for s in self._items[1:])

    def summary(self) -> dict:
        if not self._items:
            return {"n_spectra": 0}
        wl_min = min(float(s.wavelength[0]) for s in self._items)
        wl_max = max(float(s.wavelength[-1]) for s in self._items)
        ris = self.ri_values()
        counts: dict[float, int] = {}
        for s in self._items:
            if s.ri is not None:
                counts[s.ri] = counts.get(s.ri, 0) + 1
        return {
            "n_spectra": len(self._items),
            "n_labeled": sum(1 for s in self._items if s.is_labeled),
            "n_unlabeled": sum(1 for s in self._items if not s.is_labeled),
            "n_points": self._items[0].wavelength.size,
            "wavelength_range": (wl_min, wl_max),
            "step_nm": self._items[0].step_nm,
            "ri_values": ris,
            "ri_range": (ris[0], ris[-1]) if ris else None,
            "n_repeated_ri": sum(1 for c in counts.values() if c > 1),
            "n_sentinel": sum(s.n_sentinel for s in self._items),
            "same_axis": self.same_axis(),
        }

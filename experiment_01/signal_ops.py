"""
signal_ops.py
-------------
Pure, reusable implementations of the basic discrete-time signal operations.

A discrete-time signal is stored as a ``DTSignal``: a sample vector ``x`` *and*
its index vector ``n``.  Keeping ``n`` explicit is what makes shifting, folding
and convolution mathematically correct -- the support of the signal moves,
instead of samples being silently truncated or zero-padded at the edges.

No function here does any I/O or plotting, so the same code is used by the
batch pipeline (run_experiment.py), the GUI (gui.py) and the unit tests.
"""

from dataclasses import dataclass, field

import numpy as np


# ==========================================================================
# Signal container
# ==========================================================================
@dataclass
class DTSignal:
    n: np.ndarray                 # integer sample indices (strictly increasing, step 1)
    x: np.ndarray                 # sample values x[n]
    name: str = "x[n]"
    unit: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n = np.asarray(self.n, dtype=int)
        self.x = np.asarray(self.x, dtype=float)
        if self.n.shape != self.x.shape:
            raise ValueError("n and x must have the same length")

    @classmethod
    def from_array(cls, values, n0: int = 0, name: str = "x[n]", unit: str = ""):
        values = np.asarray(values, dtype=float)
        return cls(np.arange(n0, n0 + len(values)), values, name, unit)

    def __len__(self):
        return len(self.x)

    @property
    def support(self):
        """(first index, last index) of the stored samples."""
        return int(self.n[0]), int(self.n[-1])

    def value_at(self, n: int) -> float:
        """x[n], returning 0 outside the stored support (the DT convention)."""
        idx = n - self.n[0]
        return float(self.x[idx]) if 0 <= idx < len(self.x) else 0.0


# ==========================================================================
# Helpers
# ==========================================================================
def _align(s1: DTSignal, s2: DTSignal):
    """Place both signals on the union of their supports (zero outside each support)."""
    n_lo = min(s1.n[0], s2.n[0])
    n_hi = max(s1.n[-1], s2.n[-1])
    n = np.arange(n_lo, n_hi + 1)
    x1 = np.zeros(len(n))
    x2 = np.zeros(len(n))
    x1[s1.n - n_lo] = s1.x
    x2[s2.n - n_lo] = s2.x
    return n, x1, x2


# ==========================================================================
# 1. Arithmetic operations (sample-by-sample)
# ==========================================================================
def add(s1: DTSignal, s2: DTSignal, name="x1[n] + x2[n]") -> DTSignal:
    """y[n] = x1[n] + x2[n]"""
    n, x1, x2 = _align(s1, s2)
    return DTSignal(n, x1 + x2, name, s1.unit)


def subtract(s1: DTSignal, s2: DTSignal, name="x1[n] - x2[n]") -> DTSignal:
    """y[n] = x1[n] - x2[n]"""
    n, x1, x2 = _align(s1, s2)
    return DTSignal(n, x1 - x2, name, s1.unit)


def multiply(s1: DTSignal, s2: DTSignal, name="x1[n] · x2[n]", unit=None) -> DTSignal:
    """y[n] = x1[n] · x2[n]"""
    n, x1, x2 = _align(s1, s2)
    return DTSignal(n, x1 * x2, name, s1.unit if unit is None else unit)


def amplitude_scale(s: DTSignal, A: float, name=None) -> DTSignal:
    """y[n] = A · x[n]"""
    return DTSignal(s.n.copy(), A * s.x, name or f"{A:g}·x[n]", s.unit)


# ==========================================================================
# 2. Operations on the independent variable n
# ==========================================================================
def time_shift(s: DTSignal, k: int, name=None) -> DTSignal:
    """y[n] = x[n - k].  k > 0 -> delay (shift right), k < 0 -> advance (shift left)."""
    if name is None:
        name = "x[n]" if k == 0 else (f"x[n - {k}]" if k > 0 else f"x[n + {-k}]")
    return DTSignal(s.n + k, s.x.copy(), name, s.unit)


def time_reverse(s: DTSignal, name="x[-n]") -> DTSignal:
    """y[n] = x[-n]  (folding about n = 0)."""
    return DTSignal(-s.n[::-1], s.x[::-1].copy(), name, s.unit)


def downsample(s: DTSignal, D: int, name=None) -> DTSignal:
    """y[n] = x[Dn]  (time compression / decimation without anti-alias filtering)."""
    if D < 1:
        raise ValueError("D must be a positive integer")
    keep = s.n % D == 0
    return DTSignal(s.n[keep] // D, s.x[keep].copy(), name or f"x[{D}n]", s.unit)


def upsample(s: DTSignal, L: int, name=None) -> DTSignal:
    """y[n] = x[n/L] if n is a multiple of L, else 0  (time expansion by zero insertion)."""
    if L < 1:
        raise ValueError("L must be a positive integer")
    n = np.arange(s.n[0] * L, s.n[-1] * L + 1)
    y = np.zeros(len(n))
    y[(s.n * L) - n[0]] = s.x
    return DTSignal(n, y, name or f"x[n/{L}]", s.unit)


def linear_interpolate(s_up: DTSignal, L: int, name=None) -> DTSignal:
    """Fill the zeros produced by ``upsample`` by straight lines between the original samples.

    Equivalent to convolving the zero-stuffed signal with the triangular kernel
    h[n] = 1 - |n|/L,  |n| < L.
    """
    h = 1.0 - np.abs(np.arange(-L + 1, L)) / L
    y = np.convolve(s_up.x, h)[L - 1: L - 1 + len(s_up.x)]
    return DTSignal(s_up.n.copy(), y, name or f"interp_{L}(x)", s_up.unit)


def even_odd_parts(s: DTSignal):
    """x_e[n] = (x[n] + x[-n]) / 2,   x_o[n] = (x[n] - x[-n]) / 2."""
    folded = time_reverse(s)
    n, x, xr = _align(s, folded)
    return (DTSignal(n, 0.5 * (x + xr), "x_e[n]", s.unit),
            DTSignal(n, 0.5 * (x - xr), "x_o[n]", s.unit))


# ==========================================================================
# 3. Convolution
# ==========================================================================
def convolve(s: DTSignal, h: DTSignal, name=None) -> DTSignal:
    """y[n] = sum_k x[k] h[n - k].

    Output support starts at n_x0 + n_h0 and has length N + M - 1.
    """
    y = np.convolve(s.x, h.x)
    n0 = s.n[0] + h.n[0]
    return DTSignal(np.arange(n0, n0 + len(y)), y, name or "x[n] * h[n]", s.unit)


def moving_average_kernel(M: int) -> DTSignal:
    """Causal M-point moving-average impulse response h[n] = 1/M, 0 <= n <= M-1."""
    if M < 1:
        raise ValueError("M must be a positive integer")
    return DTSignal(np.arange(M), np.full(M, 1.0 / M), f"h[n] (M={M})")


def unit_impulse(n0: int = 0) -> DTSignal:
    """delta[n - n0]"""
    return DTSignal(np.array([n0]), np.array([1.0]), f"δ[n-{n0}]")


# ==========================================================================
# 4. Signal measures
# ==========================================================================
def energy(s: DTSignal) -> float:
    """E = sum |x[n]|^2"""
    return float(np.sum(s.x ** 2))


def average_power(s: DTSignal) -> float:
    """P = E / N  over the stored support."""
    return energy(s) / len(s)


def signal_stats(s: DTSignal) -> dict:
    """Summary statistics shown in the result tables and the GUI."""
    lo, hi = s.support
    return {
        "signal": s.name,
        "N": len(s),
        "support": f"[{lo}, {hi}]",
        "mean": float(np.mean(s.x)),
        "std": float(np.std(s.x)),
        "min": float(np.min(s.x)),
        "max": float(np.max(s.x)),
        "energy": energy(s),
        "power": average_power(s),
    }


# ==========================================================================
# Registry used by the GUI to list operations with their defining equations
# ==========================================================================
OPERATIONS = {
    "Addition":           "y[n] = x₁[n] + x₂[n]",
    "Subtraction":        "y[n] = x₁[n] − x₂[n]",
    "Multiplication":     "y[n] = x₁[n] · x₂[n]",
    "Amplitude scaling":  "y[n] = A · x₁[n]",
    "Time shift":         "y[n] = x₁[n − k]",
    "Time reversal":      "y[n] = x₁[−n]",
    "Downsampling":       "y[n] = x₁[Dn]",
    "Upsampling":         "y[n] = x₁[n/L]  (zeros between samples)",
    "Convolution (MA)":   "y[n] = Σ x₁[k] h[n − k],  h[n] = 1/M",
}


def apply_operation(op: str, s1: DTSignal, s2: DTSignal, params: dict) -> DTSignal:
    """Dispatch an operation by name (used by the GUI)."""
    if op == "Addition":
        return add(s1, s2)
    if op == "Subtraction":
        return subtract(s1, s2)
    if op == "Multiplication":
        return multiply(s1, s2, unit=f"{s1.unit}·{s2.unit}")
    if op == "Amplitude scaling":
        return amplitude_scale(s1, params["A"])
    if op == "Time shift":
        return time_shift(s1, params["k"])
    if op == "Time reversal":
        return time_reverse(s1)
    if op == "Downsampling":
        return downsample(s1, params["D"])
    if op == "Upsampling":
        return upsample(s1, params["L"])
    if op == "Convolution (MA)":
        return convolve(s1, moving_average_kernel(params["M"]), name=f"x[n] * h[n], M={params['M']}")
    raise ValueError(f"Unknown operation: {op}")

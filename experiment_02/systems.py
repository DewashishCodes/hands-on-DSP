"""
systems.py
----------
The discrete-time systems under test, plus generic LTI property tests.

Each system maps a finite input x[n], n = 0 ... N-1, to an output y[n] of the
same length, starting from rest (zero initial conditions).  The test functions
treat a system as a black box, exactly as an engineer would when checking an
unknown device, so they work for any callable including a user-defined
difference equation.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import lfilter

import config


# ==========================================================================
# 1. Systems
# ==========================================================================
def moving_average(x, M=config.MA_LENGTH_M):
    """y[n] = (1/M) * sum_{k=0}^{M-1} x[n-k]   (causal, zero initial conditions)."""
    return lfilter(np.full(int(M), 1.0 / int(M)), [1.0], x)


def ewma(x, alpha=config.EWMA_ALPHA):
    """y[n] = alpha * x[n] + (1 - alpha) * y[n-1],   y[-1] = 0."""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x)


def aqi_pm25(x, breakpoints=config.AQI_PM25_BREAKPOINTS):
    """CPCB sub-index: piecewise-linear map of concentration to AQI (memoryless, nonlinear)."""
    x = np.clip(np.asarray(x, dtype=float), 0, breakpoints[-1][1])
    c_lo, c_hi, i_lo, i_hi = (np.array(col, dtype=float) for col in zip(*breakpoints))
    return np.interp(x, np.r_[c_lo[0], c_hi], np.r_[i_lo[0], i_hi])


def drifting_sensor(x, beta=config.DRIFT_BETA):
    """y[n] = (1 + beta * n) * x[n]   (gain that drifts with time: linear, time-varying)."""
    n = np.arange(len(x))
    return (1.0 + beta * n) * np.asarray(x, dtype=float)


def difference_equation(x, b=(1.0,), a=(1.0,)):
    """sum_k a_k y[n-k] = sum_k b_k x[n-k]   (general LTI system, zero initial conditions)."""
    return lfilter(np.asarray(b, float), np.asarray(a, float), x)


@dataclass
class System:
    name: str
    func: Callable
    equation: str
    application: str
    expected_linear: bool
    expected_ti: bool
    params: dict = field(default_factory=dict)

    def __call__(self, x):
        return self.func(np.asarray(x, dtype=float), **self.params)


def default_systems():
    return {
        "S1": System("24-h moving average", moving_average,
                     "y[n] = (1/M) Σ x[n−k],  k = 0 … M−1",
                     "NAAQS 24-hour average of PM2.5", True, True, {"M": config.MA_LENGTH_M}),
        "S2": System("Exponential smoother", ewma,
                     "y[n] = α x[n] + (1−α) y[n−1]",
                     "Display filter of a low-cost sensor", True, True, {"alpha": config.EWMA_ALPHA}),
        "S3": System("AQI sub-index", aqi_pm25,
                     "y[n] = I(x[n]),  piecewise linear",
                     "CPCB National Air Quality Index", False, True),
        "S4": System("Drifting sensor", drifting_sensor,
                     "y[n] = (1 + βn) x[n]",
                     "Uncalibrated sensor whose gain drifts", True, False, {"beta": config.DRIFT_BETA}),
    }


# ==========================================================================
# 2. Property tests (black-box)
# ==========================================================================
def delay(x, k):
    """x[n - k] on n = 0 ... N-1+k (k zeros inserted at the start, k >= 0)."""
    return np.r_[np.zeros(int(k)), np.asarray(x, dtype=float)]


def relative_error(lhs, rhs):
    """|| lhs - rhs || / || rhs ||  (Euclidean norm)."""
    denom = np.linalg.norm(rhs)
    return float(np.linalg.norm(lhs - rhs) / denom) if denom else float(np.linalg.norm(lhs))


def linearity_test(system, x1, x2, a=config.TEST_A, b=config.TEST_B):
    """Superposition: T{a x1 + b x2}  vs  a T{x1} + b T{x2}."""
    lhs = system(a * np.asarray(x1) + b * np.asarray(x2))
    rhs = a * system(x1) + b * system(x2)
    err = relative_error(lhs, rhs)
    return {"lhs": lhs, "rhs": rhs, "error": err, "holds": err < config.TOLERANCE}


def time_invariance_test(system, x, k=config.TEST_SHIFT_K):
    """Shift test: T{x[n-k]}  vs  y[n-k]."""
    lhs = system(delay(x, k))
    rhs = delay(system(x), k)
    err = relative_error(lhs, rhs)
    return {"lhs": lhs, "rhs": rhs, "error": err, "holds": err < config.TOLERANCE, "k": k}


def impulse_response(system, length=config.RESPONSE_LENGTH):
    """h[n] = T{δ[n]}."""
    d = np.zeros(length)
    d[0] = 1.0
    return system(d)


def step_response(system, length=config.RESPONSE_LENGTH):
    """s[n] = T{u[n]}."""
    return system(np.ones(length))


def convolution_check(system, x):
    """For an LTI system the output must equal x[n] * h[n] (truncated to N samples)."""
    h = impulse_response(system, len(x))
    y_conv = np.convolve(x, h)[: len(x)]
    y_direct = system(x)
    return {"y_direct": y_direct, "y_conv": y_conv, "error": relative_error(y_conv, y_direct)}


def classify(system, x1, x2):
    lin = linearity_test(system, x1, x2)
    ti = time_invariance_test(system, x1)
    return {"linear": lin, "ti": ti, "lti": lin["holds"] and ti["holds"]}

"""
iir.py
------
IIR filter design and implementation.

* A second-order notch and a first-order high-pass are designed directly from
  pole and zero positions, from first principles.
* Butterworth, Chebyshev I and elliptic band-passes are designed from their
  analog prototypes with the bilinear transform (SciPy supplies the prototypes;
  the bilinear mapping and pre-warping are written out here and checked).
* Filters are applied by their recursive difference equation, in direct form
  and as cascaded second-order sections, forward or forward-backward.

No function here does any I/O or plotting.
"""

import numpy as np
from scipy import signal


# ==========================================================================
# 1. Difference equation (recursive)
# ==========================================================================
def iir_filter(b, a, x):
    """a0 y[n] = sum_k b_k x[n-k] - sum_{k>=1} a_k y[n-k]   (direct form, zero initial state)."""
    b, a, x = np.asarray(b, float), np.asarray(a, float), np.asarray(x, float)
    b, a = b / a[0], a / a[0]
    M, N = len(b) - 1, len(a) - 1
    y = np.zeros(len(x))
    for n in range(len(x)):
        acc = 0.0
        for k in range(min(M, n) + 1):
            acc += b[k] * x[n - k]
        for k in range(1, min(N, n) + 1):
            acc -= a[k] * y[n - k]
        y[n] = acc
    return y


def sos_filter(sos, x):
    """Cascade of biquads (second-order sections): numerically robust for high orders."""
    y = np.asarray(x, float)
    for s in sos:
        y = signal.lfilter(s[:3], s[3:], y)
    return y


def filtfilt_sos(sos, x):
    """Forward then backward filtering: zero phase, squared magnitude (non-causal, off-line only)."""
    return sos_filter(sos, sos_filter(sos, x)[::-1])[::-1]


# ==========================================================================
# 2. Direct pole-zero designs
# ==========================================================================
def notch(f0, Q, fs):
    """Zeros on the unit circle at ±w0, poles just inside at radius r: H(z) = (1 - 2cos w0 z^-1 + z^-2)/(1 - 2r cos w0 z^-1 + r^2 z^-2)."""
    w0 = 2 * np.pi * f0 / fs
    bw = w0 / Q                                     # -3 dB bandwidth (rad/sample)
    r = 1 - bw / 2
    b = np.array([1.0, -2 * np.cos(w0), 1.0])
    a = np.array([1.0, -2 * r * np.cos(w0), r * r])
    k = np.sum(a) / np.sum(b)                       # unit gain at DC
    return b * k, a, r


def dc_blocker(fc, fs):
    """First-order high-pass: zero at z = 1, pole at z = p (y[n] = g(x[n] - x[n-1]) + p y[n-1])."""
    p = np.exp(-2 * np.pi * fc / fs)
    g = (1 + p) / 2
    return np.array([g, -g]), np.array([1.0, -p])


# ==========================================================================
# 3. Bilinear transform designs
# ==========================================================================
def prewarp(f, fs):
    """Analog frequency that the bilinear transform maps onto f:  Omega = 2 fs tan(pi f / fs)."""
    return 2 * fs * np.tan(np.pi * np.asarray(f, float) / fs)


def bilinear_zpk(z, p, k, fs):
    """s = 2 fs (1 - z^-1)/(1 + z^-1):  each s-plane root s_i maps to z_i = (2fs + s_i)/(2fs - s_i)."""
    z, p = np.asarray(z, complex), np.asarray(p, complex)
    c = 2 * fs
    zd = (c + z) / (c - z)
    pd = (c + p) / (c - p)
    zd = np.r_[zd, -np.ones(len(p) - len(z))]       # zeros at infinity go to z = -1 (Nyquist)
    kd = k * np.real(np.prod(c - z) / np.prod(c - p))
    return zd, pd, kd


def design_bandpass(family, band, fs, order, rp=1.0, rs=40.0):
    """Analog low-pass prototype -> analog band-pass (pre-warped edges) -> bilinear -> digital SOS."""
    if family == "butterworth":
        z, p, k = signal.buttap(order)
    elif family == "chebyshev1":
        z, p, k = signal.cheb1ap(order, rp)
    elif family == "elliptic":
        z, p, k = signal.ellipap(order, rp, rs)
    else:
        raise ValueError(family)
    W1, W2 = prewarp(band, fs)
    z, p, k = signal.lp2bp_zpk(z, p, k, wo=np.sqrt(W1 * W2), bw=W2 - W1)
    zd, pd, kd = bilinear_zpk(z, p, k, fs)
    return {"zpk": (zd, pd, kd), "sos": signal.zpk2sos(zd, pd, kd), "ba": signal.zpk2tf(zd, pd, kd)}


# ==========================================================================
# 4. Analysis
# ==========================================================================
def freq_response(b, a, fs, n=8192):
    w, H = signal.freqz(b, a, worN=n, fs=fs)
    return w, H


def freq_response_sos(sos, fs, n=8192):
    return signal.sosfreqz(sos, worN=n, fs=fs)


def group_delay_sos(sos, fs, n=4096):
    """Sum of the group delays of the sections (samples)."""
    w = np.linspace(0, np.pi, n, endpoint=False)[1:]
    tau = np.zeros(len(w))
    for s in sos:
        _, gd = signal.group_delay((s[:3], s[3:]), w=w)
        tau += gd
    return w * fs / (2 * np.pi), tau


def max_pole_radius(p):
    return float(np.max(np.abs(p))) if len(p) else 0.0


def band_edges(f, H, level_db=-3.0):
    A = 20 * np.log10(np.abs(H) + 1e-15)
    above = np.nonzero(A >= A.max() + level_db)[0]
    return float(f[above[0]]), float(f[above[-1]])

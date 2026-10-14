"""
circular.py
-----------
Pure implementations of circular convolution and its relatives.

Three independent methods are provided so the report can show that they agree:
the direct definition with modulo-N indexing, the circulant-matrix method, and
linear convolution followed by time-aliasing (wrapping).  No function here does
any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Circular convolution: three methods
# ==========================================================================
def _pad(v, N):
    v = np.asarray(v, dtype=float)
    if len(v) > N:
        raise ValueError("sequence longer than N")
    return np.r_[v, np.zeros(N - len(v))]


def circular_convolve(x, h, N=None):
    """Direct definition:  y[n] = sum_{m=0}^{N-1} x[m] h[(n - m) mod N]."""
    N = N or max(len(x), len(h))
    x, h = _pad(x, N), _pad(h, N)
    y = np.zeros(N)
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]
    return y


def circulant_matrix(h, N=None):
    """Matrix whose column m is h circularly shifted by m:  H[n, m] = h[(n - m) mod N]."""
    N = N or len(h)
    h = _pad(h, N)
    return np.array([[h[(n - m) % N] for m in range(N)] for n in range(N)])


def circular_convolve_matrix(x, h, N=None):
    """y = H x  with H the circulant matrix of h."""
    N = N or max(len(x), len(h))
    return circulant_matrix(h, N) @ _pad(x, N)


def wrap(y, N):
    """Time-aliasing: fold a long sequence into N samples, y_N[n] = sum_r y[n + rN]."""
    out = np.zeros(N)
    for i, v in enumerate(np.asarray(y, dtype=float)):
        out[i % N] += v
    return out


def circular_via_linear(x, h, N=None):
    """N-point circular convolution = linear convolution wrapped modulo N."""
    N = N or max(len(x), len(h))
    return wrap(np.convolve(x, h), N)


def linear_via_circular(x, h):
    """Zero-pad to N >= L + M - 1 and the circular result equals the linear one."""
    N = len(x) + len(h) - 1
    return circular_convolve_matrix(x, h, N)


# ==========================================================================
# 2. Circular shift and centred kernels
# ==========================================================================
def circular_shift(x, k):
    """x[(n - k) mod N]  (rotation; equivalent to convolving with delta[n - k])."""
    return np.roll(np.asarray(x, dtype=float), k)


def centred_kernel(weights, N):
    """Place an odd-length symmetric kernel with its centre at n = 0 (wrapping the left half)."""
    w = np.asarray(weights, dtype=float)
    c = len(w) // 2
    h = np.zeros(N)
    for i, v in enumerate(w):
        h[(i - c) % N] += v
    return h


def smooth_circular(x, weights):
    return circular_convolve_matrix(x, centred_kernel(weights, len(x)))


def smooth_linear(x, weights):
    """The same smoothing done with ordinary (non-circular) convolution and zero padding."""
    return np.convolve(np.asarray(x, dtype=float), np.asarray(weights, dtype=float), mode="same")


# ==========================================================================
# 3. Wind application: runway coverage as a sum of circular convolutions
# ==========================================================================
def acceptance_kernel(speed_kt, limit_kt, N, bin_deg):
    """a[k] = 1 if the crosswind |V sin(k * bin)| for a wind k bins off the runway is <= limit."""
    ang = np.deg2rad(np.arange(N) * bin_deg)
    return (np.abs(speed_kt * np.sin(ang)) <= limit_kt + 1e-9).astype(float)


def runway_coverage(hist_by_speed, speeds, limit_kt, n_always_ok, total, bin_deg):
    """coverage[h] = (always-usable reports + sum_s (p_s circ a_s)[h]) / total.

    ``hist_by_speed[s]`` is the 36-bin direction histogram of reports at speed
    ``speeds[s]``.  The acceptance kernel is symmetric, so correlation and
    convolution coincide.
    """
    N = hist_by_speed.shape[1]
    acc = np.zeros(N)
    for p, v in zip(hist_by_speed, speeds):
        if p.sum():
            acc += circular_convolve_matrix(p, acceptance_kernel(v, limit_kt, N, bin_deg), N)
    return (acc + n_always_ok) / total

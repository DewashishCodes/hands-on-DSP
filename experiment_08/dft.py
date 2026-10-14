"""
dft.py
------
The discrete Fourier transform and its inverse, implemented from the
definition, plus the tools used to read a spectrum.

Three forward implementations are provided so the report can show that they
agree: a double loop over the definition, the DFT matrix, and the FFT.  No
function here does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. DFT and IDFT
# ==========================================================================
def dft_direct(x):
    """X[k] = sum_{n=0}^{N-1} x[n] e^{-j 2 pi k n / N}   (definition, O(N^2))."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        s = 0j
        for n in range(N):
            s += x[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = s
    return X


def dft_matrix(N):
    """W[k, n] = e^{-j 2 pi k n / N}."""
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, k) / N)


def dft(x):
    """X = W x  (matrix form of the definition)."""
    x = np.asarray(x, dtype=complex)
    return dft_matrix(len(x)) @ x


def idft(X):
    """x[n] = (1/N) sum_{k=0}^{N-1} X[k] e^{+j 2 pi k n / N}   (= (1/N) W* X)."""
    X = np.asarray(X, dtype=complex)
    return np.conj(dft_matrix(len(X))) @ X / len(X)


def fft(x, n=None):
    return np.fft.fft(x, n)


# ==========================================================================
# 2. Reading a spectrum
# ==========================================================================
def frequencies(N, fs):
    """Frequency of bin k:  f_k = k fs / N."""
    return np.arange(N) * fs / N


def amplitude_spectrum(X):
    """Single-sided amplitude of a real signal: |X[0]|/N and 2|X[k]|/N for 0 < k < N/2."""
    N = len(X)
    A = 2 * np.abs(X) / N
    A[0] /= 2
    return A[: N // 2 + 1]


def bin_amplitude_phase(X, k):
    """Amplitude and phase of the sinusoid A cos(2 pi k n / N + phi) represented by bins k and N-k."""
    N = len(X)
    return float(2 * np.abs(X[k]) / N), float(np.angle(X[k]))


def keep_bins(X, bins):
    """Zero every bin except +k and -k (N - k) for k in ``bins``: an ideal frequency-domain filter."""
    N = len(X)
    mask = np.zeros(N, bool)
    for k in bins:
        mask[k % N] = mask[(-k) % N] = True
    return np.where(mask, X, 0)


def hann(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def detrend(x, order):
    n = np.arange(len(x))
    p = np.polyfit(n, x, order)
    return x - np.polyval(p, n), np.polyval(p, n), p


# ==========================================================================
# 3. Properties
# ==========================================================================
def parseval_error(x, X):
    """| sum |x|^2 - (1/N) sum |X|^2 |  relative to the energy."""
    e_t = np.sum(np.abs(x) ** 2)
    e_f = np.sum(np.abs(X) ** 2) / len(X)
    return float(abs(e_t - e_f) / e_t)


def symmetry_error(X):
    """For real x:  X[N - k] = X*[k]."""
    N = len(X)
    return float(np.max(np.abs(X[(-np.arange(N)) % N] - np.conj(X))) / np.max(np.abs(X)))

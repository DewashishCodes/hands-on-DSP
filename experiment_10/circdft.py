"""
circdft.py
----------
Circular convolution computed with the DFT, compared with the direct
modulo-N sum, plus the kernels and the cyclic-prefix construction.

No function here does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Circular convolution: direct and via the DFT
# ==========================================================================
def circ_direct(x, h):
    """y[n] = sum_m x[m] h[((n - m))_N]   (definition, N^2 multiplications)."""
    x, h = np.asarray(x, float), np.asarray(h, float)
    N = len(x)
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N      # (n - m) mod N
    return h[idx] @ x


def circ_dft(x, h):
    """y = IDFT{ DFT{x} . DFT{h} }  (both of length N, no zero padding)."""
    x, h = np.asarray(x, float), np.asarray(h, float)
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h, len(x)), len(x))


def circ_dft_matrix(x, h):
    """The same product written with the DFT matrix, as it is done by hand."""
    N = len(x)
    k = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(k, k) / N)
    return np.real(np.conj(W) @ ((W @ x) * (W @ np.asarray(h, float))) / N)


def linear_same(x, h_centred, half):
    """Ordinary (zero-padded) convolution, centred: what a non-circular smoother does at the year ends."""
    y = np.convolve(np.asarray(x, float), np.roll(h_centred, half)[: 2 * half + 1])
    return y[half: half + len(x)]


def cyclic_prefix_conv(x, h_causal):
    """Prepend the last M-1 samples of x, convolve linearly, discard the prefix: equals circular convolution."""
    M = len(h_causal)
    xp = np.r_[x[-(M - 1):], x] if M > 1 else np.asarray(x, float)
    y = np.convolve(xp, h_causal)
    return y[M - 1: M - 1 + len(x)]


# ==========================================================================
# 2. Kernels (centred at n = 0, wrapped to length N, unit sum)
# ==========================================================================
def centred(weights, N):
    w = np.asarray(weights, float)
    c = len(w) // 2
    h = np.zeros(N)
    for i, v in enumerate(w):
        h[(i - c) % N] += v
    return h / h.sum()


def boxcar(M, N):
    return centred(np.ones(int(M)), N)


def gaussian(sigma, N):
    half = int(np.ceil(4 * sigma))
    t = np.arange(-half, half + 1)
    return centred(np.exp(-0.5 * (t / sigma) ** 2), N)


def harmonic_kernel(K, N):
    """Kernel whose DFT is 1 for |k| <= K and 0 elsewhere (Dirichlet kernel): keeping K harmonics."""
    H = np.zeros(N)
    H[: K + 1] = 1
    H[N - K:] = 1
    return np.real(np.fft.ifft(H))


def frequency_response(h):
    """|H[k]| for k = 0 ... N/2: the gain applied to the k-th annual harmonic."""
    return np.abs(np.fft.rfft(h))


# ==========================================================================
# 3. Helpers
# ==========================================================================
def fill_circular(x):
    """Fill missing calendar days by circular linear interpolation."""
    x = np.asarray(x, float)
    N = len(x)
    ok = ~np.isnan(x)
    idx = np.arange(N)
    xp = np.r_[idx[ok] - N, idx[ok], idx[ok] + N]
    fp = np.r_[x[ok], x[ok], x[ok]]
    return np.interp(idx, xp, fp)


def roughness(x):
    """RMS of the day-to-day change, circularly."""
    return float(np.sqrt(np.mean((np.roll(x, -1) - x) ** 2)))

"""
fastconv.py
-----------
Linear convolution computed with the DFT, and the tools around it.

Three routes are implemented so the report can compare them: direct
time-domain convolution, single-block DFT convolution with zero padding, and
overlap-add block convolution (how real-time reverbs work).  No function here
does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Convolution: direct and via the DFT
# ==========================================================================
def conv_direct(x, h):
    """y[n] = sum_k x[k] h[n-k]  (time domain, L*M multiplications)."""
    return np.convolve(np.asarray(x, float), np.asarray(h, float))


def conv_dft(x, h, N=None):
    """y = IDFT_N{ DFT_N{x} DFT_N{h} }.

    With N >= L + M - 1 the result equals linear convolution.  With a smaller N the
    output is the N-point circular convolution, i.e. the linear result folded
    (time-aliased) onto N samples.
    """
    x, h = np.asarray(x, float), np.asarray(h, float)
    L, M = len(x), len(h)
    N = N or L + M - 1
    X = np.fft.rfft(x, N) if N >= L else np.fft.rfft(_fold(x, N), N)
    H = np.fft.rfft(h, N) if N >= M else np.fft.rfft(_fold(h, N), N)
    return np.fft.irfft(X * H, N)


def _fold(v, N):
    out = np.zeros(N)
    for i, s in enumerate(v):
        out[i % N] += s
    return out


def next_fast_length(n):
    """Smallest 2^a 3^b 5^c >= n (FFT-friendly length)."""
    best = 1 << int(np.ceil(np.log2(n)))
    f5 = 1
    while f5 <= best:
        f35 = f5
        while f35 <= best:
            q = -(-n // f35)
            p2 = 1 << max(0, int(np.ceil(np.log2(q))))
            if f35 * p2 < best:
                best = f35 * p2
            f35 *= 3
        f5 *= 5
    return best


def overlap_add(x, h, block):
    """Split x into blocks of length B, convolve each with h via an FFT of size >= B + M - 1, add the tails."""
    x, h = np.asarray(x, float), np.asarray(h, float)
    L, M = len(x), len(h)
    N = next_fast_length(block + M - 1)
    H = np.fft.rfft(h, N)
    y = np.zeros(L + M - 1)
    for start in range(0, L, block):
        seg = x[start: start + block]
        ys = np.fft.irfft(np.fft.rfft(seg, N) * H, N)[: len(seg) + M - 1]
        y[start: start + len(ys)] += ys
    return y, N


# ==========================================================================
# 2. Cost model
# ==========================================================================
def mults_direct(L, M):
    return L * M


def mults_fft(N):
    """Real multiplications: three real FFTs of length N plus N/2 complex products (about 2 N log2 N)."""
    return 3 * (N / 2) * np.log2(N) * 2 + 3 * N


def mults_ola(L, M, block):
    N = next_fast_length(block + M - 1)
    blocks = int(np.ceil(L / block))
    return blocks * (2 * (N / 2) * np.log2(N) * 2 + 3 * N) + mults_fft(N) / 3


# ==========================================================================
# 3. Measures and room acoustics
# ==========================================================================
def rel_error(a, b):
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]) / np.linalg.norm(b[:n]))


def snr_db(ref, est):
    n = min(len(ref), len(est))
    return float(10 * np.log10(np.sum(ref[:n] ** 2) / np.sum((ref[:n] - est[:n]) ** 2)))


def schroeder_db(h):
    """Backward-integrated energy decay curve in dB (Schroeder 1965)."""
    e = np.cumsum(h[::-1] ** 2)[::-1]
    return 10 * np.log10(e / e[0] + 1e-30)


def reverberation_time(edc, fs, lo=-5.0, hi=-25.0):
    """RT60 from a straight-line fit of the decay curve between lo and hi dB (T20 when hi = -25)."""
    i0, i1 = int(np.argmax(edc <= lo)), int(np.argmax(edc <= hi))
    t = np.arange(i0, i1) / fs
    slope, icpt = np.polyfit(t, edc[i0:i1], 1)            # dB per second
    return float(-60.0 / slope), slope, icpt, i0, i1


def direct_to_reverberant(h, fs, window_ms=2.5):
    """Energy of the direct sound (±window around the peak) relative to everything after it, in dB."""
    k = int(np.argmax(np.abs(h)))
    w = int(window_ms * 1e-3 * fs)
    direct = np.sum(h[max(0, k - w): k + w] ** 2)
    late = np.sum(h[k + w:] ** 2)
    return float(10 * np.log10(direct / late)), k


def deconvolve(y, h, L, eps):
    """Recover x from y = x * h by regularised spectral division:  X = Y H* / (|H|^2 + eps max|H|^2)."""
    N = next_fast_length(len(y))
    Y, H = np.fft.rfft(y, N), np.fft.rfft(h, N)
    P = np.abs(H) ** 2
    X = Y * np.conj(H) / (P + eps * P.max())
    return np.fft.irfft(X, N)[:L]

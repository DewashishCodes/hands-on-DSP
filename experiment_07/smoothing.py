"""
smoothing.py
------------
Smoothing filters written as linear constant-coefficient difference equations.

The core routine solves  sum_k a_k y[n-k] = sum_k b_k x[n-k]  by direct
iteration, sample by sample, exactly as one would by hand, including any
initial conditions.  Every smoother in the experiment (moving average,
exponential moving average, cascaded EMA, Holt's trend-corrected smoother,
MACD) is built on it.  No function here does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. The general difference equation, solved by iteration
# ==========================================================================
def difference_equation(b, a, x, y_init=None, x_init=None):
    """y[n] = (1/a0) * ( sum_{k=0}^{M} b_k x[n-k] - sum_{k=1}^{N} a_k y[n-k] ).

    y_init = [y[-1], y[-2], ...] and x_init = [x[-1], x[-2], ...] are the
    initial conditions (zero if omitted, i.e. the system starts at rest).
    """
    b = np.asarray(b, float)
    a = np.asarray(a, float)
    x = np.asarray(x, float)
    M, N = len(b) - 1, len(a) - 1
    xp = np.zeros(M) if x_init is None else np.r_[np.asarray(x_init, float), np.zeros(M)][:M]
    yp = np.zeros(N) if y_init is None else np.r_[np.asarray(y_init, float), np.zeros(N)][:N]
    y = np.zeros(len(x))
    for n in range(len(x)):
        xs = np.r_[x[n], xp]                    # x[n], x[n-1], ..., x[n-M]
        acc = np.dot(b, xs) - np.dot(a[1:], yp)
        y[n] = acc / a[0]
        if M:
            xp = np.r_[x[n], xp[:-1]]
        if N:
            yp = np.r_[y[n], yp[:-1]]
    return y


# ==========================================================================
# 2. Smoothers
# ==========================================================================
def ma_coeffs(N):
    """Non-recursive (FIR) N-point moving average: b = [1/N] * N, a = [1]."""
    return np.full(int(N), 1.0 / int(N)), np.array([1.0])


def ma_recursive_coeffs(N):
    """The same moving average written recursively: y[n] = y[n-1] + (x[n] - x[n-N]) / N."""
    b = np.zeros(int(N) + 1)
    b[0], b[-1] = 1.0 / N, -1.0 / N
    return b, np.array([1.0, -1.0])


def ema_coeffs(alpha):
    """y[n] = alpha x[n] + (1 - alpha) y[n-1]."""
    return np.array([alpha]), np.array([1.0, -(1.0 - alpha)])


def cascade_coeffs(b1, a1, b2, a2):
    """Two filters in series: multiply the numerator and denominator polynomials."""
    return np.convolve(b1, b2), np.convolve(a1, a2)


def holt_coeffs(alpha, beta):
    """Holt's linear (level + trend) smoother as one second-order difference equation.

    l[n] = alpha x[n] + (1 - alpha)(l[n-1] + t[n-1]),  t[n] = beta (l[n] - l[n-1]) + (1 - beta) t[n-1].
    Eliminating t gives  l[n] + a1 l[n-1] + a2 l[n-2] = alpha x[n] - alpha (1 - beta) x[n-1].
    """
    a1 = alpha + alpha * beta - 2.0
    a2 = 1.0 - alpha
    return np.array([alpha, -alpha * (1.0 - beta)]), np.array([1.0, a1, a2])


def ema(x, alpha, y0="first"):
    """EMA with the usual initial condition y[-1] = x[0] (or 0 / any number)."""
    b, a = ema_coeffs(alpha)
    y_init = [x[0]] if y0 == "first" else [float(y0)]
    return difference_equation(b, a, x, y_init=y_init)


def holt(x, alpha, beta):
    """Holt's smoother started at level x[0] with zero trend."""
    b, a = holt_coeffs(alpha, beta)
    # with l[-1] = l[-2] = x[0] and x[-1] = x[0] the trend starts at zero
    return difference_equation(b, a, x, y_init=[x[0], x[0]], x_init=[x[0]])


def macd(x, fast, slow, signal):
    ef, es = ema(x, 2 / (fast + 1)), ema(x, 2 / (slow + 1))
    line = ef - es
    sig = ema(line, 2 / (signal + 1))
    return ef, es, line, sig


# ==========================================================================
# 3. System analysis of a difference equation
# ==========================================================================
def impulse_response(b, a, L):
    d = np.zeros(L)
    d[0] = 1.0
    return difference_equation(b, a, d)


def step_response(b, a, L):
    return difference_equation(b, a, np.ones(L))


def frequency_response(b, a, w):
    """H(e^{jw}) = B(e^{jw}) / A(e^{jw})."""
    z = np.exp(-1j * np.asarray(w))
    B = sum(bk * z ** k for k, bk in enumerate(b))
    A = sum(ak * z ** k for k, ak in enumerate(a))
    return B / A


def poles_zeros(b, a):
    """Roots of the polynomials in z (after multiplying through by the highest power)."""
    nb, na = len(b) - 1, len(a) - 1
    order = max(nb, na)
    bz = np.r_[b, np.zeros(order - nb)]
    az = np.r_[a, np.zeros(order - na)]
    return np.roots(az), np.roots(bz)


def is_stable(b, a):
    p, _ = poles_zeros(b, a)
    return bool(np.all(np.abs(p) < 1 - 1e-12)) if len(p) else True


def noise_gain(b, a, L=5000):
    """Ratio of output to input variance for white noise:  sum h[n]^2."""
    return float(np.sum(impulse_response(b, a, L) ** 2))


def dc_lag(b, a):
    """Low-frequency group delay  sum n h[n] / sum h[n]  (samples)."""
    h = impulse_response(b, a, 5000)
    return float(np.sum(np.arange(len(h)) * h) / np.sum(h))


def ops_per_sample(b, a):
    """Multiplications per output sample of the direct implementation."""
    return int(np.count_nonzero(b) + np.count_nonzero(a[1:]))


def zero_input_zero_state(b, a, x, y_init):
    """Split the complete response into the part due to initial conditions and the part due to the input."""
    y_full = difference_equation(b, a, x, y_init=y_init)
    y_zs = difference_equation(b, a, x)
    y_zi = difference_equation(b, a, np.zeros(len(x)), y_init=y_init)
    return y_full, y_zi, y_zs


def crossings(u):
    """Indices where u changes sign (a crossover of two smoothed lines)."""
    s = np.sign(u)
    return np.nonzero((s[1:] * s[:-1]) < 0)[0] + 1

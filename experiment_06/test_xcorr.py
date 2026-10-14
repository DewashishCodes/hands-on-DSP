"""
test_xcorr.py
-------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_xcorr
"""

import unittest

import numpy as np

import xcorr as xc

RNG = np.random.default_rng(6)


class TestDefinition(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        x = np.array([1.0, 3.0, 2.0, 0.0])
        y = np.array([0.0, 1.0, 3.0, 2.0])          # y[n] = x[n - 1]
        lags, r = xc.crosscorr_direct(y, x, 3)
        np.testing.assert_allclose(r, [0, 0, 2, 9, 14, 9, 2])
        self.assertEqual(lags[np.argmax(r)], 1)

    def test_three_methods_agree(self):
        x, y = RNG.normal(size=150), RNG.normal(size=150)
        l1, a = xc.crosscorr_direct(y, x, 40)
        l2, b = xc.crosscorr_conv(y, x)
        l3, c = xc.crosscorr_fft(y, x, 40)
        np.testing.assert_allclose(b[(l2 >= -40) & (l2 <= 40)], a, atol=1e-9)
        np.testing.assert_allclose(c, a, atol=1e-9)

    def test_swap_reverses_lags(self):
        x, y = RNG.normal(size=80), RNG.normal(size=80)
        _, ryx = xc.crosscorr_direct(y, x, 20)
        _, rxy = xc.crosscorr_direct(x, y, 20)
        np.testing.assert_allclose(ryx, rxy[::-1])

    def test_cauchy_schwarz_bound(self):
        x, y = RNG.normal(size=200), RNG.normal(size=200)
        _, rho = xc.ccf(y, x, 100)
        self.assertTrue(np.all(np.abs(rho) <= 1 + 1e-12))

    def test_autocorrelation_is_special_case(self):
        x = RNG.normal(size=60)
        _, r = xc.crosscorr_direct(x, x, 10)
        np.testing.assert_allclose(r, r[::-1])
        self.assertEqual(int(np.argmax(r)), 10)


class TestDelayEstimation(unittest.TestCase):
    def test_integer_delay_recovered(self):
        x = RNG.normal(size=500)
        y = np.r_[np.zeros(7), x[:-7]]
        lags, rho = xc.ccf(y, x, 30)
        self.assertEqual(xc.peak_lag(lags, rho)[0], 7)

    def test_delay_in_noise(self):
        x = RNG.normal(size=2000)
        y = np.r_[np.zeros(12), x[:-12]] + RNG.normal(0, 2.0, 2000)
        lags, rho = xc.ccf(y, x, 40)
        self.assertEqual(xc.peak_lag(lags, rho)[0], 12)

    def test_fractional_delay_by_parabola(self):
        n = np.arange(2000)
        x = np.sin(2 * np.pi * n / 40) * np.exp(-((n - 1000) / 300) ** 2)
        y = np.sin(2 * np.pi * (n - 3.4) / 40) * np.exp(-((n - 3.4 - 1000) / 300) ** 2)
        lags, rho = xc.ccf(y, x, 15)
        self.assertAlmostEqual(xc.peak_lag(lags, rho, (-10, 10))[1], 3.4, delta=0.1)

    def test_periodic_signal_is_ambiguous(self):
        n = np.arange(1000)
        x = np.sin(2 * np.pi * n / 12)
        y = np.sin(2 * np.pi * (n - 3) / 12)
        lags, rho = xc.ccf(y, x, 30)
        peaks = [k for k, v in xc.local_peaks(lags, rho) if v > 0.9]
        self.assertIn(3, peaks)
        self.assertIn(15, peaks)

    def test_haversine(self):
        self.assertAlmostEqual(xc.haversine_km(0, 0, 1, 0), 111.19, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_smoothing.py
-----------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_smoothing
"""

import unittest

import numpy as np
from scipy.signal import lfilter

import smoothing as sm

RNG = np.random.default_rng(7)


class TestDifferenceEquation(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        b, a = sm.ema_coeffs(0.5)
        y = sm.difference_equation(b, a, [80.0, 82.0, 81.0, 85.0], y_init=[80.0])
        np.testing.assert_allclose(y, [80.0, 81.0, 81.0, 83.0])

    def test_matches_lfilter(self):
        b, a = [0.2, 0.1, -0.05], [1.0, -0.6, 0.08]
        x = RNG.normal(size=300)
        np.testing.assert_allclose(sm.difference_equation(b, a, x), lfilter(b, a, x), atol=1e-12)

    def test_recursive_ma_equals_fir(self):
        x = RNG.normal(size=200)
        np.testing.assert_allclose(sm.difference_equation(*sm.ma_recursive_coeffs(10), x),
                                   sm.difference_equation(*sm.ma_coeffs(10), x), atol=1e-12)

    def test_zero_input_plus_zero_state(self):
        b, a = sm.ema_coeffs(0.1)
        x = RNG.normal(size=100)
        y, yzi, yzs = sm.zero_input_zero_state(b, a, x, [5.0])
        np.testing.assert_allclose(y, yzi + yzs, atol=1e-12)
        np.testing.assert_allclose(yzi, 5.0 * 0.9 ** np.arange(1, 101), atol=1e-12)

    def test_holt_matches_two_equation_form(self):
        x = np.cumsum(RNG.normal(size=150)) + 50
        al, be = 0.2, 0.1
        l, t, out = x[0], 0.0, []
        for v in x:
            ln = al * v + (1 - al) * (l + t)
            t = be * (ln - l) + (1 - be) * t
            l = ln
            out.append(l)
        np.testing.assert_allclose(sm.holt(x, al, be), out, atol=1e-9)


class TestSystemProperties(unittest.TestCase):
    def test_ema_impulse_and_pole(self):
        b, a = sm.ema_coeffs(0.25)
        np.testing.assert_allclose(sm.impulse_response(b, a, 10), 0.25 * 0.75 ** np.arange(10))
        p, _ = sm.poles_zeros(b, a)
        self.assertAlmostEqual(p[0], 0.75)

    def test_unit_dc_gain(self):
        for b, a in (sm.ma_coeffs(7), sm.ema_coeffs(0.3), sm.holt_coeffs(0.3, 0.1)):
            self.assertAlmostEqual(abs(sm.frequency_response(b, a, [0.0])[0]), 1.0)

    def test_noise_gain_and_lag_formulas(self):
        for L in (1, 4, 9):
            bm, am = sm.ma_coeffs(2 * L + 1)
            be, ae = sm.ema_coeffs(1 / (L + 1))
            self.assertAlmostEqual(sm.noise_gain(bm, am), 1 / (2 * L + 1))
            self.assertAlmostEqual(sm.noise_gain(be, ae), 1 / (2 * L + 1), places=6)
            self.assertAlmostEqual(sm.dc_lag(be, ae), L, places=6)

    def test_holt_has_no_lag_on_a_ramp(self):
        x = 0.05 * np.arange(800) + 70
        y = sm.holt(x, 0.1, 0.05)
        self.assertLess(abs(x[-1] - y[-1]), 1e-6)

    def test_unstable_pole_diverges(self):
        y = sm.impulse_response([1.0], [1.0, -1.05], 200)
        self.assertGreater(y[-1], 1e3)
        self.assertFalse(sm.is_stable([1.0], [1.0, -1.05]))

    def test_crossings(self):
        self.assertEqual(list(sm.crossings(np.array([1, 2, -1, -2, 3]))), [2, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)

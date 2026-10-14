"""
test_systems.py
---------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_systems
"""

import unittest

import numpy as np

import config
import systems as sy

RNG = np.random.default_rng(11)
X1 = np.abs(RNG.normal(40, 20, 200))      # non-negative, like concentrations
X2 = np.abs(RNG.normal(15, 8, 200))
S = sy.default_systems()


class TestClassification(unittest.TestCase):
    def test_moving_average_is_lti(self):
        self.assertTrue(sy.classify(S["S1"], X1, X2)["lti"])

    def test_ewma_is_lti(self):
        self.assertTrue(sy.classify(S["S2"], X1, X2)["lti"])

    def test_aqi_is_nonlinear_but_time_invariant(self):
        r = sy.classify(S["S3"], X1, X2)
        self.assertFalse(r["linear"]["holds"])
        self.assertTrue(r["ti"]["holds"])

    def test_drift_is_linear_but_time_varying(self):
        r = sy.classify(S["S4"], X1, X2)
        self.assertTrue(r["linear"]["holds"])
        self.assertFalse(r["ti"]["holds"])

    def test_squarer_is_nonlinear(self):
        sq = sy.System("square", lambda x: x ** 2, "", "", False, True)
        self.assertFalse(sy.linearity_test(sq, X1, X2)["holds"])

    def test_nonzero_initial_condition_breaks_linearity(self):
        """y[n] = 0.5 y[n-1] + x[n] with y[-1] = 1 is only incrementally linear."""
        def sys_ic(x):
            y, prev = np.zeros(len(x)), 1.0
            for i, v in enumerate(x):
                prev = 0.5 * prev + v
                y[i] = prev
            return y
        s = sy.System("ic", sys_ic, "", "", False, True)
        self.assertFalse(sy.linearity_test(s, X1, X2)["holds"])


class TestResponses(unittest.TestCase):
    def test_ma_impulse_response_is_rectangular(self):
        h = sy.impulse_response(S["S1"], 30)
        np.testing.assert_allclose(h[:24], 1 / 24)
        np.testing.assert_allclose(h[24:], 0)

    def test_ewma_impulse_response_is_geometric(self):
        a = config.EWMA_ALPHA
        h = sy.impulse_response(S["S2"], 20)
        np.testing.assert_allclose(h, a * (1 - a) ** np.arange(20))

    def test_step_is_running_sum_of_impulse(self):
        for key in ("S1", "S2"):
            np.testing.assert_allclose(sy.step_response(S[key], 40),
                                       np.cumsum(sy.impulse_response(S[key], 40)))

    def test_lti_output_equals_convolution(self):
        for key in ("S1", "S2"):
            self.assertLess(sy.convolution_check(S[key], X1)["error"], 1e-12)

    def test_dc_gain_is_one(self):
        for key in ("S1", "S2"):
            self.assertAlmostEqual(sy.step_response(S[key], 400)[-1], 1.0, places=9)


class TestAQI(unittest.TestCase):
    def test_breakpoints(self):
        np.testing.assert_allclose(sy.aqi_pm25([0, 30, 60, 90, 120, 250]), [0, 50, 100, 200, 300, 400])

    def test_hand_calculation(self):
        """Matches the worked example in the Calculations section."""
        self.assertAlmostEqual(sy.aqi_pm25([45])[0], 75.0)
        self.assertAlmostEqual(sy.aqi_pm25([90])[0], 200.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

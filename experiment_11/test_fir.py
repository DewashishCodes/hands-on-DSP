"""
test_fir.py
-----------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_fir
"""

import unittest

import numpy as np
from scipy.signal import firwin, lfilter

import fir

RNG = np.random.default_rng(11)
FS = 125.0


class TestDesign(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section (N = 5, fc = fs/4, Hamming)."""
        np.testing.assert_allclose(fir.ideal_lowpass(FS / 4, FS, 5), [0.0, 1 / np.pi, 0.5, 1 / np.pi, 0.0], atol=1e-12)
        np.testing.assert_allclose(fir.window("hamming", 5), [0.08, 0.54, 1.0, 0.54, 0.08], atol=1e-12)

    def test_matches_scipy_firwin(self):
        for w, sw in (("hamming", "hamming"), ("hann", "hann"), ("blackman", "blackman"), ("rectangular", "boxcar")):
            np.testing.assert_allclose(fir.design_lowpass(4.0, FS, 101, w), firwin(101, 4.0, fs=FS, window=sw), atol=1e-12)

    def test_linear_phase_symmetry(self):
        self.assertTrue(fir.is_linear_phase(fir.design_bandpass(0.5, 4.0, FS, 251)))

    def test_group_delay_is_constant(self):
        f, tau = fir.group_delay(fir.design_lowpass(4.0, FS, 101), FS)
        np.testing.assert_allclose(tau[f < 3], 50.0, atol=1e-6)

    def test_unit_dc_gain(self):
        self.assertAlmostEqual(np.sum(fir.design_lowpass(0.5, FS, 501)), 1.0)

    def test_stopband_follows_window_table(self):
        b = fir.design_lowpass(4.0, FS, 251, "blackman")
        self.assertLess(fir.measure_lowpass(b, 4.0, FS, 1.5)["stop_db"], -70)

    def test_kaiser_meets_spec(self):
        N, beta = fir.kaiser_parameters(60, 0.3, FS)
        b = fir.design_lowpass(0.65, FS, N, "kaiser", beta)
        self.assertLess(fir.measure_lowpass(b, 0.65, FS, 0.15)["stop_db"], -57)


class TestFiltering(unittest.TestCase):
    def test_difference_equation_matches_lfilter(self):
        b, x = RNG.normal(size=31), RNG.normal(size=500)
        np.testing.assert_allclose(fir.fir_filter(b, x), lfilter(b, 1, x), atol=1e-12)

    def test_passband_tone_passes_and_stopband_tone_is_removed(self):
        t = np.arange(4000) / FS
        b = fir.design_bandpass(0.5, 4.0, FS, 251)
        y = fir.fir_filter(b, np.sin(2 * np.pi * 1.5 * t))[500:]
        z = fir.fir_filter(b, np.sin(2 * np.pi * 15 * t))[500:]
        self.assertAlmostEqual(np.std(y) * np.sqrt(2), 1.0, delta=0.02)
        self.assertLess(np.std(z), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

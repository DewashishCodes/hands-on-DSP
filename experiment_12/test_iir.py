"""
test_iir.py
-----------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_iir
"""

import unittest

import numpy as np
from scipy import signal

import iir

RNG = np.random.default_rng(12)
FS = 160.0


class TestDesign(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        b, a, r = iir.notch(40.0, 20.0, FS)                       # w0 = pi/2, so cos w0 = 0
        self.assertAlmostEqual(r, 1 - np.pi / 80)
        np.testing.assert_allclose(a, [1.0, 0.0, r * r], atol=1e-12)
        np.testing.assert_allclose(b / b[0], [1.0, 0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(iir.prewarp(40.0, FS) / (2 * np.pi * 40.0), 4 / np.pi)

    def test_bilinear_matches_scipy(self):
        for fam, ref in (("butterworth", signal.butter(4, (8, 13), "bandpass", fs=FS, output="sos")),
                         ("elliptic", signal.ellip(4, 1, 40, (8, 13), "bandpass", fs=FS, output="sos"))):
            d = iir.design_bandpass(fam, (8, 13), FS, 4, 1, 40)
            _, H = signal.sosfreqz(d["sos"], 2048, fs=FS)
            _, H2 = signal.sosfreqz(ref, 2048, fs=FS)
            np.testing.assert_allclose(np.abs(H), np.abs(H2), atol=1e-10)

    def test_prewarped_edges_are_exact(self):
        d = iir.design_bandpass("butterworth", (50, 70), FS, 4)
        f, H = signal.sosfreqz(d["sos"], 16384, fs=FS)
        lo, hi = iir.band_edges(f, H)
        self.assertAlmostEqual(lo, 50, delta=0.02)
        self.assertAlmostEqual(hi, 70, delta=0.02)

    def test_all_poles_inside_unit_circle(self):
        for fam in ("butterworth", "chebyshev1", "elliptic"):
            self.assertLess(iir.max_pole_radius(iir.design_bandpass(fam, (8, 13), FS, 4)["zpk"][1]), 1)

    def test_notch_removes_tone_and_keeps_alpha(self):
        b, a, _ = iir.notch(60.0, 30.0, FS)
        t = np.arange(3200) / FS
        y = iir.iir_filter(b, a, np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 10 * t))
        np.testing.assert_allclose(y[1000:], np.sin(2 * np.pi * 10 * t[1000:]), atol=0.01)


class TestImplementation(unittest.TestCase):
    def test_difference_equation_matches_lfilter(self):
        b, a = signal.butter(3, 0.2)
        x = RNG.normal(size=400)
        np.testing.assert_allclose(iir.iir_filter(b, a, x), signal.lfilter(b, a, x), atol=1e-12)

    def test_zero_phase_has_no_lag(self):
        sos = iir.design_bandpass("butterworth", (8, 13), FS, 4)["sos"]
        t = np.arange(4000) / FS
        x = np.sin(2 * np.pi * 10 * t)
        y = iir.filtfilt_sos(sos, x)
        np.testing.assert_allclose(y[800:3200], x[800:3200], atol=0.02)

    def test_dc_blocker_removes_offset(self):
        b, a = iir.dc_blocker(0.5, FS)
        y = iir.iir_filter(b, a, np.full(3000, 50.0))
        self.assertLess(abs(y[-1]), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

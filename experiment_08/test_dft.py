"""
test_dft.py
-----------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_dft
"""

import unittest

import numpy as np

import dft as D

RNG = np.random.default_rng(8)


class TestTransform(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        X = D.dft([0.1, 2.5, 0.6, -3.2])
        np.testing.assert_allclose(X, [0.0, -0.5 - 5.7j, 1.4, -0.5 + 5.7j], atol=1e-12)

    def test_three_methods_agree(self):
        x = RNG.normal(size=40)
        np.testing.assert_allclose(D.dft_direct(x), np.fft.fft(x), atol=1e-9)
        np.testing.assert_allclose(D.dft(x), np.fft.fft(x), atol=1e-9)

    def test_idft_inverts(self):
        x = RNG.normal(size=64) + 1j * RNG.normal(size=64)
        np.testing.assert_allclose(D.idft(D.dft(x)), x, atol=1e-12)

    def test_parseval(self):
        x = RNG.normal(size=100)
        self.assertLess(D.parseval_error(x, D.dft(x)), 1e-12)

    def test_conjugate_symmetry_for_real_input(self):
        self.assertLess(D.symmetry_error(D.dft(RNG.normal(size=50))), 1e-12)

    def test_linearity(self):
        x, y = RNG.normal(size=32), RNG.normal(size=32)
        np.testing.assert_allclose(D.dft(2 * x - 3 * y), 2 * D.dft(x) - 3 * D.dft(y), atol=1e-10)

    def test_circular_shift_is_phase_ramp(self):
        x = RNG.normal(size=24)
        k = np.arange(24)
        np.testing.assert_allclose(D.dft(np.roll(x, 5)), D.dft(x) * np.exp(-2j * np.pi * k * 5 / 24), atol=1e-10)


class TestSpectrumReading(unittest.TestCase):
    def test_amplitude_and_phase_of_cosine(self):
        N, k0 = 120, 10
        n = np.arange(N)
        X = D.dft(3.0 * np.cos(2 * np.pi * k0 * n / N + 0.7))
        A, ph = D.bin_amplitude_phase(X, k0)
        self.assertAlmostEqual(A, 3.0)
        self.assertAlmostEqual(ph, 0.7)

    def test_leakage_when_not_on_a_bin(self):
        n = np.arange(115)
        A = D.amplitude_spectrum(D.dft(np.cos(2 * np.pi * n / 12)))
        self.assertLess(A.max(), 0.95)

    def test_keep_bins_extracts_sinusoid(self):
        N = 96
        n = np.arange(N)
        s = np.cos(2 * np.pi * 8 * n / N)
        x = s + 0.5 * np.cos(2 * np.pi * 30 * n / N)
        np.testing.assert_allclose(D.idft(D.keep_bins(D.dft(x), [8])).real, s, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)

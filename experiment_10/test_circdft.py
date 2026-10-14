"""
test_circdft.py
---------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_circdft
"""

import unittest

import numpy as np

import circdft as cd

RNG = np.random.default_rng(10)


class TestCircularViaDFT(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        x, h = [20.0, 30.0, 40.0, 30.0], [0.5, 0.25, 0.0, 0.25]
        np.testing.assert_allclose(cd.circ_dft(x, h), [25.0, 30.0, 35.0, 30.0], atol=1e-12)
        np.testing.assert_allclose(np.fft.fft(x), [120, -20, 0, -20], atol=1e-12)

    def test_dft_equals_direct(self):
        x, h = RNG.normal(size=101), RNG.normal(size=101)
        np.testing.assert_allclose(cd.circ_dft(x, h), cd.circ_direct(x, h), atol=1e-10)
        np.testing.assert_allclose(cd.circ_dft_matrix(x, h), cd.circ_direct(x, h), atol=1e-9)

    def test_odd_and_even_lengths(self):
        for N in (8, 9, 365):
            x, h = RNG.normal(size=N), RNG.normal(size=N)
            np.testing.assert_allclose(cd.circ_dft(x, h), cd.circ_direct(x, h), atol=1e-10)

    def test_cyclic_prefix_turns_linear_into_circular(self):
        x, hc = RNG.normal(size=50), RNG.normal(size=7)
        np.testing.assert_allclose(cd.cyclic_prefix_conv(x, hc), cd.circ_dft(x, np.r_[hc, np.zeros(43)]), atol=1e-10)

    def test_rotation_equivariance(self):
        x = RNG.normal(size=365)
        h = cd.boxcar(31, 365)
        np.testing.assert_allclose(cd.circ_dft(np.roll(x, 100), h), np.roll(cd.circ_dft(x, h), 100), atol=1e-10)


class TestKernels(unittest.TestCase):
    def test_unit_sum_preserves_mean(self):
        x = RNG.normal(size=365) + 25
        for h in (cd.boxcar(31, 365), cd.gaussian(7, 365), cd.harmonic_kernel(3, 365)):
            self.assertAlmostEqual(cd.circ_dft(x, h).sum(), x.sum(), places=8)

    def test_harmonic_kernel_keeps_low_harmonics_exactly(self):
        k = np.arange(365)
        x = 3 * np.cos(2 * np.pi * k / 365) + np.cos(2 * np.pi * 2 * k / 365) + 0.5 * np.cos(2 * np.pi * 9 * k / 365)
        y = cd.circ_dft(x, cd.harmonic_kernel(3, 365))
        np.testing.assert_allclose(y, x - 0.5 * np.cos(2 * np.pi * 9 * k / 365), atol=1e-10)

    def test_boxcar_null(self):
        H = cd.frequency_response(cd.boxcar(73, 365))     # 365/73 = 5: harmonic 5 falls on a null
        self.assertLess(H[5], 1e-12)

    def test_fill_circular_wraps(self):
        x = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
        np.testing.assert_allclose(cd.fill_circular(x), [8 / 3, 2.0, 3.0, 4.0, 10 / 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)

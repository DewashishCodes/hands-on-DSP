"""
test_circular.py
----------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_circular
"""

import unittest

import numpy as np

import circular as cc

RNG = np.random.default_rng(4)


class TestCircularConvolution(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        x = [83, 192, 158, 171]
        h = [0.5, 0.25, 0, 0.25]
        np.testing.assert_allclose(cc.circular_convolve(x, h), [132.25, 156.25, 169.75, 145.75])

    def test_three_methods_agree(self):
        x, h = RNG.normal(size=12), RNG.normal(size=12)
        a = cc.circular_convolve(x, h)
        np.testing.assert_allclose(cc.circular_convolve_matrix(x, h), a)
        np.testing.assert_allclose(cc.circular_via_linear(x, h), a)

    def test_matches_dft_product(self):
        x, h = RNG.normal(size=16), RNG.normal(size=16)
        ref = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))
        np.testing.assert_allclose(cc.circular_convolve(x, h), ref, atol=1e-12)

    def test_commutative(self):
        x, h = RNG.normal(size=10), RNG.normal(size=10)
        np.testing.assert_allclose(cc.circular_convolve(x, h), cc.circular_convolve(h, x))

    def test_zero_padding_gives_linear(self):
        x, h = RNG.normal(size=7), RNG.normal(size=4)
        np.testing.assert_allclose(cc.linear_via_circular(x, h), np.convolve(x, h))

    def test_short_N_aliases(self):
        x, h = RNG.normal(size=7), RNG.normal(size=4)
        self.assertFalse(np.allclose(cc.circular_convolve(x, h, 7)[:3], np.convolve(x, h)[:3]))

    def test_delta_shift_is_rotation(self):
        x = RNG.normal(size=9)
        d = np.zeros(9)
        d[3] = 1
        np.testing.assert_allclose(cc.circular_convolve(x, d), cc.circular_shift(x, 3))

    def test_circulant_rows_are_rotations(self):
        H = cc.circulant_matrix([1, 2, 3, 4])
        np.testing.assert_allclose(H[:, 1], [4, 1, 2, 3])


class TestSmoothing(unittest.TestCase):
    def test_circular_preserves_total(self):
        p = RNG.integers(0, 100, 36).astype(float)
        self.assertAlmostEqual(cc.smooth_circular(p, [0.25, 0.5, 0.25]).sum(), p.sum())

    def test_linear_loses_mass_at_seam(self):
        p = RNG.integers(1, 100, 36).astype(float)
        self.assertLess(cc.smooth_linear(p, [0.25, 0.5, 0.25]).sum(), p.sum())

    def test_rotation_equivariance(self):
        p = RNG.normal(size=36)
        w = [0.25, 0.5, 0.25]
        np.testing.assert_allclose(cc.smooth_circular(cc.circular_shift(p, 5), w),
                                   cc.circular_shift(cc.smooth_circular(p, w), 5))


class TestRunway(unittest.TestCase):
    def test_acceptance_kernel_is_symmetric(self):
        a = cc.acceptance_kernel(15, 10, 36, 10)
        np.testing.assert_allclose(a[1:], a[1:][::-1])

    def test_calm_only_is_fully_usable(self):
        H = np.zeros((3, 36))
        self.assertTrue(np.all(cc.runway_coverage(H, [5, 10, 15], 10, 50, 50, 10) == 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

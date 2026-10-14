"""
test_autocorr.py
----------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_autocorr
"""

import unittest

import numpy as np

import autocorr as ac

RNG = np.random.default_rng(5)


class TestDefinition(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        x = np.array([2.0, -1.0, 3.0, 1.0])
        np.testing.assert_allclose(ac.autocorr_direct(x, 3), [15.0, -2.0, 5.0, 2.0])

    def test_three_methods_agree(self):
        x = RNG.normal(size=200)
        a = ac.autocorr_direct(x, 50)
        np.testing.assert_allclose(ac.autocorr_conv(x)[199:250], a, atol=1e-9)
        np.testing.assert_allclose(ac.autocorr_fft(x, 50), a, atol=1e-9)

    def test_even_symmetry(self):
        r = ac.autocorr_conv(RNG.normal(size=64))
        np.testing.assert_allclose(r, r[::-1], atol=1e-12)

    def test_maximum_at_zero_lag(self):
        r = ac.autocorr_conv(RNG.normal(size=64))
        self.assertTrue(np.all(np.abs(r) <= r[63] + 1e-12))

    def test_r0_is_energy(self):
        x = RNG.normal(size=100)
        self.assertAlmostEqual(ac.autocorr_direct(x, 0)[0], np.sum(x ** 2))


class TestFeatures(unittest.TestCase):
    def test_period_of_sinusoid(self):
        n = np.arange(2400)
        x = np.sin(2 * np.pi * n / 132)
        rho = ac.acf(x, 300)
        k, kr, v = ac.dominant_period(rho, (60, 200))
        self.assertAlmostEqual(kr, 132, delta=0.5)

    def test_sinusoid_acf_is_cosine(self):
        n = np.arange(5000)
        rho = ac.acf(np.sin(2 * np.pi * n / 50), 100)
        np.testing.assert_allclose(rho[:100], np.cos(2 * np.pi * np.arange(100) / 50), atol=0.03)

    def test_white_noise_mostly_inside_bounds(self):
        x = RNG.normal(size=4000)
        rho = ac.acf(x, 100)
        frac = np.mean(np.abs(rho[1:]) > ac.confidence_bound(4000))
        self.assertLess(frac, 0.12)

    def test_periodicity_survives_noise(self):
        n = np.arange(3000)
        x = np.sin(2 * np.pi * n / 128) + RNG.normal(0, 1.0, 3000)
        k, kr, v = ac.dominant_period(ac.acf(x, 300), (60, 200))
        self.assertAlmostEqual(kr, 128, delta=4)

    def test_biased_estimate_tapers(self):
        x = RNG.normal(size=500)
        b = ac.acf(x, 490)
        u = ac.acf(x, 490, unbiased=True)
        self.assertLess(np.abs(b[480:]).max(), np.abs(u[480:]).max())


if __name__ == "__main__":
    unittest.main(verbosity=2)

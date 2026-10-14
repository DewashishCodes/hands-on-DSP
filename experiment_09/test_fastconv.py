"""
test_fastconv.py
----------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_fastconv
"""

import unittest

import numpy as np

import fastconv as fc

RNG = np.random.default_rng(9)


class TestDFTConvolution(unittest.TestCase):
    def test_hand_example(self):
        """Matches the worked example in the Calculations section."""
        x, h = [1.0, 2.0, 3.0], [1.0, 0.5]
        np.testing.assert_allclose(fc.conv_dft(x, h), [1.0, 2.5, 4.0, 1.5], atol=1e-12)
        np.testing.assert_allclose(fc.conv_dft(x, h, 3), [2.5, 2.5, 4.0], atol=1e-12)

    def test_equals_direct_with_enough_padding(self):
        x, h = RNG.normal(size=500), RNG.normal(size=120)
        np.testing.assert_allclose(fc.conv_dft(x, h), np.convolve(x, h), atol=1e-10)

    def test_larger_N_only_adds_zeros(self):
        x, h = RNG.normal(size=100), RNG.normal(size=30)
        y = fc.conv_dft(x, h, 256)
        np.testing.assert_allclose(y[:129], np.convolve(x, h), atol=1e-10)
        np.testing.assert_allclose(y[129:], 0, atol=1e-10)

    def test_short_N_is_time_aliased_linear(self):
        x, h = RNG.normal(size=100), RNG.normal(size=30)
        N = 110
        lin = np.convolve(x, h)
        folded = lin[:N].copy()
        folded[: len(lin) - N] += lin[N:]
        np.testing.assert_allclose(fc.conv_dft(x, h, N), folded, atol=1e-10)

    def test_overlap_add_is_exact(self):
        x, h = RNG.normal(size=3000), RNG.normal(size=200)
        for B in (64, 500, 1024):
            y, _ = fc.overlap_add(x, h, B)
            np.testing.assert_allclose(y, np.convolve(x, h), atol=1e-10)

    def test_next_fast_length(self):
        self.assertEqual(fc.next_fast_length(158499), 160000)
        self.assertEqual(fc.next_fast_length(1000), 1000)
        self.assertEqual(fc.next_fast_length(1025), 1080)


class TestRoomTools(unittest.TestCase):
    def test_rt60_of_exponential_decay(self):
        fs = 8000
        T = 0.5
        t = np.arange(int(1.5 * fs)) / fs
        h = RNG.normal(size=len(t)) * 10 ** (-3 * t / T)       # amplitude falls 60 dB in T seconds
        rt = fc.reverberation_time(fc.schroeder_db(h), fs)[0]
        self.assertAlmostEqual(rt, T, delta=0.03)

    def test_deconvolution_recovers_input_without_noise(self):
        x, h = RNG.normal(size=2000), np.r_[1.0, 0.0, 0.5, 0.25]
        y = np.convolve(x, h)
        np.testing.assert_allclose(fc.deconvolve(y, h, len(x), 1e-12), x, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

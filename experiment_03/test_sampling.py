"""
test_sampling.py
----------------
Verifies the theory of Section 3 against the implementation.

    python -m unittest -v test_sampling
"""

import unittest

import numpy as np

import sampling as sp

FS = 48_000
t = np.arange(4800) / FS


class TestMovingAverage(unittest.TestCase):
    def test_dc_gain_is_one(self):
        y = sp.moving_average(np.full(100, 3.0), 6, centred=False)
        np.testing.assert_allclose(y[5:], 3.0)

    def test_nulls_at_multiples_of_fs_over_M(self):
        for M in (3, 6, 12):
            self.assertAlmostEqual(sp.ma_frequency_response(M, FS / M, FS), 0.0, places=9)

    def test_response_matches_fft_of_kernel(self):
        M = 6
        H = np.abs(np.fft.rfft(np.full(M, 1 / M), 4096))
        f = np.fft.rfftfreq(4096, 1 / FS)
        np.testing.assert_allclose(sp.ma_frequency_response(M, f, FS), H, atol=1e-9)

    def test_tone_at_null_is_removed(self):
        x = np.sin(2 * np.pi * 8000 * t)                 # 8 kHz = 48 kHz / 6
        y = sp.moving_average(x, 6, centred=False)
        self.assertLess(np.max(np.abs(y[10:])), 1e-9)

    def test_hand_calculation(self):
        y = sp.moving_average(np.array([0.2, 0.5, -0.1, 0.4]), 3, centred=False)
        np.testing.assert_allclose(y, [0.2 / 3, 0.7 / 3, 0.2, 0.8 / 3])


class TestSampling(unittest.TestCase):
    def test_alias_frequency(self):
        self.assertAlmostEqual(sp.alias_frequency(3100, 4000), 900)
        self.assertAlmostEqual(sp.alias_frequency(1500, 4000), 1500)
        self.assertAlmostEqual(sp.alias_frequency(5000, 4000), 1000)

    def test_undersampled_tone_appears_at_alias(self):
        x = np.sin(2 * np.pi * 3100 * t)
        xs = sp.sample(x, FS, 4000)
        f = np.fft.rfftfreq(len(xs), 1 / 4000)
        self.assertAlmostEqual(f[np.argmax(np.abs(np.fft.rfft(xs)))], 900, delta=5)

    def test_sample_length(self):
        self.assertEqual(len(sp.sample(np.zeros(4800), FS, 8000)), 800)


class TestReconstruction(unittest.TestCase):
    def setUp(self):
        self.x = np.sin(2 * np.pi * 1000 * t)            # well inside the 8 kHz band
        self.xs = sp.sample(self.x, FS, 8000)

    def test_all_methods_pass_through_samples(self):
        for m in ("Zero-order hold", "Linear", "Sinc (band-limited)"):
            y = sp.reconstruct(self.xs, 6, len(self.x), m)
            np.testing.assert_allclose(y[600:4200:6], self.xs[100:700], atol=1e-6)

    def test_sinc_beats_linear_beats_hold(self):
        snr = {m: sp.snr_db(self.x, sp.reconstruct(self.xs, 6, len(self.x), m), 600)
               for m in ("Zero-order hold", "Linear", "Sinc (band-limited)")}
        self.assertGreater(snr["Sinc (band-limited)"], snr["Linear"])
        self.assertGreater(snr["Linear"], snr["Zero-order hold"])
        self.assertGreater(snr["Sinc (band-limited)"], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)

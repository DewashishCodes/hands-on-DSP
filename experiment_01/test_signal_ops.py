"""
test_signal_ops.py
------------------
Verifies every identity from the Theory section against the implementation.

    python -m unittest -v test_signal_ops
"""

import unittest

import numpy as np

import signal_ops as ops

RNG = np.random.default_rng(7)


def rand_signal(N=20, n0=0):
    return ops.DTSignal.from_array(RNG.normal(size=N), n0)


class TestArithmetic(unittest.TestCase):
    def test_addition_is_commutative(self):
        a, b = rand_signal(), rand_signal(15, 3)
        np.testing.assert_allclose(ops.add(a, b).x, ops.add(b, a).x)

    def test_addition_on_union_of_supports(self):
        a = ops.DTSignal.from_array([1, 2, 3], 0)
        b = ops.DTSignal.from_array([10, 20], 2)
        y = ops.add(a, b)
        self.assertEqual(y.support, (0, 3))
        np.testing.assert_allclose(y.x, [1, 2, 13, 20])

    def test_subtraction_inverts_addition(self):
        a, b = rand_signal(), rand_signal()
        np.testing.assert_allclose(ops.subtract(ops.add(a, b), b).x, a.x)

    def test_multiplication_by_ones_is_identity(self):
        a = rand_signal()
        ones = ops.DTSignal(a.n, np.ones(len(a)))
        np.testing.assert_allclose(ops.multiply(a, ones).x, a.x)

    def test_energy_scales_by_A_squared(self):
        a = rand_signal()
        self.assertAlmostEqual(ops.energy(ops.amplitude_scale(a, 3.0)), 9.0 * ops.energy(a))


class TestIndexOperations(unittest.TestCase):
    def test_delay_moves_support_right(self):
        a = rand_signal(10)
        y = ops.time_shift(a, 4)
        self.assertEqual(y.support, (4, 13))
        self.assertEqual(y.value_at(4), a.value_at(0))

    def test_shift_then_inverse_shift_is_identity(self):
        a = rand_signal()
        y = ops.time_shift(ops.time_shift(a, 5), -5)
        np.testing.assert_array_equal(y.n, a.n)
        np.testing.assert_allclose(y.x, a.x)

    def test_reversal_twice_is_identity(self):
        a = rand_signal(10, 2)
        y = ops.time_reverse(ops.time_reverse(a))
        np.testing.assert_array_equal(y.n, a.n)
        np.testing.assert_allclose(y.x, a.x)

    def test_reversal_maps_n_to_minus_n(self):
        a = rand_signal(6, 1)
        y = ops.time_reverse(a)
        for n in range(1, 7):
            self.assertEqual(y.value_at(-n), a.value_at(n))

    def test_even_plus_odd_reconstructs(self):
        a = rand_signal(9)
        xe, xo = ops.even_odd_parts(a)
        recon = ops.add(xe, xo)
        for n in a.n:
            self.assertAlmostEqual(recon.value_at(n), a.value_at(n))

    def test_downsample_after_upsample_is_identity(self):
        a = rand_signal(12)
        y = ops.downsample(ops.upsample(a, 3), 3)
        np.testing.assert_allclose(y.x, a.x)

    def test_downsample_length(self):
        self.assertEqual(len(ops.downsample(rand_signal(168), 3)), 56)

    def test_upsample_inserts_L_minus_1_zeros(self):
        y = ops.upsample(ops.DTSignal.from_array([1, 2, 3]), 3)
        np.testing.assert_allclose(y.x, [1, 0, 0, 2, 0, 0, 3])

    def test_linear_interpolation_passes_through_samples(self):
        a = rand_signal(8)
        rec = ops.linear_interpolate(ops.upsample(a, 3), 3)
        np.testing.assert_allclose(rec.x[::3], a.x)


class TestConvolution(unittest.TestCase):
    def test_output_length_is_N_plus_M_minus_1(self):
        y = ops.convolve(rand_signal(168), ops.moving_average_kernel(24))
        self.assertEqual(len(y), 168 + 24 - 1)

    def test_convolution_with_impulse_is_identity(self):
        a = rand_signal()
        np.testing.assert_allclose(ops.convolve(a, ops.unit_impulse(0)).x, a.x)

    def test_convolution_with_shifted_impulse_is_delay(self):
        a = rand_signal(10)
        y = ops.convolve(a, ops.unit_impulse(3))
        self.assertEqual(y.support, (3, 12))

    def test_convolution_is_commutative(self):
        a, b = rand_signal(10), rand_signal(4)
        np.testing.assert_allclose(ops.convolve(a, b).x, ops.convolve(b, a).x)

    def test_moving_average_preserves_constant(self):
        c = ops.DTSignal.from_array(np.full(50, 27.5))
        y = ops.convolve(c, ops.moving_average_kernel(24))
        np.testing.assert_allclose(y.x[23:50], 27.5)

    def test_hand_calculation(self):
        """Matches the worked example in the Calculations section."""
        x = ops.DTSignal.from_array([23.1, 23.0, 22.8, 22.4])
        y = ops.convolve(x, ops.moving_average_kernel(3))
        np.testing.assert_allclose(y.x, [7.7, 15.3667, 22.9667, 22.7333, 15.0667, 7.4667],
                                   atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import numpy as np
import numpy.testing as npt
import tensorflow as tf


from hhpinn._sobolev3reg import sobolev3reg


class TestSobolev3Reg:
    def test__simple_case__should_have_expected_constant_values(self):
        x = tf.Variable([
            [1.0, 27.5],
            [2.3, 1.35],
            [4.4, 12.3],
        ])
        def u(x, training=False): return (tf.matmul(x, [[1], [4]]))**3
        s3reg_fn = sobolev3reg(u)

        s3_fn = s3reg_fn(x)

        # Results are computed using Mathematica.
        # dxxx, dxxy, dxyy, dxyx, dyyy, dyyx, dyxx, dyxy.
        # The derivatives sum of squares is constant in this case.
        desired = (
            6**2 + 24**2 + 96**2 + 24**2 + 384**2 + 96**2 + 24**2 + 96**2
        )

        npt.assert_allclose(s3_fn, desired, rtol=1e-7, atol=1e-7)

    def test__nonlinear_function__should_have_expected_3rd_derivatives(self):
        data = tf.random.uniform((2000000, 2))
        x, y = tf.split(data, 2, axis=1)

        def u(data, training=False):
            x, y = tf.split(data, 2, axis=1)
            uu = tf.sin(x + 4*y) + (5*x + 2*y)**4
            return uu

        s3reg_fn = sobolev3reg(u)

        s3reg = s3reg_fn(data)

        # Results are computed using Mathematica.
        # dxxx, dxxy, dxyy, dxyx, dyyy, dyyx, dyxx, dyxy.
        # The derivatives sum of squares is constant in this case.
        derivs = (
            3000 * (5 * x + 2 * y) - 1 * tf.cos(x + 4 * y),
            1200 * (5 * x + 2 * y) - 4 * tf.cos(x + 4 * y),
            480 * (5 * x + 2 * y) - 16 * tf.cos(x + 4 * y),
            1200 * (5 * x + 2 * y) - 4 * tf.cos(x + 4 * y),
            192 * (5 * x + 2 * y) - 64 * tf.cos(x + 4 * y),
            480 * (5 * x + 2 * y) - 16 * tf.cos(x + 4 * y),
            1200 * (5 * x + 2 * y) - 4 * tf.cos(x + 4 * y),
            480 * (5 * x + 2 * y) - 16 * tf.cos(x + 4 * y),
        )

        sum_of_sq = 0.0

        for der in derivs:
            sum_of_sq += der**2

        npt.assert_allclose(s3reg, sum_of_sq, rtol=1e-6, atol=1e-7)

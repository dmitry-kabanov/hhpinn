import numpy as np
import numpy.testing as npt
import tensorflow as tf


from hhpinn._sobolev3reg import sobolev3reg


class TestSobolev3Reg:
    def test__linear_function__should_have_zero_third_derivatives(self):
        x = tf.Variable([
            [1.0, 27.5],
            [2.3, 1.35],
            [4.4, 12.3],
        ])
        def u(x): return (tf.matmul(x, [[1], [4]]))**3
        s3reg_fn = sobolev3reg(u)

        s3_fn = s3reg_fn(x)

        # Results are computed using Mathematica.
        # dxxx, dxxy, dxyy, dxyx, dyyy, dyyx, dyxx, dyxy.
        # The derivatives sum of squares is constant in this case.
        desired = (
            6**2 + 24**2 + 96**2 + 24**2 + 384**2 + 96**2 + 24**2 + 96**2
        )

        npt.assert_allclose(s3_fn, desired, rtol=1e-7, atol=1e-7)

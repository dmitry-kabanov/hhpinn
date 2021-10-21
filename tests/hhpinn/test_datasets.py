import numpy as np
import numpy.testing as npt

from hhpinn.datasets import TGV2DPlusTrigonometricFlow


class TestTGV2DPlusTrigonometricFlow:
    def test_total_flow_is_sum_of_components(self):
        ds = TGV2DPlusTrigonometricFlow()

        x, tot, pot, sol = ds.load_data_on_grid(grid_size=(51, 51))

        npt.assert_allclose(tot, pot + sol, rtol=1e-7, atol=1e-7)

    def test_flow_components_are_orthogonal_in_L2_sense(self):
        ds = TGV2DPlusTrigonometricFlow()

        for s in [13, 21, 34, 55, 89, 144, 233]:
            x, tot, pot, sol = ds.load_data_on_grid(grid_size=(s, s))

            # Compute pointwise dot product at each given point.
            dot_prod_pw = np.zeros(len(tot))
            for i in range(len(tot)):
                dot_prod_pw[i] = np.dot(pot[i], sol[i])

            # Now compute inner product, which is effectively computed
            # here through mid-point quadrature rule (discarding the volume
            # of the integration domain).
            inner_prod = np.mean(dot_prod_pw)

            # Assert that inner product of potential and solenoidal fields
            # is equal to zero, that is, the fields are orthogonal.
            npt.assert_allclose(inner_prod, 0.0, rtol=1e-7, atol=1e-7)

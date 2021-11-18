import numpy as np
import numpy.testing as npt

from hhpinn.datasets import TGV2DPlusTrigonometricFlow
from hhpinn.datasets import RibeiroEtal2016Dataset


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


class TestRibeiroEtal2016:
    def test_grid_size_param_is_used(self):
        grid_size = (11, 11)
        ds = RibeiroEtal2016Dataset(grid_size)

        assert ds.grid_size == grid_size

        phi = ds.generate_phi_on_grid()
        psi = ds.generate_psi_on_grid()

        assert phi.shape == grid_size
        assert psi.shape == grid_size

    def test_constants(self):
        ds = RibeiroEtal2016Dataset((5, 5))

        assert ds.p0 == (+3.0, -3.0)
        assert ds.p1 == (-3.0, -3.0)
        assert ds.p2 == (+0.0, +3.0)

        assert ds.lb == -6.0
        assert ds.ub == +6.0

    def test_mean_for_potential_component_maximum(self):
        grid_size = (101, 101)
        ds = RibeiroEtal2016Dataset(grid_size)
        n_samples = 20

        phi_samples = []
        for i in range(n_samples):
            phi_i = ds.generate_phi_on_grid()
            phi_samples.append(phi_i)

        phi_average = np.mean(phi_samples, axis=0)
        assert phi_average.shape == grid_size

        max = np.max(phi_average)
        npt.assert_allclose(max, 1.0, rtol=1e-6, atol=1e-6)

        min = np.min(phi_average)
        npt.assert_allclose(min, -1.0, rtol=1e-6, atol=1e-6)

    def test_mean_for_solenoidal_field_maximum(self):
        grid_size = (101, 101)
        ds = RibeiroEtal2016Dataset(grid_size)
        n_samples = 20

        psi_samples = []
        for i in range(n_samples):
            psi_i = ds.generate_psi_on_grid()
            psi_samples.append(psi_i)

        psi_average = np.mean(psi_samples, axis=0)
        assert psi_average.shape == grid_size

        max = np.max(psi_average)
        npt.assert_allclose(max, 1.0, rtol=1e-6, atol=1e-6)

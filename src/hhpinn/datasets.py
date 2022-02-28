"""Module contains functions for working with datasets."""
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


class TGV2D:
    """Dataset based on Taylor--Green 2D vortex data.

    Parameters
    ----------
    N : int, optional (default 10)
        Number of measurements to choose from the generated data.
    domain : ndarray with shape (2,)
        Default is [2 pi, 2 pi].
    random_seed : int, optional (default 10)
        Random seed that is used to generate random locations.
    """

    def __init__(
        self, N=10, domain=np.asarray([2.0 * np.pi, 2.0 * np.pi]), random_seed=10
    ):
        self.N = N
        self.domain = domain
        self.random_seed = random_seed

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Choose randomly `self.N` location points and generate data."""
        np.random.seed(self.random_seed)

        N = self.N
        domain = self.domain

        xP = np.random.random_sample((N, 2)) * domain
        uP = np.zeros((N, 2))
        for i in range(N):
            uP[i] = self._vortex(xP[i])

        return xP, uP

    def load_data_on_grid(self, grid_size=(11, 11)) -> Tuple[np.ndarray, np.ndarray]:
        """Returns data in sklearn format `(X, y)` generated on the uniform grid."""
        # Note that in the `np.mgrid` notation a:b:n*1j means "create n points
        # between `a` and `b` with `b` inclusive".
        D = self.domain
        Yg, Xg = np.mgrid[
            0.0 : D[1] : grid_size[1] * 1j, 0.0 : D[0] : grid_size[0] * 1j
        ]

        result = self.eval_on_grid(Xg, Yg)

        X_col = Xg.flatten()[:, None]
        Y_col = Yg.flatten()[:, None]

        X = np.hstack((X_col, Y_col))

        u = result[0].flatten()[:, None]
        v = result[1].flatten()[:, None]
        y = np.hstack((u, v))

        return X, y

    def eval_on_grid(self, X_grid, Y_grid):
        """Evaluate vortex data on the grid (`X_grid`, `Y_grid`)."""
        result = np.zeros((2,) + X_grid.shape, dtype=float)
        for j in range(X_grid.shape[0]):
            for i in range(X_grid.shape[1]):
                x = np.asarray([X_grid[j, i], Y_grid[j, i]])
                result[:, j, i] = self._vortex(x)
        return result

    def _vortex(self, x):
        """Compute vortex field value at point `x`."""
        return np.asarray([np.cos(x[0]) * np.sin(x[1]), -np.sin(x[0]) * np.cos(x[1])])


class TGV2DPlusTrigonometricFlow:
    """Dataset based on Taylor--Green 2D vortex data plus trigonometric flow.

    Taylor-Green 2D vortex gives divergence-free (solenoidal) part of the flow
    while trigonometric part gives curl-free (potential) part of the flow.

    Parameters
    ----------
    N : int, optional (default 10)
        Number of measurements to choose from the generated data.
    domain : ndarray with shape (2,)
        Default is [2 pi, 2 pi].
    random_seed : int, optional (default 10)
        Random seed that is used to generate random locations.
    """

    def __init__(
        self, N=10, domain=np.asarray([2.0 * np.pi, 2.0 * np.pi]), random_seed=10
    ):
        self.N = N
        self.domain = domain
        self.random_seed = random_seed

        self.factor = 0.5  # Multiplier of the curl_free field.

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Choose randomly `self.N` location points and generate data."""
        np.random.seed(self.random_seed)

        N = self.N
        domain = self.domain

        xP = np.random.random_sample((N, 2)) * domain
        uP = np.zeros((N, 2))
        curl_free_uP = np.zeros((N, 2))
        div_free_uP = np.zeros((N, 2))
        for i in range(N):
            div_free_uP[i] = self._vortex(xP[i])
            curl_free_uP[i] = self._curl_free(xP[i])
            uP[i] = curl_free_uP[i] + div_free_uP[i]

        return xP, uP, curl_free_uP, div_free_uP

    def load_data_on_grid(
        self, grid_size=(11, 11)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns data in sklearn format `(X, y)` generated on the uniform grid."""
        # Note that in the `np.mgrid` notation a:b:n*1j means "create n points
        # between `a` and `b` with `b` inclusive".
        D = self.domain
        Yg, Xg = np.mgrid[
            0.0 : D[1] : grid_size[1] * 1j, 0.0 : D[0] : grid_size[0] * 1j
        ]

        result, curl_free_result, div_free_result = self.eval_on_grid(Xg, Yg)

        X_col = Xg.flatten()[:, None]
        Y_col = Yg.flatten()[:, None]

        X = np.hstack((X_col, Y_col))

        u = result[0].flatten()[:, None]
        v = result[1].flatten()[:, None]
        y = np.hstack((u, v))

        cfu = curl_free_result[0].flatten()[:, None]
        cfv = curl_free_result[1].flatten()[:, None]
        cfy = np.hstack((cfu, cfv))

        dfu = div_free_result[0].flatten()[:, None]
        dfv = div_free_result[1].flatten()[:, None]
        dfy = np.hstack((dfu, dfv))

        return X, y, cfy, dfy

    def eval_on_grid(self, X_grid, Y_grid):
        """Evaluate vortex data on the grid (`X_grid`, `Y_grid`)."""
        result = np.zeros((2,) + X_grid.shape, dtype=float)
        curl_free_result = np.zeros((2,) + X_grid.shape, dtype=float)
        div_free_result = np.zeros((2,) + X_grid.shape, dtype=float)
        for j in range(X_grid.shape[0]):
            for i in range(X_grid.shape[1]):
                x = np.asarray([X_grid[j, i], Y_grid[j, i]])
                curl_free_result[:, j, i] = self._curl_free(x)
                div_free_result[:, j, i] = self._vortex(x)
                result[:, j, i] = curl_free_result[:, j, i] + div_free_result[:, j, i]

        return result, curl_free_result, div_free_result

    def _curl_free(self, x):
        """Compute potential flow at point `x`."""
        return self.factor * np.asarray(
            [
                -2 * np.sin(2 * x[0]) * np.cos(3 * x[1]),
                -3 * np.cos(2 * x[0]) * np.sin(3 * x[1]),
            ]
        )

    def _vortex(self, x):
        """Compute vortex field value at point `x`."""
        return np.asarray([np.cos(x[0]) * np.sin(x[1]), -np.sin(x[0]) * np.cos(x[1])])


class RibeiroEtal2016Dataset:
    """Dataset from the paper https://doi.org/10.1016/j.cag.2016.01.001

    Note that we implement and use this dataset in a different manner from
    their paper. In their paper, they use fixed grid 100x100 to create
    measurements, where each sample of that size is randomized by randomizing
    the locations of the "origins" of the Gaussians.

    We here do not randomize the centers of the Gaussians.

    """
    def __init__(self):
        self.p0 = (+3.0, -3.0)  # Source center in potential field.
        self.p1 = (-3.0, -3.0)  # Sink center in potential field.
        self.p2 = (+0.0, +3.0)  # Vortex center in solenoidal field.

        self.lb, self.ub = (-6.0, 6.0)

        self.domain = ((-6.0, 6.0), (-6.0, 6.0))

    def generate_phi_on_grid(self, grid_size=(11, 11)):
        x0, y0 = self.p0
        x1, y1 = self.p1

        x = np.linspace(self.lb, self.ub, grid_size[0])
        y = np.linspace(self.lb, self.ub, grid_size[1])
        self.xx, self.yy = np.meshgrid(x, y)

        xx, yy = self.xx, self.yy

        source = np.exp(-0.5 * ((xx-x0)**2 + (yy-y0)**2))
        sink = -np.exp(-0.5 * ((xx-x1)**2 + (yy-y1)**2))
        return source + sink

    def generate_psi_on_grid(self, grid_size=(11, 11)):
        x = np.linspace(self.lb, self.ub, grid_size[0])
        y = np.linspace(self.lb, self.ub, grid_size[1])
        self.xx, self.yy = np.meshgrid(x, y)
        xx, yy = self.xx, self.yy
        x2, y2 = self.p2

        return np.exp(-0.5 * ((xx-x2)**2 + (yy - y2)**2))

    def plot_phi(self):
        """Plot potential (phi) component of the vector field.

        Returns
        -------
        fig : plt.Figure
            Handle to matplotlib Figure object.
        """
        phi = self.generate_phi_on_grid()

        fig = plt.figure()
        plt.pcolormesh(self.xx, self.yy, phi)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.colorbar()
        plt.tight_layout(pad=0.1)

        return fig

    def plot_psi(self):
        """Plot solenoidal (psi) component of the vector field.

        Returns
        -------
        fig : plt.Figure
            Handle to matplotlib Figure object.
        """
        psi = self.generate_psi_on_grid()

        fig = plt.figure()
        plt.pcolormesh(self.xx, self.yy, psi)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.colorbar()
        plt.tight_layout(pad=0.1)

        return fig

    def generate_potential_velocity_on_grid(self, grid_size=(11, 11)):
        """Return u, v for potential field."""
        x0, y0 = self.p0
        x1, y1 = self.p1

        x = np.linspace(self.lb, self.ub, grid_size[0])
        y = np.linspace(self.lb, self.ub, grid_size[1])
        self.xx, self.yy = np.meshgrid(x, y)
        xx, yy = self.xx, self.yy

        source = np.exp(-0.5 * ((xx-x0)**2 + (yy-y0)**2))
        sink = -np.exp(-0.5 * ((xx-x1)**2 + (yy-y1)**2))

        # Derivatives of the expression inside Gaussians:
        dexp0_dx = -(xx-x0)
        dexp1_dx = -(xx-x1)
        dexp0_dy = -(yy-y0)
        dexp1_dy = -(yy-y1)

        u = source * dexp0_dx + sink * dexp1_dx
        v = source * dexp0_dy + sink * dexp1_dy

        return u, v

    def generate_solenoidal_velocity_on_grid(self, grid_size=(11, 11)):
        """Return u, v for potential field."""
        x2, y2 = self.p2

        x = np.linspace(self.lb, self.ub, grid_size[0])
        y = np.linspace(self.lb, self.ub, grid_size[1])
        self.xx, self.yy = np.meshgrid(x, y)
        xx, yy = self.xx, self.yy

        psi = np.exp(-0.5 * ((xx-x2)**2 + (yy - y2)**2))

        # Derivatives of the expression inside the Gaussian:
        dpsi_dx = -(xx-x2)
        dpsi_dy = -(yy-y2)

        u = -psi * dpsi_dy
        v = +psi * dpsi_dx

        return u, v

    def load_data_on_grid(self, grid_size=(11, 11)):
        x = np.linspace(self.lb, self.ub, grid_size[0])
        y = np.linspace(self.lb, self.ub, grid_size[1])
        self.xx, self.yy = np.meshgrid(x, y)
        xx, yy = self.xx, self.yy

        X = np.column_stack((
            np.reshape(xx, (-1, 1)),
            np.reshape(yy, (-1, 1))
        ))

        u_pot, v_pot = self.generate_potential_velocity_on_grid(grid_size)

        U_pot = np.column_stack((
            np.reshape(u_pot, (-1, 1)),
            np.reshape(v_pot, (-1, 1)),
        ))

        u_sol, v_sol = self.generate_solenoidal_velocity_on_grid(grid_size)

        U_sol = np.column_stack((
            np.reshape(u_sol, (-1, 1)),
            np.reshape(v_sol, (-1, 1))
        ))

        U = U_pot + U_sol

        return X, U, U_pot, U_sol


    def compute_inner_product(self, grid_size=(11, 11)):
        """Compute inner product of subfields."""

        u_pot, v_pot = self.generate_potential_velocity_on_grid(grid_size)
        u_sol, v_sol = self.generate_solenoidal_velocity_on_grid(grid_size)

        vel_pot = np.column_stack((
            np.reshape(u_pot, (-1, 1)),
            np.reshape(v_pot, (-1, 1)),
        ))

        vel_sol = np.column_stack((
            np.reshape(u_sol, (-1, 1)),
            np.reshape(v_sol, (-1, 1)),
        ))

        dot_prod_pw = np.sum(vel_pot * vel_sol, axis=1)

        ip = np.mean(dot_prod_pw)

        return ip


class TomaharuSudaDataset:
    """Dataset based on the talk by Tomaharu Suda.

    The scalar potential phi = 0.5 (x**2 + y**2) and solenoidal part (-y, x)^T.

    See slide 9 from presentation at
    http://math.bu.edu/keio2018/talks/TSuda.pdf

    Parameters
    ----------
    N : int, optional (default 10)
        Number of measurements to choose from the generated data.
    domain : ndarray with shape (2,)
        Default is [(-10, 10), (-10, 10)].
    random_seed : int, optional (default 10)
        Random seed that is used to generate random locations.
    """

    def __init__(
        self, N=10, domain=np.asarray([(-10, 10), (-10, 10)]), random_seed=10
    ):
        self.N = N
        self.domain = domain
        self.random_seed = random_seed

        self.factor = 0.5  # Multiplier of the curl_free field.

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Choose randomly `self.N` location points and generate data."""
        np.random.seed(self.random_seed)

        N = self.N
        domain = self.domain

        xP = np.random.random_sample((N, 2)) * domain
        uP = np.zeros((N, 2))
        curl_free_uP = np.zeros((N, 2))
        div_free_uP = np.zeros((N, 2))
        for i in range(N):
            div_free_uP[i] = self._vortex(xP[i])
            curl_free_uP[i] = self._curl_free(xP[i])
            uP[i] = curl_free_uP[i] + div_free_uP[i]

        return xP, uP, curl_free_uP, div_free_uP

    def load_data_on_grid(
        self, grid_size=(11, 11)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns data in sklearn format `(X, y)` generated on the uniform grid."""
        # Note that in the `np.mgrid` notation a:b:n*1j means "create n points
        # between `a` and `b` with `b` inclusive".
        D = self.domain
        Yg, Xg = np.mgrid[
            D[0][0] : D[0][1] : grid_size[1] * 1j, D[1][0] : D[1][1] : grid_size[0] * 1j
        ]

        result, curl_free_result, div_free_result = self.eval_on_grid(Xg, Yg)

        X_col = Xg.flatten()[:, None]
        Y_col = Yg.flatten()[:, None]

        X = np.hstack((X_col, Y_col))

        u = result[0].flatten()[:, None]
        v = result[1].flatten()[:, None]
        y = np.hstack((u, v))

        cfu = curl_free_result[0].flatten()[:, None]
        cfv = curl_free_result[1].flatten()[:, None]
        cfy = np.hstack((cfu, cfv))

        dfu = div_free_result[0].flatten()[:, None]
        dfv = div_free_result[1].flatten()[:, None]
        dfy = np.hstack((dfu, dfv))

        return X, y, cfy, dfy

    def eval_on_grid(self, X_grid, Y_grid):
        """Evaluate vortex data on the grid (`X_grid`, `Y_grid`)."""
        result = np.zeros((2,) + X_grid.shape, dtype=float)
        curl_free_result = np.zeros((2,) + X_grid.shape, dtype=float)
        div_free_result = np.zeros((2,) + X_grid.shape, dtype=float)
        for j in range(X_grid.shape[0]):
            for i in range(X_grid.shape[1]):
                x = np.asarray([X_grid[j, i], Y_grid[j, i]])
                curl_free_result[:, j, i] = self._curl_free(x)
                div_free_result[:, j, i] = self._vortex(x)
                result[:, j, i] = curl_free_result[:, j, i] + div_free_result[:, j, i]

        return result, curl_free_result, div_free_result

    def _curl_free(self, x):
        """Compute potential flow at point `x`."""
        return np.asarray([-x[0], -x[1]])

    def _vortex(self, x):
        """Compute vortex field value at point `x`."""
        return np.asarray([-x[1], x[0]])

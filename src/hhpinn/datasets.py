"""Module contains functions for working with datasets."""
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


class TGV2DPlusPotentialPart:
    """Dataset based on Taylor--Green 2D vortex data plus potential flow

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
            uP[i] = self._vortex(xP[i]) + self._curl_free(xP[i])

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

    def _curl_free(self, x):
        """Compute potential flow at point `x`."""
        return np.asarray(
            [
                -2 * np.sin(2 * x[0]) * np.cos(3 * x[1]),
                -3 * np.cos(2 * x[0]) * np.sin(3 * x[1]),
            ]
        )

"""Functions that can are used to compute performance scores."""
import logging

import numpy as np


logger = logging.getLogger()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Return mean squared error."""
    if y_true.shape != y_pred.shape:
        logger.warning("`y_true` and `y_pred` have different shapes")

    errors = np.linalg.norm(y_true - y_pred, 2, axis=1)
    assert len(errors) == len(y_true)
    error_mse = np.mean(errors)

    return error_mse


def rel_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Return relative mean squared error."""
    if y_true.shape != y_pred.shape:
        logger.warning("`y_true` and `y_pred` have different shapes")

    num = np.sum(np.linalg.norm(y_true - y_pred, 2, axis=1)**2)
    den = np.sum(np.linalg.norm(y_true, 2, axis=1)**2)
    error_mse = num / den

    return error_mse


def rel_root_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Return relative root mean squared error."""
    return np.sqrt(rel_mse(y_true, y_pred))


def rel_pw_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Return normalized pointwise error.

    The error is computed as
    $$
    e(x_i) = \sqrt{
        || y(x_i) - \hat y(x_i) ||^2 / (\sum_j || y(x_j) ||^2),
    }
    $$
    that is, in terms of discretized L2 norms.

    This function should give results consistent with `rel_root_mse`.
    """
    err = np.linalg.norm(y_true - y_pred, 2, axis=1)**2
    rel_err = np.sqrt(
        err / np.sum(np.linalg.norm(y_true, 2, axis=1)**2)
    )

    return rel_err

"""Functions that can are used to compute performance scores."""
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    errors = np.linalg.norm(y_true - y_pred, 2, axis=1)
    error_mse = np.mean(errors)

    return error_mse


def rel_pw_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Return normalized pointwise error: e(x) = || y(x) - \hat y(x) ||."""
    err = np.linalg.norm(y_true - y_pred, 2, axis=1)
    rel_err = err / np.mean(np.linalg.norm(y_true, 2, axis=1))

    return rel_err

import numpy as np
import numpy.testing as npt


from hhpinn.scoring import rel_mse


def test_rel_mse__equal_inputs__give_zero_error():
    x = np.random.random((1000, 2))

    actual = rel_mse(x, x)
    desired = 0.0

    npt.assert_allclose(actual, desired, atol=1e-7)


def test_rel_mse__error_equal_true_values__gives_unity():
    # When predicted values are zero, the relative error should be unity.
    x = np.array([[1.0, 5.0], [4.0, 2.0]])
    y = np.zeros_like(x)

    actual = rel_mse(x, y)
    desired = 1.0

    npt.assert_allclose(actual, desired, atol=1e-7)


def test_rel_mse__given_input__works_correctly():
    x = np.array([
        [1.0, 2.0],
        [7.0, 1.0],
        [5.0, 3.0],
    ])
    y = np.array([
        [2.0, 2.0],
        [3.0, 9.0],
        [4.0, 6.0],
    ])

    actual = rel_mse(x, y)

    resid = x - y

    num, den = 0.0, 0.0
    for i in range(len(x)):
        num += (resid[i, 0])**2 + (resid[i, 1])**2
        den += (x[i, 0])**2 + (x[i, 1])**2

    desired = num / den

    npt.assert_allclose(actual, desired)


def test_rel_mse__random_input__check_works_correctly():
    N = 3
    x = np.random.random(size=(N, 2))
    y = np.random.random(size=(N, 2))

    actual = rel_mse(x, y)

    num, den = 0.0, 0.0
    for i in range(N):
        num += (x[i, 0] - y[i, 0])**2 + (x[i, 1] - y[i, 1])**2
        den += (x[i, 0])**2 + (x[i, 1])**2

    desired = num / den

    npt.assert_allclose(actual, desired, rtol=1e-6, atol=1e-8)

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


try:
    from IPython import get_ipython
    ip = str(get_ipython())
    if "zmqshell" in ip:
        plt.style.use("seaborn")
        plt.style.use("seaborn-notebook")
    else:
        plt.style.use("seaborn")
        plt.style.use("seaborn-notebook")
except (ImportError, NameError):
    # Running in console Python
    plt.style.use("seaborn")
    plt.style.use("seaborn-paper")
    matplotlib.rcParams["figure.figsize"] = (6, 3.7)

FIGSIZE_DEFAULT = matplotlib.rcParams["figure.figsize"]
FIGSIZE_WIDE = (1.5*FIGSIZE_DEFAULT[0], FIGSIZE_DEFAULT[1])


def plot_stream_field_2D(N, domain, x_values, u_values, true_values=None):
    if len(N) != 2 or len(domain) != 2:
        raise ValueError("Should be two-dimensional")

    if x_values.shape[1] != 2 or u_values.shape[1] != 2:
        raise ValueError("Should be two-dimensional")

    X = np.reshape(x_values[:, 0], N)
    Y = np.reshape(x_values[:, 1], N)
    U = np.reshape(u_values[:, 0], N)
    V = np.reshape(u_values[:, 1], N)

    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(1, 1)
    lw = 5.0 * speed / speed.max()
    al = 2.0 * speed.max()

    if true_values is not None:
        # Plot circles at the locations.
        plt.scatter(x_values[:, 0], x_values[:, 1], s=10, c="red")

        for i in range(len(x_values)):
            ax.arrow(
                np.real(x_values[i, 0]),
                np.real(x_values[i, 1]),
                np.real(true_values[i, 0]) / al,
                np.real(true_values[i, 1]) / al,
                head_width=0.08,
                head_length=0.08,
                width=0.03,
                fc="b",
                ec="b",
                clip_on=False,
            )

    ax.streamplot(
        X,
        Y,
        U,
        V,
        linewidth=lw,
        color="k",
        density=1.0,
        arrowstyle="->",
        arrowsize=1,
    )

    # We check here if the domain of type [0, Lx]x[0, Ly] or [Lx1, Lx2]x[Ly1, Ly2].
    if isinstance(domain[0], (int, float)):
        xleft = -0.02 * domain[0]
        xright = 1.02 * domain[0]
        yleft = -0.02 * domain[0]
        yright = 1.02 * domain[1]
        Lx = float(domain[0])
        Ly = float(domain[1])
    else:
        xleft = 1.02 * domain[0][0]
        xright = 1.02 * domain[0][1]
        yleft = 1.02 * domain[1][0]
        yright = 1.02 * domain[1][1]
        Lx = float(domain[0][1] - domain[0][0])
        Ly = float(domain[1][1] - domain[1][0])

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(xleft, xright)
    ax.set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax.set_aspect("equal")
    fig.tight_layout(pad=0.1)


def plot_true_and_pred_stream_fields(grid_size, domain, x, true_u, pred_u):
    """Plot true and predicted stream fields `true_u` and `pred_u`.

    Parameters
    ----------
    grid_size : tuple (nx: int, ny: int)
        Grid size.
    domain : tuple (Lx, Ly)
        Domain size.
    x, true_u, pred_u : ndarray (nx*ny, 2)
        Locations, true and predicted vector fields.

    """
    if len(grid_size) != 2 or len(domain) != 2:
        raise ValueError("Should be two-dimensional")

    if x.shape[1] != 2:
        raise ValueError("Should be two-dimensional")

    if true_u.shape != x.shape:
        raise ValueError("Shape mismatch")

    if pred_u.shape != x.shape:
        raise ValueError("Shape mismatch")

    X = np.reshape(x[:, 0], grid_size)
    Y = np.reshape(x[:, 1], grid_size)
    true_U = np.reshape(true_u[:, 0], grid_size)
    true_V = np.reshape(true_u[:, 1], grid_size)
    pred_U = np.reshape(pred_u[:, 0], grid_size)
    pred_V = np.reshape(pred_u[:, 1], grid_size)

    true_speed = np.sqrt(true_U**2 + true_V**2)
    pred_speed = np.sqrt(pred_U**2 + pred_V**2)
    true_lw = 5.0 * true_speed / true_speed.max()
    pred_lw = 5.0 * pred_speed / pred_speed.max()

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    kw_props = dict(
        color="k",
        density=1.0,
        arrowstyle="->",
        arrowsize=1,
    )

    # We check here if the domain of type [0, Lx]x[0, Ly] or [Lx1, Lx2]x[Ly1, Ly2].
    if isinstance(domain[0], (int, float)):
        xleft = -0.02 * domain[0]
        xright = 1.02 * domain[0]
        yleft = -0.02 * domain[0]
        yright = 1.02 * domain[1]
        Lx = float(domain[0])
        Ly = float(domain[1])
    else:
        xleft = 1.02 * domain[0][0]
        xright = 1.02 * domain[0][1]
        yleft = 1.02 * domain[1][0]
        yright = 1.02 * domain[1][1]
        Lx = float(domain[0][1] - domain[0][0])
        Ly = float(domain[1][1] - domain[1][0])

    ax[0].streamplot(X, Y, true_U, true_V, linewidth=true_lw, **kw_props)
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_xlim(xleft, xright)
    ax[0].set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax[0].set_aspect("equal")

    ax[1].streamplot(X, Y, pred_U, pred_V, linewidth=pred_lw, **kw_props)
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_xlim(xleft, xright)
    ax[1].set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax[1].set_aspect("equal")

    fig.tight_layout(pad=0.1)


def plot_true_and_two_pred_stream_fields(grid_size, domain, x, true_u,
                                         pred_u_1, pred_u_2):
    """Plot true and predicted stream fields `true_u` and `pred_u`.

    Parameters
    ----------
    grid_size : tuple (nx: int, ny: int)
        Grid size.
    domain : tuple (Lx, Ly)
        Domain size.
    x, true_u, pred_u : ndarray (nx*ny, 2)
        Locations, true and predicted vector fields.

    """
    if len(grid_size) != 2 or len(domain) != 2:
        raise ValueError("Should be two-dimensional")

    if x.shape[1] != 2:
        raise ValueError("Should be two-dimensional")

    if true_u.shape != x.shape:
        raise ValueError("Shape mismatch")

    if pred_u_1.shape != x.shape:
        raise ValueError("Shape mismatch")

    if pred_u_2.shape != x.shape:
        raise ValueError("Shape mismatch")

    X = np.reshape(x[:, 0], grid_size)
    Y = np.reshape(x[:, 1], grid_size)
    true_U = np.reshape(true_u[:, 0], grid_size)
    true_V = np.reshape(true_u[:, 1], grid_size)
    pred_U_1 = np.reshape(pred_u_1[:, 0], grid_size)
    pred_V_1 = np.reshape(pred_u_1[:, 1], grid_size)
    pred_U_2 = np.reshape(pred_u_2[:, 0], grid_size)
    pred_V_2 = np.reshape(pred_u_2[:, 1], grid_size)

    true_speed = np.sqrt(true_U**2 + true_V**2)
    pred_speed_1 = np.sqrt(pred_U_1**2 + pred_V_1**2)
    pred_speed_2 = np.sqrt(pred_U_2**2 + pred_V_2**2)
    true_lw = 4.0 * true_speed / true_speed.max()
    pred_lw_1 = 4.0 * pred_speed_1 / pred_speed_1.max()
    pred_lw_2 = 4.0 * pred_speed_2 / pred_speed_2.max()

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=FIGSIZE_WIDE)

    kw_props = dict(
        color="k",
        density=1.0,
        arrowstyle="->",
        arrowsize=1,
    )

    # We check here if the domain of type [0, Lx]x[0, Ly] or [Lx1, Lx2]x[Ly1, Ly2].
    if isinstance(domain[0], (int, float)):
        xleft = -0.02 * domain[0]
        xright = 1.02 * domain[0]
        yleft = -0.02 * domain[0]
        yright = 1.02 * domain[1]
        Lx = float(domain[0])
        Ly = float(domain[1])
    else:
        xleft = 1.02 * domain[0][0]
        xright = 1.02 * domain[0][1]
        yleft = 1.02 * domain[1][0]
        yright = 1.02 * domain[1][1]
        Lx = float(domain[0][1] - domain[0][0])
        Ly = float(domain[1][1] - domain[1][0])

    ax[0].streamplot(X, Y, true_U, true_V, linewidth=true_lw, **kw_props)
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_xlim(xleft, xright)
    ax[0].set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax[0].set_aspect("equal")

    ax[1].streamplot(X, Y, pred_U_1, pred_V_1, linewidth=pred_lw_1, **kw_props)
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_xlim(xleft, xright)
    ax[1].set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax[1].set_aspect("equal")

    ax[2].streamplot(X, Y, pred_U_2, pred_V_2, linewidth=pred_lw_2, **kw_props)
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")
    ax[2].set_xlim(xleft, xright)
    ax[2].set_ylim(yleft, yright)
    if 0.98 <= Lx / Ly <= 1.02:
        ax[2].set_aspect("equal")

    fig.tight_layout(pad=0.1)


def plot_error_field_2D(inputs, errors, grid_size, locs=[], vmin=None,
                      vmax=None, cbar_label=None):
    plot_scalar_field(inputs, errors, grid_size, locs, vmin,
                      vmax, cbar_label)


def plot_scalar_field(inputs, errors, grid_size, locs=[], vmin=None,
                      vmax=None, cbar_label=None):
    assert inputs.ndim == 2
    assert np.prod(errors.shape) == errors.shape[0]
    assert inputs.shape[1] == 2

    if len(locs):
        assert locs.ndim == 2
        assert locs.shape[1] == 2

    xg = inputs[:, 0].reshape(grid_size)
    yg = inputs[:, 1].reshape(grid_size)
    err_ug = errors.reshape(grid_size)

    if vmin is None:
        vmin = np.min(err_ug)

    if vmax is None:
        vmax = np.max(err_ug)

    if np.any(errors < vmin):
        # Some values in the `errors` array are lower than the argument `vmin`.
        logger.warning("Truncation of values due to `vmin`")

    if np.any(errors > vmax):
        # Some values in the `errors` array are lower than the argument `vmin`.
        logger.warning("Truncation of values due to `vmax`")

    fig = plt.figure()
    # Use `shading="nearest"` to plot field with the same dimensions
    # as points `xg` and `yg`.
    plt.pcolormesh(xg, yg, err_ug, shading="nearest", vmin=vmin, vmax=vmax)
    # It is important that `colorbar` is invoked immediately after
    # `pcolormesh`, otherwise, the range of the colorbar can be affected
    # by the following optional plotting of the points `locs`.
    plt.colorbar(label=cbar_label)
    if len(locs):
        plt.scatter(locs[:, 0], locs[:, 1], s=30, c="red")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    fig.tight_layout(pad=0.3)

    return fig

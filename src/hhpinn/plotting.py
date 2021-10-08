import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.02*domain[0], 1.02*domain[0])
    ax.set_ylim(-0.02*domain[1], 1.02*domain[1])
    if 0.98 <= domain[0] / domain[1] <= 1.02:
        ax.set_aspect("equal")
    fig.tight_layout(pad=0.1)


def plot_error_field_2D(inputs, errors, grid_size, locs=[], vmax=None):
    assert inputs.ndim == 2
    assert errors.ndim == 1
    assert inputs.shape[1] == 2

    if len(locs):
        assert locs.ndim == 2
        assert locs.shape[1] == 2

    xg = inputs[:, 0].reshape(grid_size)
    yg = inputs[:, 1].reshape(grid_size)
    err_ug = errors.reshape(grid_size)

    if vmax is None:
        vmax = np.max(err_ug)

    fig = plt.figure()
    # Use `shading="nearest"` to plot field with the same dimensions
    # as points `xg` and `yg`.
    plt.pcolormesh(xg, yg, err_ug, shading="nearest", vmax=vmax)
    # It is important that `colorbar` is invoked immediately after
    # `pcolormesh`, otherwise, the range of the colorbar can be affected
    # by the following optional plotting of the points `locs`.
    plt.colorbar()
    if len(locs):
        plt.scatter(locs[:, 0], locs[:, 1], s=30, c="red")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    fig.tight_layout(pad=0.3)

    return fig

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle


def plot_stream_field_2D(N, domain, x_values, u_values):
    if len(N) != 2 or len(domain) != 2:
        raise ValueError("Should be two-dimensional")

    if x_values.shape[1] != 2 or u_values.shape[1] != 2:
        raise ValueError("Should be two-dimensional")

    X = np.reshape(x_values[:, 0], N)
    Y = np.reshape(x_values[:, 1], N)
    U = np.reshape(u_values[:, 0], N)
    V = np.reshape(u_values[:, 1], N)

    speed = np.sqrt(U**2 + V**2)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    color = 2.0 * np.log(np.hypot(U, V))
    lw = 5.0 * speed / speed.max()
    al = 5.0 * speed.max()
    for i in range(len(x_values)):
        ax.add_artist(
            Circle(np.real(x_values[i]), 0.01, color="red")
        )
        ax.arrow(
            np.real(x_values[i, 0]),
            np.real(x_values[i, 1]),
            np.real(u_values[i, 0]) / al,
            np.real(u_values[i, 1]) / al,
            head_width=0.015,
            head_length=0.015,
            width=0.005,
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
        density=2.0,
        arrowstyle="->",
        arrowsize=0.5,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.1*domain[0], 1.1*domain[0])
    ax.set_ylim(-0.1*domain[1], 1.1*domain[1])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.1)

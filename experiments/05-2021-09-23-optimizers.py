#!/usr/bin/env python
# %% [markdown]
# # 05-2021-09-23 Optimizers to mitigate vanishing-gradient problem

# %% Imports

# %%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from typing import List

from hhpinn import StreamFunctionPINN
from hhpinn.utils import render_figure


# %% md
# ## Global variables

# %%
OUTDIR = "_output"

CONFIGS = [
    [10],
    [20],
    [50],
    [100],
    [1000],
]

RESULT_MODEL_TEMPLATE = os.path.join(OUTDIR, "model-{:d}")


# %%
try:
    from IPython import get_ipython
    ip = str(get_ipython())
    if "zmqshell" in ip:
        args = dict(save=True)
    else:
        args = dict(save=True)
except (ImportError, NameError):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--save",
        "-s",
        action="store_true",
        default=False,
        help="Save figures to disk",
    )
    args = vars(p.parse_args())

# %% [markdown]
# ## Load data
#
# We use Taylor--Green vortex with 10 measurements and random seed 10.

# %%
ds = hhpinn.datasets.TGV2D()
train_x, train_u = ds.load_data()

# %%
models: List[StreamFunctionPINN] = []

# %% md
# ## Run

# %% md
# We train models with increasing number of neurons using ADAM optimizer.

# %%
if not os.listdir(OUTDIR):
    models = []
    for i, c in enumerate(CONFIGS):
        model = StreamFunctionPINN(
            hidden_layers=c,
            epochs=1000,
            learning_rate=0.01,
            save_grad_norm=True,
            optimizer="adam",
        )
        models.append(model)
        model.fit(train_x, train_u)
        savedir = RESULT_MODEL_TEMPLATE.format(i)
        os.makedirs(savedir)
        model.save(savedir)

# %% md
# ## Processing results

# %%
# Load models from disk.
models = []
for i, c in enumerate(CONFIGS):
    m = StreamFunctionPINN.load(
        RESULT_MODEL_TEMPLATE.format(i)
    )
    models.append(m)

# %%
# Define styles
styles = ["-", "--", "o", "s", "."]

# %%
# Plot loss history.
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.plot(
        range(1, len(models[i].history["loss"])+1, 50),
        models[i].history["loss"][::50],
        styles[i],
        label=c)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "loss-history.pdf"),
    save=args["save"]
)

# %% Plot gradient norms during training.
step = 50
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_inf_norm"])+1, step),
        models[i].history["grad_inf_norm"][::step],
        styles[i],
        label=c)
plt.xlabel("Epochs")
plt.ylabel("Gradient Inf norm")
plt.legend(loc="upper right")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-inf-history.pdf"),
    save=args["save"]
)


# %% Plot gradient Euclidean norms during training.
step = 50
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_l2_norm"])+1, step),
        models[i].history["grad_l2_norm"][::step],
        styles[i],
        label=c)
plt.xlabel("Epochs")
plt.ylabel("Gradient 2-norm")
plt.legend(loc="upper right")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-euclid-history.pdf"),
    save=args["save"]
)

# %%
model = models[1]

grid_size = (11, 11)
test_x, test_u = ds.load_data_on_grid(grid_size)
pred_u = model.predict(test_x)

hhpinn.plotting.plot_stream_field_2D(
    grid_size, ds.domain, test_x, test_u
)

render_figure(
)

hhpinn.plotting.plot_stream_field_2D(
    grid_size, ds.domain, test_x, pred_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field.pdf"),
    save=args["save"],
)

err_u = np.linalg.norm(pred_u - test_u, 2, axis=1)

xg = test_x[:, 0].reshape(grid_size)
yg = test_x[:, 1].reshape(grid_size)
err_ug = err_u.reshape(grid_size)

plt.figure()
plt.pcolormesh(xg, yg, err_ug)
plt.scatter(train_x[:, 0], train_x[:, 1], s=30, c="red")
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout(pad=0.1)

render_figure(
    to_file=os.path.join("_assets", "error-field.pdf"),
    save=args["save"]
)

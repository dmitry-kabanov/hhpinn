#!/usr/bin/env python
# %% [markdown]
# # 06-2021-09-27 Preprocessing output variables
#
# There is a clear problem with gradient. The idea is that maybe
# we need to preprocess the output variables to stabilize the computation
# of the derivatives of the stream functions.
#
# ## Imports

# %%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from typing import List

from hhpinn import StreamFunctionPINN
from hhpinn.utils import render_figure


# %% [markdown]
# ## Global variables

# %%
OUTDIR = "_output"

CONFIGS = [
    ([10], "sgd"),
    ([100], "sgd"),
    ([1000], "sgd"),
    ([10], "adam"),
    ([100], "adam"),
    ([1000], "adam"),
]

# Grid size for test data.
GRID_SIZE = (11, 11)

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
# We use Taylor--Green vortex with 10 measurements and random seed 10 for
# training in the domain $[0; 2\pi]^2$.
# The test dataset is defined on the uniform grid.

# %%
ds = hhpinn.datasets.TGV2D()
train_x, train_u = ds.load_data()
test_x, test_u = ds.load_data_on_grid(GRID_SIZE)


# %%
models: List[StreamFunctionPINN] = []

# %% [markdown]
# ## Run
#
# We train models with different configurations: number of neurons and
# optimizers (SGD or ADAM).

# %%
if not os.listdir(OUTDIR):
    models = []
    for i, (h, opt) in enumerate(CONFIGS):
        model = StreamFunctionPINN(
            hidden_layers=h,
            epochs=1000,
            learning_rate=0.01,
            save_grad_norm=True,
            save_grad=100,
            optimizer=opt,
            preprocessing="standardization-both",
        )
        models.append(model)
        model.fit(train_x, train_u)
        savedir = RESULT_MODEL_TEMPLATE.format(i)
        os.makedirs(savedir)
        model.save(savedir)
else:
    print("OUTDIR not empty, skipping computations")

# %% [markdown]
# ## Processing results
#
# ### Preparation for processing

# %%
# Load models from disk.
models = []
for i, __ in enumerate(CONFIGS):
    m = StreamFunctionPINN.load(
        RESULT_MODEL_TEMPLATE.format(i)
    )
    models.append(m)

# %%
# Define styles
styles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 5))]
# Step between epochs for plotting.
step = 50

# %% [markdown]
# ### Plot loss history

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["loss"])+1, step),
        models[i].history["loss"][::step],
        linestyle=styles[i],
        label=str(c))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "loss-history.pdf"),
    save=args["save"]
)

# %% [markdown]
# ### Plot gradient infinity norm during training

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_inf_norm"])+1, step),
        models[i].history["grad_inf_norm"][::step],
        linestyle=styles[i],
        label=c)
plt.xlabel("Epochs")
plt.ylabel("Gradient Inf norm")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-inf-history.pdf"),
    save=args["save"]
)

# %% [markdown]
# ### Plot gradient Euclidean norm during training

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_l2_norm"])+1, step),
        models[i].history["grad_l2_norm"][::step],
        linestyle=styles[i],
        label=c)
plt.xlabel("Epochs")
plt.ylabel("Gradient 2-norm")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-euclid-history.pdf"),
    save=args["save"]
)

# %% [markdown]
# ### Plot gradient distributions
# #### Model [1000], "ADAM"

# %%
n_model = 5
model = models[n_model]
assert model.hidden_layers == [1000]
assert model.optimizer == "adam"

idx = list(model.history["grad"].keys())
snapshots = [idx[0], idx[int(len(idx)*0.5)], idx[-1]]
# snapshots = idx
fig, axes = plt.subplots(nrows=1, ncols=len(snapshots), sharey=True,
                         figsize=(24, 3))
for i, e in enumerate(snapshots):
    g = model.history["grad"][e]
    axes[i].hist(g, density=True)
    axes[i].set_xlabel(r"Gradient components")
    axes[i].set_xlim((-0.02, 0.02))
    axes[i].set_title("Epoch %d" % e)
axes[0].set_ylabel(r"Density")
fig.tight_layout(pad=0.1)

render_figure(
    to_file=os.path.join("_assets", f"grad-density-model={n_model}.pdf"),
    save=args["save"],
)

# %% [markdown]
# ### Plot predictability of largest models
# We want to predict the following field:

# %%
hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, test_u
)

render_figure(
)

# %% [markdown]
# #### SGD and ADAM models, predicted field

# %%
model_sgd = models[2]
model_adam = models[5]
assert model_sgd.hidden_layers == model_adam.hidden_layers
pred_u_sgd = model_sgd.predict(test_x)
pred_u_adam = model_adam.predict(test_x)

# %%
hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_sgd, test_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model-sgd.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_adam, test_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model-adam.pdf"),
    save=args["save"],
)

# %% [markdown]
# #### Error fields in SGD and ADAM models

# %% [markdown]
# This is a comparison of the pointwise error fields between a SGD and
# ADAM-trained models.
# We can see that the ADAM models is more accurate.

# %%
err_u_sgd = np.linalg.norm(pred_u_sgd - test_u, 2, axis=1)
hhpinn.plotting.plot_error_field_2D(test_x, err_u_sgd, train_x, GRID_SIZE)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=sgd.pdf"),
    save=args["save"]
)

err_u_adam = np.linalg.norm(pred_u_adam - test_u, 2, axis=1)
hhpinn.plotting.plot_error_field_2D(test_x, err_u_adam, train_x, GRID_SIZE)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=adam.pdf"),
    save=args["save"]
)

# %%

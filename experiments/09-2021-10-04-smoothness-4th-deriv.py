#!/usr/bin/env python
# %% [markdown]
# # 09-2021-10-04 Adding regularizer on fourth derivatives
#
# This is to try to prove a hypothesis that adding regularizer on higher-order
# derivatives will improve predicted fields.
# This paper:
# Sadati et al. *Hard vs soft constraints in the full field reconstruction of
# incompressible flow kinematics from noisy scattered velocimetry data*, 2011,
# proposes to use fourth-order derivative regularizer.

# %% [markdown]
# ## Imports

# %%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from collections import namedtuple
from typing import List

from hhpinn import StreamFunctionPINN
from hhpinn.utils import render_figure
from hhpinn.scoring import rel_pw_error


# %% [markdown]
# ## Global variables

# %%
OUTDIR = "_output"

# Configurations of the neural networks:
# hidden-layers, optimizer, multiplier of Sobolev4 regularizer.
Config = namedtuple("Config", ["hl", "opt", "s4"])
CONFIGS = [
    Config([2000], "adam", 0e-0),
    Config([2000], "adam", 1e-6),
    Config([2000], "adam", 1e-4),
    Config([2000], "adam", 1e-3),
    Config([2000], "adam", 1e-2),
    Config([2000], "adam", 1e-1),
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
ds = hhpinn.datasets.TGV2D(N=10)
train_x, train_u = ds.load_data()
test_x, test_u = ds.load_data_on_grid(GRID_SIZE)


# %%
models: List[StreamFunctionPINN] = []

# %% [markdown]
# ## Run
#
# We train models with different configurations (see `CONFIGS`).
# %%
if not os.listdir(OUTDIR):
    models = []
    for i, c in enumerate(CONFIGS):
        model = StreamFunctionPINN(
            hidden_layers=c.hl,
            epochs=1000,
            l2=0,
            s4=c.s4,
            learning_rate=0.01,
            save_grad_norm=True,
            save_grad=100,
            optimizer=c.opt,
        )
        models.append(model)
        model.fit(train_x, train_u)
        savedir = RESULT_MODEL_TEMPLATE.format(i)
        os.makedirs(savedir)
        model.save(savedir)
else:
    print("OUTDIR not empty, skipping computations")

# %% [markdown]
# ## Preparation for processing

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
# ## Plot loss history

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
# ## Plot gradient infinity norm during training

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
# ## True field
# We want to predict the following field:

# %%
hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, test_u
)

render_figure(
    to_file=os.path.join("_assets", "true-field.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## Errors of all models

# %%
error_mse_list = []
print("Mean Squared Errors")
print("-------------------")
for i, (c, model) in enumerate(zip(CONFIGS, models)):
    pred = model.predict(test_x)
    errors = np.linalg.norm(pred - test_u, 2, axis=1)
    error_mse = np.mean(errors)
    print("{:} Model {:44s} {:.2e}".format(i, str(c), error_mse))
    error_mse_list.append(error_mse)

plt.figure()
plt.plot(error_mse_list, "o")
plt.xlabel("Model index")
plt.ylabel("Prediction MSE")
plt.tight_layout(pad=0.1)
best_model_idx = np.argmin(error_mse_list)
print()
print("Best model index: ", best_model_idx)

render_figure(
    to_file=os.path.join("_assets", "pred-mse-vs-model.pdf"),
    save=args["save"],
)

# %% [markdown]
# The problem with the above results is that they are a bit random as they
# change from one run to another due to randomness in simulations:
# all models are initialized randomly and randomness is different for each
# model.

# %% [markdown]
# ## Predicted fields of the first, last, and best models

# %%
model_first = models[0]
model_last = models[-1]
model_best = models[best_model_idx]
pred_u_first = model_first.predict(test_x)
pred_u_last = model_last.predict(test_x)
pred_u_best = model_best.predict(test_x)

# %%
hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_first, test_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=first.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_last, test_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=last.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_best, test_u
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=best.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## Error fields of the first, last, and best models

# %% [markdown]
# This is a comparison of the pointwise error fields between
# different models.

# %%
err_u_first = rel_pw_error(test_u, pred_u_first)
hhpinn.plotting.plot_error_field_2D(test_x, err_u_first, GRID_SIZE, train_x)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=first.pdf"),
    save=args["save"]
)

err_u_last = rel_pw_error(test_u, pred_u_last)
hhpinn.plotting.plot_error_field_2D(test_x, err_u_last, GRID_SIZE, train_x)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=last.pdf"),
    save=args["save"]
)

err_u_best = rel_pw_error(test_u, pred_u_best)
hhpinn.plotting.plot_error_field_2D(test_x, err_u_best, GRID_SIZE, train_x)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=best.pdf"),
    save=args["save"]
)


# %% [markdown]
# ## Divergence fields of the first and last models
#
# Sanity checks that the neural networks construct divergence-free fields.
# Computations are done in single-precision, which has a machine epsilon of
# $1.19 \times 10^{-7}$.

# %%
div_field_first = model_first.compute_divergence(test_x)
div_field_first_norm = np.linalg.norm(div_field_first, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, div_field_first, GRID_SIZE, locs=train_x)
print(f"Divergence field first Linf-norm: {div_field_first_norm:.2e}")

div_field_last = model_last.compute_divergence(test_x)
div_field_last_norm = np.linalg.norm(div_field_last, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, div_field_last, GRID_SIZE, locs=train_x)
print(f"Divergence field last Linf-norm: {div_field_last_norm:.2e}")

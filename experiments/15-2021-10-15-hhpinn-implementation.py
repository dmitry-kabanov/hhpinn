# %% [markdown]
# # 15-2021-10-15 HHPINN2D
#
# I implement a neural network for Helmholtz-Hodge decomposition
# of two-dimensional vector fields here.

# %% [markdown]
# ## Imports

# %%
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import hhpinn

from collections import namedtuple
from typing import List

from hhpinn import HHPINN2D
from hhpinn.utils import render_figure
from hhpinn.scoring import mse, rel_mse, rel_pw_error


# %% [markdown]
# ## Global variables

# %%
OUTDIR = "_output"

# Configurations of the neural networks:
# hidden-layers, optimizer, multiplier of Sobolev4 regularizer.
Config = namedtuple("Config", ["hl", "opt", "s4"])
CONFIGS = [
    Config([150], "adam", 0e-0),
    Config([150], "adam", 1e-4),
    Config([150], "adam", 1e-3),
    Config([150], "adam", 1e-2),
    Config([150], "adam", 1e-1),
    Config([150], "adam", 1e0),
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
ds = hhpinn.datasets.TGV2DPlusTrigonometricFlow(N=200)
train_x, train_u, train_u_curl_free, train_u_div_free = ds.load_data()
test_x, test_u, test_u_curl_free, test_u_div_free = ds.load_data_on_grid(GRID_SIZE)


# %%
models: List[HHPINN2D] = []

# %% [markdown]
# ## Run
#
# We train models with different configurations (see `CONFIGS`).
# %%
if not os.listdir(OUTDIR):
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.1,
    #     decay_steps=300,
    #     decay_rate=0.1,
    # )
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [200, 500, 1000], [0.1, 0.05, 0.01, 0.001]
    )
    start = time.time()
    models = []
    for i, c in enumerate(CONFIGS):
        model = HHPINN2D(
            hidden_layers=c.hl,
            epochs=3000,
            l2=0,
            s4=c.s4,
            optimizer=c.opt,
            learning_rate=lr,
            save_grad_norm=True,
            save_grad=100,
        )
        models.append(model)
        model.fit(train_x, train_u, validation_data=(test_x, test_u))
        savedir = RESULT_MODEL_TEMPLATE.format(i)
        os.makedirs(savedir)
        model.save(savedir)
    end = time.time()
    print(f"Computation time: {end-start:.2f} sec")
else:
    print("OUTDIR not empty, skipping computations")

# %% [markdown]
# ## Preparation for processing

# %%
# Load models from disk.
models = []
for i, __ in enumerate(CONFIGS):
    m = HHPINN2D.load(RESULT_MODEL_TEMPLATE.format(i))
    models.append(m)

# %%
# Define styles
styles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 5)), (0, (8, 8))]
# Step between epochs for plotting.
step = 1

# %% [markdown]
# ## Plot loss history

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["loss"]) + 1, step),
        models[i].history["loss"][::step],
        linestyle=styles[i],
        label=str(c),
    )
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(to_file=os.path.join("_assets", "loss-history.pdf"), save=args["save"])

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["misfit"]) + 1, step),
        models[i].history["misfit"][::step],
        linestyle=styles[i],
        label=str(c),
    )
plt.xlabel("Epochs")
plt.ylabel("Misfit loss")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["sobolev4"]) + 1, step),
        models[i].history["sobolev4"][::step],
        linestyle=styles[i],
        label=str(c),
    )
plt.xlabel("Epochs")
plt.ylabel("Sobolev4 loss")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

# %% [markdown]
# ## Plot validation loss history

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["val_loss"]) + 1, step),
        models[i].history["val_loss"][::step],
        linestyle=styles[i],
        label=str(c),
    )
plt.xlabel("Epochs")
plt.ylabel("Validation loss")
plt.legend(loc="upper left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "val-loss-history.pdf"), save=args["save"]
)


# %% [markdown]
# ## Plot gradient infinity norm during training

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_phi_inf_norm"]) + 1, step),
        models[i].history["grad_phi_inf_norm"][::step],
        linestyle=styles[i],
        label=c,
    )
plt.xlabel("Epochs")
plt.ylabel("Gradient phi Inf norm")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-phi-inf-history.pdf"), save=args["save"]
)

# %%
plt.figure()
for i, c in enumerate(CONFIGS):
    plt.semilogy(
        range(1, len(models[i].history["grad_psi_inf_norm"]) + 1, step),
        models[i].history["grad_psi_inf_norm"][::step],
        linestyle=styles[i],
        label=c,
    )
plt.xlabel("Epochs")
plt.ylabel("Gradient psi Inf norm")
plt.legend(loc="lower left")
plt.tight_layout(pad=0.3)

render_figure(
    to_file=os.path.join("_assets", "grad-psi-inf-history.pdf"), save=args["save"]
)

# %% [markdown]
# ## Errors of all models

# %%
error_mse_list = []
print("Mean squared errors on test dataset")
print("-----------------------------------")
for i, (c, model) in enumerate(zip(CONFIGS, models)):
    pred = model.predict(test_x)
    error_mse = mse(test_u, pred)
    # Sanity check that the validation loss at the end of training
    # is the same as prediction MSE here because I use the same data.
    assert error_mse == model.history["val_loss"][-1]
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
# My observation is that the best coefficient with Sobolev4 regularizer
# for the solenoidal part is equal to 1e-4 or 1e-3 in all runs.

# %% [markdown]
# ## True full field

# %%
hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, test_u)

render_figure(
    to_file=os.path.join("_assets", "true-field.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## Predicted fields of the first, last, and best models

# %%
model_first = models[0]
model_last = models[-1]
model_best = models[best_model_idx]
pred_u_first = model_first.predict(test_x)
pred_u_last = model_last.predict(test_x)
pred_u_best, pred_u_best_curl_free, pred_u_best_div_free = model_best.predict(
    test_x, return_separate_fields=True
)

# %%
hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, pred_u_first, test_u)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=first.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, pred_u_last, test_u)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=last.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, pred_u_best, test_u)

render_figure(
    to_file=os.path.join("_assets", "pred-field-model=best.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## True and predicted best field for potential (curl-free) part

# %%
hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, test_u_curl_free)

render_figure(
    to_file=os.path.join("_assets", "true-field-curl-free.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(
    GRID_SIZE, ds.domain, test_x, pred_u_best_curl_free
)

render_figure(
    to_file=os.path.join("_assets", "pred-field-curl-free.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## True and predicted best field for solenoidal (div-free) part

# %%
hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, test_u_div_free)

render_figure(
    to_file=os.path.join("_assets", "true-field-div-free.pdf"),
    save=args["save"],
)

hhpinn.plotting.plot_stream_field_2D(GRID_SIZE, ds.domain, test_x, pred_u_best_div_free)

render_figure(
    to_file=os.path.join("_assets", "pred-field-div-free.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## Error fields of the first, last, and best models

# %% [markdown]
# This is a comparison of the pointwise error fields between
# different models.
# The errors are relative and shown in the same scale to make it easy
# to compare them.

# %%
err_u_first = rel_pw_error(test_u, pred_u_first)
err_u_last = rel_pw_error(test_u, pred_u_last)
err_u_best = rel_pw_error(test_u, pred_u_best)
err_max = np.max((err_u_first, err_u_last, err_u_best))

hhpinn.plotting.plot_error_field_2D(
    test_x,
    err_u_first,
    GRID_SIZE,
    train_x,
    vmin=0.0,
    vmax=err_max,
    cbar_label="Fraction",
)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=first.pdf"), save=args["save"]
)

hhpinn.plotting.plot_error_field_2D(
    test_x,
    err_u_last,
    GRID_SIZE,
    train_x,
    vmin=0.0,
    vmax=err_max,
    cbar_label="Fraction",
)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=last.pdf"), save=args["save"]
)

hhpinn.plotting.plot_error_field_2D(
    test_x,
    err_u_best,
    GRID_SIZE,
    train_x,
    vmin=0.0,
    vmax=err_max,
    cbar_label="Fraction",
)

render_figure(
    to_file=os.path.join("_assets", "error-field-model=best.pdf"), save=args["save"]
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

# %% [markdown]
# # 19-2021-11-18 Sequential training with Sobolev 3 regularizer
#
# I implement here sequential training, where first one neural network is
# trained, and then the other one on the residual.
# The difference from the previous experiment is that here I use regularizers
# to smooth the 3rd derivative of the potential fields.

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

from hhpinn.models import SequentialHHPINN2D
from hhpinn.plotting import plot_true_and_pred_stream_fields
from hhpinn.utils import render_figure
from hhpinn.scoring import mse, rel_pw_error


# %% [markdown]
# ## Global variables

# %%
OUTDIR = "_output"

# Configurations of the neural networks:
# hidden-layers, optimizer, multiplier of orthogonality regularizer.
Config = namedtuple("Config", ["hl", "opt", "s3"])
CONFIGS = [
    Config([20], "adam", 0e-0),
    Config([50], "adam", 1e-4),
    Config([100], "adam", 1e-3),
    Config([150], "adam", 1e-2),
    Config([200], "adam", 1e-1),
    Config([250], "adam", 1e-0),
]

# Grid size for validation data.
VAL_GRID_SIZE = (11, 11)

# Grid size for test data.
TEST_GRID_SIZE = (101, 101)

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
val_x, val_u, val_u_curl_free, val_u_div_free = ds.load_data_on_grid(VAL_GRID_SIZE)
test_x, test_u, test_u_curl_free, test_u_div_free = ds.load_data_on_grid(
    TEST_GRID_SIZE
)

# %%
models: List[SequentialHHPINN2D] = []

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
        model = SequentialHHPINN2D(
            hidden_layers=c.hl,
            epochs=3000,
            l2=0.0,
            s3=c.s3,
            s4=0.0,
            ip=0.0,
            optimizer=c.opt,
            learning_rate=lr,
            save_grad_norm=True,
            save_grad=100,
        )
        models.append(model)
        model.fit(train_x, train_u, validation_data=(val_x, val_u))
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
    m = SequentialHHPINN2D.load(RESULT_MODEL_TEMPLATE.format(i))
    models.append(m)

# %% [markdown]
# ## MSE of all models

# %%
error_mse_list = []
print("Mean squared errors on test dataset")
print("-----------------------------------")
for i, (c, model) in enumerate(zip(CONFIGS, models)):
    pred = model.predict(test_x)
    error_mse = mse(test_u, pred)
    # Sanity check that the validation loss at the end of training
    # is the same as prediction MSE here because I use the same data.
    # assert error_mse == model.history["val_loss"][-1]
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
#
# We can see here, that all errors are significantly larger than for the
# simultaneous model with orthogonality regularizer, where the best error was
# about 0.2.

# %% [markdown]
# ## True and predicted fields: first, last, and best models

# %%
model_first = models[0]
model_last = models[-1]
model_best = models[best_model_idx]
pred_u_first, pred_u_first_curl_free, pred_u_first_div_free = model_first.predict(
    test_x, return_separate_fields=True
)
pred_u_last, pred_u_last_curl_free, pred_u_last_div_free = model_last.predict(
    test_x, return_separate_fields=True
)
pred_u_best, pred_u_best_curl_free, pred_u_best_div_free = model_best.predict(
    test_x, return_separate_fields=True
)

# %%
plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u, pred_u_first)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-model=first.pdf"),
    save=args["save"],
)

plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u, pred_u_last)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-model=last.pdf"),
    save=args["save"],
)

plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u, pred_u_best)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-model=best.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## True and predicted best field for potential (curl-free) part

# %%
plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_curl_free, pred_u_first_curl_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-curl-free-model=first.pdf"),
    save=args["save"],
)

plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_curl_free, pred_u_last_curl_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-curl-free-model=last.pdf"),
    save=args["save"],
)


plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_curl_free, pred_u_best_curl_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-curl-free-model=best.pdf"),
    save=args["save"],
)


# %% [markdown]
# ## True and predicted best field for solenoidal (div-free) part

# %%
plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_div_free, pred_u_first_div_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-div-free-model=first.pdf"),
    save=args["save"],
)

plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_div_free, pred_u_last_div_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-div-free-model=last.pdf"),
    save=args["save"],
)

plot_true_and_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_div_free, pred_u_best_div_free
)

render_figure(
    to_file=os.path.join("_assets", "true-vs-pred-div-free-model=best.pdf"),
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
    TEST_GRID_SIZE,
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
    TEST_GRID_SIZE,
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
    TEST_GRID_SIZE,
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
hhpinn.plotting.plot_error_field_2D(test_x, div_field_first, TEST_GRID_SIZE, locs=train_x)
print(f"Divergence field first Linf-norm: {div_field_first_norm:.2e}")

div_field_last = model_last.compute_divergence(test_x)
div_field_last_norm = np.linalg.norm(div_field_last, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, div_field_last, TEST_GRID_SIZE, locs=train_x)
print(f"Divergence field last Linf-norm: {div_field_last_norm:.2e}")


# %% [markdown]
# ## Curl of potential (curl-free) fields of the first, last, and best models
#
# Sanity checks that the neural networks construct the potential part of
# the fields which really satisfies the curl-free condition.
# Computations are done in single-precision, which has a machine epsilon of
# $1.19 \times 10^{-7}$.
# Anything close to this number can be considered zero.

# %%
pot_field_first = model_first.compute_curl_of_potential_field(test_x)
pot_field_first_norm = np.linalg.norm(pot_field_first, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, pot_field_first, TEST_GRID_SIZE, locs=train_x)
print(f"Curl of potential field, first model, Linf-norm: {pot_field_first_norm:.2e}")

pot_field_last = model_last.compute_curl_of_potential_field(test_x)
pot_field_last_norm = np.linalg.norm(pot_field_last, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, pot_field_last, TEST_GRID_SIZE, locs=train_x)
print(f"Curl of potential field, last model, Linf-norm: {pot_field_last_norm:.2e}")

pot_field_best = model_best.compute_curl_of_potential_field(test_x)
pot_field_best_norm = np.linalg.norm(pot_field_best, np.Inf)
hhpinn.plotting.plot_error_field_2D(test_x, pot_field_best, TEST_GRID_SIZE, locs=train_x)
print(f"Curl of potential field, best model, Linf-norm: {pot_field_best_norm:.2e}")

# %% [markdown]
# ## Orthogonality of fields

# %%
ip_field_first = model_first.compute_inner_product(test_x)
hhpinn.plotting.plot_error_field_2D(test_x, ip_field_first, TEST_GRID_SIZE)
print(f"Inner product, first model: {ip_field_first.mean():.3f}")

ip_field_last = model_last.compute_inner_product(test_x)
hhpinn.plotting.plot_error_field_2D(test_x, ip_field_last, TEST_GRID_SIZE)
print(f"Inner product, last model: {ip_field_last.mean():.3f}")

ip_field_best = model_best.compute_inner_product(test_x)
hhpinn.plotting.plot_error_field_2D(test_x, ip_field_best, TEST_GRID_SIZE)
print(f"Inner product, best model: {ip_field_best.mean():.3f}")

# %%

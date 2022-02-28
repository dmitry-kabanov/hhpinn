# %% [markdown]
# # 27-2022-02-14 One layer network with Ribeiro dataset.
#
# I want to see the performance of the neural network on RibeiroEtAl2016 dataset
# that it probably a bit easier to fit.
# Besides, the train data are given on the uniform grid.
#
# For the definition of the dataset, please see the corresponding code in
# `hhpinn/datasets.py`.

# %% [markdown]
# ## Imports

# %%
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import hhpinn

from collections import namedtuple
from pathlib import Path
from typing import List

from hhpinn import HHPINN2D
from hhpinn.plotting import plot_true_and_pred_stream_fields
from hhpinn.plotting import plot_true_and_two_pred_stream_fields
from hhpinn.utils import render_figure
from hhpinn.scoring import rel_root_mse, rel_pw_error


# %% [markdown]
# ## Global variables

# %%
OUTDIR = "_output"

# Configurations of the neural networks:
# hidden-layers, optimizer, multiplier of orthogonality regularizer.
Config = namedtuple("Config", ["hl", "opt", "ip"])
CONFIGS = [
    Config([64], "adam", 0.0),
    Config([64], "adam", 1e-3),
    Config([64], "adam", 1e-1),
    Config([64], "adam", 1e-0),
    Config([64], "adam", 1e+1),
    Config([64], "adam", 1e+2),
]

# Grid size for training data.
TRAIN_GRID_SIZE = (51, 51)

# Grid size for validation data.
VAL_GRID_SIZE = (16, 16)

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

# %%
ds = hhpinn.datasets.RibeiroEtal2016Dataset()
train_x, train_u, train_u_curl_free, train_u_div_free = ds.load_data_on_grid(TRAIN_GRID_SIZE)
val_x, val_u, val_u_curl_free, val_u_div_free = ds.load_data_on_grid(VAL_GRID_SIZE)
test_x, test_u, test_u_curl_free, test_u_div_free = ds.load_data_on_grid(TEST_GRID_SIZE)


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
        [200, 500, 1000, 10000], [0.1, 0.05, 0.01, 0.001, 0.0001]
    )
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     [200, 500, 1000], [0.1, 0.09, 0.05, 0.01]
    # )
    #lr = 1e-1
    start = time.time()
    models = []
    for i, c in enumerate(CONFIGS):
        model = HHPINN2D(
            hidden_layers=c.hl,
            epochs=100000,
            l2=0,
            s3=0,
            s4=0,
            ip=c.ip,
            G=32,
            use_uniform_grid_for_regs=True,
            use_batch_normalization=True,
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
if not models:
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
test_u_pot = test_u_curl_free
test_u_sol = test_u_div_free
pot_mse_list = []
sol_mse_list = []
error_mse_list = []
lambda_list = []
print("Relative RMSE on test dataset")
print("-----------------------------------")
print("{} Model {:44s} {:>6s} {:>5s} {:>5s}".format("", "config", "Total", "Pot", "Sol"))
for i, (c, model) in enumerate(zip(CONFIGS, models)):
    pred = model.predict(test_x)
    error_mse = rel_root_mse(test_u, pred)
    pred_pot, pred_sol = model.predict_separate_fields(test_x)
    pot_rmse = rel_root_mse(test_u_pot, pred_pot)
    sol_rmse = rel_root_mse(test_u_sol, pred_sol)
    print("{:} Model {:44s} {:5.0f} {:>5.0f} {:>5.0f}".format(
            i, str(c), error_mse*100, pot_rmse*100, sol_rmse*100
        )
    )
    lambda_list.append(c.ip)
    error_mse_list.append(error_mse)
    pot_mse_list.append(pot_rmse)
    sol_mse_list.append(sol_rmse)

sci_fmt = lambda x: r"\num{%.0e}" % x
per_fmt = lambda x: "%d" % (x*100)
formatters = [sci_fmt, per_fmt, per_fmt, per_fmt]
header = [r"$\lambda$", r"$e_{\text{tot}}$", r"$e_{\text{pot}}$",
           r"$e_{\text{sol}}$"]
err_df = pd.DataFrame({
    "lambda": lambda_list,
    "rrmse": error_mse_list,
    "pot_rrmse": pot_mse_list,
    "sol_rrmse": sol_mse_list,
})
err_df.index = err_df.index + 1
err_df.to_latex(
    Path("_assets") / "ribeiro-errors.tex",
    header=header,
    formatters=formatters,
    escape=False
)


plt.figure()
plt.plot(error_mse_list, "o", label="Total")
plt.plot(pot_mse_list, "s", label="Potential")
plt.plot(sol_mse_list, "*", label="Solenoidal")
plt.xlabel("Model index")
plt.ylabel("Relative prediction RMSE")
plt.legend(loc="upper right")
plt.tight_layout(pad=0.1)
best_model_idx = np.argmin(error_mse_list)
print()
print("Best model index (wrt total error): ", best_model_idx)
best_model_idx = np.argmin(pot_mse_list + sol_mse_list)
print("Best model index (wrt subfields error): ", best_model_idx)

render_figure(
    to_file=os.path.join("_assets", "pred-rel-rmse-vs-model.pdf"),
    save=args["save"],
)

# %% [markdown]
# ## True and predicted fields of the first, last, and best models

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

plot_true_and_two_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u,
    pred_u_first, pred_u_best
)

render_figure(
    to_file=os.path.join("_assets", "recon-total-first-best.pdf"),
    save=args["save"],
)

render_figure(
    to_file=os.path.join("_assets", "recon-total-first-best.png"),
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

plot_true_and_two_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_curl_free,
    pred_u_first_curl_free, pred_u_best_curl_free
)

render_figure(
    to_file=os.path.join("_assets", "recon-pot-first-best.pdf"),
    save=args["save"],
)

render_figure(
    to_file=os.path.join("_assets", "recon-pot-first-best.png"),
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

plot_true_and_two_pred_stream_fields(
    TEST_GRID_SIZE, ds.domain, test_x, test_u_div_free,
    pred_u_first_div_free, pred_u_best_div_free
)

render_figure(
    to_file=os.path.join("_assets", "recon-sol-first-best.pdf"),
    save=args["save"],
)

render_figure(
    to_file=os.path.join("_assets", "recon-sol-first-best.png"),
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
max_err_u_first = np.linalg.norm(err_u_first, np.Inf)
max_err_u_last = np.linalg.norm(err_u_last, np.Inf)
max_err_u_best = np.linalg.norm(err_u_best, np.Inf)

print(f"Max relative pointwise error, first model: {max_err_u_first:.3f}")
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

print(f"Max relative pointwise error, last model: {max_err_u_last:.3f}")
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

print(f"Max relative pointwise error, best model: {max_err_u_best:.3f}")
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
div_Linf_norm_list = []
print("Linf norm of divergence fields on test dataset")
print("Closer to zero is better")
print("-----------------------------------")
print("{} {:>8s} {:>10s}".format("", "ip", "Linf-norm"))
for i, (c, model) in enumerate(zip(CONFIGS, models)):
    div_field = model.compute_divergence(test_x)
    div_Linf_norm = np.linalg.norm(div_field, np.Inf)
    print("{:} {:>8.1e} {:>8.1e}".format(i, c.ip, div_Linf_norm))
    div_Linf_norm_list.append(div_Linf_norm)

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

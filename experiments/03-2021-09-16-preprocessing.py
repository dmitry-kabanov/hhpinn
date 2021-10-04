# %%
# %pwd

# %% [markdown]
# # 03-2021-09-16-preprocessing

# %% [markdown]
# ## Setup

# %%
# # %load code/experiments/03-2021-09-16-preprocessing.py
# #!/usr/bin/env python
# %matplotlib inline
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from typing import Dict, List

from hhpinn import HodgeHelmholtzPINN
from hhpinn.utils import render_figure

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

ds = hhpinn.datasets.TGV2D()
train_x, train_u = ds.load_data()

# %%
models: List[HodgeHelmholtzPINN] = []

# %% [markdown]
# ## Compute

# %%
if not os.listdir(OUTDIR):
    models = []
    for i, c in enumerate(CONFIGS):
        model = HodgeHelmholtzPINN(
            hidden_layers=c,
            epochs=1000,
            learning_rate=0.01,
        )
        models.append(model)
        model.fit(train_x, train_u)
        savedir = RESULT_MODEL_TEMPLATE.format(i)
        os.makedirs(savedir)
        model.save(savedir)

# %% [markdown]
# ## Results

# %%
args = {"save": True}

# %% [markdown]
# Load the results from disc:

# %%
ds = hhpinn.datasets.TGV2D()
train_x, train_u = ds.load_data()

models: List[HodgeHelmholtzPINN] = []
for i, c in enumerate(CONFIGS):
    m = HodgeHelmholtzPINN.load(
        RESULT_MODEL_TEMPLATE.format(i)
    )
    models.append(m)

# %%
styles = ["-", "--", "o", "s", "."]
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

# %% [markdown]
# Now let's plot true and predicted field. Also the pointwise error field.

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

err_u = np.linalg.norm(pred_u - test_u, 2, axis=1)

xg = test_x[:, 0].reshape(grid_size)
yg = test_x[:, 1].reshape(grid_size)
err_ug = err_u.reshape(grid_size)

plt.figure()
plt.pcolormesh(xg, yg, err_ug)
plt.scatter(train_x[:, 0], train_x[:, 1], s=10, c="red")
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout(pad=0.1)

render_figure(
    to_file=os.path.join("_assets", "error-field.pdf"),
    save=args["save"]
)

# %%

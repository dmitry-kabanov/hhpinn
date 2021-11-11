# %% [markdown]
# # 17-2021-10-29 RibeiroEtal2016 dataset
#
# I implement vector field from the paper:
# Ribeiro, P. C., de Campos Velho, H. F., & Lopes, H. (2016)
# Helmholtz–Hodge decomposition and the analysis of 2D vector field ensembles.
# Computers & Graphics, 55, 80-96, doi:10.1016/j.cag.2016.01.001

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
# hidden-layers, optimizer, multiplier of orthogonality regularizer.
Config = namedtuple("Config", ["hl", "opt", "ip"])
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
ds = hhpinn.datasets.RibeiroEtal2016Dataset()

# %% [markdown]
# ## Scalar potential field

# %%
ds.plot_phi()

# %% [markdown]
# ## Vector potential field

# %%
ds.plot_psi()

# %% [markdown]
# ## Are subfields orthogonal?
# If the following method returns a value close to zero, then yes.

# %%
ds.compute_inner_product()

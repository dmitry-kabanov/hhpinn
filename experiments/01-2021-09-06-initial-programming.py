#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from typing import Dict

from hhpinn.utils import render_figure


OUTDIR = "_output"

RESULT_MODEL = os.path.join(OUTDIR, "model.pkl")


def main(args=None):
    args = parse_args(args)

    if not has_computed():
        print("Running compute()")
        compute()
    else:
        print("Running plot()")
        plot(args)


def parse_args(args=None) -> Dict:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--save",
        "-s",
        action="store_true",
        default=False,
        help="Save figures to disk",
    )
    args = vars(p.parse_args(args))

    return args


def has_computed():
    if os.listdir(OUTDIR):
        return True
    else:
        return False


def compute():
    ds = hhpinn.datasets.TGV2D()
    train_x, train_u = ds.load_data()

    model = hhpinn.models.HodgeHelmholtzPINN(
        hidden_layers=[20],
        epochs=20000,
        learning_rate=0.01,
    )

    model.fit(train_x, train_u)

    model.save(RESULT_MODEL)


def plot(args: Dict):
    ds = hhpinn.datasets.TGV2D()
    train_x, train_u = ds.load_data()

    model = hhpinn.models.HodgeHelmholtzPINN.load(RESULT_MODEL)

    plt.figure()
    plt.plot(model.history["loss"], "-", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.1)

    render_figure(
        to_file=os.path.join("_assets", "loss-history.pdf"),
        save=args["save"]
    )

    grid_size = (11, 11)
    test_x, test_u = ds.load_data_on_grid(grid_size)
    pred_u = model.predict(test_x)

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

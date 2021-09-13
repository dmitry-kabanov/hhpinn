#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import hhpinn

from typing import Dict


OUTDIR = "_output"


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
        hidden_layers=[10],
        epochs=50,
        learning_rate=0.01,
    )

    model.fit(train_x, train_u)

    plt.figure()
    plt.plot(model.history["loss"], "-", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.1)

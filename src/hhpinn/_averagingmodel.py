"""
Baseline model that simply averages its input and predicts a constant field.
"""
import numpy as np


class AveragingModel:
    def __init__(self):
        pass

    def fit(self, x, y):
        self.y_mean_ = np.mean(y, axis=0)
        assert len(self.y_mean_) == y.shape[1]

    def predict(self, x_new):
        """Predict a constant mean field irrespective of `x_new`."""
        y_pred = np.empty_like(x_new)
        for i in range(len(x_new)):
            y_pred[i] = self.y_mean_

        return y_pred

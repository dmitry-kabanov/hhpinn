import os
import pickle

import numpy as np
import tensorflow as tf

from typing import Dict, List, Union

from sklearn.preprocessing import StandardScaler

from hhpinn.scoring import mse

from ._averagingmodel import AveragingModel
from ._hhpinn2d import HHPINN2D
from ._sequentialhhpinn2d import SequentialHHPINN2D


class StreamFunctionPINN:
    """Physics-informed neural network for divergence-free 2D vector fields."""

    def __init__(
        self,
        hidden_layers=[10],
        epochs=50,
        l2=0.0,
        s4=0.0,
        optimizer="sgd",
        learning_rate=0.01,
        preprocessing="identity",
        save_grad_norm=False,
        save_grad: int = 0,
    ):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.l2 = l2
        self.s4 = s4
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.preprocessing = preprocessing
        self.save_grad_norm = save_grad_norm
        self.save_grad = save_grad
        self._nparams = 9

        self.model = None
        self.history: Dict[str, Union[Dict, List]] = {}
        self.transformer = None
        self.transformer_output = None

    def get_params(self):
        params = {
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "l2": self.l2,
            "s4": self.s4,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "preprocessing": self.preprocessing,
            "save_grad_norm": self.save_grad_norm,
            "save_grad": self.save_grad,
        }
        assert len(params) == self._nparams

        return params

    def build_model(self) -> tf.keras.models.Model:
        """Build and return Keras model with given hyperparameters."""
        inp = tf.keras.layers.Input(2)
        x = inp
        for neurons in self.hidden_layers:
            x = tf.keras.layers.Dense(
                neurons,
                activation="tanh",
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            )(x)

        out = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=False,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
        )(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        return model

    def fit(self, x, y, validation_data=None):
        # Preprocess training data.
        if self.preprocessing == "identity":
            xs = x
            ys = y
        elif self.preprocessing == "standardization":
            self.transformer = StandardScaler()
            self.transformer.fit(x)
            xs = self.transformer.transform(x)
            ys = y
        elif self.preprocessing == "standardization-both":
            self.transformer = StandardScaler()
            self.transformer.fit(x)
            xs = self.transformer.transform(x)
            self.transformer_output = StandardScaler()
            self.transformer_output.fit(y)
            ys = self.transformer_output.transform(y)
        else:
            raise ValueError("Unknown values for preprocessing")

        # Training data should be `tf.Variable`s, so that the GradientTape
        # could watch them automatically.
        x_train = tf.Variable(xs, dtype=tf.float32)
        y_train = tf.Variable(ys, dtype=tf.float32)

        xmin = (xs[:, 0].min(), xs[:, 1].min())
        xmax = (xs[:, 0].max(), xs[:, 1].max())

        # Instantiate model.
        model = self.build_model()
        self.model = model

        if self.l2 < 0.0:
            raise ValueError("Multiplier of L2 regularizer should be non-negative")

        if self.s4 < 0.0:
            raise ValueError("Muliplier of S4 regularizer should be non-negative")

        # Instantiate optimizer.
        if self.optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unknown value for optimizer")
        self.opt = opt

        # Dictionary for recording training history.
        self.history = {"loss": [], "misfit": [], "sobolev4": []}

        if self.save_grad_norm:
            self.history["grad_inf_norm"] = []
            self.history["grad_l2_norm"] = []

        if self.save_grad:
            self.history["grad"] = {}

        if validation_data:
            self.history["val_loss"] = []

        for e in range(self.epochs):
            with tf.GradientTape() as tape_loss:
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t1:
                    t1.watch(x_train)
                    psi = model(x_train)

                # Compute velocity predictions from the stream function `psi`.
                stream_func_grad = t1.gradient(psi, x_train)

                y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])
                misfit = tf.norm(y_pred - y_train, 2, axis=1) ** 2

                xmin = (0.0, 0.0)
                xmax = (2*np.pi, 2*np.pi)
                x_colloc = tf.Variable(
                    np.random.uniform(xmin, xmax, size=(256, 2)), dtype=tf.float32, trainable=False
                )

                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t4:
                    t4.watch(x_colloc)
                    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t3:
                        t3.watch(x_colloc)
                        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t2:
                            t2.watch(x_colloc)
                            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t1:
                                t1.watch(x_colloc)
                                psi = model(x_colloc)

                            # Compute velocity predictions from the stream function `psi`.
                            stream_func_grad = t1.gradient(psi, x_colloc)

                            y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])
                            u, v = tf.split(y_pred, 2, axis=1)

                        grad_u = t2.gradient(u, x_colloc)
                        grad_v = t2.gradient(v, x_colloc)

                        du_dx, du_dy = tf.split(grad_u, 2, axis=1)
                        dv_dx, dv_dy = tf.split(grad_v, 2, axis=1)

                    grad_du_dx = t3.gradient(du_dx, x_colloc)
                    grad_du_dy = t3.gradient(du_dy, x_colloc)
                    grad_dv_dx = t3.gradient(dv_dx, x_colloc)
                    grad_dv_dy = t3.gradient(dv_dy, x_colloc)

                    d2u_dxx, d2u_dxy = tf.split(grad_du_dx, 2, axis=1)
                    d2u_dyx, d2u_dyy = tf.split(grad_du_dy, 2, axis=1)
                    d2v_dxx, d2v_dxy = tf.split(grad_dv_dx, 2, axis=1)
                    d2v_dyx, d2v_dyy = tf.split(grad_dv_dy, 2, axis=1)

                    # reg_3 = (
                    #     d2u_dxx ** 2
                    #     + d2u_dxy ** 2
                    #     + d2u_dxx ** 2
                    #     + d2u_dyy ** 2
                    #     + d2v_dxx ** 2
                    #     + d2v_dxy ** 2
                    #     + d2v_dxx ** 2
                    #     + d2v_dyy ** 2
                    # )

                grad_d2u_dxx = t4.gradient(d2u_dxx, x_colloc)
                grad_d2u_dxy = t4.gradient(d2u_dxy, x_colloc)
                grad_d2u_dyx = t4.gradient(d2u_dyx, x_colloc)
                grad_d2u_dyy = t4.gradient(d2u_dyy, x_colloc)
                grad_d2v_dxx = t4.gradient(d2v_dxx, x_colloc)
                grad_d2v_dxy = t4.gradient(d2v_dxy, x_colloc)
                grad_d2v_dyx = t4.gradient(d2v_dyx, x_colloc)
                grad_d2v_dyy = t4.gradient(d2v_dyy, x_colloc)

                reg_4 = (
                    tf.reduce_sum(grad_d2u_dxx**2, axis=1)
                    + tf.reduce_sum(grad_d2u_dxy**2, axis=1)
                    + tf.reduce_sum(grad_d2u_dyx**2, axis=1)
                    + tf.reduce_sum(grad_d2u_dyy**2, axis=1)
                    + tf.reduce_sum(grad_d2v_dxx**2, axis=1)
                    + tf.reduce_sum(grad_d2v_dxy**2, axis=1)
                    + tf.reduce_sum(grad_d2v_dyx**2, axis=1)
                    + tf.reduce_sum(grad_d2v_dyy**2, axis=1)
                )

                loss = tf.reduce_mean(misfit) + self.s4 * tf.reduce_mean(reg_4)

            grad = tape_loss.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grad, model.trainable_variables))

            self.history["loss"].append(loss.numpy())
            self.history["misfit"].append(tf.reduce_mean(misfit).numpy())
            self.history["sobolev4"].append(tf.reduce_mean(reg_4).numpy())

            print("Epoch: {:d} | Loss: {:.1e}".format(e, loss.numpy()))

            if self.save_grad_norm:
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad])
                self.history["grad_inf_norm"].append(
                    np.linalg.norm(flat_grad, ord=np.inf)
                )
                self.history["grad_l2_norm"].append(np.linalg.norm(flat_grad, ord=2))

            if self.save_grad and (((e + 1) % self.save_grad == 0) or e == 0):
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad])
                self.history["grad"][e] = flat_grad

            if validation_data:
                val_pred = self.predict(validation_data[0])
                val_loss = mse(validation_data[1], val_pred)
                self.history["val_loss"].append(val_loss)

    def predict(self, x_new):
        if self.preprocessing == "identity":
            x_new_s = x_new
        else:
            x_new_s = self.transformer.transform(x_new)

        x_var = tf.Variable(x_new_s, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_var)
            psi = self.model(x_var)

        # Compute velocity predictions from the stream function `psi`.
        stream_func_grad = tape.gradient(psi, x_var)
        y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

        result = y_pred.numpy()

        if self.preprocessing == "standardization-both":
            result = self.transformer_output.inverse_transform(result)

        return result

    def compute_divergence(self, x_new):
        if self.preprocessing == "identity":
            x_new_s = x_new
        else:
            x_new_s = self.transformer.transform(x_new)

        # We need input as `tf.Variable` to be able to record operations
        # inside a gradient tape.
        x_var = tf.Variable(x_new_s, dtype=tf.float32)

        with tf.GradientTape(
            persistent=True, watch_accessed_variables=False
        ) as div_tape:
            div_tape.watch(x_var)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x_var)
                psi = self.model(x_var)

            # Compute velocity predictions from the stream function `psi`.
            stream_func_grad = tape.gradient(psi, x_var)
            y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])
            u, v = tf.split(y_pred, 2, axis=1)

        grad_u = div_tape.gradient(u, x_var)
        grad_v = div_tape.gradient(v, x_var)

        du_dx = grad_u[:, 0]
        dv_dy = grad_v[:, 1]

        divergence = du_dx + dv_dy

        result = divergence.numpy()

        del div_tape

        # if self.preprocessing == "standardization-both":
        #     result = self.transformer_output.inverse_transform(result)

        return result

    def save(self, dirname):
        filename = os.path.join(dirname, "model_params.pkl")
        params = self.get_params()
        with open(filename, "wb") as fh:
            pickle.dump(params, fh)

        if self.model:
            self.model.save(os.path.join(dirname, "model"))

        if self.history:
            history_file = os.path.join(dirname, "history.pkl")
            with open(history_file, "wb") as fh:
                pickle.dump(self.history, fh)

        if self.transformer:
            tfile = os.path.join(dirname, "transformer.pkl")
            with open(tfile, "wb") as fh:
                pickle.dump(self.transformer, fh)

        if self.transformer_output:
            tfile = os.path.join(dirname, "transformer_output.pkl")
            with open(tfile, "wb") as fh:
                pickle.dump(self.transformer_output, fh)

    @classmethod
    def load(cls, dirname):
        filename = os.path.join(dirname, "model_params.pkl")
        with open(filename, "rb") as fh:
            params = pickle.load(fh)

        obj = cls(**params)

        # Load Keras model if its folder exists.
        keras_model = os.path.join(dirname, "model")
        if os.path.exists(keras_model):
            obj.model = tf.keras.models.load_model(keras_model)

        history_file = os.path.join(dirname, "history.pkl")
        if os.path.exists(history_file):
            with open(history_file, "rb") as fh:
                obj.history = pickle.load(fh)

        tfile = os.path.join(dirname, "transformer.pkl")
        if os.path.exists(tfile):
            with open(tfile, "rb") as fh:
                obj.transformer = pickle.load(fh)

        tfile = os.path.join(dirname, "transformer_output.pkl")
        if os.path.exists(tfile):
            with open(tfile, "rb") as fh:
                obj.transformer_output = pickle.load(fh)

        return obj

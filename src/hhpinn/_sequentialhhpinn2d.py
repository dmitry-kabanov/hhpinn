import os
import pickle

import numpy as np
import tensorflow as tf

from typing import Dict, List, Tuple, Union

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model

from hhpinn.scoring import mse
from hhpinn._sobolev3reg import sobolev3reg


class SequentialHHPINN2D:
    """Neural network for Helmholtz--Hodge decomposition of 2D vector fields.

    This physics-informed neural network (PINN) learns from given
    vector dataset R^2 \to R^2 two networks that represent potential and
    divergence-free parts of the underlying vector field.

    This network first trains the solenoidal part first and then the potential
    part, that is, the training is sequential.

    """

    def __init__(
        self,
        hidden_layers=[10],
        epochs=50,
        l2=0.0,
        s3=0.0,
        s4=0.0,
        ip=0.0,
        optimizer="sgd",
        learning_rate=0.01,
        preprocessing="identity",
        save_grad_norm=False,
        save_grad: int = 0,
    ):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.l2 = l2
        self.s3 = s3
        self.s4 = s4
        self.ip = ip
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.preprocessing = preprocessing
        self.save_grad_norm = save_grad_norm
        self.save_grad = save_grad
        self._nparams = 11

        self.model_phi: Union[Model, None] = None
        self.model_psi: Union[Model, None] = None
        self.history: Dict[str, Union[Dict, List]] = {}
        self.transformer = None
        self.transformer_output = None

    def get_params(self):
        params = {
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "l2": self.l2,
            "s3": self.s3,
            "s4": self.s4,
            "ip": self.ip,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "preprocessing": self.preprocessing,
            "save_grad_norm": self.save_grad_norm,
            "save_grad": self.save_grad,
        }
        assert len(params) == self._nparams

        return params

    def build_model(self) -> Model:
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

        model = Model(inputs=inp, outputs=out)

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

        # Instantiate models.
        model_phi = self.build_model()
        self.model_phi = model_phi
        model_psi = self.build_model()
        self.model_psi = model_psi

        if self.l2 < 0.0:
            raise ValueError("Multiplier of L2 regularizer should be non-negative")

        if self.s3 < 0.0:
            raise ValueError("Muliplier of S3 regularizer should be non-negative")

        if self.s4 < 0.0:
            raise ValueError("Muliplier of S4 regularizer should be non-negative")

        # Instantiate optimizer.
        if self.optimizer == "sgd":
            opt_phi = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            opt_psi = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            opt_phi = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            opt_psi = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unknown value for optimizer")
        self.opt_phi = opt_phi
        self.opt_psi = opt_psi

        # Dictionary for recording training history.
        self.history = {
            "loss": [], "misfit": [], "sobolev4": [], "sobolev3_phi": [],
            "sobolev3_psi": [],
        }

        if self.save_grad_norm:
            self.history["grad_phi_inf_norm"] = []
            self.history["grad_psi_inf_norm"] = []

        if self.save_grad:
            self.history["grad_phi"] = {}
            self.history["grad_psi"] = {}

        if validation_data:
            self.history["val_phi_loss"] = []
            self.history["val_psi_loss"] = []

        tape_kw = dict(persistent=True, watch_accessed_variables=False)

        s3reg_phi_fn = sobolev3reg(model_phi)
        s3reg_psi_fn = sobolev3reg(model_psi)

        G = 8
        xx = np.linspace(xmin[0], xmax[0], num=G)
        yy = np.linspace(xmin[1], xmax[1], num=G)
        XX, YY = np.meshgrid(xx, yy)

        x_colloc = tf.Variable(
            np.column_stack((np.reshape(XX, (-1, 1)), np.reshape(YY, (-1, 1)))),
            dtype=tf.float32
        )
        assert x_colloc.shape == (G*G, 2)

        # Training loop for solenoidal (\psi) network.
        for e in range(self.epochs):
            with tf.GradientTape(persistent=True) as tape_loss:
                with tf.GradientTape(**tape_kw) as t1:
                    t1.watch(x_train)
                    psi = model_psi(x_train)

                # Divergence-free part in 2D is defined by stream function:
                # u = +∂psi_∂y, v = -∂psi_∂x.
                stream_func_grad = t1.gradient(psi, x_train)
                div_free_part = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

                u_pred = div_free_part
                misfit = tf.norm(u_pred - y_train, 2, axis=1) ** 2

                s3reg_psi = s3reg_psi_fn(x_colloc)
                loss = tf.reduce_mean(misfit) + \
                    self.s3 * tf.reduce_mean(s3reg_psi)

            grad_psi = tape_loss.gradient(loss, model_psi.trainable_variables)
            opt_psi.apply_gradients(zip(grad_psi, model_psi.trainable_variables))

            if validation_data:
                val_pred = self.predict(validation_data[0])
                val_loss = mse(validation_data[1], val_pred)
                self.history["val_psi_loss"].append(val_loss)

            print(f"Epoch {e:d}")

        # Residual output training data are defined by the original dataset
        # and the prediction of the solenoidal (\psi) network.
        y_train_resid = y_train - model_psi(x_train)

        # Training loop for the potential (\phi) network.
        for e in range(self.epochs):
            with tf.GradientTape(persistent=True) as tape_loss:
                with tf.GradientTape(**tape_kw) as t1:
                    t1.watch(x_train)
                    phi = model_phi(x_train)

                # Potential (curl-free part) is a gradient of scalar-valued
                # function phi.
                u_pot = t1.gradient(phi, x_train)

                u_pred = u_pot
                misfit = tf.norm(u_pred - y_train_resid, 2, axis=1) ** 2

                s3reg_phi = s3reg_phi_fn(x_colloc)
                loss = tf.reduce_mean(misfit) + \
                    self.s3 * tf.reduce_mean(s3reg_phi)

            grad_phi = tape_loss.gradient(loss, model_phi.trainable_variables)
            opt_phi.apply_gradients(zip(grad_phi, model_phi.trainable_variables))

            if validation_data:
                val_pred = self.predict(validation_data[0])
                val_loss = mse(validation_data[1], val_pred)
                self.history["val_phi_loss"].append(val_loss)

    def predict(self, x_new, return_separate_fields=False):
        if self.preprocessing == "identity":
            x_new_s = x_new
        else:
            x_new_s = self.transformer.transform(x_new)

        x_var = tf.Variable(x_new_s, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x_var)
            phi = self.model_phi(x_var)
            psi = self.model_psi(x_var)

        curl_free_part = tape.gradient(phi, x_var)
        stream_func_grad = tape.gradient(psi, x_var)
        div_free_part = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

        y_pred = curl_free_part + div_free_part
        result = y_pred.numpy()

        if self.preprocessing == "standardization-both":
            raise ValueError("This is buggy, not completely implemented")
            result = self.transformer_output.inverse_transform(result)

        if return_separate_fields:
            return result, curl_free_part.numpy(), div_free_part.numpy()
        else:
            return result

    def predict_separate_fields(
        self, x_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict potential and solenoidal vector fields separately."""
        __, result_pot, result_sol = self.predict(x_new, return_separate_fields=True)

        return result_pot, result_sol

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
                psi = self.model_psi(x_var)

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

    def compute_curl_of_potential_field(self, x_new):
        r"""Compute curl of potential field represented by \nabla \phi.

        Note that here we use that :math:`\nabla \phi = (\partial_x \phi,
        \partial_y \phi, 0)` and :math:`\phi` does not depend on z-coordinate
        that gives the following formula for computing the curl:

            \nabla \times \nabla \phi = \left(
                \partial_{xy} \phi - \partial_{yx} \phi
            \right) \vec k,

        which shows that the curl will be zero if \phi is smooth enough that
        the mixed derivatives do not depend on the order of differentiation.

        """
        if self.preprocessing == "identity":
            x_new_s = x_new
        else:
            x_new_s = self.transformer.transform(x_new)

        # We need input as `tf.Variable` to be able to record operations
        # inside a gradient tape.
        x_var = tf.Variable(x_new_s, dtype=tf.float32)

        with tf.GradientTape(
            persistent=True, watch_accessed_variables=False
        ) as curl_tape:
            curl_tape.watch(x_var)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x_var)
                phi = self.model_phi(x_var)

            pot_part = tape.gradient(phi, x_var)

            u, v = tf.split(pot_part, 2, axis=1)

        grad_u = curl_tape.gradient(u, x_var)
        grad_v = curl_tape.gradient(v, x_var)

        du_dy = grad_u[:, 1]
        dv_dx = grad_v[:, 0]

        curl = du_dy - dv_dx

        result = curl.numpy()

        del curl_tape

        # if self.preprocessing == "standardization-both":
        #     result = self.transformer_output.inverse_transform(result)

        return result

    def compute_inner_product(self, x_new: np.ndarray) -> np.ndarray:
        pot_field, sol_field = self.predict_separate_fields(x_new)

        ip = tf.reduce_sum(pot_field * sol_field, axis=1)
        result = ip.numpy()

        return result

    def save(self, dirname):
        filename = os.path.join(dirname, "model_params.pkl")
        params = self.get_params()
        with open(filename, "wb") as fh:
            pickle.dump(params, fh)

        if self.model_phi:
            self.model_phi.save(os.path.join(dirname, "model_phi"))

        if self.model_psi:
            self.model_psi.save(os.path.join(dirname, "model_psi"))

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
        keras_model_phi = os.path.join(dirname, "model_phi")
        if os.path.exists(keras_model_phi):
            obj.model_phi = tf.keras.models.load_model(keras_model_phi,
                                                       compile=False)

        # Load Keras model if its folder exists.
        keras_model_psi = os.path.join(dirname, "model_psi")
        if os.path.exists(keras_model_psi):
            obj.model_psi = tf.keras.models.load_model(keras_model_psi,
                                                       compile=False)

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

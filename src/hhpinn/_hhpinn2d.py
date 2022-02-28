import os
import pickle

import numpy as np
import tensorflow as tf

from typing import Dict, List, Tuple, Union

from hhpinn.scoring import mse
from hhpinn.transformer import Transformer
from hhpinn._sobolev3reg import sobolev3reg


class HHPINN2D:
    """Neural network for Helmholtz--Hodge (HH) decomposition of 2D vector fields.

    This physics-informed neural network (PINN) learns from given
    vector dataset R^2 \to R^2 two networks that represent potential and
    divergence-free parts of the underlying vector field.

    """

    def __init__(
        self,
        hidden_layers=[10],
        epochs=50,
        l2=0.0,
        s3=0.0,
        s4=0.0,
        ip=0.0,
        G=8,
        use_batch_normalization=False,
        use_uniform_grid_for_regs=True,
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
        self.G = G
        self.use_uniform_grid_for_regs = use_uniform_grid_for_regs
        self.use_batch_normalization = use_batch_normalization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.preprocessing = preprocessing
        self.save_grad_norm = save_grad_norm
        self.save_grad = save_grad
        self._nparams = 14

        self.model_phi: Union[tf.keras.Model, None] = None
        self.model_psi: Union[tf.keras.Model, None] = None
        self.history: Dict[str, Union[Dict, List]] = {}
        self.transformer = None

    def get_params(self):
        params = {
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "l2": self.l2,
            "s3": self.s3,
            "s4": self.s4,
            "ip": self.ip,
            "G": self.G,
            "use_uniform_grid_for_regs": self.use_uniform_grid_for_regs,
            "use_batch_normalization": self.use_batch_normalization,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "preprocessing": self.preprocessing,
            "save_grad_norm": self.save_grad_norm,
            "save_grad": self.save_grad,
        }
        assert len(params) == self._nparams

        return params

    def build_model(self):
        """Build and return Keras model with given hyperparameters."""
        inp = tf.keras.layers.Input(2)
        x = inp
        for i, neurons in enumerate(self.hidden_layers):
            x = tf.keras.layers.Dense(
                neurons,
                activation="tanh",
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
                bias_regularizer=tf.keras.regularizers.l2(l2=self.l2),
            )(x)
            if self.use_batch_normalization and i+1 != len(self.hidden_layers):
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.2)(x)

        out = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=False,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2),
        )(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        return model

    def fit(self, x, y, validation_data=None, verbose=1):
        # Preprocess training data.
        # (xs, ys): scaled versions of (x, y)
        self.transformer = Transformer(self.preprocessing)
        xs, ys = self.transformer.fit_transform(x, y)

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

        # Nullify the dictionary for recording training history.
        self.history["loss"] = []
        self.history["misfit"] = []
        self.history["sobolev4"] = []
        self.history["ip"] = []

        if self.save_grad_norm:
            self.history["grad_phi_inf_norm"] = []
            self.history["grad_psi_inf_norm"] = []

        if self.save_grad:
            self.history["grad_phi"] = {}
            self.history["grad_psi"] = {}

        if validation_data:
            self.history["val_loss"] = []

        G = self.G

        xx = np.linspace(xmin[0], xmax[0], num=G)
        yy = np.linspace(xmin[1], xmax[1], num=G)
        XX, YY = np.meshgrid(xx, yy)
        x_colloc_grid = tf.Variable(
            np.column_stack(
                (np.reshape(XX, (-1, 1)), np.reshape(YY, (-1, 1)))
            ),
            dtype=tf.float32,
            trainable=False,
        )

        train_step_fn = self.train_step_wrapper(
            opt_phi, opt_psi, x_colloc_grid, xmin, xmax
        )

        for e in range(self.epochs):
            loss, misfit_mean, ip_reg_mean, grad_phi, grad_psi = (
                train_step_fn(x_train, y_train)
            )

            self.history["loss"].append(loss.numpy())
            self.history["misfit"].append(misfit_mean.numpy())
            self.history["ip"].append(ip_reg_mean.numpy())

            if self.save_grad_norm:
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad_phi
                                            if g is not None])
                self.history["grad_phi_inf_norm"].append(
                    np.linalg.norm(flat_grad, ord=np.inf)
                )
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad_psi
                                            if g is not None])
                self.history["grad_psi_inf_norm"].append(
                    np.linalg.norm(flat_grad, ord=np.inf)
                )

            if self.save_grad and (((e + 1) % self.save_grad == 0) or e == 0):
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad_phi
                                            if g is not None])
                self.history["grad_phi"][e] = flat_grad
                flat_grad = np.concatenate([g.numpy().ravel() for g in grad_psi
                                            if g is not None])
                self.history["grad_psi"][e] = flat_grad

            val_loss = 0.0
            if validation_data:
                val_pred = self.predict(validation_data[0])
                val_loss = mse(validation_data[1], val_pred)
                self.history["val_loss"].append(val_loss)

            if verbose:
                msg = "Epoch: {:05d} | Loss: {:.1e}".format(e, loss.numpy())
                if validation_data:
                    msg += " | Val_loss: {:.1e}".format(val_loss)
                print(msg)

            # Sanity checks that the variables for models used in this method
            # are the same as the properties of the object.
            tf.debugging.assert_equal(
                self.model_phi.trainable_variables[-1],
                model_phi.trainable_variables[-1]
            )

            tf.debugging.assert_equal(
                self.model_psi.trainable_variables[-1],
                model_psi.trainable_variables[-1]
            )

    def train_step_wrapper(self, opt_phi, opt_psi, x_colloc_grid, xmin, xmax):
        model_phi: tf.keras.Model = self.model_phi
        model_psi: tf.keras.Model = self.model_psi

        x_colloc_grid = x_colloc_grid

        opt_phi = opt_phi
        opt_psi = opt_psi

        tape_kw = dict(persistent=True, watch_accessed_variables=False)

        s3reg_phi_fn = sobolev3reg(model_phi)
        s3reg_psi_fn = sobolev3reg(model_psi)

        xmin, xmax = xmin, xmax

        @tf.function
        def train_step(x_train, y_train):
            with tf.GradientTape(persistent=True) as tape_loss:
                with tf.GradientTape(**tape_kw) as t1:
                    t1.watch(x_train)
                    phi = model_phi(x_train, training=True)
                    psi = model_psi(x_train, training=True)

                # Potential (curl-free) part is the gradient of `phi`.
                curl_free_part = t1.gradient(phi, x_train)

                # Solenoidal (divergence-free) part in 2D is defined
                # by stream function: u = ∂psi_∂y, v = -∂psi_∂x.
                stream_func_grad = t1.gradient(psi, x_train)
                div_free_part = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

                u_pred = curl_free_part + div_free_part
                misfit = tf.norm(u_pred - y_train, 2, axis=1) ** 2
                misfit_mean = tf.reduce_mean(misfit)

                # s3reg_phi_mean = tf.Variable(0.0, dtype=tf.float32)
                # s3reg_psi_mean = tf.Variable(0.0, dtype=tf.float32)
                #if self.s3:
                x_colloc = x_colloc_grid
                s3reg_phi = s3reg_phi_fn(x_colloc)
                s3reg_phi_mean = tf.reduce_mean(s3reg_phi)
                s3reg_psi = s3reg_psi_fn(x_colloc)
                s3reg_psi_mean = tf.reduce_mean(s3reg_psi)

                # ip_reg_mean = tf.Variable(0.0, dtype=tf.float32)
                # if self.ip:
                if self.use_uniform_grid_for_regs:
                    x_colloc_ip = x_colloc_grid
                else:
                    x_colloc_ip = tf.Variable(
                        np.random.uniform(xmin, xmax, size=(256, 2)),
                        dtype=tf.float32,
                        trainable=False,
                    )

                with tf.GradientTape(
                    persistent=True, watch_accessed_variables=False
                ) as t1:
                    t1.watch(x_colloc_ip)
                    phi = model_phi(x_colloc_ip, training=True)
                    psi = model_psi(x_colloc_ip, training=True)

                # Potential (curl-free part) is a gradient of scalar-valued
                # function phi.
                pot_part = t1.gradient(phi, x_colloc_ip)

                # Divergence-free part in 2D is defined by stream function:
                # u = ∂psi_∂y, v = -∂psi_∂x.
                stream_func_grad = t1.gradient(psi, x_colloc_ip)
                sol_part = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

                ip_reg = tf.square(tf.reduce_sum(pot_part * sol_part, axis=1))
                ip_reg_mean = tf.reduce_mean(ip_reg)

                loss = (
                    misfit_mean
                    + self.s3 * (s3reg_phi_mean + s3reg_psi_mean)
                    + self.ip * ip_reg_mean
                )

            grad_phi = tape_loss.gradient(loss, model_phi.trainable_variables)
            opt_phi.apply_gradients(zip(grad_phi, model_phi.trainable_variables))

            grad_psi = tape_loss.gradient(loss, model_psi.trainable_variables)
            opt_psi.apply_gradients(zip(grad_psi, model_psi.trainable_variables))

            return loss, misfit_mean, ip_reg_mean, grad_phi, grad_psi

        return train_step

    def predict(self, x_new, return_separate_fields=False):
        if (self.model_phi is None) or (self.model_psi is None):
            raise RuntimeError("You must call `fit` method first")
        if self.transformer is None:
            raise RuntimeError("You must call `fit` method first")

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

    def predict_scalar_fields(self, x_new):
        if (self.model_phi is None) or (self.model_psi is None):
            raise RuntimeError("You must call `fit` method first")
        if self.transformer is None:
            raise RuntimeError("You must call `fit` method first")

        x_new_s = self.transformer.transform(x_new)

        x_var = tf.Variable(x_new_s, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x_var)
            phi = self.model_phi(x_var)
            psi = self.model_psi(x_var)

        return phi.numpy(), psi.numpy()

    def compute_divergence(self, x_new):
        if (self.model_phi is None) or (self.model_psi is None):
            raise RuntimeError("You must call `fit` method first")
        if self.transformer is None:
            raise RuntimeError("You must call `fit` method first")

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
        if (self.model_phi is None) or (self.model_psi is None):
            raise RuntimeError("You must call `fit` method first")
        if self.transformer is None:
            raise RuntimeError("You must call `fit` method first")

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

    @classmethod
    def load(cls, dirname):
        filename = os.path.join(dirname, "model_params.pkl")
        with open(filename, "rb") as fh:
            params = pickle.load(fh)

        obj = cls(**params)

        # Load Keras model if its folder exists.
        keras_model_phi = os.path.join(dirname, "model_phi")
        if os.path.exists(keras_model_phi):
            obj.model_phi = tf.keras.models.load_model(keras_model_phi)

        # Load Keras model if its folder exists.
        keras_model_psi = os.path.join(dirname, "model_psi")
        if os.path.exists(keras_model_psi):
            obj.model_psi = tf.keras.models.load_model(keras_model_psi)

        history_file = os.path.join(dirname, "history.pkl")
        if os.path.exists(history_file):
            with open(history_file, "rb") as fh:
                obj.history = pickle.load(fh)

        tfile = os.path.join(dirname, "transformer.pkl")
        if os.path.exists(tfile):
            with open(tfile, "rb") as fh:
                obj.transformer = pickle.load(fh)

        return obj

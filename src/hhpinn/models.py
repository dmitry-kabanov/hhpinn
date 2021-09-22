import os
import pickle

import tensorflow as tf

from sklearn.preprocessing import StandardScaler


class HodgeHelmholtzPINN:
    """Physics-informed neural network for learning fluid flows."""
    def __init__(self, hidden_layers=[10], epochs=50, learning_rate=0.01,
                 preprocessing="identity"):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.preprocessing = preprocessing
        self._nparams = 4

        self.model = None
        self.history = None
        self.transformer = None

    def get_params(self):
        params = {
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "preprocessing": self.preprocessing,
        }
        assert len(params) == self._nparams

        return params

    def build_model(self) -> tf.keras.models.Model:
        """Build and return Keras model with given hyperparameters."""
        inp = tf.keras.layers.Input(2)
        x = inp
        for neurons in self.hidden_layers:
            x = tf.keras.layers.Dense(neurons, activation="tanh",
                                      kernel_initializer="glorot_normal")(x)

        out = tf.keras.layers.Dense(1, activation=None,
                                    use_bias=False,
                                    kernel_initializer="glorot_normal")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        return model

    def fit(self, x, y):
        if self.preprocessing == "identity":
            xs = x
            ys = y
        elif self.preprocessing == "standardization":
            self.transformer = StandardScaler()
            self.transformer.fit(x)
            xs = self.transformer.transform(x)
            ys = y
        else:
            raise ValueError("Unknown values for preprocessing")

        x_train = tf.Variable(xs, dtype=tf.float32)
        y_train = tf.Variable(ys, dtype=tf.float32)

        model = self.build_model()
        self.model = model

        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.history = {"loss": []}

        for e in range(self.epochs):
            with tf.GradientTape(persistent=True) as tape:
                psi = model(x_train)

                # Compute velocity predictions from the stream function `psi`.
                stream_func_grad = tape.gradient(psi, x_train)
                y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

                misfit = y_pred - y_train
                misfit_sq = tf.norm(misfit, 2, axis=1)**2
                loss = tf.reduce_mean(misfit_sq)

            grad = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grad, model.trainable_variables))

            self.history["loss"].append(loss.numpy())

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

        return y_pred.numpy()

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

        return obj

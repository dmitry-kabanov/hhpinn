import pickle

import tensorflow as tf


class HodgeHelmholtzPINN:
    """Physics-informed neural network for learning fluid flows."""
    def __init__(self, hidden_layers=[10], epochs=50, learning_rate=0.01):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._nparams = 3

        self.model = None

    def get_params(self):
        params = {
            "hidden_layers": self.hidden_layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
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
                                    kernel_initializer="glorot_normal")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)

        return model

    def fit(self, x, y):
        x_train = tf.Variable(x, dtype=tf.float32)
        y_train = tf.Variable(y, dtype=tf.float32)

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
        x_var = tf.Variable(x_new, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_var)
            psi = self.model(x_var)

        # Compute velocity predictions from the stream function `psi`.
        stream_func_grad = tape.gradient(psi, x_var)
        y_pred = tf.matmul(stream_func_grad, [[0, -1], [1, 0]])

        return y_pred.numpy()

    def save(self, filename):
        with open(filename, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as fh:
            obj = pickle.load(fh)

        return obj

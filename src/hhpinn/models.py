import tensorflow as tf


class HodgeHelmholtzPINN:
    """Physics-informed neural network for learning fluid flows."""
    def __init__(self, hidden_layers=[10], epochs=50, learning_rate=0.01):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def build_model(self) -> tf.keras.models.Model:
        """Build and return Keras model with given hyperparameters."""
        inp = tf.keras.layers.Input(2)
        x = inp
        for neurons in self.hidden_layers:
            x = tf.keras.layers.Dense(neurons, activation="tanh")(x)

        out = tf.keras.layers.Dense(1, activation=None)(x)

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
            with tf.GradientTape() as tape:
                y_pred = model(x_train)

                # Build the curl operation
                stream_func_grad = tape.gradient(y_pred, x_train)
                y_pred = tf.stack(
                    [stream_func_grad[:, 1], -stream_func_grad[:, 0]]
                )
                misfit = y_pred - y_train
                loss = tf.reduce_mean(misfit**2, axis=1)

            grad = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grad, model.trainable_variables))

            self.history["loss"].append(loss.numpy())

    def predict(self, x_new):
        x_var = tf.Variable(x_new, dtype=tf.float32)
        with tf.GradientTape(watch_access_variables=False) as tape:
            y_pred = self.model(x_var)

        curl_res = tape.gradient(y_pred, x_var)

        return curl_res.numpy()

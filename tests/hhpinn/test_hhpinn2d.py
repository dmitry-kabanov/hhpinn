import numpy as np
import numpy.testing as npt
import tensorflow as tf

from hhpinn import HHPINN2D


class TestHHPINN2D:
    def test_default_hyperparameters(self):
        sut = HHPINN2D()

        assert sut.hidden_layers == [10]
        assert sut.epochs == 50
        assert sut.l2 == 0.0
        assert sut.optimizer == "sgd"
        assert sut.learning_rate == 0.01
        assert sut.preprocessing == "identity"
        assert sut.save_grad_norm is False
        assert sut.save_grad == 0

    def test_saves_hyperparameters(self):
        HIDDEN_LAYERS = [27, 8, 101]
        EPOCHS = 22
        L2 = 1.27e-3
        S4 = 2.12e-1
        OPTIMIZER = "adam"
        LEARNING_RATE = 0.031
        PREPROCESSING = "standardization"
        SAVE_GRAD_NORM = True
        SAVE_GRAD = 100

        sut = HHPINN2D(
            hidden_layers=HIDDEN_LAYERS, epochs=EPOCHS,
            l2=L2,
            s4=S4,
            optimizer=OPTIMIZER,
            learning_rate=LEARNING_RATE,
            preprocessing=PREPROCESSING,
            save_grad_norm=SAVE_GRAD_NORM,
            save_grad=SAVE_GRAD
        )

        npt.assert_equal(sut.hidden_layers, HIDDEN_LAYERS)
        npt.assert_equal(sut.epochs, EPOCHS)
        npt.assert_equal(sut.l2, L2)
        npt.assert_equal(sut.s4, S4)
        npt.assert_equal(sut.optimizer, OPTIMIZER)
        npt.assert_equal(sut.learning_rate, LEARNING_RATE)
        npt.assert_equal(sut.preprocessing, PREPROCESSING)
        npt.assert_equal(sut.save_grad_norm, SAVE_GRAD_NORM)
        npt.assert_equal(sut.save_grad, SAVE_GRAD)

    def test_nonlinear_model_has_correct_number_of_neurons(self):
        sut = HHPINN2D(hidden_layers=[3])

        # Expected number of neurons.
        # Note that the last zero is due to lack of bias in the output layer.
        exp_neurons = 3 * 2 + 3 + 1 * 3 + 0

        actual_neurons = 0
        for w in sut.build_model().trainable_variables:
            actual_neurons += np.prod(w.shape)

        assert actual_neurons == exp_neurons

    def test_nonlinear_model_has_correct_number_of_neurons_two_layers(self):
        sut = HHPINN2D(hidden_layers=[3, 7])

        # Expected number of neurons. There is no bias in the output layer
        # as it does not play any role in training and prediction
        # due to differentiation of the model.
        exp_neurons = 3 * 2 + 3 + 7 * 3 + 7 + 1 * 7 + 0

        actual_neurons = 0
        for w in sut.build_model().trainable_variables:
            actual_neurons += np.prod(w.shape)

        assert actual_neurons == exp_neurons

    def test_save_load_before_training(self, tmpdir):
        # Variable `tmpdir` is a `pytest` fixture.
        m1 = HHPINN2D()

        m1.save(tmpdir)
        m2 = HHPINN2D.load(tmpdir)

        npt.assert_equal(m1.get_params(), m2.get_params())

    def test_save_load_after_training(self, tmpdir):
        m1 = HHPINN2D(hidden_layers=[10], epochs=5, learning_rate=0.1)
        m1.fit(np.random.normal(size=(10, 2)), np.random.normal(size=(10, 2)))

        m1.save(tmpdir)
        m2 = HHPINN2D.load(tmpdir)

        for i in range(len(m1.model_phi.trainable_variables)):
            npt.assert_array_equal(
                m1.model_phi.trainable_variables[i], m2.model_phi.trainable_variables[i]
            )

        for i in range(len(m1.model_psi.trainable_variables)):
            npt.assert_array_equal(
                m1.model_psi.trainable_variables[i], m2.model_psi.trainable_variables[i]
            )

    def test_save_load_after_training_history_is_saved(self, tmpdir):
        m1 = HHPINN2D(hidden_layers=[10], epochs=5, learning_rate=0.1)
        m1.fit(np.random.normal(size=(10, 2)), np.random.normal(size=(10, 2)))

        m1.save(tmpdir)
        m2 = HHPINN2D.load(tmpdir)

        assert m1.history == m2.history

    def test_save_load_after_training_transformer_is_saved(self, tmpdir):
        m1 = HHPINN2D(
            hidden_layers=[10],
            epochs=5,
            learning_rate=0.1,
            preprocessing="standardization",
        )
        m1.fit(np.random.normal(size=(10, 2)), np.random.normal(size=(10, 2)))

        m1.save(tmpdir)
        m2 = HHPINN2D.load(tmpdir)

        x_new = np.random.normal(size=(10, 2))

        npt.assert_array_equal(
            m1.transformer.transform(x_new), m2.transformer.transform(x_new)
        )

    def test_connected_correctly(self):
        # Relatively rigid test to check that MLP network is wired correctly.
        model = HHPINN2D(hidden_layers=[3])
        nn = model.build_model()

        w1 = np.array([[3.0, 4.0], [7.0, 5.0], [1.0, 6.0]]).T
        b1 = np.array([1.0, 2.0, 3.0])
        o1 = np.array([[0.5, 1.2, 1.9]]).T
        nn.set_weights([w1, b1, o1])

        x = np.array([[27.0, 15.0], [3.0, 2.0], [-1, 2.0], [7.0, 4.0]])

        desired = np.tanh(x @ w1 + b1) @ o1
        actual = nn(x)

        npt.assert_allclose(actual, desired, rtol=1e-7, atol=1e-7)

    def test_default_optimizer_used(self):
        model = HHPINN2D(epochs=3)
        x = np.random.random(size=(10, 2))
        y = np.random.random(size=(10, 2))

        model.fit(x, y)

        assert isinstance(model.opt_phi, tf.keras.optimizers.SGD)
        assert isinstance(model.opt_psi, tf.keras.optimizers.SGD)

    def test_non_default_optimizer_used(self):
        model = HHPINN2D(epochs=3, optimizer="adam")
        x = np.random.random(size=(10, 2))
        y = np.random.random(size=(10, 2))

        model.fit(x, y)

        assert isinstance(model.opt_phi, tf.keras.optimizers.Adam)
        assert isinstance(model.opt_psi, tf.keras.optimizers.Adam)

    def test_prediction_is_divergence_free(self):
        model = HHPINN2D(epochs=3, optimizer="adam")
        x = np.random.random(size=(10, 2))
        y = np.random.random(size=(10, 2))

        model.fit(x, y)

        x_new = np.random.random(size=(4000, 2))
        f = model.compute_divergence(x_new)

        assert f.ndim == 1
        assert f.shape[0] == x_new.shape[0]
        assert np.linalg.norm(f, np.Inf) <= 1e-7

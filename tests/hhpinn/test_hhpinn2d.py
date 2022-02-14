import numpy as np
import numpy.testing as npt
import tensorflow as tf

from hhpinn import AveragingModel, HHPINN2D
from hhpinn.datasets import TGV2DPlusTrigonometricFlow

from hhpinn.scoring import mse


TEST_GRID_SIZE = (11, 11)


class TestHHPINN2D:
    def test_default_hyperparameters(self):
        sut = HHPINN2D()

        assert sut.hidden_layers == [10]
        assert sut.epochs == 50
        assert sut.l2 == 0.0
        assert sut.use_batch_normalization is False
        assert sut.optimizer == "sgd"
        assert sut.learning_rate == 0.01
        assert sut.preprocessing == "identity"
        assert sut.save_grad_norm is False
        assert sut.save_grad == 0

    def test_saves_hyperparameters(self):
        HIDDEN_LAYERS = [27, 8, 101]
        EPOCHS = 22
        L2 = 1.27e-3
        S3 = 8.15e-8
        S4 = 2.12e-1
        USE_BATCH_NORMALIZATION = True
        OPTIMIZER = "adam"
        LEARNING_RATE = 0.031
        PREPROCESSING = "standardization"
        SAVE_GRAD_NORM = True
        SAVE_GRAD = 100

        sut = HHPINN2D(
            hidden_layers=HIDDEN_LAYERS,
            epochs=EPOCHS,
            l2=L2,
            s3=S3,
            s4=S4,
            use_batch_normalization=USE_BATCH_NORMALIZATION,
            optimizer=OPTIMIZER,
            learning_rate=LEARNING_RATE,
            preprocessing=PREPROCESSING,
            save_grad_norm=SAVE_GRAD_NORM,
            save_grad=SAVE_GRAD,
        )

        npt.assert_equal(sut.hidden_layers, HIDDEN_LAYERS)
        npt.assert_equal(sut.epochs, EPOCHS)
        npt.assert_equal(sut.l2, L2)
        npt.assert_equal(sut.s3, S3)
        npt.assert_equal(sut.s4, S4)
        npt.assert_equal(sut.use_batch_normalization, USE_BATCH_NORMALIZATION)
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

    def test_prediction_components_are_orthogonal_in_L2_sense(self):
        # Test that potential and solenoidal components of the total field
        # are orthogonal to each other.
        # We compute inner product via Monte Carlo method for different
        # sample sizes and check that the result is close to zero.
        model = HHPINN2D(epochs=300, ip=100.0, optimizer="adam")

        x = np.random.random(size=(10, 2))
        u = np.random.random(size=(10, 2))

        model.fit(x, u)

        ip = []
        for N in [512, 1024, 2048]:
            x_new = np.random.random(size=(N, 2))
            pred_u_poten, pred_u_solen = model.predict_separate_fields(x_new)

            dot_prod_pw = np.zeros(len(pred_u_poten))
            for i in range(len(pred_u_poten)):
                dot_prod_pw[i] = np.dot(pred_u_poten[i], pred_u_solen[i])
            ip.append(np.mean(dot_prod_pw))

        # Use very loose tolerance as it is difficult to get accurate results
        # in each test run due to randomness.
        npt.assert_allclose(ip, 0.0, rtol=1e-1, atol=1e-1)

    def test_prediction_total_field_is_sum_of_components(self):
        model = HHPINN2D(epochs=3)

        x = np.random.random(size=(10, 2))
        u = np.random.random(size=(10, 2))
        x_new = np.random.random(size=(2000, 2))

        model.fit(x, u)

        pred_tot = model.predict(x_new)
        pred_pot, pred_sol = model.predict_separate_fields(x_new)

        npt.assert_allclose(pred_tot, pred_pot + pred_sol, rtol=1e-7, atol=2e-7)

    def test_larger_network_has_smaller_loss(self):
        # Test that the neural network with larger number of degrees of freedom
        # has smaller training loss.
        # This test only works on average, that's why I run training
        # TRIALS times and average the trials results and check the mean
        # training loss.
        TRIALS = 5
        EPOCHS = 50

        ds = TGV2DPlusTrigonometricFlow(N=10)
        train_x, train_u, __, __ = ds.load_data()

        losses_small = []
        losses_large = []

        for __ in range(TRIALS):
            model_small = HHPINN2D(hidden_layers=[10], epochs=EPOCHS, optimizer="adam")
            model_large = HHPINN2D(hidden_layers=[10, 5], epochs=EPOCHS,
                                   optimizer="adam")
            model_small.fit(train_x, train_u)
            model_large.fit(train_x, train_u)

            loss_small = model_small.history["loss"]
            loss_large = model_large.history["loss"]

            losses_small.append(loss_small)
            losses_large.append(loss_large)

        mean_loss_small = np.mean(losses_small, axis=0)
        mean_loss_large = np.mean(losses_large, axis=0)
        assert len(mean_loss_small) == EPOCHS

        # Larger model should give on average smaller training loss
        # as overfitting is easier with larger number of degrees of freedom.
        npt.assert_array_less(mean_loss_large[-10:], mean_loss_small[-10:])

    def test_hhpinn2d_is_better_than_averaging_model(self):
        # We need to check it statistically as sometimes HHPINN2D
        # gives worse predictions, which depends on the randomized
        # parameter initialization.
        ds = TGV2DPlusTrigonometricFlow(N=50)
        train_x, train_u, __, __ = ds.load_data()
        test_x, test_u, __, __ = ds.load_data_on_grid(TEST_GRID_SIZE)

        fails = 0

        for __ in range(5):
            try:
                model_aver = AveragingModel()
                model_pinn = HHPINN2D(
                    hidden_layers=[10], epochs=50, optimizer="adam", learning_rate=1e-2)

                model_aver.fit(train_x, train_u)
                model_pinn.fit(train_x, train_u, verbose=0)

                pred_aver = model_aver.predict(test_x)
                pred_pinn = model_pinn.predict(test_x)

                err_aver = mse(test_u, pred_aver)
                err_pinn = mse(test_u, pred_pinn)

                del model_aver
                del model_pinn

                assert err_pinn < err_aver
            except AssertionError:
                fails += 1

        assert fails <= 1

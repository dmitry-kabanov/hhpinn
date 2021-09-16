import numpy as np
import numpy.testing as npt

from hhpinn.models import HodgeHelmholtzPINN


class TestHodgeHelmholtzPINN:
    def test_saves_hyperparameters(self):
        HIDDEN_LAYERS = [27, 8, 101]
        EPOCHS = 22
        LEARNING_RATE = 0.031

        sut = HodgeHelmholtzPINN(
            hidden_layers=HIDDEN_LAYERS, epochs=EPOCHS, learning_rate=LEARNING_RATE
        )

        npt.assert_equal(sut.hidden_layers, HIDDEN_LAYERS)
        npt.assert_equal(sut.epochs, EPOCHS)
        npt.assert_equal(sut.learning_rate, LEARNING_RATE)

    def test_linear_model_returns_correct_output(self):
        sut = HodgeHelmholtzPINN(
            hidden_layers=[], epochs=0
        )
        sut.fit(np.zeros((10, 2)), np.zeros((10, 2)))
        weights = sut.model.trainable_variables[0]

        test_pred = sut.predict(np.random.normal(size=(10, 2)))

        ux = test_pred[:, 0]
        uy = test_pred[:, 1]

        for i in range(len(ux)):
            npt.assert_allclose(ux[i], weights[1])

        for i in range(len(uy)):
            npt.assert_allclose(uy[i], -weights[0])

    def test_save_load_before_training(self, tmpdir):
        # Variable `tmpdir` is a `pytest` fixture.
        m1 = HodgeHelmholtzPINN()

        m1.save(tmpdir)
        m2 = HodgeHelmholtzPINN.load(tmpdir)

        npt.assert_equal(m1.hidden_layers, m2.hidden_layers)

    def test_save_load_after_training(self, tmpdir):
        m1 = HodgeHelmholtzPINN(hidden_layers=[10], epochs=5, learning_rate=0.1)
        m1.fit(np.random.normal(size=(10, 2)), np.random.normal(size=(10, 2)))

        m1.save(tmpdir)
        m2 = HodgeHelmholtzPINN.load(tmpdir)

        for i in range(len(m1.model.trainable_variables)):
            npt.assert_array_equal(m1.model.trainable_variables[i],
                             m2.model.trainable_variables[i])

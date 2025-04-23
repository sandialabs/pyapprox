import unittest
import numpy as np
from functools import partial


from pyapprox.surrogates.nonlinear.mlp import (
    NeuralNetwork,
    SigmoidActivation,
    TanhActivation,
    RELUActivation,
    IdentityActivation,
    Layer,
    NNMSELoss,
    NNMSEL2ReguarlizedLoss,
    MultiTaskNeuralNetwork,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.model import ModelFromVectorizedCallable


def _approx_fprimeprime(xk, f, epsilon, args=()):
    # Do not recommend using other than to get rough estimate of second
    # derivative when debugging
    # keep eps > 1e-4 because it is squared
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (
            f(*((xk + d,) + args)) - 2 * f0 + f(*((xk - d,) + args))
        ) / d[k] ** 2
        ei[k] = 0.0
    return grad


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def setup_single_hidden_layer_mlp(
        self, activation, loss, nvars, nqoi, width
    ):
        bkd = NumpyMixin
        layers = [
            Layer(activation, nvars, width),
            Layer(IdentityActivation(bkd), width, nqoi),
        ]
        network = NeuralNetwork(layers)
        network.set_optimizer(
            network.default_optimizer(ncandidates=1, method="trust-constr")
        )
        network.set_loss(loss)
        return network

    def setup_single_hidden_layer_mlp_with_training_data(
        self, activation, loss
    ):
        bkd = NumpyMixin
        nvars = 3
        nqoi = 2
        width = 3
        network = self.setup_single_hidden_layer_mlp(
            activation, loss, nvars, nqoi, width
        )
        train_samples = bkd.asarray(
            np.random.uniform(0, 1, (nvars, nvars * 11))
        )
        train_values = bkd.stack(
            [bkd.sum(train_samples ** (ii + 2), axis=0) for ii in range(nqoi)],
            axis=1,
        )
        network._set_training_data(train_samples, train_values)
        return network

    def check_nn_loss_gradients(self, activation, loss):
        network = self.setup_single_hidden_layer_mlp_with_training_data(
            activation, loss
        )
        errors = network._loss.check_apply_jacobian(
            network.hyp_list().get_active_opt_params()[:, None], disp=False
        )
        assert errors.min() / errors.max() < 1e-6

    def test_nn_loss_gradient(self):
        bkd = NumpyMixin
        self.check_nn_loss_gradients(SigmoidActivation(bkd), NNMSELoss())
        self.check_nn_loss_gradients(TanhActivation(bkd), NNMSELoss())
        self.check_nn_loss_gradients(
            RELUActivation(bkd), NNMSEL2ReguarlizedLoss(0.5)
        )

    def check_nn_input_gradients(self, activation, loss):
        network = self.setup_single_hidden_layer_mlp_with_training_data(
            activation, loss
        )
        x0 = network._ctrain_samples[:, :1]
        errors = network.check_apply_jacobian(x0, disp=False)
        assert errors.min() / errors.max() < 1e-6

    def test_nn_input_gradient(self):
        bkd = NumpyMixin
        self.check_nn_input_gradients(SigmoidActivation(bkd), NNMSELoss())

    def test_fit(self):
        bkd = NumpyMixin

        def fun(x):
            return bkd.cos(np.pi * np.sum(x, axis=0))[:, None]

        model = ModelFromVectorizedCallable(1, 1, fun)

        nqoi, nvars, width = 1, 1, 5
        activation, loss = SigmoidActivation(bkd), NNMSELoss()
        network = self.setup_single_hidden_layer_mlp(
            activation, loss, nvars, nqoi, width
        )

        nsamples = 50
        lb, ub = -1, 1
        training_samples = bkd.linspace(lb, ub, nsamples)[None, :]
        training_vals = model(training_samples)
        network.fit(training_samples, training_vals)
        validation_samples = bkd.array(
            np.random.uniform(lb, ub, (nvars, int(1e4)))
        )
        validation_values = fun(validation_samples)
        error = bkd.norm(
            validation_values - network(validation_samples)
        ) / bkd.norm(validation_values)
        # print(error)
        assert error < 1e-4

        # import matplotlib.pyplot as plt

        # ax = plt.figure().gca()
        # model.plot_surface(ax, [-1, 1])
        # network.plot_surface(ax, [-1, 1])
        # plt.plot(training_samples[0], training_vals, "o")
        # plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)

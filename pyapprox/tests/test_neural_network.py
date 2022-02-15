import unittest
import numpy as np
from functools import partial


from pyapprox.neural_networks import (
    NeuralNetwork, sigmoid_function, sigmoid_gradient,
    sigmoid_second_derivative, flatten_nn_parameters
)
from pyapprox.optimization import check_gradients, approx_jacobian


def _approx_fprimeprime(xk, f, epsilon, args=()):
    # Do not recommend using other than to get rough estimate of second
    # derivative when debugging
    # keep eps > 1e-4 because it is squared
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk), ), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (
            f(*((xk + d,) + args)) - 2*f0 + f(*((xk - d,) + args))) / d[k]**2
        ei[k] = 0.0
    return grad


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def check_nn_loss_gradients(self, activation_fun):
        nvars = 3
        nqoi = 2

        opts = {'activation_func': activation_fun, 'layers': [nvars, 3, nqoi],
                'loss_func': 'squared_loss', 'lag_mult': 0.5}
        # No hidden layers works
        # opts = {'activation_func':'sigmoid', 'layers':[nvars, nqoi],
        #        'loss_func':'squared_loss'}
        network = NeuralNetwork(opts)

        # train_samples = np.linspace(0, 1, 11)[None, :]
        train_samples = np.random.uniform(0, 1, (nvars, nvars*11))
        train_values = np.hstack(
            [np.sum(train_samples**(ii+2), axis=0)[:, None]
             for ii in range(nqoi)])
        obj = partial(
            network.objective_function, train_samples, train_values)
        jac = partial(
            network.objective_jacobian, train_samples, train_values)
        parameters = np.random.normal(0, 1, (network.nparams))

        disp = True
        # disp = False

        def fun(x):
            return np.sum(obj(x))

        zz = parameters[:, None]

        errors = check_gradients(fun, jac, zz, plot=False, disp=disp, rel=True,
                                 direction=None, jacp=None)
        # make sure gradient changes by six orders of magnitude
        assert np.log10(errors.max())-np.log10(errors.min()) > 6


    def check_nn_input_gradients(self, activation_fun):
        nvars = 3
        nqoi = 2

        opts = {'activation_func': activation_fun, 'layers': [nvars, 3, nqoi],
                'loss_func': 'squared_loss'}
        # No hidden layers works
        # opts = {'activation_func':'sigmoid', 'layers':[nvars, nqoi],
        #        'loss_func':'squared_loss'}
        network = NeuralNetwork(opts)

        # train_samples = np.linspace(0, 1, 11)[None, :]
        train_samples = np.random.uniform(0, 1, (nvars, nvars*11))
        train_values = np.hstack(
            [np.sum(train_samples**(ii+2), axis=0)[:, None]
             for ii in range(nqoi)])
        obj = partial(
            network.objective_function, train_samples, train_values)
        jac = partial(
            network.objective_jacobian, train_samples, train_values)
        parameters = np.random.normal(0, 1, (network.nparams))

        disp = True
        # disp = False

        jac = partial(network.gradient_wrt_inputs, parameters, store=True)
        x0 = train_samples[:, :1]

        def fun(x):
            return network.forward_propagate(x, parameters).T

        errors = check_gradients(fun, jac, x0, plot=False, disp=disp, rel=True,
                                 direction=None, jacp=None)
        # print(np.log10(errors.max())-np.log10(errors.min()))
        assert np.log10(errors.max())-np.log10(errors.min()) > 5.7  # 6

    def test_nn_loss_gradient(self):
        self.check_nn_loss_gradients('sigmoid')
        self.check_nn_loss_gradients('tanh')
        self.check_nn_loss_gradients('relu')

    def test_nn_input_gradient(self):
        self.check_nn_input_gradients('sigmoid')
        self.check_nn_input_gradients('tanh')
        self.check_nn_input_gradients('relu')

    def test_1_layer_gradient_wrt_inputs(self):
        nqoi, nvars = 2, 3
        A = np.random.normal(0, 1, (nqoi, nvars))
        b = np.random.normal(0, 1, (nqoi, 1))

        def fun(x):
            return sigmoid_function(A.dot(x)+b)

        def jac(x):
            return sigmoid_gradient(A.dot(x)+b)*A
            # return np.diag(sigmoid_gradient(A.dot(x)+b)[:, 0]).dot(A)

        x0 = np.random.normal(0, 1, (nvars, 1))
        assert np.allclose(approx_jacobian(fun, x0), jac(x0))

        def g(x): return sigmoid_function(x)[:, None]
        assert np.allclose(_approx_fprimeprime(x0[:, 0], g, 1e-4)[:, 0, :],
                           approx_jacobian(sigmoid_gradient, x0[:, 0]))
        assert np.allclose(
            np.diag(approx_jacobian(sigmoid_gradient, x0[:, 0])),
            sigmoid_second_derivative(x0)[:, 0])

        def f(x):
            x = x[:, None]
            return fun(x)

        def hess(x):
            return sigmoid_second_derivative(A.dot(x)+b)*A**2

        assert np.allclose(
            hess(x0), _approx_fprimeprime(x0[:, 0], f, 1e-4)[:, 0, :])

    def test_fit(self):
        def fun(x):
            return np.cos(2*np.pi*np.sum(x, axis=0))[:, None]

        nqoi, nhl_nodes = 1, 3
        nvars, nsamples = 1, 25
        # nvars, nsamples = 2, 30
        lb, ub = -1, 1
        training_samples = np.random.uniform(lb, ub, (nvars, nsamples))
        # training_samples = np.linspace(lb, ub, nsamples)[None, :]
        training_vals = fun(training_samples)

        opts = {'activation_func': 'tanh',
                'layers': [nvars, nhl_nodes, nqoi],
                'loss_func': 'squared_loss'}
        network = NeuralNetwork(opts)
        nrestarts = 10
        x0 = np.random.uniform(-1, 2, (network.nparams, nrestarts))
        optimizer_opts = {"method": "L-BFGS-B",
                          "options": {"maxiter": 1000}}
        results = network.fit(training_samples, training_vals, x0, verbose=1,
                              opts=optimizer_opts)

        expanded_params = network.expand_parameters(network.parameters)
        assert np.allclose(
            network.parameters, flatten_nn_parameters(*expanded_params))

        validation_samples = np.random.uniform(lb, ub, (nvars, int(1e4)))
        validation_values = fun(validation_samples)
        error = (
            np.linalg.norm(validation_values-network(validation_samples)) /
            np.linalg.norm(validation_values))
        print(error)
        # assert error < 0.04

        import matplotlib.pyplot as plt
        xx = np.linspace(lb, ub, 101)
        # for x in x0.T:
        #     network.set_parameters(x)
        #     plt.plot(xx, network(xx[None, :])[:, 0], lw=0.5, c='g')
        # plt.show()

        plt.plot(xx, fun(xx[None, :])[:, 0], 'r')
        plt.plot(training_samples[0, :], training_vals[:, 0], 'ro')
        network_vals = network(xx[None, :])
        plt.plot(xx, network_vals[:, 0], '--b')
        # selected compled solves for ploting
        comp_results = [res for res in results if res.success]
        for res in comp_results:
            network.set_parameters(res.x)
            plt.plot(xx, network(xx[None, :])[:, 0], lw=0.5, c='k')
        incomp_results = [res for res in results if
                          (not res.success and "precision" in res.message)]
        for res in incomp_results:
            network.set_parameters(res.x)
            plt.plot(xx, network(xx[None, :])[:, 0], lw=0.5, c='g')
        plt.ylim(network_vals.min()-0.1*abs(network_vals.min()),
                 network_vals.max()+0.1*abs(network_vals.max()))
        # plt.show()


if __name__ == "__main__":
    neural_network_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestNeuralNetwork)
    unittest.TextTestRunner(verbosity=2).run(neural_network_test_suite)

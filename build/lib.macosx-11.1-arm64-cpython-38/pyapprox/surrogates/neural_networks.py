#!/usr/bin/env python
import numpy as np
from functools import partial
from scipy import optimize
from scipy.special import expit


def sigmoid_function(samples):
    # return 1/(1 + np.exp(-samples))
    return expit(samples)


def sigmoid_gradient(samples):
    return sigmoid_function(samples)*(1.0-sigmoid_function(samples))


def sigmoid_second_derivative(samples):
    exp_x = np.exp(samples)
    return -exp_x*(exp_x-1)/(exp_x+1)**3


def tanh_function(samples):
    return np.tanh(samples)


def tanh_gradient(samples):
    return 1-np.tanh(samples)**2


def relu_function(samples):
    return np.maximum(0, samples)


def relu_gradient(samples):
    grad = np.zeros_like(samples)
    grad[samples > 0] = 1
    return grad


def squared_loss_function(approx_values, values):
    """
    Parameters
    ----------
    approx_values : np.ndarray(nqoi, nsamples)
        The output of the approximation
    values : np.ndarray(nqoi, nsamples)
        The training data
    """
    return 0.5*np.sum((approx_values - values)**2, axis=0)/values.shape[1]


def squared_loss_gradient_numerator_convention(approx_values, values):
    """
    Parameters
    ----------
    approx_values : np.ndarray(nqoi, nsamples)
        The output of the approximation
    values : np.ndarray(nqoi, nsamples)
        The training data
    """
    return ((approx_values - values)/values.shape[1]).T


def flatten_nn_parameters(Wmats, bvecs, flatten_ord="C"):
    parameters = []
    for ii in range(1, len(Wmats)):
        parameters.append(Wmats[ii].flatten(order=flatten_ord))
        parameters.append(bvecs[ii])
    return np.hstack(parameters)


def __generate_initial_nn_parameters_glorot(layers, factor=2):
    nlayers = len(layers)
    parameters = []
    for ii in range(nlayers-1):
        ub = np.sqrt(factor / (layers[ii] + layers[ii+1]))
        lb = -ub
        parameters.append(
            np.random.uniform(lb, ub, (layers[ii]*layers[ii+1])))
        parameters.append(np.random.uniform(lb, ub, (layers[ii+1])))
    return np.hstack(parameters)[:, None]


def generate_initial_nn_parameters_glorot(layers, nrepeats=1, factor=2):
    return np.hstack([__generate_initial_nn_parameters_glorot(layers, factor)
                      for nn in range(nrepeats)])


class NeuralNetwork(object):
    def __init__(self, opts):
        self.opts = opts
        self.parse_options()

        # storage for intermediate calculations used when
        # calculating the gradient
        self.derivative_info = None
        self.last_parameters = None

    def parse_options(self):
        activation_functions = {
            'sigmoid': [sigmoid_function, sigmoid_gradient],
            'tanh': [tanh_function, tanh_gradient],
            'relu': [relu_function, relu_gradient]
        }
        activation_type = self.opts['activation_func']
        if activation_type not in activation_functions:
            msg = f"Activation func {activation_type} not "
            msg += "supported"
            raise Exception(msg)
        self.afunc, self.agrad = activation_functions[
            activation_type]

        loss_functions = {
            'squared_loss':
            [squared_loss_function,
             squared_loss_gradient_numerator_convention]}
        loss_type = self.opts['loss_func']
        if loss_type not in loss_functions:
            msg = f"Loss func {loss_type} not "
            msg += "supported"
            raise Exception(msg)
        self.Cfunc, self.Cgrad = loss_functions[loss_type]

        # determines whether to add activation function on last layer
        self.output_activation = self.opts.get("output_activation", False)

        self.layers = self.opts['layers']
        self.nlayers = len(self.layers)
        self.nvars = self.layers[0]
        self.nqoi = self.layers[-1]
        assert self.nlayers >= 2
        self.parameters = None

        self.nparams = 0
        for ii in range(self.nlayers-1):
            self.nparams += self.layers[ii]*self.layers[ii+1]
            self.nparams += self.layers[ii+1]

        self.var_trans = self.opts.get("var_trans", None)
        self.lag_mult = self.opts.get("lag_mult", 0)

    def expand_parameters(self, parameters):
        parameters = parameters.squeeze()
        assert parameters.ndim == 1
        assert self.nparams == parameters.shape[0]
        Wmats, bvecs = [None], [None]
        lb = 0
        for ii in range(self.nlayers-1):
            ub = lb + self.layers[ii]*self.layers[ii+1]
            Wmats.append(parameters[lb:ub].reshape(
                (self.layers[ii+1], self.layers[ii])))
            lb = ub
            ub = lb + self.layers[ii+1]
            bvecs.append(parameters[lb:ub])
            lb = ub
        return Wmats, bvecs

    def forward_propagate(self, train_samples, parameters):
        Wmats, bvecs = self.expand_parameters(parameters)
        self.derivative_info = []

        # Input Layer
        yout = train_samples
        self.derivative_info.append([yout, yout])

        # Hidden Layers
        for ii in range(1, self.nlayers-1):
            # python broadcasting is used to add bvecs[ii] to all columns
            uout = Wmats[ii].dot(yout) + bvecs[ii][:, None]
            yout = self.afunc(uout)
            self.derivative_info.append([uout, yout])

        # Output layer
        ii = self.nlayers-1
        # python broadcasting is used to add bvecs[ii] to all columns
        uout = Wmats[ii].dot(yout) + bvecs[ii][:, None]
        if self.output_activation:
            yout = self.afunc(uout)
        else:
            yout = uout.copy()
        self.derivative_info.append([uout, yout])
        return yout.T  # keep pyapprox convention for outputs

    def update_parameter_gradient(self, dC_dW, dC_db, jacobian, ub):
        lb = ub - dC_db.shape[0]
        jacobian[lb:ub] = dC_db
        ub = lb
        lb = ub - np.prod(dC_dW.shape)
        # must flatten with order="F" to account for fact that we
        # are using numerator convention which transposes denominator
        # of derivative, i.e. W. Or equivalently use
        # jacobian[lb:ub] = dC_dW.T.flatten()
        jacobian[lb:ub] = dC_dW.flatten(order="F")
        ub = lb
        return jacobian, ub

    def layer_backwards_propgate(self, delta_l, Wmats, layer, jacobian, ub,
                                 activation=True):
        u_l, y_l = self.derivative_info[layer-1]
        dC_dW = y_l.dot(delta_l)
        dC_db = np.sum(delta_l, axis=0)
        delta_lm1 = (delta_l.dot(Wmats[layer]))*self.agrad(u_l.T)
        jacobian, ub = self.update_parameter_gradient(
            dC_dW, dC_db, jacobian, ub)
        return delta_lm1, jacobian, ub

    def backwards_propagate(self, train_values, parameters):
        Wmats, bvecs = self.expand_parameters(parameters)
        # The true jacobian has shape (1, nparams)
        # but scipy optimize requires a 1D ndarray
        jacobian = np.empty((self.nparams))

        # Gradient of loss with resepect to y_L
        u_l, y_l = self.derivative_info[-1]
        # train_values.T converts from pyapprox convention to NN convention
        dC_yl = self.Cgrad(y_l, train_values.T)
        if self.output_activation:
            delta_l = dC_yl*self.agrad(u_l.T)
        else:
            delta_l = dC_yl
        delta_l, jacobian, ub = self.layer_backwards_propgate(
            delta_l, Wmats, self.nlayers-1, jacobian, self.nparams,
            self.output_activation)
        for layer in range(self.nlayers-2, 0, -1):
            delta_l, jacobian, ub = self.layer_backwards_propgate(
                delta_l, Wmats, layer, jacobian, ub)
        return jacobian

    def gradient_wrt_inputs(self, parameters, sample, store=False):
        """
        Compute gradient of final layer outputs with respect to first layer
        inputs
        """
        assert sample.shape[1] == 1
        self.forward_propagate(sample, parameters)
        Wmats, bvecs = self.expand_parameters(parameters)
        z_L = self.derivative_info[-1][0]   # (nsamples, width[L])
        if self.output_activation:
            da_dz_diag = self.agrad(z_L)
        else:
            da_dz_diag = np.ones_like(z_L)
        dz_da = Wmats[-1]
        jacobian = da_dz_diag*dz_da
        self.jac_info = []
        for ll in range(self.nlayers-2, 0, -1):
            z_ll = self.derivative_info[ll][0]
            dz_da = Wmats[ll]
            da_dz_diag = self.agrad(z_ll)
            tmp = da_dz_diag*dz_da
            if store:
                self.jac_info.append(tmp)
            jacobian = jacobian.dot(tmp)
        return jacobian

    def second_derivatives_wrt_inputs(self, parameters, sample):
        """
        Compute second derivative of final layer outputs with respect
        to first layer inputs. The output is the diagonal of the hessian
        for each qoi. Useful for physics informed NNs.
        """
        raise NotImplementedError("Not tested")
        assert len(self.jac_info) == self.nlayers-2
        assert sample.shape[1] == 1
        self.forward_propagate(sample, parameters)
        Wmats, bvecs = self.expand_parameters(parameters)
        z_L = self.derivative_info[-1][0]   # (nsamples, width[L])
        da_dz_diag = sigmoid_second_derivative(z_L)
        dz_da = Wmats[-1]
        cnt = 0
        derivs = (da_dz_diag*dz_da**2).dot(self.jac_info[cnt])
        for ll in range(self.nlayers-2, 0, -1):
            cnt += 1
            z_ll = self.derivative_info[ll][0]
            dz_da = Wmats[ll]
            da_dz_diag = sigmoid_second_derivative(z_ll)
            if cnt < self.nlayers-2:
                derivs += (da_dz_diag*dz_da**2).dot(self.jac_info[cnt])
        return derivs

    def objective_jacobian(self, train_samples, train_values,
                           parameters):
        if (self.last_parameters is None or
                not np.allclose(parameters, self.last_parameters, rtol=1e-12)):
            self.forward_propagate(train_samples, parameters)
        grad = self.backwards_propagate(train_values, parameters)
        grad += self.lag_mult*parameters.squeeze()/train_samples.shape[0]
        return grad

    def objective_function(self, train_samples, train_values,
                           parameters):
        assert train_values.ndim == 2
        assert train_values.shape[0] == train_samples.shape[1]
        self.last_parameters = parameters
        approx_values = self.forward_propagate(
            train_samples, parameters)
        loss = np.sum(self.Cfunc(approx_values.T, train_values.T))
        loss += self.lag_mult*0.5*parameters.squeeze().dot(
            parameters.squeeze())/train_samples.shape[0]
        return loss

    def _parse_data(self, samples, values):
        assert samples.shape[0] == self.layers[0]
        assert values.ndim == 2
        assert values.shape[0] == samples.shape[1]
        assert values.shape[1] == self.layers[-1]
        if self.var_trans is not None:
            train_samples = self.var_trans.map_to_canonical(samples)
        else:
            train_samples = samples
        train_values = values
        ntrain_samples = train_values.shape[0]
        return train_samples, train_values, ntrain_samples

    def fit(self, samples, values, x0=None, opts={}, verbose=0):
        self.train_samples, self.train_values, self.ntrain_samples = (
            self._parse_data(samples, values))
        obj = partial(self.objective_function, self.train_samples,
                      self.train_values)
        jac = partial(self.objective_jacobian, self.train_samples,
                      self.train_values)
        if x0 is None:
            x0 = generate_initial_nn_parameters_glorot(self.layers, 1)
        elif np.isscalar(x0):
            x0 = generate_initial_nn_parameters_glorot(self.layers, x0)
        if x0.ndim != 2:
            raise ValueError(f"x0.ndim={x0.ndim} but must equal 2")
        if x0.shape[0] != self.nparams:
            msg = f"x0.shape[0] is {x0.shape[0]} but must be {self.nparams}"
            raise ValueError(msg)

        if verbose > 0:
            print("No. Parameters", self.nparams)
            print("No. Samples", self.ntrain_samples)

        results = []
        for x in x0.T:
            results.append(optimize.minimize(obj, x, args=(), jac=jac, **opts))
            if verbose > 0:
                print(f"Repeat {len(results)}/{x0.shape[1]}", "Loss",
                      results[-1].fun, "Success", results[-1].success,
                      "gnorm", np.linalg.norm(results[-1].jac))
        completed_solves = [res.success for res in results]
        if opts.get("enforce_success", False) and not np.any(completed_solves):
            print([res.message for res in results])
            raise RuntimeError("No optimizations converged")
        obj_vals = [res.fun for res in results]
        II = np.argmin(obj_vals)
        self.set_parameters(results[II].x)
        if verbose > 0:
            print("No. restarts completed",
                  np.where(completed_solves)[0].shape[0], f"/ {x0.shape[1]}")
            print("Losses")
            print(obj_vals)
        return results

    def set_parameters(self, parameters):
        if parameters.shape != (self.nparams,):
            msg = f"parameters shape is {parameters.shape}"
            msg += f" but must have shape {(self.nparams,)}"
            raise ValueError(msg)
        self.parameters = parameters

    def __call__(self, samples):
        if self.var_trans is not None:
            canonical_samples = self.var_trans.map_to_canonical(samples)
        else:
            canonical_samples = samples.copy()
        return self.forward_propagate(canonical_samples, self.parameters)

    def __repr__(self):
        rep = "MLP({0})".format(
            ",".join([str(layer) for layer in self.layers]))
        return rep


class MultiTaskNeuralNetwork(NeuralNetwork):
    def backwards_propagate(self, train_values, parameters, task):
        Wmats, bvecs = self.expand_parameters(parameters)
        # The true jacobian has shape (1, nparams)
        # but scipy optimize requires a 1D ndarray
        jacobian = np.empty((self.nparams))

        # Gradient of loss with resepect to y_L
        u_l, y_l = self.derivative_info[-1]
        # train_values.T converts from pyapprox convention to NN convention
        dC_yl = self.Cgrad(y_l, train_values.T)
        if self.output_activation:
            delta_l = dC_yl*self.agrad(u_l.T)
        else:
            delta_l = dC_yl
        delta_l[:, :task] = 0
        delta_l[:, task+1:] = 0
        delta_l, jacobian, ub = self.layer_backwards_propgate(
            delta_l, Wmats, self.nlayers-1, jacobian, self.nparams,
            self.output_activation)
        for layer in range(self.nlayers-2, 0, -1):
            delta_l, jacobian, ub = self.layer_backwards_propgate(
                delta_l, Wmats, layer, jacobian, ub)
        return jacobian

    def objective_jacobian(self,
                           train_samples_per_model, train_values_per_model,
                           parameters):
        grad = 0
        task = 0
        for train_samples, train_values in zip(
                train_samples_per_model, train_values_per_model):
            self.forward_propagate(train_samples, parameters)
            grad += self.backwards_propagate(train_values, parameters, task)
            task += 1
        grad += self.lag_mult*parameters.squeeze()/train_samples[0].shape[0]
        # TODO should train_samples[0].shape[0] be ntrain_samples
        return grad

    def objective_function(self, train_samples_per_model,
                           train_values_per_model,
                           parameters):
        self.last_parameters = parameters
        loss = 0
        task = 0
        # assert self.layers[-1] == 1  # TODO assumes 1 qoi for each task
        for train_samples, train_values in zip(
                train_samples_per_model, train_values_per_model):
            approx_values = self.forward_propagate(
                train_samples, parameters)[:, task:task+1]
            loss += np.sum(self.Cfunc(approx_values.T, train_values.T))
            task += 1
        loss += self.lag_mult*0.5*parameters.squeeze().dot(
            parameters.squeeze())/train_samples[0].shape[0]
        # TODO should train_samples[0].shape[0] be ntrain_samples
        return loss

    def _parse_data(self, samples_per_model, values_per_model):
        assert isinstance(samples_per_model, list)
        assert isinstance(values_per_model, list)
        assert len(values_per_model) == self.layers[-1]
        assert len(values_per_model) == len(samples_per_model)
        for samples, values in zip(samples_per_model, values_per_model):
            assert samples.shape[0] == self.layers[0]
            assert values.shape[0] == samples.shape[1]
            assert values.ndim == 2
        if self.var_trans is not None:
            train_samples = [
                self.var_trans.map_to_canonical(samples)
                for samples in samples_per_model]
        else:
            train_samples = samples_per_model
        train_values = values_per_model
        ntrain_samples = sum([samples.shape[1] for samples in train_samples])
        return train_samples, train_values, ntrain_samples

    def __repr__(self):
        rep = "MultiTaskNN({0})".format(
            ",".join([str(layer) for layer in self.layers]))
        return rep

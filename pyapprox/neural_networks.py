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
    return 0.5*np.sum((approx_values - values)**2, axis=1)/values.shape[0]


def squared_loss_gradient(approx_values, values):
    return (approx_values - values)/values.shape[0]


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
            [squared_loss_function, squared_loss_gradient]}
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
        assert train_samples.ndim == 2
        Wmats, bvecs = self.expand_parameters(parameters)
        aout = train_samples.T
        self.derivative_info = [[aout, aout]]
        for ii in range(1, self.nlayers-1):
            zout = Wmats[ii].dot(aout.T).T + bvecs[ii]
            aout = self.afunc(zout)
            self.derivative_info.append([zout, aout])
        ii = self.nlayers-1
        zout = Wmats[ii].dot(aout.T).T + bvecs[ii]
        if self.output_activation:
            aout = self.afunc(zout)
        else:
            aout = zout.copy()
        self.derivative_info.append([zout, aout])
        return zout

    def backwards_propagate(self, train_values, parameters):
        """
        Compute gradient of cost function with respect to hyper-parameters
        """
        jacobian = np.empty((self.nparams))
        Wmats, bvecs = self.expand_parameters(parameters)
        dC_da = self.Cgrad(self.derivative_info[-1][1], train_values)
        z_L = self.derivative_info[-1][0]      # (nsamples, width[L])
        if self.output_activation:
            dC_dz = (dC_da*self.agrad(z_L)).T  # (width[L], nsamples)
        else:
            dC_dz = (dC_da*np.ones_like(z_L)).T
        dC_db = np.sum(dC_dz, axis=1)          # (width[L])
        a_Lm1 = self.derivative_info[-2][1]    # (nsamples, width[L-1])
        dC_dW = dC_dz.dot(a_Lm1)               # (width[L], width[L-1])
        ub = self.nparams
        lb = ub - dC_db.shape[0]
        jacobian[lb:ub] = dC_db
        ub = lb
        lb = ub - np.prod(dC_dW.shape)
        jacobian[lb:ub] = dC_dW.flatten()
        ub = lb
        for ll in range(self.nlayers-2, 0, -1):
            dC_dz_prev = dC_dz.copy()
            z_ll = self.derivative_info[ll][0]
            a_llm1 = self.derivative_info[ll-1][1]
            dC_dz = Wmats[ll+1].T.dot(dC_dz_prev)*self.agrad(z_ll).T
            dC_dW = dC_dz.dot(a_llm1)
            dC_db = np.sum(dC_dz, axis=1)
            lb = ub - dC_db.shape[0]
            jacobian[lb:ub] = dC_db
            ub = lb
            lb = ub - np.prod(dC_dW.shape)
            jacobian[lb:ub] = dC_dW.flatten()
            ub = lb
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
            da_dz_diag = self.agrad(z_L).T
        else:
            da_dz_diag = np.ones_like(z_L).T
        dz_da = Wmats[-1]
        jacobian = da_dz_diag*dz_da
        self.jac_info = []
        for ll in range(self.nlayers-2, 0, -1):
            z_ll = self.derivative_info[ll][0]
            dz_da = Wmats[ll]
            da_dz_diag = self.agrad(z_ll).T
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
        da_dz_diag = sigmoid_second_derivative(z_L).T
        dz_da = Wmats[-1]
        cnt = 0
        derivs = (da_dz_diag*dz_da**2).dot(self.jac_info[cnt])
        for ll in range(self.nlayers-2, 0, -1):
            cnt += 1
            z_ll = self.derivative_info[ll][0]
            dz_da = Wmats[ll]
            da_dz_diag = sigmoid_second_derivative(z_ll).T
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
        loss = np.sum(self.Cfunc(approx_values, train_values))
        loss += self.lag_mult*0.5*parameters.squeeze().dot(
            parameters.squeeze())/train_samples.shape[0]
        return loss

    def fit(self, samples, values, x0=None, opts={}, verbose=0):
        assert samples.shape[0] == self.layers[0]
        if self.var_trans is not None:
            self.train_samples = self.var_trans.map_to_canonical_space(samples)
        else:
            self.train_samples = samples
        self.train_values = values
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
            print("No. Samples", self.train_samples.shape[1])

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
            canonical_samples = self.var_trans.map_to_canonical_space(samples)
        else:
            canonical_samples = samples.copy()
        return self.forward_propagate(canonical_samples, self.parameters)

    def __repr__(self):
        rep = "MLP({0})".format(
            ",".join([str(layer) for layer in self.layers]))
        return rep

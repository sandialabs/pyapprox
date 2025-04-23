from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from scipy.special import expit

from pyapprox.surrogates.loss import LossFunction
from pyapprox.surrogates.regressor import OptimizedRegressor
from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.hyperparameter import (
    HyperParameterList,
    HyperParameter,
    IdentityHyperParameterTransform,
)


class Activation(ABC):
    def __init__(self, bkd: BackendMixin = NumpyMixin):
        self._bkd = bkd

    @abstractmethod
    def __call__(self, mat: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jacobian_diag(self, mat: Array) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class IdentityActivation(Activation):
    def __call__(self, mat: Array) -> Array:
        return mat

    def jacobian_diag(self, mat: Array) -> Array:
        return 1.0


class SigmoidActivation(Activation):
    def __call__(self, mat: Array) -> Array:
        return expit(mat)

    def jacobian_diag(self, mat: Array) -> Array:
        return self(mat) * (1.0 - self(mat))


class TanhActivation(Activation):
    def __call__(self, mat: Array) -> Array:
        return np.tanh(mat)

    def jacobian_diag(self, mat: Array) -> Array:
        return 1 - np.tanh(mat) ** 2


class RELUActivation(Activation):
    def __call__(self, mat: Array) -> Array:
        return np.maximum(0, mat)

    def jacobian_diag(self, mat: Array) -> Array:
        grad = np.zeros_like(mat)
        grad[mat > 0] = 1
        return grad


class Layer:
    def __init__(self, activation: Activation, inwidth: int, outwidth: int):
        if not isinstance(activation, Activation):
            raise ValueError("activation must be an instance of Activation")
        self._bkd = activation._bkd
        self._activation = activation
        self._inwidth = inwidth
        self._outwidth = outwidth

        transform = IdentityHyperParameterTransform(backend=self._bkd)
        self._weights = HyperParameter(
            "weights",
            inwidth * outwidth,
            self._bkd.asarray(np.random.normal(0, 1, (inwidth * outwidth,))),
            (-np.inf, np.inf),
            transform,
            fixed=False,
            backend=self._bkd,
        )

        self._biases = HyperParameter(
            "biases",
            outwidth,
            self._bkd.asarray(np.random.normal(0, 1, (outwidth,))),
            (-np.inf, np.inf),
            transform,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._weights, self._biases])

    def __repr__(self):
        return "{0}(shape=[{1},{2}], {3})".format(
            self.__class__.__name__,
            self._outwidth,
            self._inwidth,
            self._activation,
        )

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _Wmat(self) -> Array:
        return self._weights.get_values().reshape(
            self._outwidth, self._inwidth
        )

    def _bvec(self) -> Array:
        return self._biases.get_values()[:, None]

    def __call__(self, yout: Array, store_jac: bool = False) -> Array:
        uout = self._Wmat() @ yout + self._bvec()
        yout = self._activation(uout)
        if store_jac:
            self._uout = uout
            self._yout = yout
        return yout


class NNLoss(LossFunction):
    @abstractmethod
    def _loss_from_values(self, approx_values: Array, values: Array) -> float:
        # computes loss for each sample
        raise NotImplementedError

    @abstractmethod
    def _loss_jacobian_from_values(
        self, approx_values: Array, values: Array
    ) -> Array:
        # computes loss for each sample
        raise NotImplementedError

    def jacobian_implemented(self) -> bool:
        return True

    def _update_parameter_gradient(self, dC_dW, dC_db, jacobian, ub):
        lb = ub - dC_db.shape[0]
        jacobian[lb:ub] = dC_db
        ub = lb
        lb = ub - np.prod(dC_dW.shape)
        # must flatten with order="F" to account for fact that we
        # are using numerator convention which transposes denominator
        # of derivative, i.e.
        # jacobian[lb:ub] = dC_dW.flatten(order="F")
        # Or equivalently use
        jacobian[lb:ub] = dC_dW.T.flatten()
        ub = lb
        return jacobian, ub

    def _layer_backwards_propgate(
        self,
        delta_l: Array,
        ll: int,
        jacobian: Array,
        ub: int,
    ) -> Tuple[Array, Array, int]:
        layer_l = self._model._layers[ll]
        if ll > 0:
            layer_lm1 = self._model._layers[ll - 1]
            u_l, y_l = layer_lm1._uout, layer_lm1._yout
            activation_values = layer_lm1._activation.jacobian_diag(u_l.T)
        else:
            u_l, y_l = self._train_samples, self._train_samples
            activation_values = 1.0
        dC_dW = y_l @ delta_l
        dC_db = self._bkd.sum(delta_l, axis=0)
        delta_lm1 = (delta_l @ layer_l._Wmat()) * activation_values
        jacobian, ub = self._update_parameter_gradient(
            dC_dW, dC_db, jacobian, ub
        )
        return delta_lm1, jacobian, ub

    def _backwards_propagate(self, train_values: Array) -> Array:
        jacobian = self._bkd.ones((self._model.hyp_list().nactive_vars(),))

        # Gradient of loss with resepect to y_L
        u_l, y_l = self._model._layers[-1]._uout, self._model._layers[-1]._yout
        # train_values.T converts from pyapprox convention to NN convention
        dC_yl = self._loss_jacobian_from_values(y_l, train_values.T)
        delta_l = dC_yl * self._model._layers[-1]._activation.jacobian_diag(
            u_l.T
        )
        ub = self._model.hyp_list().nactive_vars()
        for ll in range(self._model._nlayers - 1, -1, -1):
            delta_l, jacobian, ub = self._layer_backwards_propgate(
                delta_l, ll, jacobian, ub
            )
        return jacobian[None, :]

    def _set_batch(self, train_samples: Array, train_values: Array):
        self._train_samples = train_samples
        self._train_values = train_values

    def _loss_values(self, params: Array) -> Array:
        if not hasattr(self, "_train_samples"):
            raise ValueError("Must call _set_batch()")
        self._model._hyp_list.set_active_opt_params(params[:, 0])
        yout = self._model._forward_propagate(self._train_samples)
        ntrain_samples = self._train_values.shape[0]
        return (
            self._bkd.sum(self._loss_from_values(yout.T, self._train_values))[
                None, None
            ]
            / ntrain_samples
        ) + self._regularization(params)

    def _jacobian(self, params: Array) -> Array:
        if (
            self._model._hyp_list.nactive_vars()
            != self._model._hyp_list.nvars()
        ):
            raise NotImplementedError(
                "jacobian only supported if all parameters are active."
                "Need to change self._update_parameter_gradient "
            )
        self._model._hyp_list.set_active_opt_params(params[:, 0])
        self._model._forward_propagate(self._train_samples, store_jac=True)
        jac = self._backwards_propagate(self._train_values)
        ntrain_samples = self._train_values.shape[0]
        return jac / ntrain_samples + self._regularization_jacobian(params)

    def _regularization(self, params: Array) -> Array:
        return 0.0

    def _regularization_jacobian(self, params: Array) -> Array:
        return 0.0


class NNMSELoss(NNLoss):
    def _loss_from_values(self, approx_values: Array, values: Array) -> float:
        # approx_values shape assumed to follow pyapprox
        # convention (nsamples, nqoi)
        return 0.5 * self._bkd.sum((approx_values - values) ** 2, axis=1)

    def _loss_jacobian_from_values(
        self, approx_values: Array, values: Array
    ) -> Array:
        # approx_values shape assumed to follow pyapprox
        # convention (nsamples, nqoi)
        # numerator convention
        return (approx_values - values).T


class NNMSEL2ReguarlizedLoss(NNMSELoss):
    def __init__(self, lag_mult: float):
        super().__init__()
        self._lag_mult = lag_mult

    def _regularization(self, params: Array) -> Array:
        ntrain_samples = self._train_values.shape[0]
        return self._lag_mult * 0.5 * params.T @ params / ntrain_samples

    def _regularization_jacobian(self, params: Array) -> Array:
        ntrain_samples = self._train_values.shape[0]
        return self._lag_mult * params.T / ntrain_samples


class NeuralNetwork(OptimizedRegressor):
    def __init__(
        self,
        layers: List[Layer],
    ):
        self._parse_layers(layers)
        super().__init__(layers[0]._bkd)

        # storage for intermediate calculations used when
        # calculating the gradient
        self._derivative_info = None
        self._last_parameters = None

    def nparams(self) -> int:
        return self._nparams

    def _parse_layers(self, layers: List[Layer]):
        self._layers = layers
        self._nlayers = len(self._layers)
        for layer in self._layers:
            if not isinstance(layer, Layer):
                raise ValueError(
                    "layer {0} must be an instance of Layer".format(layer)
                )
        for ii in range(self._nlayers - 1):
            if self._layers[ii]._outwidth != self._layers[ii + 1]._inwidth:
                raise ValueError("Layers are inconsistent")
        self._hyp_list = sum([layer.hyp_list() for layer in self._layers])

    def nqoi(self) -> int:
        return self._layers[-1]._outwidth

    def nvars(self) -> int:
        return self._layers[0]._inwidth

    def _forward_propagate(
        self, train_samples: Array, store_jac: bool = False
    ) -> Array:
        yout = train_samples
        for ii, layer in enumerate(self._layers):
            yout = layer(yout, store_jac)
        return yout

    def _set_training_data(self, train_samples: Array, train_values: Array):
        super()._set_training_data(train_samples, train_values)
        self._loss._set_batch(self._ctrain_samples, self._ctrain_values)

    def jacobian_implemented(self) -> bool:
        return True

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute gradient of final layer outputs with respect to first layer
        inputs
        """
        assert sample.shape[1] == 1
        self._forward_propagate(sample, store_jac=True)
        jacobian = None
        for ll in range(self._nlayers - 1, -1, -1):
            z_ll = self._layers[ll]._uout
            dz_da = self._layers[ll]._Wmat()
            da_dz_diag = self._layers[ll]._activation.jacobian_diag(z_ll)
            tmp = da_dz_diag * dz_da
            if jacobian is None:
                jacobian = tmp
            else:
                jacobian = jacobian @ tmp
        return jacobian

    def _values(self, samples: Array) -> Array:
        canonical_samples = self._in_trans.map_to_canonical(samples)
        return self._forward_propagate(canonical_samples).T

    def __repr__(self):
        rep = "MLP({0})".format(
            ",".join([str(layer) for layer in self._layers])
        )
        return rep


class MultiTaskNeuralNetwork(NeuralNetwork):
    def backwards_propagate(self, train_values, parameters, task):
        Wmats, bvecs = self._expand_parameters(parameters)
        # The true jacobian has shape (1, nparams)
        # but scipy optimize requires a 1D ndarray
        jacobian = np.empty((self._nparams))

        # Gradient of loss with resepect to y_L
        u_l, y_l = self._layers[-1]._uout, self._layers[-1]._yout
        # train_values.T converts from pyapprox convention to NN convention
        dC_yl = self._Cgrad(y_l, train_values.T)
        if self._output_activation:
            delta_l = dC_yl * self._agrad(u_l.T)
        else:
            delta_l = dC_yl
        delta_l[:, :task] = 0
        delta_l[:, task + 1 :] = 0
        delta_l, jacobian, ub = self._layer_backwards_propgate(
            delta_l,
            Wmats,
            self._nlayers - 1,
            jacobian,
            self._nparams,
            self._output_activation,
        )
        for layer in range(self._nlayers - 2, 0, -1):
            delta_l, jacobian, ub = self._layer_backwards_propgate(
                delta_l, Wmats, layer, jacobian, ub
            )
        return jacobian

    def objective_jacobian(
        self, train_samples_per_model, train_values_per_model, parameters
    ):
        grad = 0
        task = 0
        for train_samples, train_values in zip(
            train_samples_per_model, train_values_per_model
        ):
            self._forward_propagate(train_samples, parameters)
            grad += self._backwards_propagate(train_values, parameters, task)
            task += 1
        grad += (
            self._lag_mult * parameters.squeeze() / train_samples[0].shape[0]
        )
        # TODO should train_samples[0].shape[0] be ntrain_samples
        return grad

    def objective_function(
        self, train_samples_per_model, train_values_per_model, parameters
    ):
        self._last_parameters = parameters
        loss = 0
        task = 0
        # assert self._layers[-1] == 1  # TODO assumes 1 qoi for each task
        for train_samples, train_values in zip(
            train_samples_per_model, train_values_per_model
        ):
            approx_values = self._forward_propagate(train_samples, parameters)[
                :, task : task + 1
            ]
            loss += np.sum(self._Cfunc(approx_values.T, train_values.T))
            task += 1
        loss += (
            self._lag_mult
            * 0.5
            * parameters.squeeze().dot(parameters.squeeze())
            / train_samples[0].shape[0]
        )
        # TODO should train_samples[0].shape[0] be ntrain_samples
        return loss

    def _parse_data(self, samples_per_model, values_per_model):
        assert isinstance(samples_per_model, list)
        assert isinstance(values_per_model, list)
        assert len(values_per_model) == self._layers[-1]
        assert len(values_per_model) == len(samples_per_model)
        for samples, values in zip(samples_per_model, values_per_model):
            assert samples.shape[0] == self._layers[0]
            assert values.shape[0] == samples.shape[1]
            assert values.ndim == 2
        if self._var_trans is not None:
            train_samples = [
                self._var_trans.map_to_canonical(samples)
                for samples in samples_per_model
            ]
        else:
            train_samples = samples_per_model
        train_values = values_per_model
        ntrain_samples = sum([samples.shape[1] for samples in train_samples])
        return train_samples, train_values, ntrain_samples

    def __repr__(self):
        rep = "MultiTaskNN({0})".format(
            ",".join([str(layer) for layer in self._layers])
        )
        return rep


def __generate_initial_nn_parameters_glorot(layers, factor=2):
    nlayers = len(layers)
    parameters = []
    for ii in range(nlayers - 1):
        ub = np.sqrt(factor / (layers[ii] + layers[ii + 1]))
        lb = -ub
        parameters.append(
            np.random.uniform(lb, ub, (layers[ii] * layers[ii + 1]))
        )
        parameters.append(np.random.uniform(lb, ub, (layers[ii + 1])))
    return np.hstack(parameters)[:, None]


def generate_initial_nn_parameters_glorot(layers, nrepeats=1, factor=2):
    return np.hstack(
        [
            __generate_initial_nn_parameters_glorot(layers, factor)
            for nn in range(nrepeats)
        ]
    )

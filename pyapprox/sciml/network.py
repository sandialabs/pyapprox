import pickle

from pyapprox.sciml.util._torch_wrappers import (
    asarray, array, randperm, cumsum, ones, copy)
from pyapprox.sciml.transforms import IdentityValuesTransform
from pyapprox.sciml.optimizers import LBFGSB
from pyapprox.sciml.integraloperators import (
    DenseAffineIntegralOperator, DenseAffineIntegralOperatorFixedBias,
    FourierConvolutionOperator)
from pyapprox.sciml.activations import (IdentityActivation, TanhActivation)
from pyapprox.sciml.layers import Layer


class CERTANN():
    def __init__(self, nvars, layers, activations, var_trans=None,
                 values_trans=None, optimizer=None):
        """
        A quadrature based nerual operator.

        Parameters
        ----------
        nvars : integer
            The dimension of the input samples

        layers : list[Layer] (nlayers)
            A list of layers

        activations : list[Activation] (nlayers)
            A list of activation functions for each layer

        var_trans : ValuesTransform
            A transformation applied to the inputs, e.g. to map them to [-1, 1]

        values_trans : ValuesTransform
            A transformation applied to the outputs, e.g. to normalize
            the training values to have mean zero and unit variance

        optimizer : Optimizer
            An opimizer used to fit the network.
        """
        self._nvars = nvars  # dimension of input samples
        # for layer in layers:
        #     if not isinstance(layer, Layer):
        #         raise ValueError("Layer type provided is not supported")
        if isinstance(layers, Layer):
            self._layers = [layers]  # list of kernels for each layer
        else:
            self._layers = layers
        self._nlayers = len(self._layers)
        if callable(activations):
            activations = [activations for nn in range(self._nlayers)]
        if len(activations) != self._nlayers:
            raise ValueError("incorrect number of activations provided")
        self._activations = activations  # activation functions for each layer
        if optimizer is None:
            optimizer = LBFGSB()
        self._optimizer = optimizer

        if var_trans is None:
            self._var_trans = IdentityValuesTransform()
        else:
            self._var_trans = var_trans
        if values_trans is None:
            self._values_trans = IdentityValuesTransform()
        else:
            self._values_trans = values_trans

        self._hyp_list = sum([layer._hyp_list for layer in self._layers])

    def _forward(self, input_samples):
        if input_samples.shape[0] != self._nvars:
            raise ValueError("input_samples has the wrong shape")
        y_samples = copy(input_samples)
        for kk in range(self._nlayers):
            u_samples = self._layers[kk](y_samples)
            y_samples = self._activations[kk](u_samples)
        return y_samples

    def _loss(self, batches=1, batch_index=0):
        ntrain_samples = self._canonical_train_samples.shape[-1]
        batch_sizes = ones((batches+1,)) * int(ntrain_samples / batches)
        batch_sizes[0] = 0
        batch_sizes[1:(ntrain_samples % batches)] += 1
        batch_arr = cumsum(batch_sizes, dim=0)

        if batch_index == 0:    # shuffle at beginning of epoch
            shuffle = randperm(ntrain_samples)
            self._canonical_train_samples = (
                self._canonical_train_samples[..., shuffle])
            self._canonical_train_values = (
                self._canonical_train_values[..., shuffle])

        idx0 = int(batch_arr[batch_index].item())
        idx1 = int(batch_arr[batch_index+1].item())
        batch_approx_values = self._forward(
            self._canonical_train_samples[..., idx0:idx1])
        batch_canonical_values = self._canonical_train_values[..., idx0:idx1]
        return ((batch_approx_values-batch_canonical_values)**2).sum()/(
                ntrain_samples)

    def _fit_objective(self, active_opt_params_np, batches=1, batch_index=0):
        active_opt_params = asarray(
            active_opt_params_np, requires_grad=True)
        self._hyp_list.set_active_opt_params(active_opt_params)
        nll = self._loss(batches=batches, batch_index=batch_index)
        nll.backward()
        val = nll.item()
        # copy is needed because zero_ is called
        nll_grad = active_opt_params.grad.detach().numpy().copy()
        active_opt_params.grad.zero_()
        # must set requires grad to False after gradient is computed
        # otherwise when evaluate_posterior will fail because it will
        # still think the hyper_params require grad. Extra copies could be
        # avoided by doing this after fit is complete. However then fit
        # needs to know when torch is being used
        for hyp in self._hyp_list.hyper_params:
            hyp.detach()
        return val, nll_grad

    def _set_training_data(self, train_samples: array, train_values: array):
        if train_samples.shape[0] != self._nvars:
            raise ValueError("train_samples has the wrong shape {0}".format(
                train_samples.shape))
        if train_samples.shape[-1] != train_values.shape[-1]:
            raise ValueError("train_values has the wrong shape {0}".format(
                train_values.shape))

        self.train_samples = train_samples
        self.train_values = train_values
        self._canonical_train_samples = asarray(
            self._var_trans.map_to_canonical(train_samples))
        self._canonical_train_values = asarray(
            self._values_trans.map_to_canonical(train_values))

    def fit(self, train_samples: array, train_values: array, verbosity=0,
            tol=1e-5):
        self._set_training_data(train_samples, train_values)
        self._optimizer.set_objective_function(self._fit_objective)
        self._optimizer.set_bounds(self._hyp_list.get_active_opt_bounds())
        self._optimizer.set_verbosity(verbosity)
        self._optimizer.set_tolerance(tol)
        res = self._optimizer.optimize(self._hyp_list.get_active_opt_params())
        self._hyp_list.set_active_opt_params(res.x)

    def save(self, filename):
        '''
        To load, use pyapprox.sciml.network.load(filename)
        '''
        pickle.dump(self, open(filename, 'wb'))

    def __call__(self, input_samples):
        return self._forward(asarray(input_samples))

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self._hyp_list._short_repr())


def load(filename):
    return pickle.load(open(filename, 'rb'))


def initialize_homogeneous_transform_NO(
        niop_layers, hidden_width, ninputs, noutputs, kmax,
        convolution_op=FourierConvolutionOperator,
        hidden_activation=TanhActivation, use_affine_block=True):
    """
    Initialize the layers of a FNO
    """
    iops = [
        convolution_op(kmax) for nn in range(niop_layers)]
    if not use_affine_block:
        layers = [Layer([iop]) for iop in iops]
    else:
        layers = [
            Layer([iops[nn], DenseAffineIntegralOperator(
                hidden_width, hidden_width)])
            for nn in range(niop_layers)]
    activations = [hidden_activation() for nn in range(niop_layers)]
    if hidden_width != ninputs:
        layers = (
            [DenseAffineIntegralOperatorFixedBias(ninputs, hidden_width)] +
            layers +
            [DenseAffineIntegralOperatorFixedBias(hidden_width, noutputs)])
        activations = (
            [IdentityActivation()]+activations+[IdentityActivation()])
    network = CERTANN(ninputs, layers, activations)
    return network

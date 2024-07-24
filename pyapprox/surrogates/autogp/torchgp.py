import torch

from pyapprox.surrogates.kernels._kernels import Kernel
from pyapprox.surrogates.autogp.trends import Monomial
from pyapprox.util.transforms import Transform
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess, MOExactGaussianProcess, MOPeerExactGaussianProcess,
    MOICMPeerExactGaussianProcess)
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.autogp.variationalgp import (
    InducingSamples, InducingGaussianProcess)


class TorchGPFitMixin:
    def _fit_objective(self, active_opt_params_np):
        # todo change to follow call and jacobian api used by new optimize
        # classes

        # this is only place where torch should be called explicitly
        # as we are using its functionality to compute the gradient of their
        # negative log likelihood. We could replace this with a grad
        # computed analytically
        active_opt_params = torch.as_tensor(
            active_opt_params_np, dtype=torch.double).requires_grad_(True)
        nll = self._neg_log_likelihood(active_opt_params)
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
        for hyp in self.hyp_list.hyper_params:
            hyp.detach()
        return val, nll_grad


class TorchExactGaussianProcess(
        TorchLinAlgMixin, TorchGPFitMixin, ExactGaussianProcess):
    # Mixins must be first if defining an abstractmethod
    # And init of all nonmixin classes must be called explicitly in this
    # classes __init__
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 mean: Monomial = None,
                 kernel_reg: float = 0):
        super().__init__(nvars, kernel, var_trans, values_trans,
                         mean, kernel_reg)


class TorchMOExactGaussianProcess(
        TorchLinAlgMixin, TorchGPFitMixin, MOExactGaussianProcess):
    # Mixins must be first if defining an abstractmethod
    # And init of all nonmixin classes must be called explicitly in this
    # classes __init__
    def __init__(self,
                 nvars: int,
                 kernel: Kernel = None,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 kernel_reg: float = 0):
        super().__init__(nvars, kernel, var_trans, values_trans,
                         None, kernel_reg)


class TorchMOPeerExactGaussianProcess(
        TorchLinAlgMixin, TorchGPFitMixin, MOPeerExactGaussianProcess):
    # Mixins must be first if defining an abstractmethod
    # And init of all nonmixin classes must be called explicitly in this
    # classes __init__
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 kernel_reg: float = 0):
        super().__init__(nvars, kernel, var_trans, values_trans,
                         None, kernel_reg)


class TorchMOICMPeerExactGaussianProcess(
        TorchLinAlgMixin, TorchGPFitMixin, MOICMPeerExactGaussianProcess):
    # Mixins must be first if defining an abstractmethod
    # And init of all nonmixin classes must be called explicitly in this
    # classes __init__
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 output_kernel: Kernel,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 kernel_reg: float = 0):
        super().__init__(nvars, kernel, output_kernel, var_trans, values_trans,
                         kernel_reg)


class TorchInducingSamples(InducingSamples, TorchLinAlgMixin):
    def __init__(self, nvars, ninducing_samples, inducing_variable=None,
                 inducing_samples=None, inducing_sample_bounds=None,
                 noise=None):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._IdentityHyperParameterTransform = (
            TorchIdentityHyperParameterTransform)
        self._LogHyperParameterTransform = (
            TorchLogHyperParameterTransform)
        super().__init__(nvars, ninducing_samples, inducing_variable,
                         inducing_samples, inducing_sample_bounds,
                         noise)


class TorchInducingGaussianProcess(
        TorchLinAlgMixin, TorchGPFitMixin, InducingGaussianProcess):
    def __init__(self, nvars,
                 kernel,
                 inducing_samples,
                 kernel_reg=0,
                 var_trans=None,
                 values_trans=None):
        super().__init__(nvars, kernel, inducing_samples,
                         var_trans, values_trans, kernel_reg)

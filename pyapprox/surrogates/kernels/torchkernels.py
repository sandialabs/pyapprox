import math

import torch

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.hyperparameter.torchhyperparameter import (
    TorchIdentityHyperParameterTransform, TorchLogHyperParameterTransform,
    TorchHyperParameter, TorchHyperParameterList)
from pyapprox.surrogates.kernels._kernels import (
    MaternKernel, ConstantKernel, GaussianNoiseKernel, PeriodicMaternKernel,
    SphericalCovariance, SphericalCovarianceHyperParameter)
from pyapprox.util.transforms.torchtransforms import (
    TorchSphericalCorrelationTransform)


class TorchAutogradMixin:
    def _autograd_fun(self, active_params_opt):
        active_params_opt.requires_grad = True
        self.hyp_list.set_active_opt_params(active_params_opt)
        return self(self._X)

    def jacobian(self, X):
        self._X = X
        return torch.autograd.functional.jacobian(
            self._autograd_fun, self.hyp_list.get_active_opt_params())


class TorchConstantKernel(
        ConstantKernel, TorchAutogradMixin, TorchLinAlgMixin):
    def __init__(self, constant, constant_bounds=None,
                 transform=TorchIdentityHyperParameterTransform()):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        super().__init__(constant, transform, constant_bounds)


class TorchGaussianNoiseKernel(
        GaussianNoiseKernel, TorchAutogradMixin, TorchLinAlgMixin):
    def __init__(self, constant, constant_bounds=None):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        super().__init__(
            constant, TorchLogHyperParameterTransform(), constant_bounds)


class TorchMaternKernel(MaternKernel, TorchAutogradMixin, TorchLinAlgMixin):
    def __init__(self, nu: float,
                 lenscale, lenscale_bounds, nvars: int):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        super().__init__(nu, lenscale, lenscale_bounds, nvars,
                         TorchLogHyperParameterTransform())


class TorchPeriodicMaternKernel(PeriodicMaternKernel, TorchLinAlgMixin):
    def __init__(self, nu: float, period, period_bounds,
                 lenscale, lenscale_bounds):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        super().__init__(
            nu, period, period_bounds, lenscale, lenscale_bounds,
            TorchLogHyperParameterTransform(),
            TorchLogHyperParameterTransform())


class TorchSphericalCovarianceHyperParameter(
        SphericalCovarianceHyperParameter, TorchLinAlgMixin):
    def __init__(self, hyper_params):
        self._SphericalCorrelationTransform = (
            TorchSphericalCorrelationTransform)
        self._IdentityHyperParameterTransform = (
            TorchIdentityHyperParameterTransform)
        super().__init__(hyper_params)


class TorchSphericalCovariance(SphericalCovariance, TorchLinAlgMixin):
    def __init__(self, noutputs,
                 radii=1, radii_bounds=[1e-1, 1],
                 angles=math.pi/2, angle_bounds=[0, math.pi],
                 radii_transform=TorchIdentityHyperParameterTransform(),
                 angle_transform=TorchIdentityHyperParameterTransform()):
        self._SphericalCorrelationTransform = (
            TorchSphericalCorrelationTransform)
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._SphericalCovarianceHyperParameter = (
            TorchSphericalCovarianceHyperParameter)
        self._IdentityHyperParameterTransform = (
            TorchIdentityHyperParameterTransform)
        super().__init__(noutputs, radii_transform, angle_transform,
                         radii, radii_bounds, angles, angle_bounds)

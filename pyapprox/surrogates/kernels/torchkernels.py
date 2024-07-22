import math

import torch

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
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
    pass


class TorchGaussianNoiseKernel(
        GaussianNoiseKernel, TorchAutogradMixin, TorchLinAlgMixin):
    pass


class TorchMaternKernel(MaternKernel, TorchAutogradMixin, TorchLinAlgMixin):
    pass


class TorchPeriodicMaternKernel(PeriodicMaternKernel, TorchLinAlgMixin):
    pass


class TorchSphericalCovarianceHyperParameter(
        SphericalCovarianceHyperParameter, TorchLinAlgMixin):
    def __init__(self, hyper_params):
        self._SphericalCorrelationTransform = (
            TorchSphericalCorrelationTransform)
        super().__init__(hyper_params)


class TorchSphericalCovariance(SphericalCovariance, TorchLinAlgMixin):
    def __init__(self, noutputs,
                 radii=1, radii_bounds=[1e-1, 1],
                 angles=math.pi/2, angle_bounds=[0, math.pi],
                 radii_transform=None,
                 angle_transform=None):
        self._SphericalCorrelationTransform = (
            TorchSphericalCorrelationTransform)
        super().__init__(noutputs, radii_transform, angle_transform,
                         radii, radii_bounds, angles, angle_bounds)

import math

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.kernels._kernels import (
    ConstantKernel, GaussianNoiseKernel, MaternKernel, PeriodicMaternKernel,
    SphericalCovariance, SphericalCovarianceHyperParameter)
from pyapprox.util.hyperparameter.numpyhyperparameter import (
    NumpyIdentityHyperParameterTransform, NumpyLogHyperParameterTransform,
    NumpyHyperParameter, NumpyHyperParameterList)
from pyapprox.util.transforms.numpytransforms import (
    NumpySphericalCorrelationTransform)


class NumpyConstantKernel(ConstantKernel, NumpyLinAlgMixin):
    def __init__(self, constant, constant_bounds=None,
                 transform=NumpyIdentityHyperParameterTransform()):
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        super().__init__(constant, transform, constant_bounds)


class NumpyGaussianNoiseKernel(GaussianNoiseKernel, NumpyLinAlgMixin):
    def __init__(self, constant, constant_bounds=None):
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        super().__init__(
            constant, NumpyLogHyperParameterTransform(), constant_bounds)


class NumpyMaternKernel(MaternKernel, NumpyLinAlgMixin):
    def __init__(self, nu: float,
                 lenscale, lenscale_bounds, nvars: int):
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        super().__init__(nu, lenscale, lenscale_bounds, nvars,
                         NumpyLogHyperParameterTransform())


class NumpyPeriodicMaternKernel(PeriodicMaternKernel, NumpyLinAlgMixin):
    def __init__(self, nu: float, period, period_bounds,
                 lenscale, lenscale_bounds):
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        super().__init__(
            nu, period, period_bounds, lenscale, lenscale_bounds,
            NumpyLogHyperParameterTransform(),
            NumpyLogHyperParameterTransform())


class NumpySphericalCovarianceHyperParameter(
        SphericalCovarianceHyperParameter, NumpyLinAlgMixin):
    def __init__(self, hyper_params):
        self._SphericalCorrelationTransform = (
            NumpySphericalCorrelationTransform)
        self._IdentityHyperParameterTransform = (
            NumpyIdentityHyperParameterTransform)
        super().__init__(hyper_params)


class NumpySphericalCovariance(SphericalCovariance, NumpyLinAlgMixin):
    def __init__(self, noutputs,
                 radii=1, radii_bounds=[1e-1, 1],
                 angles=math.pi/2, angle_bounds=[0, math.pi],
                 radii_transform=NumpyIdentityHyperParameterTransform(),
                 angle_transform=NumpyIdentityHyperParameterTransform()):
        self._SphericalCorrelationTransform = (
            NumpySphericalCorrelationTransform)
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        self._SphericalCovarianceHyperParameter = (
            NumpySphericalCovarianceHyperParameter)
        self._IdentityHyperParameterTransform = (
            NumpyIdentityHyperParameterTransform)
        super().__init__(noutputs, radii_transform, angle_transform,
                         radii, radii_bounds, angles, angle_bounds)

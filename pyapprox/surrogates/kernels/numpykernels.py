import math

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.kernels._kernels import (
    ConstantKernel, GaussianNoiseKernel, MaternKernel, PeriodicMaternKernel,
    SphericalCovariance, SphericalCovarianceHyperParameter)
from pyapprox.util.transforms.numpytransforms import (
    NumpySphericalCorrelationTransform)


class NumpyConstantKernel(ConstantKernel, NumpyLinAlgMixin):
    pass


class NumpyGaussianNoiseKernel(GaussianNoiseKernel, NumpyLinAlgMixin):
    pass


class NumpyMaternKernel(MaternKernel, NumpyLinAlgMixin):
    pass


class NumpyPeriodicMaternKernel(PeriodicMaternKernel, NumpyLinAlgMixin):
    pass


class NumpySphericalCovarianceHyperParameter(
        SphericalCovarianceHyperParameter, NumpyLinAlgMixin):
    def __init__(self, hyper_params):
        self._SphericalCorrelationTransform = (
            NumpySphericalCorrelationTransform)
        super().__init__(hyper_params)


class NumpySphericalCovariance(SphericalCovariance, NumpyLinAlgMixin):
    def __init__(self, noutputs,
                 radii=1, radii_bounds=[1e-1, 1],
                 angles=math.pi/2, angle_bounds=[0, math.pi],
                 radii_transform=None,
                 angle_transform=None):
        self._SphericalCorrelationTransform = (
            NumpySphericalCorrelationTransform)
        super().__init__(noutputs, radii_transform, angle_transform,
                         radii, radii_bounds, angles, angle_bounds)

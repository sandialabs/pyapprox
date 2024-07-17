from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.transforms._transforms import (
    IdentityTransform, StandardDeviationTransform,
    NSphereCoordinateTransform, SphericalCorrelationTransform)


NumpyIdentityTransform = IdentityTransform


class NumpyStandardDeviationTransform(
        StandardDeviationTransform, NumpyLinAlgMixin):
    pass


class NumpyNSphereCoordinateTransform(
        NSphereCoordinateTransform, NumpyLinAlgMixin):
    pass


class NumpySphericalCorrelationTransform(
        SphericalCorrelationTransform, NumpyLinAlgMixin):
    def __init__(self, noutputs):
        self._NSphereCoordinateTransform = NumpyNSphereCoordinateTransform
        super().__init__(noutputs)

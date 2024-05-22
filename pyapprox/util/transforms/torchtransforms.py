from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.transforms._transforms import (
    IdentityTransform, StandardDeviationTransform,
    NSphereCoordinateTransform, SphericalCorrelationTransform)


TorchIdentityTransform = IdentityTransform


class TorchStandardDeviationTransform(
        StandardDeviationTransform, TorchLinAlgMixin):
    pass


class TorchNSphereCoordinateTransform(
        NSphereCoordinateTransform, TorchLinAlgMixin):
    pass


class TorchSphericalCorrelationTransform(
        SphericalCorrelationTransform, TorchLinAlgMixin):
    def __init__(self, noutputs):
        self._NSphereCoordinateTransform = TorchNSphereCoordinateTransform
        super().__init__(noutputs)

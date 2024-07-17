from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.hyperparameter._hyperparameter import (
    IdentityHyperParameterTransform, LogHyperParameterTransform,
    HyperParameter, HyperParameterList)


class TorchIdentityHyperParameterTransform(
        IdentityHyperParameterTransform, TorchLinAlgMixin):
    pass


class TorchLogHyperParameterTransform(
        LogHyperParameterTransform, TorchLinAlgMixin):
    pass


class TorchHyperParameter(HyperParameter, TorchLinAlgMixin):
    pass


class TorchHyperParameterList(HyperParameterList, TorchLinAlgMixin):
    pass

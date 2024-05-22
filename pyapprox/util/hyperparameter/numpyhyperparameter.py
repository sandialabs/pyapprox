from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.hyperparameter._hyperparameter import (
    IdentityHyperParameterTransform, LogHyperParameterTransform,
    HyperParameter, HyperParameterList)


class NumpyIdentityHyperParameterTransform(
        IdentityHyperParameterTransform, NumpyLinAlgMixin):
    pass


class NumpyLogHyperParameterTransform(
        LogHyperParameterTransform, NumpyLinAlgMixin):
    pass


class NumpyHyperParameter(HyperParameter, NumpyLinAlgMixin):
    pass


class NumpyHyperParameterList(HyperParameterList, NumpyLinAlgMixin):
    pass

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.hyperparameter.numpyhyperparameter import (
    NumpyHyperParameter, NumpyHyperParameterList,
    NumpyIdentityHyperParameterTransform)
from pyapprox.surrogates.autogp.trends import Monomial


class NumpyMonomial(Monomial, NumpyLinAlgMixin):
    def __init__(self, nvars, degree, coefs, coef_bounds,
                 name="MonomialCoefficients"):
        self._HyperParameter = NumpyHyperParameter
        self._HyperParameterList = NumpyHyperParameterList
        transform = NumpyIdentityHyperParameterTransform()
        super().__init__(nvars, degree, coefs, coef_bounds, transform, name)

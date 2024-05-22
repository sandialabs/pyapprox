from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.hyperparameter.torchhyperparameter import (
    TorchHyperParameter, TorchHyperParameterList,
    TorchIdentityHyperParameterTransform)
from pyapprox.surrogates.autogp.trends import Monomial


class TorchMonomial(Monomial, TorchLinAlgMixin):
    def __init__(self, nvars, degree, coefs, coef_bounds,
                 name="MonomialCoefficients"):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        transform = TorchIdentityHyperParameterTransform()
        super().__init__(nvars, degree, coefs, coef_bounds, transform, name)

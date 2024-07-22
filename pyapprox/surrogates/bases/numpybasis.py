from pyapprox.surrogates.bases._basis import MonomialBasis
from pyapprox.surrogates.bases._basisexp import BasisExpansion
from pyapprox.surrogates.bases._linearsystemsolvers import (
    LstSqSolver, OMPSolver)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.hyperparameter.numpyhyperparameter import (
    NumpyHyperParameter, NumpyHyperParameterList,
    NumpyIdentityHyperParameterTransform)


class NumpyMonomialBasis(MonomialBasis, NumpyLinAlgMixin):
    pass


class NumpyLstSqSolver(LstSqSolver, NumpyLinAlgMixin):
    pass


class NumpyOMPSolver(OMPSolver, NumpyLinAlgMixin):
    pass


class NumpyBasisExpansion(BasisExpansion, NumpyLinAlgMixin):
    def __init__(self, basis, solver=NumpyLstSqSolver(), nqoi=1,
        self._transform = NumpyIdentityHyperParameterTransform()
        super().__init__(basis, solver, nqoi, coef_bounds)

from abc import ABC, abstractmethod
import itertools

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.utilities import get_random_k_fold_sample_indices
from pyapprox.surrogates.bases.basisexp import Regressor, BasisExpansion
from pyapprox.surrogates.bases.linearsystemsolvers import OMPSolver


class StructureParameterIterator(ABC):
    """Iterate over a models structural parameters, e.g. polynomial degree."""

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    @abstractmethod
    def set_structure_params(
        self, regressor: Regressor, structure_params: list
    ):
        raise NotImplementedError


class PolynomialDegreeIterator(StructureParameterIterator):
    def __init__(self, degrees: list, pnorms: list):
        self._degrees = degrees
        self._pnorms = pnorms
        self._param_iter = itertools.product(*[self._degrees, self._pnorms])

    def __next__(self):
        return next(self._param_iter)

    def set_structure_params(
        self, regressor: BasisExpansion, structure_params: list
    ):
        degree = structure_params[0]
        pnorm = structure_params[1]
        indices = regressor.basis.set_hyperbolic_indices(
            regressor.nvars(), degree, pnorm
        )
        # need to call regressor.set_basis to make sure hyperparameter list is updated
        # correctly
        regressor.set_basis(regressor.basis)


class OMPNTermsIterator(StructureParameterIterator):
    def __init__(self, nterms: list):
        self._nterms = nterms
        self._param_iter = iter(self._nterms)

    def __next__(self):
        return next(self._param_iter)

    def set_structure_params(
        self, regressor: BasisExpansion, structure_params: list
    ):
        if not isinstance(regressor._solver, OMPSolver):
            raise ValueError("solver must be type OMPSolver.")
        regressor._solver.set_max_nonzeros(structure_params)


class GridSearchStructureParameterIterator(StructureParameterIterator):
    def __init__(self, iterators: list[StructureParameterIterator]):
        self._iterators = iterators
        self._product_iterator = itertools.product(*self._iterators)

    def __next__(self):
        return next(self._product_iterator)

    def set_structure_params(
        self, regressor: Regressor, structure_params: list
    ):
        if len(structure_params) != len(self._iterators):
            raise ValueError(
                "len(structure_params) {0} != len(self._iterators {1}".format(
                    len(structure_params), len(self._iterators)
                )
            )
        for ii, it in enumerate(self._iterators):
            it.set_structure_params(regressor, structure_params[ii])


class CrossValidation(ABC):
    def __init__(self, train_samples, train_values, regressor):
        self.regressor = regressor
        self._bkd = regressor._bkd
        self._train_samples = train_samples
        self._train_values = train_values
        self._ntrain_samples = self._train_samples.shape[1]
        if train_values.shape[1] > 1:
            raise NotImplementedError("nQoI > 1 is not supported yet.")

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0})".format(self.__class__.__name__)


class KFoldCrossValidation(CrossValidation):
    def __init__(
        self,
        train_samples,
        train_values,
        regressor,
        nfolds=5,
        random_folds=True,
    ):
        super().__init__(train_samples, train_values, regressor)
        self._nfolds = nfolds
        self._fold_sample_indices = get_random_k_fold_sample_indices(
            self._ntrain_samples, self._nfolds, random_folds, bkd=self._bkd
        )

    def run(self):
        sum_sq_residuals = 0
        fold_residuals = []
        for indices in self._fold_sample_indices:
            self.regressor.fit(
                self._train_samples[:, indices], self._train_values[indices]
            )
            test_samples = self._bkd._la_delete(
                self._train_samples, indices, axis=1
            )
            test_values = self._bkd._la_delete(
                self._train_values, indices, axis=0
            )
            fold_residuals.append(self.regressor(test_samples) - test_values)
            sum_sq_residuals += self._bkd._la_sum(
                fold_residuals[-1] ** 2, axis=0
            )
        cv_score = self._bkd._la_sqrt(
            sum_sq_residuals / self._ntrain_samples    
        )
        return cv_score

    def __repr__(self):
        return "{0}(K={1}, nsamples={2})".format(
            self.__class__.__name__, self._nfolds, self._ntrain_samples
        )


class CrossValidationStructureSearch:
    def __init__(
        self,
        cross_validator: CrossValidation,
        structure_iterator: StructureParameterIterator,
    ):
        self._cross_validator = cross_validator
        self._structure_iterator = structure_iterator
        self._bkd = self._cross_validator.regressor._bkd

    def run(self):
        results = []
        for structure_params in self._structure_iterator:
            self._structure_iterator.set_structure_params(
                self._cross_validator.regressor, structure_params
            )
            results.append((self._cross_validator.run(), structure_params))
        best_idx = self._bkd._la_argmin(
            self._bkd._la_asarray([result[0] for result in results])
        )
        return results[best_idx][1], results, best_idx

    def __repr__(self):
        return "{0}(cv={1}, search={2})".format(
            self.__class__.__name__,
            self._cross_validator,
            self._structure_iterator,
        )

from abc import ABC, abstractmethod
import itertools
from typing import Tuple


from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.surrogates.regressor import Regressor
from pyapprox.surrogates.affine.basisexp import BasisExpansion
from pyapprox.surrogates.affine.linearsystemsolvers import OMPSolver


def get_random_k_fold_sample_indices(
    nsamples: int,
    nfolds: int,
    random: bool = True,
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    sample_indices = bkd.arange(nsamples, dtype=int)
    if random is True:
        sample_indices = bkd.asarray(
            np.random.permutation(sample_indices), dtype=int
        )

    fold_sample_indices = [bkd.empty(0, dtype=int) for kk in range(nfolds)]
    nn = 0
    while nn < nsamples:
        for jj in range(nfolds):
            fold_sample_indices[jj] = bkd.hstack(
                [fold_sample_indices[jj], sample_indices[nn]]
            )
            nn += 1
            if nn >= nsamples:
                break
    if bkd.unique(bkd.hstack(fold_sample_indices)).shape[0] != nsamples:
        raise RuntimeError()
    return fold_sample_indices


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
        regressor.basis().set_hyperbolic_indices(degree, pnorm)
        # need to call regressor.set_basis to make sure hyperparameter
        # list is updated correctly
        regressor.set_basis(regressor.basis())


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
            test_samples = self._bkd.delete(
                self._train_samples, indices, axis=1
            )
            test_values = self._bkd.delete(self._train_values, indices, axis=0)
            fold_residuals.append(self.regressor(test_samples) - test_values)
            sum_sq_residuals += self._bkd.sum(fold_residuals[-1] ** 2, axis=0)
        cv_score = self._bkd.sqrt(sum_sq_residuals / self._ntrain_samples)
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
        best_idx = self._bkd.argmin(
            self._bkd.asarray([result[0] for result in results])
        )
        return results[best_idx][1], results, best_idx

    def __repr__(self):
        return "{0}(cv={1}, search={2})".format(
            self.__class__.__name__,
            self._cross_validator,
            self._structure_iterator,
        )


def get_cross_validation_rsquared_coefficient_of_variation(
    cv_score: float, train_vals: Array, bkd: BackendMixin = NumpyMixin
) -> float:
    r"""
    cv_score = :math:`N^{-1/2}\left(\sum_{n=1}^N e_n\right^{1/2}` where
    :math:`e_n` are the cross  validation residues at each test point and
    :math:`N` is the number of traing vals

    We define r_sq as

    .. math:: 1-\frac{N^{-1}\left(\sum_{n=1}^N e_n\right)}/mathbb{V}\left[
              Y\right] where Y is the vector of training vals
    """
    # total sum of squares (proportional to variance)
    denom = bkd.std(train_vals)
    # the factors of 1/N in numerator and denominator cancel out
    rsq = 1 - (cv_score / denom) ** 2
    return rsq


def leave_one_out_lsq_cross_validation(
    basis_mat: Array,
    values: Array,
    alpha: float = 0,
    coef: Array = None,
    bkd: BackendMixin = NumpyMixin,
) -> Tuple[Array, float, Array]:
    """
    let :math:`x_i` be the ith row of :math:`X` and let
    :math:`\beta=(X^\top X)^{-1}X^\top y` such that the residuals
    at the training samples satisfy

    .. math:: r_i = X\beta-y

    then the leave one out cross validation errors are given by

    .. math:: e_i = \frac{r_i}{1-h_i}

    where

    :math:`h_i = x_i^\top(X^\top X)^{-1}x_i`
    """
    assert values.ndim == 2
    assert basis_mat.shape[0] > basis_mat.shape[1] + 2
    gram_mat = basis_mat.T @ basis_mat
    gram_mat += alpha * bkd.eye(gram_mat.shape[0])
    H_mat = basis_mat @ (bkd.inv(gram_mat) @ basis_mat.T)
    H_diag = bkd.diag(H_mat)
    if coef is None:
        coef = bkd.lstsq(gram_mat, basis_mat.T @ values)
    assert coef.ndim == 2
    residuals = basis_mat @ coef - values
    cv_errors = residuals / (1 - H_diag[:, None])
    cv_score = bkd.sqrt(bkd.sum(cv_errors**2, axis=0) / basis_mat.shape[0])
    return cv_errors, cv_score, coef


def leave_many_out_lsq_cross_validation(
    basis_mat: Array,
    values: Array,
    fold_sample_indices: Array,
    alpha: float = 0,
    coef: Array = None,
    bkd: BackendMixin = NumpyMixin,
) -> Tuple[Array, float, Array]:
    nfolds = len(fold_sample_indices)
    nsamples = basis_mat.shape[0]
    cv_errors = []
    cv_score = 0
    gram_mat = basis_mat.T @ basis_mat
    gram_mat += alpha * bkd.eye(gram_mat.shape[0])
    if coef is None:
        coef = bkd.lstsq(gram_mat, basis_mat.T @ values)
    residuals = basis_mat @ coef - values
    gram_mat_inv = bkd.inv(gram_mat)
    for kk in range(nfolds):
        indices_kk = fold_sample_indices[kk]
        nvalidation_samples_kk = indices_kk.shape[0]
        assert nsamples - nvalidation_samples_kk >= basis_mat.shape[1]
        basis_mat_kk = basis_mat[indices_kk, :]
        residuals_kk = residuals[indices_kk, :]

        H_mat = bkd.eye(nvalidation_samples_kk) - basis_mat_kk @ (
            gram_mat_inv @ basis_mat_kk.T
        )
        H_mat_inv = bkd.inv(H_mat)
        cv_errors.append(H_mat_inv @ residuals_kk)
        cv_score += bkd.sum(cv_errors[-1] ** 2, axis=0)
    return cv_errors, bkd.sqrt(cv_score[0] / basis_mat.shape[0]), coef

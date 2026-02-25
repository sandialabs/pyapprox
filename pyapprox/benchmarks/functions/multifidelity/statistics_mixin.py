"""Mixin for computing multifidelity statistics via numerical quadrature.

This module provides a mixin class that adds methods for computing
covariance matrices and higher-order statistics needed for multi-fidelity
estimation. The implementations use numerical quadrature and can be
overridden with analytical formulas for efficiency.
"""

from functools import partial
from typing import Callable, Generic, List, Sequence, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)


class MultifidelityStatisticsMixin(Generic[Array]):
    """Mixin providing numerical quadrature implementations for MF statistics.

    This mixin provides methods for computing covariance matrices and
    higher-order statistics needed for multi-fidelity estimation, using
    numerical quadrature. Subclasses can override these methods with
    analytical implementations for efficiency.

    Required attributes/methods on the class using this mixin:
    - _bkd: Backend[Array] - Backend for array operations
    - _models: Sequence of model functions with __call__(samples) -> Array
    - _nmodels: int - Number of models
    - _nqoi: int - Number of QoI per model

    The models should accept samples of shape (nvars, nsamples) and return
    values of shape (nqoi, nsamples).
    """

    # Declare attributes for type checking (set by subclass)
    _bkd: Backend[Array]
    _models: Sequence[Callable[[Array], Array]]
    _nmodels: int
    _nqoi: int

    def _get_quadrature_rule(
        self, npts: int = 21
    ) -> Tuple[Array, Array]:
        """Get Gauss quadrature points and weights on [0, 1].

        Parameters
        ----------
        npts : int
            Number of quadrature points.

        Returns
        -------
        quadx : Array
            Quadrature points of shape (1, npts).
        quadw : Array
            Quadrature weights of shape (npts,).
        """
        # Create uniform marginal on [0, 1]
        marginal = UniformMarginal(0.0, 1.0, self._bkd)

        # Create Gauss basis factory from marginal
        factories = create_basis_factories([marginal], self._bkd, "gauss")
        basis = factories[0].create_basis()
        basis.set_nterms(npts)

        # Get quadrature points and weights in physical domain [0, 1]
        # points: (1, npts), weights: (npts, 1)
        quadx, weights = basis.quadrature_rule()

        # Flatten weights to (npts,)
        quadw = weights[:, 0]

        return quadx, quadw

    def _flat_fun_wrapper(self, model_idx: int, qoi_idx: int, xx: Array) -> Array:
        """Evaluate a specific QoI of a specific model.

        Parameters
        ----------
        model_idx : int
            Index of the model.
        qoi_idx : int
            Index of the QoI within the model.
        xx : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (nsamples,).
        """
        return self._models[model_idx](xx)[qoi_idx, :]

    def _get_flat_funs(self) -> List[Callable[[Array], Array]]:
        """Get list of flattened functions (one per model-QoI pair)."""
        flat_funs: List[Callable[[Array], Array]] = []
        for ii in range(self._nmodels):
            for jj in range(self._nqoi):
                flat_funs.append(partial(self._flat_fun_wrapper, ii, jj))
        return flat_funs

    def _flat_covs(self) -> List[List[float]]:
        """Extract covariance matrices between QoI of same model.

        Returns list of lists where flat_covs[model][k*nqoi + l] is
        Cov(f_model_k, f_model_l).
        """
        cov = self.covariance_matrix()
        flat_covs: List[List[float]] = []
        for ii in range(self._nmodels):
            flat_covs.append([])
            for jj in range(self._nqoi):
                for kk in range(self._nqoi):
                    flat_covs[ii].append(
                        float(cov[ii * self._nqoi + jj, ii * self._nqoi + kk])
                    )
        return flat_covs

    def _V_fun_entry(
        self,
        jj: int, kk: int, ll: int,
        means: Array,
        flat_covs: List[List[float]],
        flat_funs: List[Callable[[Array], Array]],
        xx: Array
    ) -> Array:
        """Compute V entry for variance covariance.

        V_jkl(x) = (f_jk(x) - E[f_jk]) * (f_jl(x) - E[f_jl]) - Cov(f_jk, f_jl)
        """
        idx1 = jj * self._nqoi + kk
        idx2 = jj * self._nqoi + ll
        return (
            (flat_funs[idx1](xx) - means[idx1])
            * (flat_funs[idx2](xx) - means[idx2])
            - flat_covs[jj][kk * self._nqoi + ll]
        )

    def _V_fun(
        self,
        jj1: int, kk1: int, ll1: int,
        jj2: int, kk2: int, ll2: int,
        means: Array,
        flat_covs: List[List[float]],
        flat_funs: List[Callable[[Array], Array]],
        xx: Array
    ) -> Array:
        """Compute product of two V entries for Kronecker product covariance."""
        return (
            self._V_fun_entry(jj1, kk1, ll1, means, flat_covs, flat_funs, xx)
            * self._V_fun_entry(jj2, kk2, ll2, means, flat_covs, flat_funs, xx)
        )

    def _B_fun(
        self,
        ii: int, jj: int, kk: int, ll: int,
        means: Array,
        flat_covs: List[List[float]],
        flat_funs: List[Callable[[Array], Array]],
        xx: Array
    ) -> Array:
        """Compute covariance between mean and variance estimator."""
        return (
            (flat_funs[ii](xx) - means[ii])
            * self._V_fun_entry(jj, kk, ll, means, flat_covs, flat_funs, xx)
        )

    def means(self) -> Array:
        """Compute means of all models via numerical quadrature.

        Returns
        -------
        Array
            Means of shape (nmodels, nqoi).
        """
        quadx, quadw = self._get_quadrature_rule()
        means_list = []
        for model in self._models:
            vals = model(quadx)  # (nqoi, npts)
            model_means = []
            for k in range(self._nqoi):
                model_means.append(self._bkd.dot(vals[k, :], quadw))
            means_list.append(self._bkd.hstack(model_means))
        return self._bkd.vstack(means_list)

    def covariance_matrix(self) -> Array:
        """Compute full covariance matrix via numerical quadrature.

        Returns
        -------
        Array
            Covariance matrix of shape (nmodels*nqoi, nmodels*nqoi).
        """
        quadx, quadw = self._get_quadrature_rule(npts=50)

        # Compute means
        nflat = self._nmodels * self._nqoi
        means = []
        for model in self._models:
            vals = model(quadx)  # (nqoi, npts)
            for k in range(self._nqoi):
                means.append(self._bkd.dot(vals[k, :], quadw))

        # Compute covariance
        cov = self._bkd.zeros((nflat, nflat))
        flat_funs = self._get_flat_funs()
        for i in range(nflat):
            fi_vals = flat_funs[i](quadx)
            for j in range(nflat):
                fj_vals = flat_funs[j](quadx)
                cov[i, j] = self._bkd.dot(
                    (fi_vals - means[i]) * (fj_vals - means[j]), quadw
                )

        return cov

    def covariance_of_centered_values_kronecker_product(self) -> Array:
        """Compute covariance for Kronecker product of centered values.

        Computes Cov[(f_i - E[f_i])^{otimes 2}, (f_j - E[f_j])^{otimes 2}]
        via numerical quadrature.

        Returns
        -------
        Array
            Covariance of shape (nmodels*nqoi^2, nmodels*nqoi^2).
        """
        quadx, quadw = self._get_quadrature_rule()
        flat_funs = self._get_flat_funs()
        means = self._bkd.array([
            self._bkd.sum(f(quadx) * quadw) for f in flat_funs
        ])
        flat_covs = self._flat_covs()

        n = self._nmodels * self._nqoi ** 2
        est_cov = self._bkd.zeros((n, n))

        cnt1 = 0
        for jj1 in range(self._nmodels):
            for kk1 in range(self._nqoi):
                for ll1 in range(self._nqoi):
                    cnt2 = 0
                    for jj2 in range(self._nmodels):
                        for kk2 in range(self._nqoi):
                            for ll2 in range(self._nqoi):
                                quad_val = self._V_fun(
                                    jj1, kk1, ll1,
                                    jj2, kk2, ll2,
                                    means, flat_covs, flat_funs, quadx
                                )
                                est_cov[cnt1, cnt2] = self._bkd.sum(
                                    quad_val * quadw
                                )
                                cnt2 += 1
                    cnt1 += 1

        return est_cov

    def covariance_of_mean_and_variance_estimators(self) -> Array:
        """Compute covariance between mean and variance estimators.

        Computes Cov[f_i, (f_j - E[f_j])^{otimes 2}] via numerical quadrature.

        Returns
        -------
        Array
            Covariance of shape (nmodels*nqoi, nmodels*nqoi^2).
        """
        quadx, quadw = self._get_quadrature_rule()
        flat_funs = self._get_flat_funs()
        means = self._bkd.array([
            self._bkd.sum(f(quadx) * quadw) for f in flat_funs
        ])
        flat_covs = self._flat_covs()

        n_mean = self._nmodels * self._nqoi
        n_var = self._nmodels * self._nqoi ** 2
        est_cov = self._bkd.zeros((n_mean, n_var))

        for ii in range(n_mean):
            cnt = 0
            for jj in range(self._nmodels):
                for kk in range(self._nqoi):
                    for ll in range(self._nqoi):
                        quad_val = self._B_fun(
                            ii, jj, kk, ll,
                            means, flat_covs, flat_funs, quadx
                        )
                        est_cov[ii, cnt] = self._bkd.sum(quad_val * quadw)
                        cnt += 1

        return est_cov

    def covariance_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> Array:
        """Extract covariance submatrix for subset of models and QoI.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        Array
            Submatrix of covariance.
        """
        cov = self.covariance_matrix()
        nsub_models = len(model_idx)
        nsub_qoi = len(qoi_idx)
        n = nsub_models * nsub_qoi

        cov_new = self._bkd.zeros((n, n))
        cnt1 = 0
        for jj1 in model_idx:
            for kk1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1 * self._nqoi + kk1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        idx2 = jj2 * self._nqoi + kk2
                        cov_new[cnt1, cnt2] = cov[idx1, idx2]
                        cnt2 += 1
                cnt1 += 1

        return cov_new

    def covariance_of_centered_values_kronecker_product_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> Array:
        """Get subproblem covariance for Kronecker product.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        Array
            Submatrix of shape (len(model_idx)*len(qoi_idx)^2,
                                len(model_idx)*len(qoi_idx)^2).
        """
        W = self.covariance_of_centered_values_kronecker_product()
        nsub_models = len(model_idx)
        nsub_qoi = len(qoi_idx)
        n = nsub_models * nsub_qoi ** 2

        W_new = self._bkd.zeros((n, n))
        cnt1 = 0
        for jj1 in model_idx:
            for kk1 in qoi_idx:
                for ll1 in qoi_idx:
                    cnt2 = 0
                    idx1 = jj1 * self._nqoi**2 + kk1 * self._nqoi + ll1
                    for jj2 in model_idx:
                        for kk2 in qoi_idx:
                            for ll2 in qoi_idx:
                                idx2 = (
                                    jj2 * self._nqoi**2
                                    + kk2 * self._nqoi
                                    + ll2
                                )
                                W_new[cnt1, cnt2] = W[idx1, idx2]
                                cnt2 += 1
                    cnt1 += 1

        return W_new

    def covariance_of_mean_and_variance_estimators_subproblem(
        self, model_idx: List[int], qoi_idx: List[int]
    ) -> Array:
        """Get subproblem covariance between mean and variance estimators.

        Parameters
        ----------
        model_idx : List[int]
            Indices of models to include.
        qoi_idx : List[int]
            Indices of QoI to include.

        Returns
        -------
        Array
            Submatrix of shape (len(model_idx)*len(qoi_idx),
                                len(model_idx)*len(qoi_idx)^2).
        """
        B = self.covariance_of_mean_and_variance_estimators()
        nsub_models = len(model_idx)
        nsub_qoi = len(qoi_idx)
        n_mean = nsub_models * nsub_qoi
        n_var = nsub_models * nsub_qoi ** 2

        B_new = self._bkd.zeros((n_mean, n_var))
        cnt1 = 0
        for jj1 in model_idx:
            for kk1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1 * self._nqoi + kk1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = (
                                jj2 * self._nqoi**2
                                + kk2 * self._nqoi
                                + ll2
                            )
                            B_new[cnt1, cnt2] = B[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1

        return B_new

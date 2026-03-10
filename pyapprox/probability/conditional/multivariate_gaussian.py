"""
Conditional multivariate Gaussian distributions.

Provides conditional multivariate Gaussians where the mean and covariance
are functions of a conditioning variable x.

Two parameterizations:
- ConditionalDenseCholGaussian: full Cholesky L(x), Σ = L L^T
- ConditionalLowRankCholGaussian: diagonal + low-rank, Σ = D² + V V^T
"""

import math
from typing import Any, Generic, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class ConditionalDenseCholGaussian(Generic[Array]):
    """Conditional multivariate Gaussian with full Cholesky covariance.

    q(z | x) = N(z; mu(x), L(x) L(x)^T)

    where L(x) is lower-triangular with positive diagonal
    (diagonal = exp(log_chol_diag_func(x))).

    Parameters
    ----------
    mean_func : callable
        Maps x to mean vector. nqoi = d (latent dimension).
        Must support __call__, nvars(), nqoi(), and optionally hyp_list().
    log_chol_diag_func : callable
        Maps x to log of Cholesky diagonal. nqoi = d.
    chol_offdiag_func : callable or None
        Maps x to off-diagonal Cholesky elements (row-major lower-triangular
        order). nqoi = d*(d-1)/2. None if d = 1.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        mean_func,
        log_chol_diag_func,
        chol_offdiag_func: Optional[Any],
        bkd: Backend[Array],
    ) -> None:
        self._mean_func = mean_func
        self._log_chol_diag_func = log_chol_diag_func
        self._chol_offdiag_func = chol_offdiag_func
        self._bkd = bkd

        d = mean_func.nqoi()
        self._d = d
        n_offdiag = d * (d - 1) // 2

        if log_chol_diag_func.nqoi() != d:
            raise ValueError(
                f"log_chol_diag_func must have nqoi={d}, "
                f"got {log_chol_diag_func.nqoi()}"
            )
        if d > 1:
            if chol_offdiag_func is None:
                raise ValueError("chol_offdiag_func required for d > 1")
            if chol_offdiag_func.nqoi() != n_offdiag:
                raise ValueError(
                    f"chol_offdiag_func must have nqoi={n_offdiag}, "
                    f"got {chol_offdiag_func.nqoi()}"
                )
        elif chol_offdiag_func is not None:
            raise ValueError("chol_offdiag_func must be None for d=1")

        if mean_func.nvars() != log_chol_diag_func.nvars():
            raise ValueError("mean_func and log_chol_diag_func must have same nvars")
        if chol_offdiag_func is not None:
            if mean_func.nvars() != chol_offdiag_func.nvars():
                raise ValueError("all parameter functions must have same nvars")

        # Precompute lower-triangular indices for off-diagonal elements
        # Row-major order: (1,0), (2,0), (2,1), (3,0), ...
        if d > 1:
            rows = []
            cols = []
            for i in range(d):
                for j in range(i):
                    rows.append(i)
                    cols.append(j)
            self._offdiag_rows = rows
            self._offdiag_cols = cols

        self._log_2pi = math.log(2.0 * math.pi)
        self._setup_methods()

    def _setup_methods(self) -> None:
        funcs = [self._mean_func, self._log_chol_diag_func]
        if self._chol_offdiag_func is not None:
            funcs.append(self._chol_offdiag_func)
        if all(hasattr(f, "hyp_list") for f in funcs):
            self._hyp_list = funcs[0].hyp_list()
            for f in funcs[1:]:
                self._hyp_list = self._hyp_list + f.hyp_list()
            self.hyp_list = self._get_hyp_list
            self.nparams = self._get_nparams

    def _get_hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _get_nparams(self) -> int:
        return self._hyp_list.nparams()

    def _sync_param_funcs(self) -> None:
        for func in [
            self._mean_func,
            self._log_chol_diag_func,
            self._chol_offdiag_func,
        ]:
            if func is not None and hasattr(func, "_sync_from_hyp_list"):
                func._sync_from_hyp_list()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._mean_func.nvars()

    def nqoi(self) -> int:
        return self._d

    def _build_chol_factor(self, x: Array) -> Array:
        """Build batch of lower-triangular Cholesky factors.

        Parameters
        ----------
        x : Array
            Shape (nvars, nsamples).

        Returns
        -------
        Array
            Shape (nsamples, d, d).
        """
        bkd = self._bkd
        d = self._d
        nsamples = x.shape[1]

        log_diag = self._log_chol_diag_func(x)  # (d, nsamples)
        diag_vals = bkd.exp(log_diag)  # (d, nsamples)

        if d == 1:
            # (nsamples, 1, 1)
            return bkd.reshape(diag_vals[0, :], (nsamples, 1, 1))

        offdiag = self._chol_offdiag_func(x)  # (n_offdiag, nsamples)

        # Build L row-by-row using stack (autograd-safe)
        rows = []
        offdiag_idx = 0
        for i in range(d):
            row_elements = []
            for j in range(d):
                if j < i:
                    row_elements.append(offdiag[offdiag_idx, :])
                    offdiag_idx += 1
                elif j == i:
                    row_elements.append(diag_vals[i, :])
                else:
                    row_elements.append(bkd.zeros((nsamples,)))
            # Stack elements to form row: (nsamples, d)
            rows.append(bkd.stack(row_elements, axis=1))
        # Stack rows: (nsamples, d, d)
        return bkd.stack(rows, axis=1)

    def _validate_inputs(self, x: Array, y: Array) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got {y.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(f"x first dim must be {self.nvars()}, got {x.shape[0]}")
        if y.shape[0] != self._d:
            raise ValueError(f"y first dim must be {self._d}, got {y.shape[0]}")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have same number of samples")

    def logpdf(self, x: Array, y: Array) -> Array:
        """Evaluate log probability density.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        y : Array
            Output variable. Shape: (d, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)
        self._sync_param_funcs()
        bkd = self._bkd
        d = self._d
        nsamples = x.shape[1]

        mean = self._mean_func(x)  # (d, nsamples)
        log_diag = self._log_chol_diag_func(x)  # (d, nsamples)
        L_batch = self._build_chol_factor(x)  # (nsamples, d, d)

        residuals = y - mean  # (d, nsamples)

        # Batch solve: L @ whitened = residuals
        # residuals.T -> (nsamples, d), add dim -> (nsamples, d, 1)
        res_3d = bkd.reshape(residuals.T, (nsamples, d, 1))
        whitened_3d = bkd.solve_triangular(
            L_batch, res_3d, lower=True
        )  # (nsamples, d, 1)
        whitened = whitened_3d[:, :, 0]  # (nsamples, d)

        sq_mahal = bkd.sum(whitened**2, axis=1)  # (nsamples,)
        log_det_L = bkd.sum(log_diag, axis=0)  # (nsamples,)

        logpdf = -0.5 * d * self._log_2pi - log_det_L - 0.5 * sq_mahal
        return bkd.reshape(logpdf, (1, nsamples))

    def rvs(self, x: Array) -> Array:
        """Generate random samples.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Random samples. Shape: (d, nsamples)
        """
        nsamples = x.shape[1]
        base = self._bkd.asarray(np.random.randn(self._d, nsamples).astype(np.float64))
        return self.reparameterize(x, base)

    def reparameterize(self, x: Array, base_samples: Array) -> Array:
        """Transform N(0,I) base samples to q(z|x).

        z = mu(x) + L(x) @ base_samples

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        base_samples : Array
            Standard normal samples. Shape: (d, nsamples)

        Returns
        -------
        Array
            Reparameterized samples. Shape: (d, nsamples)
        """
        self._sync_param_funcs()
        bkd = self._bkd

        mean = self._mean_func(x)  # (d, nsamples)
        L_batch = self._build_chol_factor(x)  # (nsamples, d, d)

        # Batch L @ eps via einsum: (nsamples,d,d) x (d,nsamples) -> (d,nsamples)
        # eps.T -> (nsamples, d); einsum 'nij,nj->ni' -> (nsamples, d); .T -> (d,
        # nsamples)
        eps_T = base_samples.T  # (nsamples, d)
        Leps = bkd.einsum("nij,nj->ni", L_batch, eps_T)  # (nsamples, d)

        return mean + Leps.T  # (d, nsamples)

    def kl_divergence(self, x: Array, prior) -> Array:
        """Compute KL(q(.|x) || prior) per sample.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        prior : DenseCholeskyMultivariateGaussian
            Fixed multivariate Gaussian prior with mean(), covariance_inverse(),
            and covariance_operator().log_determinant().

        Returns
        -------
        Array
            Per-sample KL divergence. Shape: (1, nsamples)
        """
        self._sync_param_funcs()
        bkd = self._bkd
        d = self._d
        nsamples = x.shape[1]

        mean_q = self._mean_func(x)  # (d, nsamples)
        log_diag = self._log_chol_diag_func(x)  # (d, nsamples)
        L_batch = self._build_chol_factor(x)  # (nsamples, d, d)

        mean_p = prior.mean()  # (d, 1)
        cov_p_inv = prior.covariance_inverse()  # (d, d)
        log_det_L_p = prior.covariance_operator().log_determinant()

        # Trace term: tr(Sigma_p^{-1} L_q L_q^T) per sample
        # = sum_{ij} (Sigma_p^{-1} @ L_q)_{ij} * (L_q)_{ij}
        # Sigma_p_inv @ L_batch: einsum 'ij,njk->nik'
        SinvL = bkd.einsum("ij,njk->nik", cov_p_inv, L_batch)  # (N,d,d)
        trace_term = bkd.einsum("nij,nij->n", SinvL, L_batch)  # (N,)

        # Quadratic term: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
        mean_diff = mean_p - mean_q  # (d, nsamples)
        Sinv_diff = bkd.dot(cov_p_inv, mean_diff)  # (d, nsamples)
        quad_term = bkd.sum(mean_diff * Sinv_diff, axis=0)  # (nsamples,)

        # Log determinant terms
        log_det_q = 2.0 * bkd.sum(log_diag, axis=0)  # (nsamples,)
        log_det_p = 2.0 * log_det_L_p

        kl = 0.5 * (trace_term + quad_term - d + log_det_p - log_det_q)
        return bkd.reshape(kl, (1, nsamples))

    def covariance(self, x: Array) -> Array:
        """Compute covariance matrices at conditioning points.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Covariance matrices. Shape: (nsamples, d, d)
        """
        self._sync_param_funcs()
        L_batch = self._build_chol_factor(x)  # (nsamples, d, d)
        return self._bkd.einsum("nij,nkj->nik", L_batch, L_batch)

    def base_distribution(self):
        """Return the base distribution for reparameterization: N(0, I_d)."""
        from pyapprox.probability.gaussian.dense import (
            DenseCholeskyMultivariateGaussian,
        )

        return DenseCholeskyMultivariateGaussian(
            self._bkd.zeros((self._d, 1)),
            self._bkd.eye(self._d),
            self._bkd,
        )

    def __repr__(self) -> str:
        return f"ConditionalDenseCholGaussian(nvars={self.nvars()}, nqoi={self.nqoi()})"


class ConditionalLowRankCholGaussian(Generic[Array]):
    """Conditional multivariate Gaussian with low-rank + diagonal covariance.

    q(z | x) = N(z; mu(x), D(x)^2 + V(x) V(x)^T)

    where D(x) = diag(exp(log_diag_func(x))) and V(x) is (d, rank).

    Reparameterization computes L = cholesky(D² + VV^T) and uses
    z = mu + L @ eps with eps ~ N(0, I_d).

    When rank = d, can represent any covariance (combined with D).

    Note: The Cholesky factorization of D² + VV^T requires the matrix
    to be positive definite. If D values become very small (log_diag
    very negative) and V is near zero, the matrix may become numerically
    singular. Users should ensure the optimizer bounds prevent this,
    e.g. by setting a lower bound on log_diag parameters.

    Parameters
    ----------
    mean_func : callable
        Maps x to mean vector. nqoi = d.
    log_diag_func : callable
        Maps x to log of diagonal stdevs. nqoi = d.
    factor_func : callable or None
        Maps x to low-rank factor columns (flattened). nqoi = d * rank.
        None if rank = 0.
    rank : int
        Rank of the low-rank component.
    bkd : Backend[Array]
        Computational backend.
    """
    # TODO: why do we support rank=0 what does this even mean?
    # is this just diagonal? if so update docs to make this distinction clear

    def __init__(
        self,
        mean_func, #TODO: all args here and in other functions must have types. Wwe should use runtime_checkable protocols incuding for factor_func
        log_diag_func,
        factor_func: Optional[Any],
        rank: int,
        bkd: Backend[Array],
    ) -> None:
        self._mean_func = mean_func
        self._log_diag_func = log_diag_func
        self._factor_func = factor_func
        self._rank = rank
        self._bkd = bkd

        d = mean_func.nqoi()
        self._d = d

        if log_diag_func.nqoi() != d:
            raise ValueError(
                f"log_diag_func must have nqoi={d}, got {log_diag_func.nqoi()}"
            )
        if rank > 0:
            if factor_func is None:
                raise ValueError("factor_func required for rank > 0")
            if factor_func.nqoi() != d * rank:
                raise ValueError(
                    f"factor_func must have nqoi={d * rank}, got {factor_func.nqoi()}"
                )
        elif factor_func is not None:
            raise ValueError("factor_func must be None for rank=0")

        if mean_func.nvars() != log_diag_func.nvars():
            raise ValueError("mean_func and log_diag_func must have same nvars")
        if factor_func is not None:
            if mean_func.nvars() != factor_func.nvars():
                raise ValueError("all parameter functions must have same nvars")

        self._log_2pi = math.log(2.0 * math.pi)
        self._setup_methods()

    def _setup_methods(self) -> None:
        funcs = [self._mean_func, self._log_diag_func]
        if self._factor_func is not None:
            funcs.append(self._factor_func)
        if all(hasattr(f, "hyp_list") for f in funcs):
            self._hyp_list = funcs[0].hyp_list()
            for f in funcs[1:]:
                self._hyp_list = self._hyp_list + f.hyp_list()
            self.hyp_list = self._get_hyp_list
            self.nparams = self._get_nparams

    def _get_hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _get_nparams(self) -> int:
        return self._hyp_list.nparams()

    def _sync_param_funcs(self) -> None:
        for func in [
            self._mean_func,
            self._log_diag_func,
            self._factor_func,
        ]:
            if func is not None and hasattr(func, "_sync_from_hyp_list"):
                func._sync_from_hyp_list()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._mean_func.nvars()

    def nqoi(self) -> int:
        return self._d

    def rank(self) -> int:
        return self._rank

    def _build_D_V(self, x: Array):
        """Build diagonal and low-rank factor from parameter functions.

        Returns
        -------
        diag_vals : Array
            Shape (d, nsamples). Diagonal stdevs (exp of log_diag_func).
        V_batch : Array or None
            Shape (nsamples, d, rank). Low-rank factor. None if rank=0.
        """
        bkd = self._bkd
        d = self._d
        nsamples = x.shape[1]

        log_diag = self._log_diag_func(x)  # (d, nsamples)
        diag_vals = bkd.exp(log_diag)  # (d, nsamples)

        V_batch = None
        if self._rank > 0:
            # factor_func output: (d*rank, nsamples)
            flat_V = self._factor_func(x)  # (d*rank, nsamples)
            # Reshape to (nsamples, d, rank)
            # flat_V.T -> (nsamples, d*rank) -> (nsamples, d, rank)
            V_batch = bkd.reshape(flat_V.T, (nsamples, d, self._rank))

        return diag_vals, V_batch

    def _validate_inputs(self, x: Array, y: Array) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got {y.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(f"x first dim must be {self.nvars()}, got {x.shape[0]}")
        if y.shape[0] != self._d:
            raise ValueError(f"y first dim must be {self._d}, got {y.shape[0]}")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have same number of samples")

    def logpdf(self, x: Array, y: Array) -> Array:
        """Evaluate log probability density.

        Uses Woodbury identity for (D^2 + V V^T)^{-1}.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        y : Array
            Output variable. Shape: (d, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)
        self._sync_param_funcs()
        bkd = self._bkd
        d = self._d
        r = self._rank
        nsamples = x.shape[1]

        mean = self._mean_func(x)  # (d, nsamples)
        diag_vals, V_batch = self._build_D_V(x)
        log_diag = self._log_diag_func(x)  # (d, nsamples)
        D2 = diag_vals**2  # (d, nsamples)

        residuals = y - mean  # (d, nsamples)

        if r == 0:
            # Diagonal case: Sigma = D^2
            whitened_sq = residuals**2 / D2  # (d, nsamples)
            sq_mahal = bkd.sum(whitened_sq, axis=0)  # (nsamples,)
            log_det = 2.0 * bkd.sum(log_diag, axis=0)  # (nsamples,)
        else:
            # Woodbury: (D^2+VV^T)^{-1} = D^{-2} - D^{-2}V M^{-1} V^T D^{-2}
            # where M = I_r + V^T D^{-2} V, shape (N, r, r)
            Dinv2 = 1.0 / D2  # (d, nsamples)
            # D^{-2} V: (N, d, r)
            Dinv2_V = V_batch * bkd.reshape(Dinv2.T, (nsamples, d, 1))  # (N, d, r)
            # M = I_r + V^T D^{-2} V: batch (N, r, r)
            M_batch = bkd.reshape(bkd.eye(r), (1, r, r)) + bkd.einsum(
                "nji,njk->nik", V_batch, Dinv2_V
            )  # (N, r, r)
            L_M = bkd.cholesky(M_batch)  # (N, r, r)

            # Sigma^{-1} res via Woodbury
            Dinv2_res = residuals * Dinv2  # (d, nsamples)
            # V^T D^{-2} res: (N, r)
            VtDinv2_res = bkd.einsum("nji,jn->ni", V_batch, Dinv2_res)  # (N, r)
            # Solve M w = V^T D^{-2} res via Cholesky
            rhs = bkd.reshape(VtDinv2_res, (nsamples, r, 1))
            w = bkd.solve_triangular(L_M, rhs, lower=True)
            w = bkd.solve_triangular(
                bkd.moveaxis(L_M, -2, -1), w, lower=False
            )  # (N, r, 1)
            # D^{-2} V @ w: (N, d)
            Dinv2Vw = bkd.einsum("nij,nj->ni", Dinv2_V, w[:, :, 0])  # (N, d)
            Sinv_res = Dinv2_res - Dinv2Vw.T  # (d, nsamples)
            sq_mahal = bkd.sum(residuals * Sinv_res, axis=0)  # (nsamples,)

            # log det via matrix det lemma: log|D^2+VV^T| = log|M| + log|D^2|
            _, log_det_M = bkd.slogdet(M_batch)  # (N,)
            log_det = log_det_M + 2.0 * bkd.sum(log_diag, axis=0)

        logpdf = -0.5 * d * self._log_2pi - 0.5 * log_det - 0.5 * sq_mahal
        return bkd.reshape(logpdf, (1, nsamples))

    def _build_chol_factor(self, x: Array) -> Array:
        """Build Cholesky factor of full covariance Σ = D² + VV^T.

        Parameters
        ----------
        x : Array
            Shape (nvars, nsamples).

        Returns
        -------
        Array
            Shape (nsamples, d, d). Lower-triangular Cholesky factor.
        """
        cov = self.covariance(x)  # (nsamples, d, d)
        return self._bkd.cholesky(cov)  # (nsamples, d, d)

    def rvs(self, x: Array) -> Array:
        """Generate random samples.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Random samples. Shape: (d, nsamples)
        """
        nsamples = x.shape[1]
        base = self._bkd.asarray(np.random.randn(self._d, nsamples).astype(np.float64))
        return self.reparameterize(x, base)

    def reparameterize(self, x: Array, base_samples: Array) -> Array:
        """Transform base samples to q(z|x).

        Computes Cholesky L of Σ = D² + VV^T, then z = mu + L @ eps.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        base_samples : Array
            Standard normal samples. Shape: (d, nsamples)

        Returns
        -------
        Array
            Reparameterized samples. Shape: (d, nsamples)
        """
        self._sync_param_funcs()
        bkd = self._bkd

        mean = self._mean_func(x)  # (d, nsamples)
        L_batch = self._build_chol_factor(x)  # (nsamples, d, d)

        eps_T = base_samples.T  # (nsamples, d)
        Leps = bkd.einsum("nij,nj->ni", L_batch, eps_T)  # (nsamples, d)

        return mean + Leps.T  # (d, nsamples)

    def kl_divergence(self, x: Array, prior) -> Array:
        """Compute KL(q(.|x) || prior) per sample.

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)
        prior : DenseCholeskyMultivariateGaussian
            Fixed multivariate Gaussian prior.

        Returns
        -------
        Array
            Per-sample KL divergence. Shape: (1, nsamples)
        """
        self._sync_param_funcs()
        bkd = self._bkd
        d = self._d
        r = self._rank
        nsamples = x.shape[1]

        mean_q = self._mean_func(x)  # (d, nsamples)
        log_diag = self._log_diag_func(x)  # (d, nsamples)
        diag_vals, V_batch = self._build_D_V(x)
        D2 = diag_vals**2  # (d, nsamples)

        mean_p = prior.mean()  # (d, 1)
        cov_p_inv = prior.covariance_inverse()  # (d, d)
        log_det_L_p = prior.covariance_operator().log_determinant()

        # Quadratic term
        mean_diff = mean_p - mean_q  # (d, nsamples)
        Sinv_diff = bkd.dot(cov_p_inv, mean_diff)  # (d, nsamples)
        quad_term = bkd.sum(mean_diff * Sinv_diff, axis=0)  # (nsamples,)

        # Trace term: tr(Sigma_p^{-1} (D^2 + V V^T))
        # = tr(Sigma_p^{-1} diag(D^2)) + tr(Sigma_p^{-1} V V^T)
        # First part: sum_i Sigma_p_inv[i,i] * D^2[i]
        diag_Sinv = bkd.diag(cov_p_inv)  # (d,)
        trace_diag = bkd.sum(bkd.reshape(diag_Sinv, (d, 1)) * D2, axis=0)  # (nsamples,)

        if r > 0:
            # tr(Sigma_p^{-1} V V^T) per sample
            # = sum_{ij} (Sigma_p^{-1} V)_{ij} * V_{ij}
            SinvV = bkd.einsum("ij,njk->nik", cov_p_inv, V_batch)  # (N,d,r)
            trace_lr = bkd.einsum("nij,nij->n", SinvV, V_batch)  # (N,)
            trace_term = trace_diag + trace_lr
        else:
            trace_term = trace_diag

        # Log determinant of Sigma_q = D^2 + V V^T
        log_det_q_diag = 2.0 * bkd.sum(log_diag, axis=0)  # (nsamples,)
        if r > 0:
            # Matrix determinant lemma:
            # log|D^2 + VV^T| = log|I_r + V^T D^{-2} V| + log|D^2|
            Dinv2 = 1.0 / D2  # (d, nsamples)
            Dinv2_V = V_batch * bkd.reshape(Dinv2.T, (nsamples, d, 1))  # (N, d, r)
            M_batch = bkd.reshape(bkd.eye(r), (1, r, r)) + bkd.einsum(
                "nji,njk->nik", V_batch, Dinv2_V
            )  # (N,r,r)
            _, log_det_M = bkd.slogdet(M_batch)  # (N,)
            log_det_q = log_det_q_diag + log_det_M
        else:
            log_det_q = log_det_q_diag

        log_det_p = 2.0 * log_det_L_p

        kl = 0.5 * (trace_term + quad_term - d + log_det_p - log_det_q)
        return bkd.reshape(kl, (1, nsamples))

    def covariance(self, x: Array) -> Array:
        """Compute covariance matrices at conditioning points.

        Σ(x) = D(x)² + V(x) V(x)^T

        Parameters
        ----------
        x : Array
            Conditioning variable. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Covariance matrices. Shape: (nsamples, d, d)
        """
        self._sync_param_funcs()
        bkd = self._bkd
        d = self._d
        nsamples = x.shape[1]
        diag_vals, V_batch = self._build_D_V(x)
        D2 = diag_vals**2  # (d, nsamples)

        # D^2 as batch diagonal: (nsamples, d) -> (nsamples, d, d)
        eye_batch = bkd.reshape(bkd.eye(d), (1, d, d))  # (1, d, d)
        cov = eye_batch * bkd.reshape(D2.T, (nsamples, d, 1))  # broadcast

        if V_batch is not None:
            cov = cov + bkd.einsum("nij,nkj->nik", V_batch, V_batch)

        return cov

    def base_distribution(self):
        """Return the base distribution: N(0, I_d)."""
        from pyapprox.probability.gaussian.dense import (
            DenseCholeskyMultivariateGaussian,
        ) #TODO: does this need to be a lazy import,
        # i.e. adding to top will load an otional dependency or can we move
        # import to top of file

        return DenseCholeskyMultivariateGaussian(
            self._bkd.zeros((self._d, 1)),
            self._bkd.eye(self._d),
            self._bkd,
        )

    def __repr__(self) -> str:
        return (
            f"ConditionalLowRankCholGaussian(nvars={self.nvars()}, "
            f"nqoi={self.nqoi()}, rank={self._rank})"
        )

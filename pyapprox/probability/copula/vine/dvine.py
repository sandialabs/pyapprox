"""
D-vine copula.

Provides the DVineCopula class which chains bivariate pair copulas
across tree levels using h-functions for density evaluation and sampling.
"""

from typing import Dict, Generic, List

import numpy as np

from pyapprox.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.probability.copula.vine.helpers import (
    compute_dvine_partial_correlations,
    correlation_from_partial_correlations,
    precision_bandwidth,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class DVineCopula(Generic[Array]):
    """
    D-vine copula with truncation.

    A D-vine on ordering (0, 1, ..., n-1) where:
    - Tree t (1-indexed): edges (e, e+t | e+1,...,e+t-1) for e=0,...,n-1-t
    - Truncation at level k: trees k+1,...,n-1 use independence copulas

    Parameters
    ----------
    pair_copulas : Dict[int, List[BivariateCopulaProtocol[Array]]]
        Dictionary mapping tree level (1-indexed) to list of pair copulas.
        Tree t has (n - t) pair copulas.
    nvars : int
        Number of variables.
    truncation_level : int
        Maximum tree level with non-trivial pair copulas.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        pair_copulas: Dict[int, List[BivariateCopulaProtocol[Array]]],
        nvars: int,
        truncation_level: int,
        bkd: Backend[Array],
    ) -> None:
        if nvars < 2 and truncation_level > 0:
            raise ValueError(f"nvars must be >= 2 for non-zero truncation, got {nvars}")
        if truncation_level < 0 or truncation_level > nvars - 1:
            raise ValueError(
                f"truncation_level must be in [0, {nvars - 1}], got {truncation_level}"
            )
        for t in range(1, truncation_level + 1):
            if t not in pair_copulas:
                raise ValueError(f"pair_copulas missing tree level {t}")
            expected = nvars - t
            actual = len(pair_copulas[t])
            if actual != expected:
                raise ValueError(
                    f"Tree {t} requires {expected} pair copulas, got {actual}"
                )
            for e, copula in enumerate(pair_copulas[t]):
                if not isinstance(copula, BivariateCopulaProtocol):
                    raise TypeError(
                        f"pair_copulas[{t}][{e}] must satisfy "
                        f"BivariateCopulaProtocol, "
                        f"got {type(copula).__name__}"
                    )

        self._pair_copulas = pair_copulas
        self._nvars = nvars
        self._truncation_level = truncation_level
        self._bkd = bkd

        all_hyps: list[Any] = []
        for t in range(1, truncation_level + 1):
            for copula in pair_copulas[t]:
                all_hyps.extend(copula.hyp_list().hyperparameters())
        if all_hyps:
            self._hyp_list = HyperParameterList(all_hyps)
        else:
            self._hyp_list = HyperParameterList([], bkd=bkd)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nparams(self) -> int:
        """Return the total number of parameters across all pair copulas."""
        return self._hyp_list.nparams()

    def truncation_level(self) -> int:
        """Return the truncation level."""
        return self._truncation_level

    def hyp_list(self) -> HyperParameterList:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def pair_copula(self, tree: int, edge: int) -> BivariateCopulaProtocol[Array]:
        """Return the pair copula at the given tree level and edge."""
        return self._pair_copulas[tree][edge]

    def npair_copulas(self) -> int:
        """Return the total number of pair copulas."""
        count = 0
        for t in range(1, self._truncation_level + 1):
            count += len(self._pair_copulas[t])
        return count

    def _validate_input(self, u: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if u.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape ({self._nvars}, nsamples), got {u.ndim}D"
            )
        if u.shape[0] != self._nvars:
            raise ValueError(f"Expected {self._nvars} variables, got {u.shape[0]}")

    def logpdf(self, u: Array) -> Array:
        """
        Evaluate the log vine copula density.

        Uses the Aas et al. (2009) forward propagation algorithm.

        Parameters
        ----------
        u : Array
            Points in (0,1)^d. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log copula density values. Shape: (1, nsamples)
        """
        self._validate_input(u)
        n = self._nvars
        nsamples = u.shape[1]
        log_density = self._bkd.zeros((1, nsamples))

        if self._truncation_level == 0:
            return log_density

        v_fwd_prev = [u[e : e + 1, :] for e in range(n)]
        v_bwd_prev = [u[e : e + 1, :] for e in range(n)]

        for t in range(1, self._truncation_level + 1):
            n_edges = n - t
            v_fwd_new: List[Array] = [self._bkd.zeros((1, 0))] * n_edges
            v_bwd_new: List[Array] = [self._bkd.zeros((1, 0))] * n_edges

            for e in range(n_edges):
                left = v_fwd_prev[e]
                right = v_bwd_prev[e + 1]
                copula = self._pair_copulas[t][e]

                uv = self._bkd.concatenate([left, right], axis=0)
                log_density = log_density + copula.logpdf(uv)

                if t < self._truncation_level:
                    v_fwd_new[e] = copula.h_function(left, right)
                    v_bwd_new[e] = copula.h_function(right, left)

            v_fwd_prev = v_fwd_new
            v_bwd_prev = v_bwd_new

        return log_density

    def sample(self, nsamples: int) -> Array:
        """
        Draw samples from the vine copula.

        Uses Aas et al. (2009) Algorithm 2: sequential sampling with
        h-function inversion.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples in (0,1)^d. Shape: (nvars, nsamples)
        """
        n = self._nvars
        T = self._truncation_level

        # TODO: do not use astype, this will break if we want to use float32
        # let backend do correct conversion
        w = self._bkd.asarray(np.random.uniform(0, 1, (n, nsamples)).astype(np.float64))

        if T == 0:
            return w

        # V_fwd[t][e] = u_e conditioned on e-1,...,e-t (predecessors)
        # V_bwd[t][e] = u_e conditioned on e+1,...,e+t (successors)
        V_fwd: Dict[int, Dict[int, Array]] = {t_: {} for t_ in range(T + 1)}
        V_bwd: Dict[int, Dict[int, Array]] = {t_: {} for t_ in range(T + 1)}

        # Variable 0
        u_0 = w[0:1, :]
        V_fwd[0][0] = u_0
        V_bwd[0][0] = u_0

        for i in range(1, n):
            K_i = min(i, T)
            v = w[i : i + 1, :]  # shape (1, nsamples)

            # Phase A: invert from deepest tree down to tree 1
            for t in range(K_i, 0, -1):
                edge = i - t
                pair_cop = self._pair_copulas[t][edge]
                if t == 1:
                    conditioning = V_bwd[0][i - 1]
                else:
                    conditioning = V_bwd[t - 1][i - t]
                v = pair_cop.h_inverse(v, conditioning)

            u_i = v
            V_fwd[0][i] = u_i
            V_bwd[0][i] = u_i

            # Phase B: update pseudo-observations for future variables
            if i < n - 1:
                for t in range(1, K_i + 1):
                    edge = i - t
                    pair_cop = self._pair_copulas[t][edge]
                    if t == 1:
                        left = V_bwd[0][i - 1]
                        right = V_fwd[0][i]
                    else:
                        left = V_bwd[t - 1][i - t]
                        right = V_fwd[t - 1][i]

                    V_fwd[t][i] = pair_cop.h_function(right, left)
                    V_bwd[t][i - t] = pair_cop.h_function(left, right)

        # Assemble result
        result = self._bkd.concatenate([V_fwd[0][i] for i in range(n)], axis=0)
        return result

    def to_correlation_matrix(self) -> Array:
        """
        Reconstruct the correlation matrix from pair copula parameters.

        Only valid when all pair copulas are BivariateGaussianCopula.

        Returns
        -------
        Array
            Correlation matrix. Shape: (nvars, nvars)

        Raises
        ------
        TypeError
            If any pair copula is not BivariateGaussianCopula.
        """
        partial_corrs: Dict[int, List[float]] = {}
        for t in range(1, self._truncation_level + 1):
            partial_corrs[t] = []
            for copula in self._pair_copulas[t]:
                if not isinstance(copula, BivariateGaussianCopula):
                    raise TypeError(
                        "to_correlation_matrix requires all pair copulas "
                        "to be BivariateGaussianCopula, got "
                        f"{type(copula).__name__}"
                    )
                arctanh_rho = copula.hyp_list().get_values()[0]
                rho = self._bkd.tanh(arctanh_rho)
                partial_corrs[t].append(self._bkd.to_float(rho))
        return correlation_from_partial_correlations(
            partial_corrs, self._nvars, self._bkd
        )

    def to_precision_matrix(self) -> Array:
        """
        Compute the precision matrix from the correlation matrix.

        Only valid when all pair copulas are BivariateGaussianCopula.

        Returns
        -------
        Array
            Precision matrix. Shape: (nvars, nvars)

        Raises
        ------
        TypeError
            If any pair copula is not BivariateGaussianCopula.
        """
        R = self.to_correlation_matrix()
        return self._bkd.inv(R)

    @classmethod
    def from_precision_matrix(
        cls,
        precision: Array,
        bkd: Backend[Array],
    ) -> "DVineCopula[Array]":
        """
        Construct a D-vine from a banded precision matrix.

        Parameters
        ----------
        precision : Array
            Symmetric positive definite precision matrix. Shape: (n, n)
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        DVineCopula
            D-vine with BivariateGaussianCopula pair copulas and
            truncation level equal to the bandwidth of the precision matrix.
        """
        n = precision.shape[0]
        k = precision_bandwidth(precision, bkd)

        if k == 0:
            return cls({}, n, 0, bkd)

        # Compute correlation matrix from precision
        Sigma = bkd.inv(precision)
        d = bkd.sqrt(bkd.get_diagonal(Sigma))
        d_inv = 1.0 / d
        R = Sigma * bkd.outer(d_inv, d_inv)

        partial_corrs = compute_dvine_partial_correlations(R, k, bkd)

        pair_copulas: Dict[int, List[BivariateCopulaProtocol[Array]]] = {}
        for t in range(1, k + 1):
            pair_copulas[t] = [
                BivariateGaussianCopula(rho, bkd) for rho in partial_corrs[t]
            ]

        return cls(pair_copulas, n, k, bkd)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DVineCopula(nvars={self._nvars}, "
            f"truncation_level={self._truncation_level}, "
            f"npair_copulas={self.npair_copulas()})"
        )

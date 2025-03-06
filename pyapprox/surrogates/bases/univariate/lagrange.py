import warnings
from typing import Tuple

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

from pyapprox.surrogates.bases.univariate.base import (
    UnivariateInterpolatingBasis,
)
from pyapprox.surrogates.bases.univariate.orthopoly import (
    Chebyshev1stKindGaussLobattoQuadratureRule,
    GaussQuadratureRule,
)


def univariate_lagrange_polynomial(
    abscissa: Array, samples: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    assert abscissa.ndim == 1
    assert samples.ndim == 1
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None] - abscissa[None, :]
    numers = samples[:, None] - abscissa[None, :]
    # values = bkd.empty((samples.shape[0], nabscissa))
    values = []
    for ii in range(nabscissa):
        # l_j(x) = prod_{i!=j} (x-x_i)/(x_j-x_i)
        denom = bkd.prod(denoms[ii, :ii]) * bkd.prod(denoms[ii, ii + 1 :])
        # denom = bkd.prod(bkd.delete(denoms[ii], ii))
        # numer = bkd.prod(bkd.delete(numers, ii, axis=1), axis=1)
        numer = bkd.prod(numers[:, :ii], axis=1) * bkd.prod(
            numers[:, ii + 1 :], axis=1
        )
        # values[:, ii] = numer/denom
        values.append(numer / denom)
    return bkd.stack(values, axis=1)
    # return values


class UnivariateLagrangeBasis(UnivariateInterpolatingBasis):
    def __init__(self, quadrature_rule, nterms=None):
        super().__init__(backend=quadrature_rule._bkd)
        self._quad_rule = quadrature_rule
        if nterms is not None:
            self.set_nterms(nterms)

    def first_derivatives_implemented(self) -> bool:
        return True

    def second_derivatives_implemented(self) -> bool:
        return True

    def set_nterms(self, nterms: int):
        self._quad_samples, self._quad_weights = self._quad_rule(nterms)
        if self._quad_samples.shape[1] != nterms:
            raise RuntimeError("quad samples have the wrong shape")

    def _values(self, samples):
        return univariate_lagrange_polynomial(
            self._quad_samples[0], samples[0], self._bkd
        )

    def _quadrature_rule(self) -> Tuple[Array, Array]:
        return self._quad_samples, self._quad_weights

    def _semideep_copy(self) -> "UnivariateLagrangeBasis":
        # do not copy quadrature rule as it may be used in multiple
        # dimensions of a tensor product and subspaces in a sparse grid
        return UnivariateLagrangeBasis(self._quad_rule, self.nterms())

    def _derivatives(self, samples: Array, order: int) -> Array:
        if order == 1:
            return self._first_derivatives(samples)
        return self._second_derivatives(samples)

    def _first_derivatives(self, samples: Array) -> Array:
        abscissa = self._quad_samples[0]
        samples = samples[0]
        nsamples = samples.shape[0]
        nabscissa = abscissa.shape[0]
        denoms = abscissa[:, None] - abscissa[None, :]
        numers = samples[:, None] - abscissa[None, :]
        derivs = self._bkd.empty((nsamples, nabscissa))
        for ii in range(nabscissa):
            denom = self._bkd.prod(denoms[ii, :ii]) * self._bkd.prod(
                denoms[ii, ii + 1 :]
            )
            # product rule for the jth 1D basis function
            numer_deriv = 0
            for jj in range(nabscissa):
                # compute deriv of kth component of product rule sum
                if ii != jj:
                    numer_deriv += self._bkd.prod(
                        self._bkd.delete(numers, (ii, jj), axis=1),
                        axis=1,
                    )
            derivs[:, ii] = numer_deriv / denom
        return derivs

    def _second_derivatives(self, samples: Array) -> Array:
        abscissa = self._quad_samples[0]
        samples = samples[0]
        nsamples = samples.shape[0]
        nabscissa = abscissa.shape[0]
        denoms = abscissa[:, None] - abscissa[None, :]
        numers = samples[:, None] - abscissa[None, :]
        derivs = self._bkd.empty((nsamples, nabscissa))
        for ii in range(nabscissa):
            denom = self._bkd.prod(denoms[ii, :ii]) * self._bkd.prod(
                denoms[ii, ii + 1 :]
            )
            numer_deriv = 0
            for jj in range(nabscissa):
                for kk in range(nabscissa):
                    if ii != jj and ii != kk and jj != kk:
                        numer_deriv += self._bkd.prod(
                            self._bkd.delete(numers, (ii, jj, kk), axis=1),
                            axis=1,
                        )
            derivs[:, ii] = numer_deriv / denom
        return derivs


class UnivariateBarycentricLagrangeBasis(UnivariateLagrangeBasis):
    def set_nterms(self, nterms: int, interval_length: float = None):
        super().set_nterms(nterms)
        self._interval_length = interval_length
        self._set_barycentric_weights()

    def _set_barycentric_weights(self):
        """
        Return barycentric weights for a sequence of samples. e.g. of sequence
        x0, x1, x2 where order represents the order in which the samples are
        added to the interpolant.

        E.g. using 3 points
        compites [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]

        Note
        ----
        If length of interval [a,b]=4C then weights will grow or decay
        exponentially at C^{-n} where n is number of points causing overflow
        or underflow.

        To minimize this effect multiply each x_j-x_k by C^{-1}. This has effect
        of rescaling all weights by C^n. In rare situations where n is so large
        randomize or use Leja ordering of the samples before computing weights.
        See Barycentric Lagrange Interpolation by
        Jean-Paul Berrut and Lloyd N. Trefethen 2004
        """
        if self._interval_length is None:
            scaling_factor = 1.0
        else:
            scaling_factor = self._interval_length / 4.0

        C_inv = 1.0 / scaling_factor
        samples = self._quad_samples[0]
        nsamples = samples.shape[0]

        weights = self._bkd.empty((nsamples, nsamples), dtype=float)
        weights[0, 0] = 1.0
        # TODO speed up by using same concept as in lagrange polynomial
        for jj in range(1, nsamples):
            weights[jj, :jj] = (
                C_inv * (samples[:jj] - samples[jj]) * weights[jj - 1, :jj]
            )
            weights[jj, jj] = self._bkd.prod(
                C_inv * (samples[jj] - samples[:jj])
            )
            weights[jj - 1, :jj] = 1.0 / weights[jj - 1, :jj]

        weights[nsamples - 1, :nsamples] = (
            1.0 / weights[nsamples - 1, :nsamples]
        )

        if not self._bkd.all(np.isfinite(weights)):
            raise RuntimeError(
                "Samples are ill conditioned. set or change scale factor"
            )
        # todo consider storing all elements of result and updating
        # if necessary
        self._bary_weights = weights[nsamples - 1]

    def _values(self, eval_samples: Array) -> Array:
        with warnings.catch_warnings():
            # ignore division by zero warning thrown when computing basis
            # at an interpolation point. It is faster to ignore then
            # only compute basis values at interpolation points
            warnings.simplefilter("ignore")
            diff = eval_samples.T - self._quad_samples
            diff_inv = 1 / diff
            # factor = self._bkd.prod(diff, axis=1)
            factor = 1 / self._bkd.sum(diff_inv * self._bary_weights, axis=1)
            basis_mat = factor[:, None] * (diff_inv * self._bary_weights)
            basis_mat[self._bkd.where(diff == 0)] = 1.0
        return basis_mat


class UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis(
    UnivariateBarycentricLagrangeBasis
):
    # TODO: not sure if I have the naming of first and second correct.
    def __init__(
        self,
        bounds: Array,
        nterms: int = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(
            Chebyshev1stKindGaussLobattoQuadratureRule(
                bounds, backend=backend
            ),
            nterms,
        )

    def _set_barycentric_weights(self):
        self._bary_weights = (-1.0) ** (self._bkd.arange(self.nterms()) % 2)
        self._bary_weights[0] /= 2
        self._bary_weights[-1] /= 2


def setup_lagrange_basis(
    basis_type: str,
    quadrature_rule=None,
    bounds: Array = None,
    backend: LinAlgMixin = NumpyLinAlgMixin,
):
    if bounds is None and quadrature_rule is None:
        raise ValueError("must specify either bounds or quadrature_rule")
    # bases that use barycentric interpolation with barycentric weights
    # numerically, can be used with any quadrature rule
    basis_dict_from_quad = {
        "lagrange": UnivariateLagrangeBasis,
        "barycentric": UnivariateBarycentricLagrangeBasis,
    }
    # bases that use barycentric interpolation with barycentric weights
    # computed exactly
    basis_dict_from_bounds = {
        "chebyhsev1": UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis
    }
    if (
        basis_type not in basis_dict_from_quad
        and basis_type not in basis_dict_from_bounds
    ):
        raise ValueError(
            "basis_type {0} not supported must be in {1}".format(
                basis_type,
                list(basis_dict_from_quad.keys())
                + list(basis_dict_from_bounds.keys()),
            )
        )
    if basis_type in basis_dict_from_quad:
        if quadrature_rule is None:
            raise ValueError(
                "{0} requires quadratrure_rule".format(basis_type)
            )
        return basis_dict_from_quad[basis_type](quadrature_rule)

    if bounds is None:
        raise ValueError("{0} requires bounds".format(basis_type))
    return basis_dict_from_bounds[basis_type](bounds, backend=backend)

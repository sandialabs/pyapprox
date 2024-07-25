import math
from abc import ABC, abstractmethod
from warnings import warn

from scipy.special import gammaln

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence,
    hermite_recurrence,
    krawtchouk_recurrence,
    hahn_recurrence,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    laguerre_recurrence,
)


class OrthonormalPolynomial1D(ABC):
    def __init__(self, backend):
        self._rcoefs = None
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend
        self._prob_meas = True

    def _ncoefs(self):
        if self._rcoefs is None:
            raise ValueError("recrusion_coefs have not been set")
        return self._rcoefs.shape[0]

    @abstractmethod
    def _get_recursion_coefficients(self, ncoefs):
        raise NotImplementedError

    def set_recursion_coefficients(self, ncoefs):
        """Compute and set the recursion coefficients of the polynomial."""
        if self._rcoefs is None or self._ncoefs() < ncoefs:
            self._rcoefs = self._bkd._la_array(
                self._get_recursion_coefficients(ncoefs)
            )

    def _opts_equal(self, other):
        return True

    def __eq__(self, other):
        return (
            self.__class__.__name__ == self.__class__.__name__
            and self._opts_equal(other)
            and self._rcoefs == other._self._rcoefs
        )

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def _values(self, samples, nmax):
        if self._rcoefs is None:
            raise ValueError("Must set recursion coefficients.")
        if nmax >= self._rcoefs.shape[0]:
            raise ValueError(
                "The number of polynomial terms requested {0} {1}".format(
                    nmax,
                    "exceeds number of rcoefs {0}".format(
                        self._rcoefs.shape[0]
                    ),
                )
            )

        nsamples = samples.shape[0]

        vals = [self._bkd._la_full((nsamples,), 1.0 / self._rcoefs[0, 1])]

        if nmax > 0:
            vals.append(
                1
                / self._rcoefs[1, 1]
                * ((samples - self._rcoefs[0, 0]) * vals[0])
            )

        for jj in range(2, nmax + 1):
            vals.append(
                1.0
                / self._rcoefs[jj, 1]
                * (
                    (samples - self._rcoefs[jj - 1, 0]) * vals[jj - 1]
                    - self._rcoefs[jj - 1, 1] * vals[jj - 2]
                )
            )
        return self._bkd._la_stack(vals, axis=1)

    def derivatives(self, samples, nmax, order, return_all=False):
        """
        Compute the first n dervivatives of the polynomial.
        """
        if order < 2:
            raise ValueError(
                "derivative order {0} must be greater than zero".format(order)
            )
        vals = self._values(samples, nmax)
        nsamples = samples.shape[0]
        nindices = nmax + 1
        a = self._rcoefs[:, 0]
        b = self._rcoefs[:, 1]

        result = self._bkd._la_empty((nsamples, nindices * (order + 1)))
        result[:, :nindices] = vals
        for _order in range(1, order + 1):
            derivs = self._bkd._la_full((nsamples, nindices), 0.0)
            for jj in range(_order, nindices):
                if jj == _order:
                    # use following expression to avoid overflow issues when
                    # computing oveflow
                    derivs[:, jj] = self._bkd._la_exp(
                        gammaln(_order + 1)
                        - 0.5
                        * self._bkd._la_sum(
                            self._bkd._la_log(b[: jj + 1] ** 2)
                        )
                    )
                else:
                    derivs[:, jj] = (
                        (samples - a[jj - 1]) * derivs[:, jj - 1]
                        - b[jj - 1] * derivs[:, jj - 2]
                        + _order * vals[:, jj - 1]
                    )
                    derivs[:, jj] *= 1.0 / b[jj]
            vals = derivs
            result[:, _order * nindices : (_order + 1) * nindices] = derivs

        if return_all:
            return result
        return result[:, order * nindices :]

    def gauss_quadrature_rule(self, npoints):
        r"""Computes Gauss quadrature from recurrence coefficients

        x, w = gauss_quadrature(npoints)

        Computes N Gauss quadrature nodes (x) and weights (w) from
        standard orthonormal recurrence coefficients.

        Parameters
        ----------
        npoints : integer
           Then number of quadrature points

        Returns
        -------
        x : array (npoints)
           The quadrature points

        w : array (npoints)
           The quadrature weights
        """
        if self._rcoefs is None:
            raise ValueError(
                "{0}: Must set recursion coefficients".format(self)
            )
        if npoints > self._ncoefs():
            raise ValueError(
                "{0}: Too many terms requested. {1}".format(
                    self,
                    "npoints={0} > ncoefs={1}".format(npoints, self._ncoefs()),
                )
            )

        a = self._rcoefs[:, 0]
        b = self._rcoefs[:, 1]

        # Form Jacobi matrix
        J = (
            self._bkd._la_diag(a[:npoints], 0)
            + self._bkd._la_diag(b[1:npoints], 1)
            + self._bkd._la_diag(b[1:npoints], -1)
        )

        x, eigvecs = self._bkd._la_eigh(J)
        if self._prob_meas:
            w = b[0] * eigvecs[0, :] ** 2
        else:
            print("A")
            w = self(x, npoints - 1)
            w = 1.0 / self._bkd._la_sum(w**2, axis=1)
        # w[~self._bkd._la_isfinite(w)] = 0.
        return x, w

    def __call__(self, samples, nmax):
        return self._values(samples, nmax)

    def _three_term_recurence(self):
        r"""
        Convert two term recursion coefficients

        .. math:: b_{n+1} p_{n+1} = (x - a_n) p_n - \sqrt{b_n} p_{n-1}

        into the equivalent
        three recursion coefficients

        .. math:: p_{n+1} = \tilde{a}_{n+1}x - \tilde{b_n}p_n - \tilde{c}_n p_{n-1}

        Returns
        -------
        abc : array (num_recursion_coeffs,3)
           The three term recursion coefficients
           :math:`\tilde{a}_n,\tilde{b}_n,\tilde{c}_n`
        """

        abc = self._bkd._la_zeros((self._ncoefs(), 3))
        abc[:, 0] = 1.0 / self._rcoefs[:, 1]
        abc[1:, 1] = self._rcoefs[:-1, 0] / self._rcoefs[1:, 1]
        abc[1:, 2] = self._rcoefs[:-1, 1] / self._rcoefs[1:, 1]
        return abc


class JacobiPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, alpha, beta, backend=None):
        super().__init__(backend)
        self._alpha = alpha
        self._beta = beta

    def _get_recursion_coefficients(self, ncoefs):
        return jacobi_recurrence(
            ncoefs,
            alpha=self._alpha,
            beta=self._beta,
            probability=self._prob_meas,
        )

    def _opts_equal(self, other):
        return self._alpha == other._alpha and self._beta == other._beta

    def __repr__(self):
        return "{0}(alpha={1}, beta={2})".format(
            self.__class__.__name__, self._alpha, self._beta
        )


class LegendrePolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(0.0, 0.0, backend=backend)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class HermitePolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, rho=0.0, prob_meas=True, backend=None):
        super().__init__(backend=backend)
        self._prob_meas = prob_meas
        self._rho = rho

    def _get_recursion_coefficients(self, ncoefs):
        return hermite_recurrence(
            ncoefs, rho=self._rho, probability=self._prob_meas
        )


class KrawtchoukPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, n, p, raisewarn=True, backend=None):
        super().__init__(backend)
        self._n = n
        self._p = p
        self._warn = raisewarn

    def _get_recursion_coefficients(self, ncoefs):
        msg = "Although bounded the Krawtchouk polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        if self._warn:
            warn(msg, UserWarning)
        ncoefs = min(ncoefs, self._n)
        return krawtchouk_recurrence(ncoefs, self._n, self._p)

    def _opts_equal(self, other):
        return self._n == other._n and self._p == other._p

    def __repr__(self):
        return "{0}(n={1}, p={2})".format(
            self.__class__.__name__, self._n, self._p
        )


class HahnPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, N, alpha, beta, raisewarn=True, backend=None):
        super().__init__(backend)
        self._N = N
        self._alpha = alpha
        self._beta = beta
        self._warn = raisewarn

    def _get_recursion_coefficients(self, ncoefs):
        msg = "Although bounded the Hahn polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        if self._warn:
            warn(msg, UserWarning)
        ncoefs = min(ncoefs, self._N)
        return hahn_recurrence(ncoefs, self._N, self._alpha, self._beta)

    def _opts_equal(self, other):
        return (
            self._N == other._N
            and self._alpha == other._alpha
            and self._beta == other._beta
        )

    def __repr__(self):
        return "{0}(N={1}, alpha={2}, beta={3})".format(
            self.__class__.__name__, self._N, self._alpha, self._beta
        )


class CharlierPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, mu, backend=None):
        super().__init__(backend)
        self._mu = mu

    def _get_recursion_coefficients(self, ncoefs):
        return charlier_recurrence(ncoefs, self._mu)


class DiscreteChebyshevPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, N, backend=None):
        super().__init__(backend)
        self._N = N

    def _get_recursion_coefficients(self, ncoefs):
        return discrete_chebyshev_recurrence(ncoefs, self._N)


class LaguerrePolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, rho, backend=None):
        super().__init__(backend)
        self._rho = rho

    def _get_recursion_coefficients(self, ncoefs):
        return laguerre_recurrence(ncoefs, self._rho)


class Chebyshev1stKindPolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(-0.5, -0.5, backend=backend)
        self._prob_meas = True

    def _get_recursion_coefficients(self, ncoefs):
        rcoefs = jacobi_recurrence(
            ncoefs,
            alpha=self._alpha,
            beta=self._beta,
            probability=self._prob_meas,
        )
        return rcoefs

    def __call__(self, samples, nmax):
        vals = super().__call__(samples, nmax)
        vals[:, 1:] /= 2**0.5
        return vals

    def gauss_quadrature_rule(self, npoints):
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w*math.pi

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class Chebyshev2ndKindPolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(0.5, 0.5, backend=backend)
        self._prob_meas = True

    def gauss_quadrature_rule(self, npoints):
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w*math.pi/2

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

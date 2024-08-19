import math
from abc import abstractmethod
from warnings import warn

from scipy.special import gammaln

from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence,
    hermite_recurrence,
    krawtchouk_recurrence,
    hahn_recurrence,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    laguerre_recurrence,
)
from pyapprox.surrogates.orthopoly.recursion_factory import (
    predictor_corrector_known_pdf,
    get_numerically_generated_recursion_coefficients_from_samples,
)
from pyapprox.surrogates.bases.univariate import (
    UnivariateBasis, UnivariateQuadratureRule
)
from pyapprox.variables.marginals import (
    get_distribution_info, get_pdf,
    transform_scale_parameters,
    is_continuous_variable,
    is_bounded_continuous_variable,
    get_probability_masses,
)
from pyapprox.util.transforms import Transform


# todo derive this from univariatebasis in surrogates.interp.tensor_prod
# this will require updating orthogonalpolybasis to pass in
# 2D row vectors rather than 1D arrays
# move univariatebasis from surrogates.interp.tensor_prod to its own file
# TODO all new classes should accept values as array (nqoi, nsamples)
class OrthonormalPolynomial1D(UnivariateBasis):
    def __init__(self, backend):
        super().__init__(backend)
        self._rcoefs = None
        self._prob_meas = True

    def _ncoefs(self):
        if self._rcoefs is None:
            raise ValueError("recrusion_coefs have not been set")
        return self._rcoefs.shape[0]

    @abstractmethod
    def _get_recursion_coefficients(self, ncoefs):
        raise NotImplementedError

    def set_nterms(self, nterms):
        """Compute and set the recursion coefficients of the polynomial."""
        if self._rcoefs is None or self._ncoefs() < nterms:
            # TODO implement increment of recursion coefficients
            self._rcoefs = self._bkd._la_array(
                self._get_recursion_coefficients(nterms)
            )
        elif self._rcoefs is not None or self._ncoefs() >= nterms:
            self._rcoefs = self._rcoefs[:nterms, :]

    def nterms(self):
        if self._rcoefs is None:
            return 0
        return self._rcoefs.shape[0]

    def _opts_equal(self, other):
        return True

    def __eq__(self, other):
        return (
            self.__class__.__name__ == self.__class__.__name__
            and self._opts_equal(other)
            and self._rcoefs == other._self._rcoefs
        )

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())

    def _values(self, samples):
        if self._rcoefs is None:
            raise ValueError("Must set recursion coefficients.")
        # samples passed in is 2D array with shape [1, nsamples]
        # so squeeze to 1D array
        samples = samples[0]
        nsamples = samples.shape[0]

        vals = [self._bkd._la_full((nsamples,), 1.0 / self._rcoefs[0, 1])]

        if self.nterms() > 1:
            vals.append(
                1
                / self._rcoefs[1, 1]
                * ((samples - self._rcoefs[0, 0]) * vals[0])
            )

        for jj in range(2, self.nterms()):
            vals.append(
                1.0
                / self._rcoefs[jj, 1]
                * (
                    (samples - self._rcoefs[jj - 1, 0]) * vals[jj - 1]
                    - self._rcoefs[jj - 1, 1] * vals[jj - 2]
                )
            )
        return self._bkd._la_stack(vals, axis=1)

    def _derivatives(self, samples, order, return_all=False):
        """
        Compute the first n dervivatives of the polynomial.
        """
        if order < 2:
            raise ValueError(
                "derivative order {0} must be greater than zero".format(order)
            )
        vals = self._values(samples)

        # samples passed in is 2D array with shape [1, nsamples]
        # so squeeze to 1D array
        samples = samples[0]
        nsamples = samples.shape[0]

        nindices = self.nterms()
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

        Unlike __call__, gauss quadrature rule can be called
        with npoints <= nterms

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
            w = self(x[None, :])[:, :npoints]
            w = 1.0 / self._bkd._la_sum(w**2, axis=1)
        # w[~self._bkd._la_isfinite(w)] = 0.
        return x[None, :], w[:, None]

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
        return "{0}(alpha={1}, beta={2}, nterms={3})".format(
            self.__class__.__name__, self._alpha, self._beta, self.nterms()
        )


class LegendrePolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(0.0, 0.0, backend=backend)

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


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
        return "{0}(n={1}, p={2}, nterms={3})".format(
            self.__class__.__name__, self._n, self._p, self.nterms()
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
        return "{0}(N={1}, alpha={2}, beta={3}, nterms={4})".format(
            self.__class__.__name__, self._N, self._alpha, self._beta,
            self.nterms()
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

    def __call__(self, samples):
        vals = super().__call__(samples)
        vals[:, 1:] /= 2**0.5
        return vals

    def gauss_quadrature_rule(self, npoints):
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w*math.pi

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class Chebyshev2ndKindPolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(0.5, 0.5, backend=backend)
        self._prob_meas = True

    def gauss_quadrature_rule(self, npoints):
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w*math.pi/2

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class DiscreteNumericOrthonormalPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, samples, weights, ortho_tol=1e-8, truncation_tol=0,
                 backend=None):
        """Compure recurrence coefficients from samples."""
        super().__init__(backend)
        self._samples = samples
        self._weights = weights
        self._ortho_tol = ortho_tol
        self._truncation_tol = truncation_tol

    def _get_recursion_coefficients(self, ncoefs):
        return get_numerically_generated_recursion_coefficients_from_samples(
            self._bkd._la_to_numpy(self._samples),
            self._bkd._la_to_numpy(self._weights), ncoefs, self._ortho_tol,
            self._truncation_tol)


class ContinuousNumericOrthonormalPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, marginal, quad_opts={}, integrate_fun=None,
                 backend=None):
        """Compure recurrence coefficients from a known PDF
        using predictor corrector method."""
        super().__init__(backend)
        # Get version var.pdf without error checking which runs much faster
        self._pdf, self._loc, self._scale, self._can_lb, self._can_ub = (
            self._parse_marginal(marginal))
        self._marginal = marginal
        self._quad_opts = self._parse_quad_opts(quad_opts, integrate_fun)

    def _parse_marginal(self, marginal):
        pdf = get_pdf(marginal)
        loc, scale = transform_scale_parameters(marginal)
        lb, ub = marginal.interval(1)
        if is_bounded_continuous_variable(marginal):
            can_lb, can_ub = -1, 1
        elif is_continuous_variable(marginal):
            can_lb = (lb-loc)/scale
            can_ub = (ub-loc)/scale
        else:
            raise ValueError("variable must be a continuous variable")
        return pdf, loc, scale, can_lb, can_ub

    def _parse_quad_opts(self, quad_opts, integrate_fun):
        if not isinstance(quad_opts, dict):
            raise ValueError("quad_opts must be a dictionary")
        if integrate_fun is not None:
            quad_opts["integrate_fun"] = integrate_fun
        return quad_opts

    def _canonical_pdf(self, x):
        return self._pdf(x*self._scale+self._loc)*self._scale

    def _get_recursion_coefficients(self, ncoefs):
        return predictor_corrector_known_pdf(
            ncoefs, self._can_lb, self._can_ub, self._canonical_pdf,
            self._quad_opts)


def setup_univariate_orthogonal_polynomial_from_marginal(
        marginal, opts=None, backend=None
):
    var_name, scales, shapes = get_distribution_info(marginal)

    # Askey polynomials that do not need scale or shape
    simple_askey_polys = {
        "uniform": LegendrePolynomial1D,
        "norm": HermitePolynomial1D,
    }
    if var_name in simple_askey_polys:
        return simple_askey_polys[var_name](backend=backend)

    # Other Askey polynomials
    # Ignore binom (KrawtchoukPolynomial1D) and hypergeom (HahnPolynomial1D)
    # because these polynomials are not defined on [-1, 1] which is assumed
    # to be the canonical domain of bounded marginals
    if var_name == "beta":
        return JacobiPolynomial1D(
            shapes["b"]-1, shapes["a"]-1, backend=backend
        )

    if var_name == "poisson":
        return CharlierPolynomial1D(*shapes, backend=backend)

    # Other continuous marginals
    if (
            is_continuous_variable(marginal) and
            var_name != "continuous_rv_sample"
    ):
        if opts is None:
            opts = {"epsrel": 1e-8, "epsabs": 1e-8, "limit": 100}
        return ContinuousNumericOrthonormalPolynomial1D(
            marginal, opts
        )

    # other discrete marginals
    if opts is None:
        opts = {"otol": 1e-8, "ptol": 1e-8}
    if hasattr(shapes, "xk"):
        xk, pk = shapes["xk"], shapes["pk"]
    else:
        xk, pk = get_probability_masses(
            marginal, opts["ptol"])

    loc, scale = transform_scale_parameters(marginal)
    xk = (xk-loc)/scale
    return DiscreteNumericOrthonormalPolynomial1D(xk, pk, opts["otol"])


class GaussQuadratureRule(UnivariateQuadratureRule):
    def __init__(self, marginal, opts=None, backend=None, store=False):
        super().__init__(backend, store)
        self._poly = setup_univariate_orthogonal_polynomial_from_marginal(
            marginal, opts=opts, backend=backend
        )

    def _quad_rule(self, nnodes):
        if self._poly.nterms() < nnodes:
            self._poly.set_nterms(nnodes)
        return self._poly.gauss_quadrature_rule(nnodes)


class AffineMarginalTransform(Transform):
    def __init__(self, marginal, enforce_bounds=False, backend=None):
        super().__init__(backend)
        self._marginal = marginal
        self._enforce_bounds = enforce_bounds

        self._loc, self._scale = transform_scale_parameters(self._marginal)

    def _check_bounds(self, user_samples):
        if (
                not self._enforce_bounds
                or not is_bounded_continuous_variable(self._marginal)
        ):
            return

        bounds = [self._loc-self._scale, self._loc+self._scale]
        if (
                self._bkd._la_any(user_samples < bounds[0]) or
                self._bkd._la_any(user_samples > bounds[1])
        ):
            raise ValueError(f'Sample outside the bounds {bounds}')

    def map_from_canonical(self, canonical_samples):
        return canonical_samples*self._scale+self._loc

    def map_to_canonical(self, user_samples):
        self._check_bounds(user_samples)
        return (user_samples-self._loc)/self._scale

    def derivatives_to_canonical(self, user_derivs, order=1):
        return user_derivs * self._scale**order

    def derivatives_from_canonical(self, canonical_derivs, order=1):
        return canonical_derivs / self._scale**order

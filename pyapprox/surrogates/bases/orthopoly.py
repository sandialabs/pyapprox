import math
from abc import abstractmethod
from warnings import warn

import scipy.special as sp
from scipy import stats

from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence,
    hermite_recurrence,
    krawtchouk_recurrence,
    hahn_recurrence,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    laguerre_recurrence,
)
from pyapprox.surrogates.bases.univariate import (
    UnivariateBasis,
    UnivariateQuadratureRule,
    UnivariateIntegrator,
    ScipyUnivariateIntegrator,
    UnivariateLagrangeBasis,
    UnivariateBarycentricLagrangeBasis,
)
from pyapprox.variables.marginals import (
    get_distribution_info,
    get_pdf,
    transform_scale_parameters,
    is_continuous_variable,
    is_bounded_continuous_variable,
    get_probability_masses,
)
from pyapprox.util.transforms import (
    UnivariateAffineTransform,
    IdentityTransform,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def _evaluate_orthonormal_polynomial_1d(rcoefs, bkd, samples):
    # samples passed in is 2D array with shape [1, nsamples]
    # so squeeze to 1D array
    samples = samples[0]
    nsamples = samples.shape[0]
    nterms = rcoefs.shape[0]

    vals = [bkd.full((nsamples,), 1.0 / rcoefs[0, 1])]

    if nterms > 1:
        vals.append(1 / rcoefs[1, 1] * ((samples - rcoefs[0, 0]) * vals[0]))

    for jj in range(2, nterms):
        vals.append(
            1.0
            / rcoefs[jj, 1]
            * (
                (samples - rcoefs[jj - 1, 0]) * vals[jj - 1]
                - rcoefs[jj - 1, 1] * vals[jj - 2]
            )
        )
    return bkd.stack(vals, axis=1)


class OrthonormalPolynomial1D(UnivariateBasis):
    def __init__(self, trans, backend):
        super().__init__(trans, backend)
        if isinstance(self._trans, IdentityTransform):
            msg = (
                "No transformation was set. Proceed with caution. "
                "User is responsible for ensuring samples are in "
                "canonical domain of the polynomial."
            )
            warn(msg, UserWarning)
        self._rcoefs = None
        self._prob_meas = True

    def _ncoefs(self):
        if self._rcoefs is None:
            raise ValueError("recursion_coefs have not been set")
        return self._rcoefs.shape[0]

    @abstractmethod
    def _get_recursion_coefficients(self, ncoefs):
        raise NotImplementedError

    def set_nterms(self, nterms):
        """Compute and set the recursion coefficients of the polynomial."""
        if self._rcoefs is None or self._ncoefs() < nterms:
            # TODO implement increment of recursion coefficients
            self._rcoefs = self._bkd.array(
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
        can_samples = self._trans.map_to_canonical(samples)
        if self._rcoefs is None:
            raise ValueError("Must set recursion coefficients.")
        return _evaluate_orthonormal_polynomial_1d(
            self._rcoefs, self._bkd, can_samples
        )

    def _derivatives(self, samples, order, return_all=False):
        """
        Compute the first n dervivatives of the polynomial.
        """
        if order < 0:
            raise ValueError(
                "derivative order {0} must be greater than zero".format(order)
            )
        vals = self._values(samples)
        can_samples = self._trans.map_to_canonical(samples)

        # samples passed in is 2D array with shape [1, nsamples]
        # so squeeze to 1D array
        can_samples = can_samples[0]
        nsamples = can_samples.shape[0]

        nindices = self.nterms()
        a = self._rcoefs[:, 0]
        b = self._rcoefs[:, 1]

        result = [vals]
        vals = vals.T
        for _order in range(1, order + 1):
            can_derivs = []
            for jj in range(_order):
                can_derivs.append(self._bkd.full((nsamples,), 0.0))
            # use following expression to avoid overflow issues when
            # computing oveflow
            can_derivs.append(
                self._bkd.full(
                    (nsamples,),
                    self._bkd.exp(
                        self._bkd.gammaln(self._bkd.asarray(_order + 1))
                        - 0.5
                        * self._bkd.sum(self._bkd.log(b[: _order + 1] ** 2))
                    ),
                )
            )
            for jj in range(_order + 1, nindices):
                can_derivs.append(
                    (
                        (can_samples - a[jj - 1]) * can_derivs[jj - 1]
                        - b[jj - 1] * can_derivs[jj - 2]
                        + _order * vals[jj - 1]
                    )
                    / b[jj]
                )
            derivs = self._trans.derivatives_from_canonical(
                self._bkd.stack(can_derivs, axis=1), _order
            )
            vals = can_derivs
            result.append(derivs)

        if return_all:
            return self._bkd.hstack(result)
        return result[-1]

    def _canonical_gauss_quadrature_rule(self, npoints):
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
        return self._canonical_gauss_quadrature_rule_from_rcoefs(
            npoints, self._rcoefs
        )

    def _canonical_gauss_quadrature_rule_from_rcoefs(self, npoints, rcoefs):
        a = rcoefs[:, 0]
        b = rcoefs[:, 1]

        # Form Jacobi matrix
        J = (
            self._bkd.diag(a[:npoints], 0)
            + self._bkd.diag(b[1:npoints], 1)
            + self._bkd.diag(b[1:npoints], -1)
        )
        x, eigvecs = self._bkd.eigh(J)
        if self._prob_meas:
            w = b[0] * eigvecs[0, :] ** 2
        else:
            w = self(x[None, :])[:, :npoints]
            w = 1.0 / self._bkd.sum(w**2, axis=1)
        # w[~self._bkd.isfinite(w)] = 0.
        return x[None, :], w[:, None]

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
        can_quad_x, can_quad_w = self._canonical_gauss_quadrature_rule(npoints)
        return self._trans.map_from_canonical(can_quad_x), can_quad_w

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

        abc = self._bkd.zeros((self._ncoefs(), 3))
        abc[:, 0] = 1.0 / self._rcoefs[:, 1]
        abc[1:, 1] = self._rcoefs[:-1, 0] / self._rcoefs[1:, 1]
        abc[1:, 2] = self._rcoefs[:-1, 1] / self._rcoefs[1:, 1]
        return abc


class JacobiPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, alpha, beta, trans=None, backend=None):
        super().__init__(trans, backend)
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

    def _canonical_gauss_lobatto_quadrature_rule(self, npoints):
        if npoints < 3:
            raise ValueError("to few points requested")
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

        N = npoints - 2
        rcoefs = self._bkd.copy(self._rcoefs[:npoints])
        # correct first b coefficient to undo sqrt in jacobi_recurrence
        # and undo setting it to 1 if self._prob_meas is True
        rcoefs[0, 1] = math.exp(
            (self._alpha + self._beta + 1.0) * math.log(2.0)
            + self._bkd.gammaln(self._bkd.asarray(self._alpha) + 1.0)
            + self._bkd.gammaln(self._bkd.asarray(self._beta) + 1.0)
            - self._bkd.gammaln(self._bkd.asarray(self._alpha + self._beta)
                                + 2.0)
        )
        rcoefs[npoints - 1, 0] = (self._alpha - self._beta) / (
            2 * N + self._alpha + self._beta + 2
        )
        rcoefs[npoints - 1, 1] = math.sqrt(
            4
            * (N + self._alpha + 1)
            * (N + self._beta + 1)
            * (N + self._alpha + self._beta + 1)
            / (
                (2 * N + self._alpha + self._beta + 1)
                * (2 * N + self._alpha + self._beta + 2) ** 2
            )
        )
        return self._canonical_gauss_quadrature_rule_from_rcoefs(
            npoints, rcoefs
        )

    def gauss_lobatto_quadrature_rule(self, npoints):
        can_quad_x, can_quad_w = self._canonical_gauss_lobatto_quadrature_rule(
            npoints
        )
        return self._trans.map_from_canonical(can_quad_x), can_quad_w


class LegendrePolynomial1D(JacobiPolynomial1D):
    def __init__(self, trans=None, backend=None):
        super().__init__(0.0, 0.0, trans=trans, backend=backend)

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class HermitePolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, rho=0.0, prob_meas=True, trans=None, backend=None):
        super().__init__(trans, backend)
        self._prob_meas = prob_meas
        self._rho = rho

    def _get_recursion_coefficients(self, ncoefs):
        return hermite_recurrence(
            ncoefs, rho=self._rho, probability=self._prob_meas
        )


class KrawtchoukPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, n, p, raisewarn=True, trans=None, backend=None):
        super().__init__(trans, backend)
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
    def __init__(
        self, N, alpha, beta, raisewarn=True, trans=None, backend=None
    ):
        super().__init__(trans, backend)
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
            self.__class__.__name__,
            self._N,
            self._alpha,
            self._beta,
            self.nterms(),
        )


class CharlierPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, mu, trans=None, backend=None):
        super().__init__(trans, backend)
        self._mu = mu

    def _get_recursion_coefficients(self, ncoefs):
        return charlier_recurrence(ncoefs, self._mu)


class DiscreteChebyshevPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, N, trans=None, backend=None):
        super().__init__(trans, backend)
        self._N = N

    def _get_recursion_coefficients(self, ncoefs):
        return discrete_chebyshev_recurrence(ncoefs, self._N)


class LaguerrePolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, rho, trans=None, backend=None):
        super().__init__(trans, backend)
        self._rho = rho

    def _get_recursion_coefficients(self, ncoefs):
        return laguerre_recurrence(ncoefs, self._rho)


class Chebyshev1stKindPolynomial1D(JacobiPolynomial1D):
    def __init__(self, trans=None, backend=None):
        # TODO: not sure if I have the naming of first and second correct.
        super().__init__(-0.5, -0.5, trans=trans, backend=backend)
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
        return quad_x, quad_w * math.pi

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class Chebyshev2ndKindPolynomial1D(JacobiPolynomial1D):
    def __init__(self, trans=None, backend=None):
        # TODO: not sure if I have the naming of first and second correct.
        super().__init__(0.5, 0.5, trans, backend=backend)
        self._prob_meas = True

    def gauss_quadrature_rule(self, npoints):
        quad_x, quad_w = super().gauss_quadrature_rule(npoints)
        return quad_x, quad_w * math.pi / 2

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class DiscreteNumericOrthonormalPolynomial1D(OrthonormalPolynomial1D):
    def __init__(
        self, samples, weights, ortho_tol=1e-8, prob_tol=0, backend=None
    ):
        """Compure recurrence coefficients from samples."""
        super().__init__(None, backend)
        self._ortho_tol = ortho_tol
        self._prob_tol = prob_tol
        self._check_samples_weights(samples, weights)
        self._samples = samples
        self._weights = weights

    def _check_samples_weights(self, samples, weights):
        if samples.ndim != 2 and samples.shape[0] != 1:
            raise ValueError("weights must be 2D column vector")
        if weights.ndim != 2 and weights.shape[1] != 1:
            raise ValueError("weights must be 2D column vector")
        if abs(weights.sum() - 1) > max(self._prob_tol, 4e-15):
            msg = f"weights sum is {weights.sum()} and so does not define "
            msg += f"a probability measure. Diff : {weights.sum()-1}"
            raise ValueError(msg)
        if weights.shape[0] != samples.shape[1]:
            raise ValueError("weights and samples are inconsistent")

    def _lanczos(self, nterms):
        nnodes = self._samples.shape[1]
        if nterms > nnodes:
            raise ValueError("Too many coefficients requested")
        alpha = self._bkd.zeros((nterms,))
        beta = self._bkd.zeros((nterms,))
        vec = self._bkd.zeros((nnodes + 1,))
        vec[0] = 1
        qii = self._bkd.zeros((nnodes + 1, nnodes + 1))
        qii[:, 0] = vec
        sqrt_w = self._bkd.sqrt(self._weights[:, 0])
        northogonalization_steps = 2
        for ii in range(nterms):
            z = self._bkd.hstack(
                [
                    vec[0] + self._bkd.sum(sqrt_w * vec[1 : nnodes + 1]),
                    sqrt_w * vec[0] + self._samples[0] * vec[1 : nnodes + 1],
                ]
            )

            if ii > 0:
                alpha[ii - 1] = vec @ z

            for jj in range(northogonalization_steps):
                z -= qii[:, : ii + 1] @ (qii[:, : ii + 1].T @ z)

            if ii < nterms:
                znorm = self._bkd.norm(z)
                # beta[ii] = znorm**2 assume we want probability measure so
                # no need to square here then take sqrt later
                beta[ii] = znorm
                vec = z / znorm
                qii[:, ii + 1] = vec
        return self._bkd.stack((alpha, beta), axis=1)

    def _check_orthonormality(self, rcoefs):
        poly_vals = _evaluate_orthonormal_polynomial_1d(
            rcoefs, self._bkd, self._samples
        )
        error = self._bkd.max(
            self._bkd.abs(
                (poly_vals.T * self._weights[:, 0]) @ poly_vals
                - self._bkd.eye(poly_vals.shape[1])
            )
        )
        if error > self._ortho_tol:
            msg = "basis created is ill conditioned. "
            msg += "Max error: {0}. Max terms: {1}, {2}".format(
                error,
                self._samples.shape[1],
                f"Terms requested {rcoefs.shape[0]}",
            )
            raise ValueError(msg)
        return rcoefs

    def _get_recursion_coefficients(self, ncoefs):
        rcoefs = self._lanczos(ncoefs)
        self._check_orthonormality(rcoefs)
        return rcoefs


class PredictorCorrector:
    def __init__(self, backend=None):
        self._integrator = None
        self._measure = None
        self._idx = None
        self._ab = None
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend

    def set_measure(self, measure):
        self._measure = measure

    def set_integrator(self, integrator):
        self._integrator = integrator

    def _integrand_0(self, x):
        return self._measure(x)

    def _integrand_1(self, x):
        pvals = _evaluate_orthonormal_polynomial_1d(self._ab, self._bkd, x)
        return (
            self._measure(x)[:, 0]
            * pvals[:, self._idx]
            * pvals[:, self._idx - 1]
        )[:, None]

    def _integrand_2(self, x):
        pvals = _evaluate_orthonormal_polynomial_1d(self._ab, self._bkd, x)
        return (self._measure(x)[:, 0] * pvals[:, self._idx] ** 2)[:, None]

    def __call__(self, nterms):
        ab = self._bkd.zeros((nterms + 1, 2))

        self._integrator.set_integrand(self._integrand_0)
        ab[0, 1] = math.sqrt(self._integrator())

        for idx in range(1, nterms + 1):
            # predict
            ab[idx, 1] = ab[idx - 1, 1]
            if idx > 1:
                ab[idx - 1, 0] = ab[idx - 2, 0]
            else:
                ab[idx - 1, 0] = 0

            self._idx = idx
            self._ab = ab[: idx + 1, :]
            self._integrator.set_integrand(self._integrand_1)
            G_idx_idxm1 = self._integrator()
            ab[idx - 1, 0] += ab[idx - 1, 1] * G_idx_idxm1
            self._integrator.set_integrand(self._integrand_2)
            G_idx_idx = self._integrator()
            ab[idx, 1] *= math.sqrt(G_idx_idx)

        return ab[:nterms, :]


class ContinuousNumericOrthonormalPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, marginal, integrator=None, trans=None, backend=None):
        """Compure recurrence coefficients from a known PDF
        using predictor corrector method."""
        super().__init__(trans, backend)
        # Get version var.pdf without error checking which runs much faster
        self._pdf, self._loc, self._scale, self._can_lb, self._can_ub = (
            self._parse_marginal(marginal)
        )
        self._marginal = marginal
        self._rcoefs_gen = PredictorCorrector(backend=self._bkd)
        if integrator is None:
            integrator = ScipyUnivariateIntegrator(backend=self._bkd)
        if not isinstance(integrator, UnivariateIntegrator):
            raise ValueError(
                "integrator must be an instance of UnivariateIntegrator"
            )
        if not integrator._bkd.bkd_equal(integrator._bkd, self._bkd):
            raise ValueError("integrator._bkd does not match self._bkd")
        integrator.set_bounds([self._can_lb, self._can_ub])
        self._rcoefs_gen.set_integrator(integrator)
        self._rcoefs_gen.set_measure(self._canonical_pdf)

    def _parse_marginal(self, marginal):
        pdf = get_pdf(marginal)
        loc, scale = transform_scale_parameters(marginal)
        lb, ub = marginal.interval(1)
        if is_bounded_continuous_variable(marginal):
            can_lb, can_ub = -1, 1
        elif is_continuous_variable(marginal):
            can_lb = (lb - loc) / scale
            can_ub = (ub - loc) / scale
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
        # pdf is from scipy so x must be converted to one d array
        return self._bkd.asarray(
            self._pdf(x[0] * self._scale + self._loc) * self._scale
        )[:, None]

    def _get_recursion_coefficients(self, ncoefs):
        return self._rcoefs_gen(ncoefs)


def setup_univariate_orthogonal_polynomial_from_marginal(
    marginal, opts={}, backend=None
):
    var_name, scales, shapes = get_distribution_info(marginal)

    trans = AffineMarginalTransform(
        marginal, enforce_bounds=True, backend=backend
    )

    # Askey polynomials that do not need scale or shape
    simple_askey_polys = {
        "uniform": LegendrePolynomial1D,
        "norm": HermitePolynomial1D,
    }
    if var_name in simple_askey_polys:
        return simple_askey_polys[var_name](trans=trans, backend=backend)

    # Other Askey polynomials
    # Ignore binom (KrawtchoukPolynomial1D) and hypergeom (HahnPolynomial1D)
    # because these polynomials are not defined on [-1, 1] which is assumed
    # to be the canonical domain of bounded marginals
    if var_name == "beta":
        return JacobiPolynomial1D(
            shapes["b"] - 1, shapes["a"] - 1, trans=trans, backend=backend
        )

    if var_name == "poisson":
        return CharlierPolynomial1D(shapes["mu"], trans=trans, backend=backend)

    # Other continuous marginals
    if is_continuous_variable(marginal) and var_name != "continuous_rv_sample":
        return ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            opts.get("integrator", None),
            trans=trans,
            backend=backend,
        )

    # other discrete marginals
    if opts is None:
        opts = {}
    if hasattr(shapes, "xk"):
        xk, pk = shapes["xk"], shapes["pk"]
    else:
        xk, pk = get_probability_masses(marginal, opts.get("ptol", 1e-8))

    loc, scale = transform_scale_parameters(marginal)
    xk = (xk - loc) / scale
    return DiscreteNumericOrthonormalPolynomial1D(
        backend.asarray(xk[None, :]),
        backend.asarray(pk[:, None]),
        opts.get("otol", 1e-8),
        opts.get("ptol", 1e-8),
        backend=backend,
    )


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

    def __repr__(self):
        return "{0}(poly={1}, bkd={2})".format(
            self.__class__.__name__, self._poly, self._bkd.__name__
        )


class GaussLegendreQuadratureRule(GaussQuadratureRule):
    """
    Gauss Quadrature rule for Lebesque integration
    (not uniform probability measure)
    """

    def __init__(self, bounds, backend=None, store=False):
        self._bounds = bounds
        marginal = stats.uniform(bounds[0], bounds[1] - bounds[0])
        super().__init__(marginal, opts=None, backend=backend, store=store)

    def _quad_rule(self, nnodes):
        if self._poly.nterms() < nnodes:
            self._poly.set_nterms(nnodes)
        quad_x, quad_w = self._poly.gauss_quadrature_rule(nnodes)
        return quad_x, quad_w * (self._bounds[1] - self._bounds[0])


class Chebyshev1stKindGaussLobattoQuadratureRule(GaussQuadratureRule):
    """Integrates functions on [a, b] with weight 1/sqrt(1-x^2)."""

    def __init__(self, bounds, backend=None, store=False):
        UnivariateQuadratureRule.__init__(self, backend, store)
        self._bounds = bounds
        loc = sum(bounds) / 2
        scale = bounds[1] - loc
        self._trans = UnivariateAffineTransform(
            loc, scale, enforce_bounds=False, backend=self._bkd
        )
        self._poly = Chebyshev1stKindPolynomial1D(
            trans=self._trans, backend=self._bkd
        )

    def _quad_rule(self, nnodes):
        self._poly.set_nterms(nnodes)
        return self._poly.gauss_lobatto_quadrature_rule(nnodes)


class Chebyshev2ndKindGaussLobattoQuadratureRule(GaussQuadratureRule):
    """Integrates functions on [a, b] with weight sqrt(1-x^2)."""

    def __init__(self, bounds, backend=None, store=False):
        UnivariateQuadratureRule.__init__(self, backend, store)
        self._bounds = bounds
        loc = sum(bounds) / 2
        scale = bounds[1] - loc
        self._trans = UnivariateAffineTransform(
            loc, scale, enforce_bounds=False, backend=self._bkd
        )
        self._poly = Chebyshev2ndKindPolynomial1D(
            trans=self._trans, backend=self._bkd
        )

    def _quad_rule(self, nnodes):
        self._poly.set_nterms(nnodes)
        return self._poly.gauss_lobatto_quadrature_rule(nnodes)


class UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis(
    UnivariateBarycentricLagrangeBasis
):
    # TODO: not sure if I have the naming of first and second correct.
    def __init__(self, bounds, nterms=None, backend=NumpyLinAlgMixin):
        super().__init__(
            Chebyshev1stKindGaussLobattoQuadratureRule(
                bounds, backend=backend
            ), nterms
        )

    def _set_barycentric_weights(self):
        self._bary_weights = (-1.0) ** (self._bkd.arange(self.nterms()) % 2)
        self._bary_weights[0] /= 2
        self._bary_weights[-1] /= 2


class AffineMarginalTransform(UnivariateAffineTransform):
    def __init__(self, marginal, enforce_bounds=False, backend=None):
        super().__init__(
            *transform_scale_parameters(marginal), enforce_bounds, backend
        )
        self._marginal = marginal

    def _check_bounds(self, user_samples):
        if not is_bounded_continuous_variable(self._marginal):
            return
        super()._check_bounds(user_samples)


# Note may need to change fourierbasis1d and trigonometricpolynomial1D to return basis that is nested. i.e. trig basis returns const + sin and cos for k=1 then sin and cos for k=2 etc. Similarly for fourier return c_0 then c_{-1} c{1} c{-2} c{2} etc. This will make them consistent with other pyapprox bases but not consistent with typical math formulation. Perhaps allow user to request either ordering use pyapprox by default. Also consider moving fourier and trig basies to univariateinterpolating bases#
class TrigonometricPolynomial1D(UnivariateBasis):
    r"""
    :math:`p(x) = a_0 + \sum_{k=1}^K a_k \cos(kx) + b_k \sin(kx)`

    .. math::
        a_0 &= 1\(2\pi) \int_{-\pi}^\pi f(x)dx\\
        a_k &= 1\pi \int_{-\pi}^\pi f(x)cos(kx)dx\\
        b_k &= 1\pi \int_{-\pi}^\pi f(x)cos(kx)dx
    """

    def __init__(self, bounds, backend=None):
        super().__init__(None, backend)
        self._bounds = None
        self._trans = None
        self.set_bounds(bounds)
        self._half_indices = None
        self._jacobian_implemented = False
        self._hessian_implemented = False

    def set_bounds(self, bounds):
        # canonical domain is [-pi, pi]
        self._bounds = bounds
        loc = sum(bounds) / 2
        scale = (bounds[1] - bounds[0]) / (2 * math.pi)
        self._trans = UnivariateAffineTransform(
            loc, scale, enforce_bounds=False, backend=self._bkd
        )

    def set_nterms(self, nterms):
        if nterms % 2 != 1:
            raise ValueError("nterms bust be an odd number")
        # half_indices is k in a_0 + \sum_{k=1}^K a_k \cos(kx) + b_k \sin(kx)
        self._half_indices = self._bkd.arange(1, (nterms - 1) // 2 + 1)[
            None, :
        ]

    def nterms(self):
        return self._half_indices.shape[1] * 2 + 1

    def _values(self, samples):
        can_samples = self._trans.map_to_canonical(samples)
        return self._bkd.hstack(
            (
                self._bkd.ones((can_samples.shape[1], 1)),
                self._bkd.cos(can_samples.T * self._half_indices),
                self._bkd.sin(can_samples.T * self._half_indices),
            )
        )


class FourierBasis1D(UnivariateBasis):
    # p(x) = sum_{k=-K}^K c_k e^{ikx}
    def __init__(self, bounds, inverse=True, backend=None):
        super().__init__(None, backend)
        self._bounds = None
        self._trans = None
        self._Kmax = None
        self._jacobian_implemented = False
        self._hessian_implemented = False
        self.set_bounds(bounds)
        if inverse:
            # compute basis to evaluate function from fourier coefs
            self._const = 1j
        else:
            # compute basis needed to compute fourier coefs with quadrature
            self._const = -1j

    def set_bounds(self, bounds):
        # canonical domain is [-pi, pi]
        self._bounds = bounds
        loc = sum(bounds) / 2
        scale = (bounds[1] - bounds[0]) / (2 * math.pi)
        self._trans = UnivariateAffineTransform(
            loc, scale, enforce_bounds=False, backend=self._bkd
        )

    def set_nterms(self, nterms):
        if nterms % 2 != 1:
            raise ValueError("nterms bust be an odd number")
        # half_indices is k in a_0 + \sum_{k=1}^K a_k \cos(kx) + b_k \sin(kx)
        self._Kmax = (nterms - 1) // 2

    def nterms(self):
        return self._Kmax * 2 + 1

    def _values(self, samples):
        can_samples = self._trans.map_to_canonical(samples)
        return self._bkd.exp(
            self._const
            * can_samples.T
            * self._bkd.arange(-self._Kmax, self._Kmax + 1)[None, :]
        )


def setup_lagrange_basis(
    basis_type,
    quadrature_rule=None,
    bounds=None,
    backend=NumpyLinAlgMixin,
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
        "chebyhsev1":
        UnivariateChebyhsev1stKindGaussLobattoBarycentricLagrangeBasis
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

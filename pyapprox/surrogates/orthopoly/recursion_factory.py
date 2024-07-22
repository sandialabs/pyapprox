from warnings import warn
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence, hermite_recurrence, krawtchouk_recurrence,
    hahn_recurrence, charlier_recurrence)
from pyapprox.surrogates.orthopoly.numeric_orthonormal_recursions import (
    predictor_corrector, get_function_independent_vars_recursion_coefficients,
    get_product_independent_vars_recursion_coefficients, lanczos)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d, numba_gammaln)
from pyapprox.variables.marginals import (
    get_distribution_info, is_continuous_variable,
    transform_scale_parameters, is_bounded_continuous_variable,
    is_bounded_discrete_variable, get_probability_masses, get_pdf)


# There is a one-to-one correspondence in these two lists
askey_poly_names = ["legendre", "hermite", "jacobi", "charlier",
                    "krawtchouk", "hahn"][:-2]
askey_variable_names = ["uniform", "norm", "beta", "poisson",
                        "binom", "hypergeom"][:-2]
# The Krawtchouk and Hahn polynomials are not defined
# on the canonical domain [-1,1]. Must use numeric recursion
# to generate polynomials on [-1,1] for consistency, so remove from Askey list


def get_askey_recursion_coefficients(poly_name, opts, num_coefs):
    if poly_name not in askey_poly_names:
        raise ValueError(f"poly_name {poly_name} not in {askey_poly_names}")

    if poly_name == "legendre":
        return jacobi_recurrence(num_coefs, alpha=0, beta=0, probability=True)

    if poly_name == "jacobi":
        return jacobi_recurrence(
            num_coefs, alpha=opts["alpha_poly"], beta=opts["beta_poly"],
            probability=True)

    if poly_name == "hermite":
        return hermite_recurrence(num_coefs, rho=0., probability=True)

    if poly_name == "krawtchouk":
        msg = "Although bounded the Krawtchouk polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        num_coefs = min(num_coefs, opts["n"])
        return krawtchouk_recurrence(num_coefs, opts["n"], opts["p"])

    if poly_name == "hahn":
        msg = "Although bounded the Hahn polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        num_coefs = min(num_coefs, opts["N"])
        return hahn_recurrence(
            num_coefs, opts["N"], opts["alpha_poly"], opts["beta_poly"])

    if poly_name == "charlier":
        return charlier_recurrence(num_coefs, opts["mu"])


def get_askey_recursion_coefficients_from_variable(var, num_coefs):
    var_name, scales, shapes = get_distribution_info(var)

    if var_name not in askey_variable_names:
        msg = f"Variable name {var_name} not in {askey_variable_names}"
        raise ValueError(msg)

    # Askey polynomials associated with continuous variables
    if var_name == "uniform":
        poly_name, opts = "legendre", {}
    elif var_name == "beta":
        poly_name = "jacobi"
        opts = {"alpha_poly": shapes["b"]-1, "beta_poly": shapes["a"]-1}
    elif var_name == "norm":
        poly_name, opts = "hermite", {}
        opts

    # Askey polynomials associated with discrete variables
    elif var_name == "binom":
        poly_name, opts = "krawtchouk", shapes
    elif var_name == "hypergeom":
        # note xk = np.arange(max(0, N-M+n), min(n, N)+1, dtype=float)
        poly_name = "hahn"
        M, n, N = [shapes[key] for key in ["M", "n", "N"]]
        opts = {"alpha_poly": -(n+1), "beta_poly": -M-1+n, "N": N}
    elif var_name == "poisson":
        poly_name, opts = "charlier", shapes

    return get_askey_recursion_coefficients(poly_name, opts, num_coefs)


def get_numerically_generated_recursion_coefficients_from_samples(
        xk, pk, num_coefs, orthonormality_tol, truncated_probability_tol=0):

    if num_coefs > xk.shape[0]:
        msg = "Number of coefs requested is larger than number of "
        msg += "probability masses"
        raise ValueError(msg)
    recursion_coeffs = lanczos(xk, pk, num_coefs, truncated_probability_tol)

    p = evaluate_orthonormal_polynomial_1d(
        np.asarray(xk, dtype=float), num_coefs-1, recursion_coeffs)
    error = np.absolute((p.T*pk).dot(p)-np.eye(num_coefs)).max()
    if error > orthonormality_tol:
        msg = "basis created is ill conditioned. "
        msg += f"Max error: {error}. Max terms: {xk.shape[0]}, "
        msg += f"Terms requested: {num_coefs}"
        raise ValueError(msg)
    return recursion_coeffs


def predictor_corrector_known_pdf(nterms, lb, ub, pdf, opts={}):
    if "quad_options" not in opts:
        tol = opts.get("orthonormality_tol", 1e-8)
        quad_options = {'epsrel': tol, 'epsabs': tol}
    else:
        quad_options = opts["quad_options"]

    return predictor_corrector(nterms, pdf, lb, ub, quad_options)


def get_recursion_coefficients_from_variable(var, num_coefs, opts):
    """
    Generate polynomial recursion coefficients by inspecting a random variable.
    """
    var_name, _, shapes = get_distribution_info(var)
    if var_name == "continuous_monomial":
        return None

    loc, scale = transform_scale_parameters(var)

    if var_name == "rv_function_indpndt_vars":
        shapes["loc"] = loc
        shapes["scale"] = scale
        return get_function_independent_vars_recursion_coefficients(
            shapes, num_coefs)

    if var_name == "rv_product_indpndt_vars":
        shapes["loc"] = loc
        shapes["scale"] = scale
        return get_product_independent_vars_recursion_coefficients(
            shapes, num_coefs)

    if (var_name in askey_variable_names and
            opts.get("numeric", False) is False):
        return get_askey_recursion_coefficients_from_variable(var, num_coefs)

    orthonormality_tol = opts.get("orthonormality_tol", 1e-8)
    truncated_probability_tol = opts.get("truncated_probability_tol", 0)

    if (not is_continuous_variable(var) or
            var.dist.name == "continuous_rv_sample"):
        if hasattr(shapes, "xk"):
            xk, pk = shapes["xk"], shapes["pk"]
        else:
            xk, pk = get_probability_masses(
                var, truncated_probability_tol)
        xk = (xk-loc)/scale

        return get_numerically_generated_recursion_coefficients_from_samples(
            xk, pk, num_coefs, orthonormality_tol, truncated_probability_tol)

    # integration performed in canonical domain so need to map back to
    # domain of pdf
    lb, ub = var.interval(1)

    # Get version var.pdf without error checking which runs much faster
    pdf = get_pdf(var)

    def canonical_pdf(x):
        # print(x, lb, ub, x*scale+loc)
        # print(var.pdf(x*scale+loc)*scale)
        # assert np.all(x*scale+loc >= lb) and np.all(x*scale+loc <= ub)
        return pdf(x*scale+loc)*scale
        # return var.pdf(x*scale+loc)*scale

    if (is_bounded_continuous_variable(var) or
            is_bounded_discrete_variable(var)):
        can_lb, can_ub = -1, 1
    elif is_continuous_variable(var):
        can_lb = (lb-loc)/scale
        can_ub = (ub-loc)/scale

    return predictor_corrector_known_pdf(
        num_coefs, can_lb, can_ub, canonical_pdf, opts)


from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class OrthonormalPolynomial1D(ABC):
    def __init__(self, backend):
        self._rcoefs = None
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend

    def _ncoefs(self):
        if self._rcoefs is None:
            raise ValueError("recrusion_coefs have not been set")
        return self._rcoefs.shape[1]

    @abstractmethod
    def _get_recursion_coefficients(self, ncoefs):
        raise NotImplementedError

    def set_recursion_coefficients(self, ncoefs):
        """Compute and set the recursion coefficients of the polynomial."""
        if self._rcoefs is None or self._ncoefs() < ncoefs:
            self._rcoefs = self._bkd._la_array(
                self._get_recursion_coefficients(ncoefs))

    def _opts_equal(self, other):
        return True

    def __eq__(self, other):
        return (self.__class__.__name__ == self.__class__.__name__ and
                self._opts_equal(other) and
                self._rcoefs == other._recursion_coefs)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def _values(self, samples, nmax):
        if self._rcoefs is None:
            raise ValueError("Must set recursion coefficients.")
        if nmax >= self._rcoefs.shape[0]:
            raise ValueError(
                "The number of polynomial terms requested {0} {1}".format(
                    nmax, "exceeds number of rcoefs {0}".format(
                        self._rcoefs.shape[0])))

        # must be initialized to zero
        vals = self._bkd._la_full((samples.shape[0], nmax+1), 0.)

        vals[:, 0] = 1.0/self._rcoefs[0, 1]

        if nmax > 0:
            vals[:, 1] = 1/self._rcoefs[1, 1] * (
                (samples - self._rcoefs[0, 0])*vals[:, 0])

        for jj in range(2, nmax+1):
            vals[:, jj] = 1.0/self._rcoefs[jj, 1]*(
                (samples-self._rcoefs[jj-1, 0])*vals[:, jj-1] -
                self._rcoefs[jj-1, 1]*vals[:, jj-2])
        return vals

    def derivatives(self, samples, nmax, order, return_all=False):
        """
        Compute the first n dervivatives of the polynomial.
        """
        if order < 2:
            raise ValueError(
                "derivative order {0} must be greater than zero".format(
                    order))
        vals = self._values(samples, nmax)
        nsamples = samples.shape[0]
        nindices = nmax+1
        a = self._rcoefs[:, 0]
        b = self._rcoefs[:, 1]

        result = self._bkd._la_empty((nsamples, nindices*(order+1)))
        result[:, :nindices] = vals
        for _order in range(1, order):
            derivs = self._la_full((nsamples, nindices), 0.)
        for jj in range(_order, nindices):
            if (jj == _order):
                # use following expression to avoid overflow issues when
                # computing oveflow
                derivs[:, jj] = self._la_exp(
                    numba_gammaln(_order+1)-0.5*self._la_sum(
                        self._la_log(b[:jj+1]**2)))
            else:

                derivs[:, jj] = (
                    (samples-a[jj-1])*derivs[:, jj-1]-b[jj-1]*derivs[:, jj-2] +
                    _order*vals[:, jj-1])
                derivs[:, jj] *= 1.0/b[jj]
        vals = derivs
        result[:, _order*nindices:(_order+1)*nindices] = derivs

        if return_all:
            return result
        return result[:, order*nindices:]

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
        if npoints > self._rcoefs.shape[0]:
            raise ValueError("Too many terms requested")

        a = self._rcoefs[:, 0]
        b = self._rcoefs[:, 1]

        # Form Jacobi matrix
        J = (self._bkd._la_diag(a[:npoints], 0) +
             self._bkd._la_diag(b[1:npoints], 1) +
             self._bkd._la_diag(b[1:npoints], -1))

        x, eigvecs = self._bkd._la_eigh(J)
        w = b[0]*eigvecs[0, :]**2
        w[~self._bkd._la_isfinite(w)] = 0.
        return x, w

    def __call__(self, samples, nmax):
        return self._values(samples, nmax)


class JacobiPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, alpha, beta, backend=None):
        super().__init__(backend)
        self._alpha = alpha
        self._beta = beta

    def _get_recursion_coefficients(self, ncoefs):
        return jacobi_recurrence(
            ncoefs, alpha=self._alpha, beta=self._beta,
            probability=True)

    def _opts_equal(self, other):
        return (self._alpha == other._alpha and self._beta == other._beta)

    def __repr__(self):
        return "{0}(alpha={1}, beta={2})".format(
            self.__class__.__name__, self._alpha, self._beta)


class LegendrePolynomial1D(JacobiPolynomial1D):
    def __init__(self, backend=None):
        super().__init__(0., 0., backend=backend)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class HermitePolynomial1D(OrthonormalPolynomial1D):
    def _get_recursion_coefficients(self, ncoefs):
        return hermite_recurrence(ncoefs, rho=0., probability=True)


class KrawtchoukPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, n, p, backend=None):
        super().__init__(backend)
        self._n = n
        self._p = p

    def _get_recursion_coefficients(self, ncoefs):
        msg = "Although bounded the Krawtchouk polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        ncoefs = min(ncoefs, self._n)
        return krawtchouk_recurrence(ncoefs, self._n, self._p)

    def _opts_equal(self, other):
        return (self._n == other._n and self._p == other._p)

    def __repr__(self):
        return "{0}(n={1}, p={2})".format(
            self.__class__.__name__, self._n, self._p)


class HahnPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, N, alpha, beta, backend=None):
        super().__init__(backend)
        self._N = N
        self._alpha = alpha
        self._beta = beta

    def _get_recursion_coefficients(self, ncoefs):
        msg = "Although bounded the Hahn polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        ncoefs = min(ncoefs, self._N)
        return hahn_recurrence(ncoefs, self._N, self._alpha, self._beta)

    def _opts_equal(self, other):
        return (self._N == other._N and self._alpha == other._alpha
                and self._beta == other._beta)

    def __repr__(self):
        return "{0}(N={1}, alpha={2}, beta={3})".format(
            self.__class__.__name__, self._N, self._alpha, self._beta)


class CharlierPolynomial1D(OrthonormalPolynomial1D):
    def __init__(self, mu, backend=None):
        super().__init__(backend)
        self._mu = mu

    def _get_recursion_coefficients(self, ncoefs):
        return charlier_recurrence(ncoefs, self._mu)


def get_askey_polynomial_from_variable(var):
    var_name, scales, shapes = get_distribution_info(var)

    # Askey polynomials associated with continuous variables
    if var_name == "uniform":
        return LegendrePolynomial1D()
    if var_name == "beta":
        return JacobiPolynomial1D(shapes["b"]-1, shapes["a"]-1)
    if var_name == "norm":
        return HermitePolynomial1D()

    # Askey polynomials associated with discrete variables
    if var_name == "binom":
        return KrawtchoukPolynomial1D(shapes["n"], shapes["p"])
    if var_name == "hypergeom":
        # note xk = np.arange(max(0, N-M+n), min(n, N)+1, dtype=float)
        M, n, N = [shapes[key] for key in ["M", "n", "N"]]
        return HahnPolynomial1D(N, -(n+1), -M-1+n)
    if var_name == "poisson":
        return CharlierPolynomial1D(shapes["mu"])

    msg = f"Variable name {var_name} not in {askey_variable_names}"
    raise ValueError(msg)

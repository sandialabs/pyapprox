import math
from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple

from scipy.stats._distn_infrastructure import rv_sample, rv_continuous
from scipy.stats import _continuous_distns, _discrete_distns
from scipy import stats
from scipy import interpolate
import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.newton import NewtonResidual, NewtonSolver


class Marginal(ABC):
    # This class implemented to allow for autograd
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend

    def _check_samples(self, samples: Array) -> Array:
        if samples.ndim != 1:
            raise ValueError(
                "samples must be 1d array but had shape{0}".format(
                    samples.shape
                )
            )
        return samples

    def _check_values(self, vals: Array) -> Array:
        if vals.ndim != 1:
            raise ValueError(
                "vals must be a 1d array but had shape{0}".format(vals.shape)
            )
        return vals

    @abstractmethod
    def _pdf(self, samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _logpdf(self, samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _cdf(self, samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _ppf(self, usamples: Array) -> Array:
        raise NotImplementedError

    def _pdf_jacobian(self, samples: Array) -> Array:
        raise NotImplementedError

    def _logpdf_jacobian(self, samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _rvs(self, nsamples: int) -> Array:
        raise NotImplementedError

    def pdf(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._pdf(samples)
        vals = self._check_values(vals)
        return vals

    def logpdf(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._logpdf(samples)
        vals = self._check_values(vals)
        return vals

    def cdf(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._cdf(samples)
        vals = self._check_values(vals)
        return vals

    def ppf(self, usamples: Array) -> Array:
        self._check_samples(usamples)
        vals = self._ppf(usamples)
        vals = self._check_values(vals)
        return vals

    def _check_jacobian(self, samples: Array, jac: Array) -> Array:
        if jac.shape != (1, samples.shape[1]):
            raise ValueError("jacobian must be a 2D row vector")
        return jac

    def pdf_jacobian(self, samples: Array) -> Array:
        if not self.pdf_jacobian_implemented():
            raise NotImplementedError("pdf_jacobian is not implemented")
        self._check_samples(samples)
        jac = self._pdf_jacobian(samples)
        return self._check_jacobian(samples, jac)

    def logpdf_jacobian(self, samples: Array) -> Array:
        if not self.logpdf_jacobian_implemented():
            raise NotImplementedError("logpdf_jacobian is not implemented")
        self._check_samples(samples)
        jac = self._logpdf_jacobian(samples)
        return self._check_jacobian(samples, jac)

    def pdf_jacobian_implemented(self) -> bool:
        return False

    def logpdf_jacobian_implemented(self) -> bool:
        return False

    def rvs(self, nsamples: int) -> Array:
        return self._check_samples(self._rvs(nsamples))

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    @abstractmethod
    def __eq__(self, other: "Marginal") -> bool:
        raise NotImplementedError


class ScipyMarginal(Marginal):
    def __init__(
        self,
        scipy_rv: stats._distn_infrastructure.rv_frozen,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._scipy_rv = scipy_rv
        super().__init__(backend)
        self._check_marginal(scipy_rv)
        self._name, self._scales, self._shapes = self._get_distribution_info()

    @abstractmethod
    def _check_marginal(self, marginal: stats._distn_infrastructure.rv_frozen):
        raise NotImplementedError

    def _pdf(self, samples: Array) -> Array:
        return self._scipy_rv.pdf(samples)

    def _logpdf(self, samples: Array) -> Array:
        return self._scipy_rv.logpdf(samples)

    def _cdf(self, samples: Array) -> Array:
        return self._scipy_rv.cdf(samples)

    def _ppf(self, usamples: Array) -> Array:
        return self._scipy_rv.ppf(usamples)

    def _rvs(self, nsamples: int) -> Array:
        return self._scipy_rv.rvs(nsamples)

    def _check_samples(self, samples: Array) -> Array:
        return self._bkd.asarray(super()._check_samples(samples))

    def _check_values(self, vals: Array) -> Array:
        return self._bkd.asarray(super()._check_values(vals))

    def _get_distribution_info(self) -> tuple[str, dict, dict]:
        """
        Get important information from a scipy.stats variable.

        Notes
        -----
        Shapes and scales can appear in either args of kwargs depending on how
        user initializes frozen object.
        """
        name = self._scipy_rv.dist.name
        shape_names = self._scipy_rv.dist.shapes
        if shape_names is not None:
            shape_names = [name.strip() for name in shape_names.split(",")]
            shape_values = [
                self._scipy_rv.args[ii]
                for ii in range(
                    min(len(self._scipy_rv.args), len(shape_names))
                )
            ]
            shape_values += [
                self._scipy_rv.kwds[shape_names[ii]]
                for ii in range(len(self._scipy_rv.args), len(shape_names))
            ]
            shapes = dict(zip(shape_names, shape_values))
        else:
            shapes = dict()

        scale_values = [
            self._scipy_rv.args[ii]
            for ii in range(len(shapes), len(self._scipy_rv.args))
        ]
        scale_values += [
            self._scipy_rv.kwds[key]
            for key in self._scipy_rv.kwds
            if key not in shapes
        ]
        if len(scale_values) == 0:
            scale_values = [0, 1]
        elif len(scale_values) == 1 and len(self._scipy_rv.args) > len(shapes):
            scale_values += [1.0]
        elif len(scale_values) == 1 and "scale" not in self._scipy_rv.kwds:
            scale_values += [1.0]
        elif len(scale_values) == 1 and "loc" not in self._scipy_rv.kwds:
            scale_values = [0] + scale_values
        scale_names = ["loc", "scale"]
        scales = dict(
            zip(scale_names, [np.atleast_1d(s) for s in scale_values])
        )

        if isinstance(self._scipy_rv.dist, float_rv_discrete):
            xk = self._scipy_rv.dist.xk.copy()
            shapes = {"xk": xk, "pk": self._scipy_rv.dist.pk}

        return name, scales, shapes

    def _raw_pdf(self, x: Array) -> Array:
        """
        Get the raw pdf of a scipy.stats variable.

        Evaluating this function avoids error checking which can
        slow evaluation significantly. Use with caution
        """
        return self._scipy._rv.dist._pdf((x - loc) / scale, **shapes) / scale

    def _transform_scale_parameters(self) -> Tuple[float, float]:
        """
        Transform scale parameters so that when any bounded variable is transformed
        to the canonical domain [-1, 1]
        """
        # copy is essential here because code below modifies scale
        loc, scale = (
            self._bkd.copy(self._scales["loc"])[0],
            self._bkd.copy(self._scales["scale"])[0],
        )
        return loc, scale

    def _shapes_equal(self, other: "ScipyMarginal") -> bool:
        return self._shapes == other._shapes

    def __eq__(self, other: Marginal) -> bool:
        """
        Determine if 2 scipy variables are equivalent

        Let
        a = beta(1,1,-1,2)
        b = beta(a=1,b=1,loc=-1,scale=2)

        then a==b will return False because .args and .kwds are different
        """
        if not isinstance(other, ScipyMarginal):
            return False
        if self._name != other._name:
            return False
        if self._scales != other._scales:
            return False
        return self._shapes_equal(other)

    def truncated_range(self, alpha: float = None) -> Array:
        if alpha is None and self.is_bounded():
            alpha = 1.0
        elif alpha is None:
            alpha = 0.99
        return self._scipy_rv.interval(alpha)

    @abstractmethod
    def is_bounded(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}(name={1})".format(self.__class__.__name__, self._name)


class ContinuousScipyMarginal(ScipyMarginal):
    def _check_marginal(self, marginal: stats._distn_infrastructure.rv_frozen):
        if not self._is_continuous_variable():
            raise ValueError("marginal is not a continous scipy variable")

    def _is_continuous_variable(self) -> bool:
        """
        Is variable continuous
        """
        return bool(
            (self._scipy_rv.dist.name in _continuous_distns._distn_names)
            or self._scipy_rv.dist.name == "continuous_rv_sample"
            or self._scipy_rv.dist.name == "continuous_monomial"
            or self._scipy_rv.dist.name == "rv_function_indpndt_vars"
            or self._scipy_rv.dist.name == "rv_product_indpndt_vars"
        )

    def is_bounded(self) -> bool:
        """
        Is variable bounded and continuous
        """
        interval = self._scipy_rv.interval(1)
        return np.isfinite(interval[0]) and np.isfinite(interval[1])

    def _transform_scale_parameters(self) -> Tuple[float, float]:
        """
        Transform scale parameters so that when any bounded variable is transformed
        to the canonical domain [-1, 1]
        """
        if not self._is_bounded():
            return super()._transform_scale_parameters()

        a, b = self._scipy_rv.interval(1)
        loc = (a + b) / 2
        scale = b - loc
        return loc, scale


class DiscreteScipyMarginal(ScipyMarginal):
    def _check_marginal(self, marginal: stats._distn_infrastructure.rv_frozen):
        if not (self._scipy_rv.dist.name in _discrete_distns._distn_names):
            raise ValueError("marginal is not a discrete scipy variable")

    def _pdf(self, samples: Array) -> Array:
        return self._scipy_rv.pmf(samples)

    def is_bounded(self) -> bool:
        interval = self._scipy_rv.interval(1)
        return np.isfinite(interval[0]) and np.isfinite(interval[1])

    def probability_masses(self, tol: float = 0) -> Array:
        """
        Get the the locations and masses of a discrete random variable.

        Parameters
        ----------
        tol : float
            Fraction of total probability in (0, 1). Can be useful with
            extracting masses when numerical precision becomes a problem
        """
        if (
            self._name == "float_rv_discrete"
            or self._name == "discrete_chebyshev"
            or self._name == "continuous_rv_sample"
        ):
            return self._scipy_rv.dist.xk.copy(), self._scipy_rv.dist.pk.copy()
        elif self._name == "hypergeom":
            M, n, N = [self._shapes[key] for key in ["M", "n", "N"]]
            xk = np.arange(max(0, N - M + n), min(n, N) + 1, dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk
        elif self._name == "binom":
            n = self._shapes["n"]
            xk = np.arange(0, n + 1, dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk
        elif (
            self._name == "nbinom"
            or self._name == "geom"
            or self._name == "logser"
            or self._name == "poisson"
            or self._name == "planck"
            or self._name == "zipf"
            or self._name == "dlaplace"
            or self._name == "skellam"
        ):
            if tol <= 0:
                raise ValueError(
                    "interval is unbounded so specify probability<1"
                )
            lb, ub = self._scipy_rv.interval(1 - tol)
            xk = np.arange(int(lb), int(ub), dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk
        elif self._name == "boltzmann":
            xk = np.arange(self._shapes["N"], dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk
        elif self._name == "randint":
            xk = np.arange(
                self._shapes["low"], self._shapes["high"], dtype=float
            )
            pk = self._scipy_rv.pmf(xk)
            return xk, pk
        else:
            raise ValueError(
                f"Variable {self._scipy_rv.dist.self._name} not supported"
            )

    def _transform_scale_parameters(self) -> Tuple[float, float]:
        """
        Transform scale parameters so that when any bounded variable is transformed
        to the canonical domain [-1, 1]
        """
        # var.interval(1) can return incorrect bounds
        xk, pk = self.probability_masses()
        a, b = xk.min(), xk.max()
        loc = (a + b) / 2
        scale = b - loc
        return loc, scale

    def _shapes_equal(self, other: "ScipyMarginal") -> bool:
        if "xk" not in self._shapes:
            return super()._shapes_equal(other)
        # xk and pk shapes are list so != comparison will not work
        not_equiv = np.any(self._shapes["xk"] != other.shapes["xk"]) or np.any(
            self._shapes["pk"] != other._shapes["pk"]
        )
        return not not_equiv


def pdf_under_affine_map(
    pdf: callable, loc: float, scale: float, y: Array
) -> Array:
    return pdf((y - loc) / scale) / scale


def pdf_derivative_under_affine_map(
    pdf_deriv: callable, loc: float, scale: float, y: Array
) -> Array:
    r"""
    Let y=g(x)=x*scale+loc and x = g^{-1}(y) = v(y) = (y-loc)/scale, scale>0
    p_Y(y)=p_X(v(y))*|dv/dy(y)|=p_X((y-loc)/scale))/scale
    dp_Y(y)/dy = dv/dy(y)*dp_X/dx(v(y))/scale = dp_X/dx(v(y))/scale**2
    """
    return pdf_deriv((y - loc) / scale) / scale**2


def beta_function(
    alpha_stat: float, beta_stat: float, bkd: LinAlgMixin = NumpyLinAlgMixin
):
    return bkd.exp(
        bkd.gammaln(alpha_stat)
        + bkd.gammaln(beta_stat)
        - bkd.gammaln(alpha_stat + beta_stat)
    )


def beta_pdf(
    alpha_stat: float,
    beta_stat: float,
    x: Array,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    # scipy implementation is slow
    const = 1.0 / beta_function(alpha_stat, beta_stat, bkd)
    return const * (x ** (alpha_stat - 1) * (1 - x) ** (beta_stat - 1))


def beta_pdf_on_ab(
    alpha_stat: float,
    beta_stat: float,
    a: float,
    b: float,
    x: Array,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    # const = 1./beta_fn(alpha_stat,beta_stat)
    # const /= (b-a)**(alpha_stat+beta_stat-1)
    # return const*((x-a)**(alpha_stat-1)*(b-x)**(beta_stat-1))
    pdf = partial(beta_pdf, alpha_stat, beta_stat)
    return pdf_under_affine_map(pdf, a, (b - a), x)


def beta_pdf_derivative(
    alpha_stat: float,
    beta_stat: float,
    x: Array,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    r"""
    x in [0, 1]
    """
    # beta_const = gamma_fn(alpha_stat+beta_stat)/(
    # gamma_fn(alpha_stat)*gamma_fn(beta_stat))

    beta_const = 1.0 / beta_function(alpha_stat, beta_stat, bkd)
    deriv = 0
    if alpha_stat > 1:
        deriv += (alpha_stat - 1) * (
            x ** (alpha_stat - 2) * (1 - x) ** (beta_stat - 1)
        )
    if beta_stat > 1:
        deriv -= (beta_stat - 1) * (
            x ** (alpha_stat - 1) * (1 - x) ** (beta_stat - 2)
        )
    deriv *= beta_const
    return deriv


def beta_pdf_derivative_on_ab(
    alpha_stat: float,
    beta_stat: float,
    a: float,
    b: float,
    x: Array,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:

    pdf_deriv = partial(beta_pdf_derivative, alpha_stat, beta_stat)
    return pdf_derivative_under_affine_map(pdf_deriv, a, b - a, x)


def gaussian_pdf(
    mean: float, var: float, x: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    r"""
    set package=sympy if want to use for symbolic calculations
    """
    return bkd.exp(-((x - mean) ** 2) / (2 * var)) / (2 * math.pi * var) ** 0.5


def gaussian_pdf_derivative(
    mean: float, var: float, x: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    return -gaussian_pdf(mean, var, x, bkd) * (x - mean) / var


class MarginalCDFNewtonResidual(NewtonResidual):
    def __init__(self, marginal):
        super().__init__(marginal._bkd)
        self._marginal = marginal

    def set_usamples(self, usamples: Array):
        if usamples.ndim != 1:
            raise ValueError("usamples must be 1D array")
        self._usamples = usamples

    def __call__(self, iterate: Array) -> Array:
        return self._marginal.cdf(iterate) - self._usamples

    def _jacobian(self, iterate: Array) -> Array:
        return self._marginal._cdf_jacobian(iterate)

    def linsolve(self, iterate: Array, res: Array) -> Array:
        # print(iterate)
        # print(res)
        # print(self._marginal._cdf_jacobian_diagonal(iterate))
        # print(self._bkd.diag(super()._jacobian(iterate)))
        return res / self._marginal._cdf_jacobian_diagonal(iterate)


class MarginalCDFNewtonSolver(NewtonSolver):
    def _bounded_line_search(
        self,
        idx: Array,
        sol: Array,
        prev_sol: Array,
        delta: Array,
        prev_residual_norm: float,
    ) -> Array:
        step_size = self._step_size
        ii = 0
        while ii < self._linesearch_maxiters:
            step_size = self._step_size / 2
            sol[idx] = prev_sol[idx] - step_size * delta[idx]
            residual = self._residual(sol)
            residual_norm = self._bkd.norm(residual)
            if residual_norm < prev_residual_norm:
                return
            ii += 1
        raise RuntimeError("Max bounded linesearch iterations reached")

    def _update_sol(self, prev_sol: Array, delta: Array) -> Array:
        sol = prev_sol - self._step_size * delta
        # make sure step size does not exceed bounds
        # (not necessary with good initial guess)
        # sol[sol < self._residual._marginal._lb] = self._residual._marginal._lb
        # sol[sol > self._residual._marginal._ub] = self._residual._marginal._ub
        return sol


class BetaMarginal(Marginal):
    def __init__(
        self,
        alpha: float,
        beta: float,
        lb: float,
        ub: float,
        nquad_samples: int = 100,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._lb = lb
        self._ub = ub
        self.set_shapes(alpha, beta)
        # quadx on [-1, 1], w(x) = 1
        # TODO cannot autograd through cdf until native bkd leggauss is called
        # It is implemented but I have to determine how to avoid circular
        # dependices
        if nquad_samples < int(self._a + self._b) + 1:
            # this condition is only sufficient for integer alpha and beta
            # but still worth checking
            raise ValueError(
                "nquad_samples must be greater than int(alpha + beta) + 1"
            )
        quadx, quadw = np.polynomial.legendre.leggauss(nquad_samples)
        self._quadx_01 = self._bkd.asarray((quadx + 1) / 2)
        self._quadw_01 = self._bkd.asarray(quadw / 2)
        self._scale = self._ub - self._lb
        self._newton_solver = MarginalCDFNewtonSolver(verbosity=0, maxiters=20)
        self._newton_solver.set_residual(MarginalCDFNewtonResidual(self))

    def set_shapes(self, alpha: float, beta: float):
        self._a = alpha
        self._b = beta
        self._scipy_rv = stats.beta(
            self._a, self._b, loc=self._lb, scale=self._ub - self._lb
        )
        self._const = 1.0 / beta_function(self._a, self._b, self._bkd)

    def rvs(self, nsamples: int) -> Array:
        return self._bkd.asarray(self._scipy_rv.rvs(nsamples))

    def _pdf_01(self, samples: Array) -> Array:
        # pdf on [0, 1]
        return beta_pdf(self._a, self._b, samples, self._bkd)

    def _pdf(self, samples: Array) -> Array:
        return self._pdf_01((samples - self._lb) / self._scale) / self._scale

    def _logpdf(self, samples: Array) -> Array:
        return self._bkd.log(self._pdf(samples))

    def _cdf(self, samples: Array) -> Array:
        # WARNING increase accuracy of quadrature rule if using non-integer
        # shape params. Current number of points assumes integrand
        # is a polynomial which is not true for non-integer shape params
        # samples s define length of integral interval [0, s] 0<=s<=1
        # map samples and weights to [-1, 1] by first mapping samples
        # in [lb, ub] to [0, 1]
        samples_01 = (samples - self._lb) / self._scale
        quadx = samples_01[:, None] * self._quadx_01[None, :]
        quadw = samples_01[:, None] * self._quadw_01[None, :]
        pdf_01_vals = self._pdf_01(quadx)
        return self._bkd.sum(pdf_01_vals * quadw, axis=1)

    def _ppf(self, usamples: Array) -> Array:
        # u samples on [0, 1]
        iterate = usamples * self._scale + self._lb
        if self._a == 1.0 and self._b == 1.0:
            return iterate
        # this funciton is used to compute gradients. Initial
        # iterate will not effect this computation, unless it is exactly
        # the answer so just use scipy as initial guess. use shift
        # smaller than newton_solver.tol
        iterate = self._bkd.asarray(self._scipy_rv.ppf(usamples) + 1e-4)
        self._newton_solver._residual.set_usamples(usamples)
        iterate = self._bkd.copy(iterate)
        vals = self._newton_solver.solve(iterate)
        vals[usamples == 0.0] = 0.0
        vals[usamples == 1.0] = 1.0
        return vals

    def _pdf_jacobian_01(self, sample: Array) -> Array:
        deriv = self._bkd.zeros(sample.shape)
        if self._a > 1:
            deriv += (self._a - 1) * (
                sample ** (self._a - 2) * (1 - sample) ** (self._b - 1)
            )
        if self._b > 1:
            deriv -= (self._b - 1) * (
                sample ** (self._a - 1) * (1 - sample) ** (self._b - 2)
            )
        return deriv * self._const

    def _pdf_jacobian(self, sample: Array) -> Array:
        return (
            self._pdf_jacobian_01((sample - self._lb) / self._scale)
            / self._scale**2
        )

    def _cdf_jacobian_diagonal(self, samples: Array) -> Array:
        samples_01 = (samples - self._lb) / self._scale
        quadx = samples_01[:, None] * self._quadx_01[None, :]
        quadw = samples_01[:, None] * self._quadw_01[None, :]
        pdf_jac = self._pdf_jacobian_01(quadx)
        cdf_jac = self._bkd.sum(pdf_jac * quadw, axis=1) / self._scale
        eps = 1e-16
        cdf_jac[self._bkd.abs(samples_01) < eps] = 1.0
        cdf_jac[self._bkd.abs(samples_01 - 1.0) < eps] = 1.0
        return cdf_jac

    def _cdf_jacobian(self, samples: Array) -> Array:
        return self._bkd.diag(self._cdf_jacobian_diagonal(samples))

    def kl_divergence(self, other: "BetaMarginal"):
        r"""
        .. math:: \mathrm{KL}(F||G)=\ln\frac{\Gamma(\alpha_{f}+\beta_{f})\Gamma(\alpha_{g})\Gamma(\beta_{g})}{\Gamma(\alpha_{g}+\beta_{g})\Gamma(\alpha_{f})\Gamma(\beta_{f})}+(\alpha_{f}-\alpha_{g})\left(\psi(\alpha_{f})-\psi(\alpha_{f}+\beta_{f})\right)+(\beta_{f}-\beta_{g})\left(\psi(\beta_{f})-\psi(\alpha_{f}+\beta_{f})\right)

        where

        .. math:: \psi(x)=\frac{d}{dx}\ln\Gamma(x)=\frac{\Gamma'(x)}{\Gamma(x)}
        """
        if self._lb != other._lb or self._ub != other._ub:
            raise ValueError("marginals must have the same bounds")
        self_sum = self._a + self._b
        other_sum = other._a + other._b
        term1_numer = (
            self._bkd.gammaln(other._a)
            + self._bkd.gammaln(other._b)
            + self._bkd.gammaln(self_sum)
        )
        term1_denom = -(
            self._bkd.gammaln(self._a)
            + self._bkd.gammaln(self._b)
            + self._bkd.gammaln(other_sum)
        )
        tmp = self._bkd.digamma(self_sum)
        term2 = (self._a - other._a) * (self._bkd.digamma(self._a) - tmp)
        term3 = (self._b - other._b) * (self._bkd.digamma(self._b) - tmp)
        # term2 = (self._a - other._a) * self._bkd.digamma(self._a)
        # term3 = (self._b - other._b) * self._bkd.digamma(self._b)
        # term3 += (self_sum - other_sum) * tmp
        return term1_numer + term1_denom + term2 + term3

    def is_bounded(self) -> bool:
        return True

    def __eq__(self, other: Marginal) -> bool:
        if not isinstance(other, BetaMarginal):
            return False
        if self._a != other._a or self._b != other._b:
            return False
        return True

    def _rvs(self, nsamples: int):
        usamples = self._bkd.asarray(np.random.uniform(0, 1, nsamples))
        return self._ppf(usamples)


class UniformMarginal(BetaMarginal):
    def __init__(
        self,
        lb: float,
        ub: float,
        nquad_samples: int = 100,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(0, 0, lb, ub, nquad_samples, backend)


class GaussianMarginal(Marginal):
    def __init__(
        self,
        mean: float,
        stdev: float,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._mean = mean
        self._stdev = stdev
        self._var = stdev**2
        self._log_const = self._bkd.log(
            1.0
            / (math.sqrt((2.0 * math.pi)) * self._bkd.asarray([self._stdev]))
        )

    def is_bounded(self) -> bool:
        return False

    def _pdf(self, samples: Array) -> Array:
        return self._bkd.exp(self._logpdf(samples))

    def _logpdf(self, samples: Array) -> Array:
        return self._log_const - (samples - self._mean) ** 2 / (
            2.0 * self._var
        )

    def _cdf(self, samples: Array) -> Array:
        return 0.5 * (
            1.0
            + self._bkd.erf(
                (samples - self._mean) / (self._stdev * math.sqrt(2))
            )
        )

    def _ppf(self, usamples: Array) -> Array:
        return (
            math.sqrt(2.0) * self._bkd.erfinv(2.0 * usamples - 1.0)
        ) * self._stdev + self._mean

    def _pdf_jacobian(self, samples: Array) -> Array:
        return (self._pdf(samples) * (self._mean - samples) / self._var)[
            None, :
        ]
        raise NotImplementedError

    def _logpdf_jacobian(self, samples: Array) -> Array:
        return (-(self._mean - samples) / self._var)[None, :]

    def pdf_jacobian_implemented(self) -> bool:
        return True

    def logpdf_jacobian_implemented(self) -> bool:
        return True

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._var

    def __eq__(self, other: Marginal) -> bool:
        if not isinstance(other, GaussianMarginal):
            return False
        if self._mean != other._mean or self._stdev != other._stdev:
            return False
        return True

    def _rvs(self, nsamples: int):
        usamples = self._bkd.asarray(np.random.uniform(0, 1, nsamples))
        return self._ppf(usamples)


# TODO make this a function of a discrete variable from samples class
class CustomDiscreteMarginal(Marginal):
    def __init__(
        self, xk: Array, pk: Array, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        super().__init__(backend)
        if xk.ndim != 1 or pk.ndim != 1:
            raise ValueError("xk and pk must be 1D arrays")
        if xk.shape != pk.shape:
            raise ValueError("xk and pk are inconsistent")
        self._xk = xk
        self._pk = pk
        idx = self._bkd.argsort(self._xk)
        self._sorted_xk = self._xk[idx]
        self._sorted_pk = self._pk[idx]
        self._ecdf = self._bkd.cumsum(self._pk[idx])
        self._interp = interpolate.interp1d(
            self._sorted_xk,
            self._ecdf,
            kind="zero",
            fill_value=(0, 1),
            bounds_error=False,
        )

    def __eq__(self, other: "CustomDiscreteMarginal") -> bool:
        return self._bkd.allclose(
            self._xk, other._xk, atol=np.finfo(float).eps * 2
        ) and self._bkd.allclose(
            self._pk, other._pk, atol=np.finfo(float).eps * 2
        )

    def is_bounded(self) -> bool:
        return True

    def nmasses(self) -> int:
        return self._xk.shape[0]

    def _pdf(self, samples: Array) -> Array:
        nsamples = samples.shape[0]
        vals = np.zeros(nsamples)
        for jj in range(nsamples):
            for ii in range(self.nmasses()):
                if np.allclose(
                    self._xk[ii], samples[jj], atol=np.finfo(float).eps * 2
                ):
                    vals[jj] = self._pk[ii]
                    break
        return vals

    def _logpdf(self, samples: Array) -> Array:
        return self._bkd.log(self._pdf(samples))

    def _cdf(self, samples: Array) -> Array:
        # CDF is currently cannot be used with autograd
        # because it uses scipy interp1d
        return self._bkd.asarray(self._interp(samples))

    def integrate_cdf(self) -> Array:
        vals = self._bkd.cumsum(
            self._bkd.diff(self._sorted_xk) * self._ecdf[:-1]
        )
        return self._bkd.hstack(self._bkd.zeros((1,)), vals)

    def moment(self, power: int) -> float:
        return (self._xk**power) @ self._pk

    def _ppf(self, usamples: Array) -> Array:
        raise NotImplementedError("TODO")

    def _rvs(self, nsamples: int) -> Array:
        return self._bkd.asarray(
            np.random.choice(self._xk, size=nsamples, p=self._pk)
        )


class EmpiricalCDF:
    def __init__(
        self,
        samples: Array,
        weights: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        if samples.ndim != 1:
            raise ValueError("samples must be a 1d array")
        self._samples = samples
        self._sorted_samples = self._bkd.sort(self._samples)
        if weights is None:
            self._ecdf = self._bkd.asarray(
                [(ii + 1) / self.nmasses() for ii in range(self.nmasses())]
            )
        else:
            assert weights.ndim == 1
            II = np.argsort(self.samples)
            self._ecdf = np.cumsum(weights[II])

        self._interp = interpolate.interp1d(
            self._sorted_samples,
            self._ecdf,
            kind="zero",
            fill_value=(0, 1),
            bounds_error=False,
        )

    def nmasses(self) -> int:
        return self._samples.shape[0]

    def __call__(self, samples: Array) -> Array:
        if samples.ndim != 1:
            raise ValueError("samples must be a 1d array")
        return self._bkd.asarray(self._interp(samples))

    def integrate_cdf(self) -> Array:
        vals = self._bkd.cumsum(
            self._bkd.diff(self._sorted_samples) * self._ecdf[:-1]
        )
        return self._bkd.hstack(self._bkd.zeros((1,)), vals)


def get_unique_variables(variables):
    """
    Get the unique 1D variables from a list of variables.
    """
    nvars = len(variables)
    unique_variables = [variables[0]]
    unique_var_indices = [[0]]
    for ii in range(1, nvars):
        found = False
        for jj in range(len(unique_variables)):
            if variables_equivalent(variables[ii], unique_variables[jj]):
                unique_var_indices[jj].append(ii)
                found = True
                break
        if not found:
            unique_variables.append(variables[ii])
            unique_var_indices.append([ii])
    return unique_variables, unique_var_indices


class float_rv_discrete(rv_sample):
    """Discrete distribution defined on locations represented as floats.

    rv_discrete in scipy only allows for integer locations.

    Currently we only guarantee that overloaded functions and cdf, ppf and
    moment work and are tested
    """

    def __init__(
        self,
        a=0,
        b=np.inf,
        name=None,
        badvalue=None,
        moment_tol=1e-8,
        values=None,
        inc=1,
        longname=None,
        shapes=None,
        seed=None,
        extradoc=None,
    ):
        # extradoc must appear in __init__ above even though it is not
        # used for backwards capability.
        super(float_rv_discrete, self).__init__(
            a,
            b,
            name,
            badvalue,
            moment_tol,
            values,
            inc,
            longname,
            shapes,
            seed=seed,
        )
        self.xk = self.xk.astype(dtype=float)

    def __new__(cls, *args, **kwds):
        return super(float_rv_discrete, cls).__new__(cls)

    def _rvs(self):
        samples = np.random.choice(self.xk, size=self._size, p=self.pk)
        return samples

    def rvs(self, *args, **kwds):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None or int or ``np.random.RandomState`` instance,
            optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        rndm = kwds.pop("random_state", None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = np.logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc * np.ones(size, "d")

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            from scipy._lib._util import check_random_state

            self._random_state = check_random_state(rndm)

        # `size` should just be an argument to _rvs(), but for, um,
        # historical reasons, it is made an attribute that is read
        # by _rvs().
        self._size = size
        vals = self._rvs(*args)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # JDJAKEM: commenting this scipy code out allows for non integer
        # locations
        # # Cast to int if discrete
        # if discrete:
        #     if size == ():
        #         vals = int(vals)
        #     else:
        #         vals = vals.astype(int)

        return vals

    def pmf(self, x):
        x = np.atleast_1d(x)
        vals = np.zeros(x.shape[0])
        for jj in range(x.shape[0]):
            for ii in range(self.xk.shape[0]):
                if np.allclose(
                    self.xk[ii], x[jj], atol=np.finfo(float).eps * 2
                ):
                    vals[jj] = self.pk[ii]
                    break
        return vals


class rv_function_indpndt_vars_gen(rv_continuous):
    """
    Custom variable representing the function of random variables.

    Used to create 1D polynomial chaos basis orthogonal to the PDFs
    of the scalar function of the 1D variables.
    """

    def _argcheck(self, fun, init_variables, quad_rules):
        return True

    def _pdf(self, x, fun, init_variables, quad_rules):
        raise NotImplementedError("Expression for PDF not known")


rv_function_indpndt_vars = rv_function_indpndt_vars_gen(
    shapes="fun, initial_variables, quad_rules",
    name="rv_function_indpndt_vars",
)


class rv_product_indpndt_vars_gen(rv_continuous):
    """
    Custom variable representing the product of random variables.

    Used to create 1D polynomial chaos basis orthogonal to the PDFs
    of the scalar product of the 1D variables.
    """

    def _argcheck(self, funs, init_variables, quad_rules):
        return True

    def _pdf(self, x, funs, init_variables, quad_rules):
        raise NotImplementedError("Expression for PDF not known")


rv_product_indpndt_vars = rv_product_indpndt_vars_gen(
    shapes="funs, initial_variables, quad_rules",
    name="rv_product_indpndt_vars",
)

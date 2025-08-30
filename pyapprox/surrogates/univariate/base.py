from abc import ABC, abstractmethod
import math
import warnings
from typing import Tuple


import numpy as np
import scipy
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.transforms import IdentityTransform, Transform


class UnivariateBasis(ABC):
    def __init__(self, trans: Transform, backend: BackendMixin):
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend
        if trans is None:
            trans = IdentityTransform(backend=self._bkd)
        self._trans = trans

    def first_derivatives_implemented(self) -> bool:
        return False

    def second_derivatives_implemented(self) -> bool:
        return False

    @abstractmethod
    def _values(samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def nterms(self) -> int:
        raise NotImplementedError

    def _check_samples(self, samples: Array):
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be a 2D row vector")

    def __call__(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._values(samples)
        if vals.ndim != 2 or vals.shape[0] != samples.shape[1]:
            raise ValueError("returned values must be a 2D array")
        return vals

    def _derivatives(self, samples: Array, order: int) -> Array:
        raise NotImplementedError

    def derivatives(self, samples: Array, order: int) -> Array:
        self._check_samples(samples)
        return self._derivatives(samples, order)

    def _quadrature_rule(self) -> Tuple[Array, Array]:
        raise NotImplementedError

    def quadrature_rule(self) -> Tuple[Array, Array]:
        samples, weights = self._quadrature_rule()
        self._check_samples(samples)
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("weights must be a 2D column vector")
        if samples.shape[1] != weights.shape[0]:
            raise ValueError("number of samples and weights are different")
        return samples, weights

    def __repr__(self) -> str:
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class Monomial1D(UnivariateBasis):
    """Univariate monomial basis."""

    def __init__(
        self,
        nterms: int = None,
        trans: Transform = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(trans, backend)
        self._nterms = None
        if nterms is not None:
            self.set_nterms(nterms)

    def first_derivatives_implemented(self) -> bool:
        return True

    def second_derivatives_implemented(self) -> bool:
        return True

    def set_nterms(self, nterms: int):
        self._nterms = nterms

    def nterms(self) -> int:
        return self._nterms

    def _values(self, samples: Array) -> Array:
        basis_matrix = samples.T ** self._bkd.arange(self._nterms)[None, :]
        return basis_matrix

    def _derivatives(self, samples: Array, order: int) -> Array:
        powers = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                self._bkd.arange(self._nterms - order),
            )
        )
        # 1 x x^2 x^3  x^4 vals
        # 0 1 2x  3x^2 4x^3 1st derivs
        # 0 0 2   6x   12x^2  2nd derivs
        consts = self._bkd.hstack(
            (
                self._bkd.zeros((order,)),
                order * self._bkd.arange(1, self._nterms - order + 1),
            )
        )
        return (samples.T ** powers[None, :]) * consts


class UnivariateInterpolatingBasis(UnivariateBasis):
    def __init__(
        self, trans: Transform = None, backend: BackendMixin = NumpyMixin
    ):
        super().__init__(trans, backend)
        self._quad_samples = None
        self._quad_weights = None

    def nterms(self) -> int:
        return self._quad_samples.shape[1]

    def __repr__(self) -> str:
        if self._quad_samples is None:
            return "{0}(bkd={1})".format(
                self.__class__.__name__, self._bkd.__name__
            )
        return "{0}(nterms={1}, bkd={2})".format(
            self.__class__.__name__, self.nterms(), self._bkd.__name__
        )

    @abstractmethod
    def _semideep_copy(self):
        raise NotImplementedError


class UnivariatePiecewisePolynomialNodeGenerator(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bounds = None
        self._bkd = backend

    def set_bounds(self, bounds: Tuple[float, float]):
        self._bounds = bounds

    def __call__(self, nnodes: int) -> Array:
        if self._bounds is None:
            raise ValueError("must call set_bounds")
        nodes = self._nodes(nnodes)
        if nodes.ndim != 2 or nodes.shape[0] != 1:
            raise RuntimeError("nodes returned must be a 2D row vector")
        return nodes

    @abstractmethod
    def _nodes(self, nnodes: int) -> Array:
        raise NotImplementedError


class UnivariateEquidistantNodeGenerator(
    UnivariatePiecewisePolynomialNodeGenerator
):
    def _nodes(self, nnodes: int) -> Array:
        return self._bkd.linspace(*self._bounds, nnodes)[None, :]


class ClenshawCurtisGrowthRuleMixin:
    def _growth_rule(self, level: int) -> int:
        if level == 0:
            return 1
        else:
            return 2**level + 1

    def _hierarchical_to_nodal_index(
        self, level: int, ll: int, ii: int
    ) -> int:
        nindices = self._growth_rule(level)
        # mid point
        if ll == 0:
            return nindices // 2
        # boundaries
        if ll == 1:
            if ii == 0:
                return 0
            return nindices - 1
        # higher level points
        return (2 * ii + 1) * 2 ** (level - ll)

    def _poly_indices_to_quad_rule_indices(self, level: int) -> Array:
        """
        Convert all 1D polynomial indices of up to and including a given level
        to their equivalent nodal index for lookup in a Clenshaw-Curtis
        quadrature rule.
        """

        quad_rule_indices = []
        nprevious_hier_indices = 0
        for ll in range(level + 1):
            nhierarchical_indices = (
                self._growth_rule(ll) - nprevious_hier_indices
            )
            for ii in range(nhierarchical_indices):
                quad_index = self._hierarchical_to_nodal_index(level, ll, ii)
                quad_rule_indices.append(quad_index)
            nprevious_hier_indices += nhierarchical_indices
        return self._bkd.asarray(quad_rule_indices, dtype=int)


class DydadicEquidistantNodeGenerator(
    ClenshawCurtisGrowthRuleMixin, UnivariatePiecewisePolynomialNodeGenerator
):
    # useful for adaptive interpolation
    def _nodes(self, nnodes: int) -> Array:
        if nnodes == 1:
            level = 0
            return self._bkd.array([[(self._bounds[0] + self._bounds[1]) / 2]])

        if not _is_power_of_two(nnodes - 1):
            raise ValueError("nnodes-1 must be a power of 2")

        level = int(round(math.log(nnodes - 1, 2), 0))
        idx = self._poly_indices_to_quad_rule_indices(level)
        return self._bkd.linspace(*self._bounds, nnodes)[None, idx]


class UnivariateQuadratureRule(ABC):
    def __init__(
        self, backend: BackendMixin = NumpyMixin, store: bool = False
    ):
        """
        Parameters
        ----------
        store : bool
            Store all quadrature rules computed. This is useful
            if repetedly calling the quadrature rule and
            the cost of computing the quadrature rule is nontrivial
        """
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend
        self._store = store
        self._quad_samples = dict()
        self._quad_weights = dict()
        self._nnodes = None

    def set_nnodes(self, nnodes: int) -> Array:
        self._nnodes = nnodes

    @abstractmethod
    def _quad_rule(self, nnodes: int) -> Array:
        raise NotImplementedError

    def __call__(self, nnodes: int = None) -> Tuple[Array, Array]:
        if nnodes is None:
            if self._nnodes is None:
                raise ValueError(
                    "If nnodes is None must first call set_nnodes"
                )
            nnodes = self._nnodes
        if self._store and nnodes in self._quad_samples:
            return self._quad_samples[nnodes], self._quad_weights[nnodes]
        quad_samples, quad_weights = self._quad_rule(nnodes)
        if self._store:
            self._quad_samples[nnodes] = quad_samples
            self._quad_weights[nnodes] = quad_weights
        return quad_samples, quad_weights

    def __repr__(self) -> str:
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__name__
        )


def _is_power_of_two(integer: int) -> bool:
    return (integer & (integer - 1) == 0) and integer != 0


class ClenshawCurtisQuadratureRule(
    ClenshawCurtisGrowthRuleMixin, UnivariateQuadratureRule
):
    """Integrates functions on [-1, 1] with weight function 1/2 or 1."""

    def __init__(
        self,
        prob_measure: bool = True,
        bounds: Tuple[float, float] = None,
        backend: BackendMixin = NumpyMixin,
        store: bool = False,
    ):
        super().__init__(backend=backend, store=store)
        self._prob_measure = prob_measure

        if bounds is None:
            msg = (
                "Bounds not set. Proceed with caution. "
                "User is responsible for ensuring samples are in "
                "canonical domain of the polynomial."
            )
            warnings.warn(msg, UserWarning)
            bounds = [-1, 1]
        self._bounds = bounds

    def _unordered_pts_wts(self, level: int) -> Tuple[Array, Array]:
        nsamples = self._growth_rule(level)

        if level == 0:
            return self._bkd.zeros((1,)), self._bkd.ones((1,))

        jj = self._bkd.arange(nsamples)
        x = -self._bkd.cos(np.pi * jj / (nsamples - 1.0))
        x[0] = -1.0
        x[-1] = 1.0
        x[nsamples // 2] = 0.0

        wt_factor = 1.0 / 2.0
        w = self._bkd.empty(nsamples)
        w[[0, -1]] = wt_factor / (nsamples * (nsamples - 2.0))
        jj = self._bkd.arange(1, nsamples - 1)
        kk = self._bkd.arange(1, (nsamples - 3) // 2 + 1)
        mysum = self._bkd.sum(
            1.0
            / (4.0 * kk[:, None] ** 2 - 1.0)
            * self._bkd.cos(
                2.0 * np.pi * kk[:, None] * jj[None, :] / (nsamples - 1.0)
            ),
            axis=0,
        )
        w[1:-1] = wt_factor * (
            2.0
            / float(nsamples - 1.0)
            * (
                1.0
                - self._bkd.cos(np.pi * jj) / (nsamples * (nsamples - 2.0))
                - 2.0 * (mysum)
            )
        )
        return x, w

    def _ordered_pts_wts(self, level: int) -> Tuple[Array, Array]:
        """
        Return the samples and weights of the Clenshaw-Curtis rule using
        polynomial ordering.

        The first point will be the middle weight of the rule. The second and
        third weights will be the left and right boundary weights. All other
        weights left of mid point will come next followed by all remaining points.
        """
        x, w = self._unordered_pts_wts(level)
        quad_indices = self._poly_indices_to_quad_rule_indices(level)
        ordered_samples_1d = x[quad_indices]
        ordered_weights_1d = w[quad_indices]
        return ordered_samples_1d, ordered_weights_1d

    def _quad_rule(self, nnodes):
        # rule requires nnodes = 2**l + 1 for l=1,2,3
        # so check n=nnodes-1 is a power of 2
        if nnodes == 1:
            level = 0
        else:
            if not _is_power_of_two(nnodes - 1):
                raise ValueError("nnodes-1 must be a power of 2")
            level = int(round(math.log(nnodes - 1, 2), 0))
        quad_samples, quad_weights = self._ordered_pts_wts(level)
        length = self._bounds[1] - self._bounds[0]
        quad_samples = (quad_samples + 1) / 2 * length + self._bounds[0]
        if not self._prob_measure:
            # taking quadweights for Lebesque measure on [-1, 1] to [a,b]
            # requires multiplying weights by (b-a)/2 but clenshaw curtis
            # weights are for uniform measure 1/2 on [-1,1] so only
            # need to multiply by (b-a)
            quad_weights *= length

        return (
            self._bkd.asarray(quad_samples)[None, :],
            self._bkd.asarray(quad_weights)[:, None],
        )


class UnivariateIntegrator(ABC):
    def __init__(self, backend: BackendMixin):
        self._integrand = None
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend

    def set_integrand(self, integrand: callable):
        self._integrand = integrand

    def set_options(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class ScipyUnivariateIntegrator(UnivariateIntegrator):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)

        self._kwargs = {}
        self._bounds = None
        self._result = None

    def set_options(self, **kwargs):
        self._kwargs = kwargs

    def set_bounds(self, bounds: Tuple[float, float]):
        if len(bounds) != 2:
            raise ValueError("bounds must be an interable with two elements")
        self._bounds = bounds

    def _scipy_integrand(self, sample: Array) -> float:
        val = self._integrand(self._bkd.atleast2d(self._bkd.asarray(sample)))
        if val.ndim != 2:
            raise RuntimeError("integrand must return 2D array with 1 element")
        return self._bkd.to_numpy(val)[0, 0]

    def __call__(self) -> float:
        result = scipy.integrate.quad(
            self._scipy_integrand, *self._bounds, **self._kwargs
        )
        self._result = result
        return result[0]


class UnivariateUnboundedIntegrator(UnivariateIntegrator):
    """
    Compute unbounded integrals by moving left and right from origin.
    Assume that integral decays towards +/- infinity. And that once integral
    over a sub interval drops below tolerance it will not increase again if
    we keep moving in same direction.

    Warning: Do not use this if your integrand is expensive. This
    algorithm priortizes speed with vectorization over minimizing the
    number of calls to the integrand
    """

    def __init__(
        self,
        quad_rule: UnivariateQuadratureRule,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend)
        self._bounds = None
        self._inteval_size = None
        self._verbosity = None
        self._nquad_samples = None
        self._adaptive = None
        self._atol = None
        self._rtol = None
        self._maxiters = None
        self.set_options()
        if not self._bkd.bkd_equal(self._bkd, quad_rule._bkd):
            raise ValueError("quad_rule backend does not match mine")
        self._quad_rule = quad_rule

    def set_options(
        self,
        interval_size: float = 2,
        nquad_samples: int = 50,
        verbosity: int = 0,
        adaptive: bool = True,
        atol: float = 1e-8,
        rtol: float = 1e-8,
        maxiters: int = 1000,
        maxinner_iters: int = 10,
    ):
        if interval_size <= 0:
            raise ValueError("Interval size must be positive")
        self._interval_size = interval_size
        self._verbosity = verbosity
        self._nquad_samples = nquad_samples
        self._adaptive = adaptive
        self._atol = atol
        self._rtol = rtol
        self._maxiters = maxiters
        self._maxinner_iters = maxinner_iters

    def set_bounds(self, bounds: Tuple[float, float]):
        if len(bounds) != 2:
            raise ValueError("bounds must be an interable with two elements")
        self._bounds = bounds
        if self._bkd.isfinite(
            self._bkd.asarray(bounds[0])
        ) and self._bkd.isfinite(self._bkd.asarray(bounds[1])):
            raise ValueError(
                "Do not use this integrator for bounded integrals"
            )

    def _initial_interval_bounds(self) -> Tuple[float, float]:
        # needs to be copy for torch
        lb, ub = self._bkd.copy(self._bkd.array(self._bounds))
        if self._bkd.isfinite(lb) and not self._bkd.isfinite(ub):
            return self._bkd.array([lb, lb + self._interval_size])
        elif not self._bkd.isfinite(lb) and self._bkd.isfinite(ub):
            return self._bkd.array([ub - self._interval_size, ub])
        return self._bkd.array(
            [-self._interval_size / 2, self._interval_size / 2]
        )

    def _integrate_interval(
        self, lb: float, ub: float, nquad_samples: int
    ) -> float:
        can_quad_x, can_quad_w = self._quad_rule(nquad_samples)
        quad_x = (can_quad_x + 1) / 2 * (ub - lb) + lb
        quad_w = can_quad_w * (ub - lb) / 2
        return self._integrand(quad_x).T @ quad_w[:, 0]

    def _adaptive_integrate_interval(self, lb: float, ub: float) -> float:
        nquad_samples = self._nquad_samples
        integral = self._integrate_interval(lb, ub, nquad_samples)
        if not self._adaptive:
            return integral
        it = 1
        while True:
            nquad_samples = (nquad_samples - 1) * 2 + 1
            prev_integral = self._bkd.copy(integral)
            integral = self._integrate_interval(lb, ub, nquad_samples)
            it += 1
            diff = self._bkd.abs(self._bkd.atleast1d(prev_integral - integral))
            if self._bkd.all(
                diff
                < self._rtol * self._bkd.abs(self._bkd.atleast1d(integral))
                + self._atol
            ):
                break
            if it >= self._maxinner_iters:
                break
        return integral

    def _left_integrate(self, lb: float, ub: float) -> float:
        integral = 0
        prev_integral = np.inf
        it = 0
        while (
            self._bkd.any(
                self._bkd.abs(
                    self._bkd.atleast1d(
                        self._bkd.asarray(integral - prev_integral)
                    )
                )
                >= self._rtol
                * self._bkd.abs(
                    self._bkd.atleast1d(self._bkd.asarray(prev_integral))
                )
                + self._atol
            )
            and lb >= self._bounds[0]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            if it == 0:
                prev_integral = self._bkd.copy(self._bkd.asarray([integral]))[
                    0
                ]
            else:
                prev_integral = self._bkd.copy(integral)
            integral += result
            ub = self._bkd.copy(lb)  # needs to be copy for torch
            lb -= self._interval_size
            it += 1
        if self._verbosity > 0:
            print(
                "nleft iters={0}, error={1}".format(
                    it,
                    self._bkd.abs(
                        self._bkd.atleast1d(
                            self._bkd.asarray(integral - prev_integral)
                        )
                    ),
                )
            )
        return integral

    def _right_integrate(self, lb: float, ub: float) -> float:
        integral = 0
        prev_integral = np.inf
        it = 0
        while (
            self._bkd.any(
                self._bkd.abs(
                    self._bkd.atleast1d(
                        self._bkd.asarray(integral - prev_integral)
                    )
                )
                >= self._rtol
                * self._bkd.abs(
                    self._bkd.atleast1d(self._bkd.asarray(prev_integral))
                )
                + self._atol
            )
            and ub <= self._bounds[1]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            if it == 0:
                prev_integral = self._bkd.copy(self._bkd.asarray([integral]))[
                    0
                ]
            else:
                prev_integral = self._bkd.copy(integral)
            integral += result
            lb = self._bkd.copy(ub)  # needs to be copy for torch
            ub += self._interval_size
            it += 1

        if self._verbosity > 0:
            print(
                "nright iters={0}, error={1}".format(
                    it,
                    self._bkd.abs(
                        self._bkd.atleast1d(
                            self._bkd.asarray(integral - prev_integral)
                        )
                    ),
                )
            )
        return integral

    def __call__(self) -> float:
        # compute left integral
        lb, ub = self._initial_interval_bounds()
        left_integral = self._left_integrate(lb, ub)
        right_integral = self._right_integrate(ub, ub + self._interval_size)
        integral = left_integral + right_integral
        if integral.shape[0] == 1:
            return integral[0]
        return integral

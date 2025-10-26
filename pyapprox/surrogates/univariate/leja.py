from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

from pyapprox.interface.model import Model
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.marginals import (
    UniformMarginal,
    GaussianMarginal,
    BetaMarginal,
    parse_marginal,
    Marginal,
    ContinuousScipyMarginal,
)
from pyapprox.optimization.minimize import Optimizer
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.univariate.base import UnivariateQuadratureRule
from pyapprox.util.visualization import get_meshgrid_function_data


def _custom_marginal_from_scipy_marginal(scipy_marginal, backend):
    if not isinstance(scipy_marginal, ContinuousScipyMarginal):
        raise ValueError(
            "scipy_marginal must be an instance of ContinuousScipyMarginal"
        )
    name = scipy_marginal._name
    marginals = {
        "uniform": UniformMarginal,
        "beta": BetaMarginal,
        "norm": GaussianMarginal,
    }
    if name not in marginals:
        raise ValueError("{0} not supported".format(name))
    if name == "norm" or name == "uniform":
        return marginals[name](
            scipy_marginal._scales["loc"],
            scipy_marginal._scales["scale"],
            backend=backend,
        )
    if name == "beta":
        return marginals[name](
            scipy_marginal._shapes["a"],
            scipy_marginal._shapes["b"],
            scipy_marginal._scales["loc"],
            scipy_marginal._scales["scale"],
            backend=backend,
        )


class LejaObjective(Model):
    def __init__(self, marginal, poly):
        super().__init__(backend=poly._bkd)
        self._poly = poly
        self._coef = None
        self._sequence = None
        self._marginal = None
        self._set_marginal(marginal)
        self._bounds = self._marginal.interval(1)

    def jacobian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._nopt_vars()

    def _nopt_vars(self) -> int:
        return 1

    def nsamples(self) -> int:
        return self._sequence.shape[1]

    def sequence(self) -> Array:
        return self._bkd.copy(self._sequence)

    @abstractmethod
    def _compute_weights(self, samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _compute_weights_jacobian(self, samples: Array) -> Array:
        raise NotImplementedError

    def _plot_bounds(self) -> Tuple[float, float]:
        return self._bkd.hstack(
            [self._marginal.interval(1 - 1e-15)] * self.nvars()
        )

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    def _set_reused_data(self):
        self._poly.set_nterms(self.nsamples() + 1)
        basis_vals = self._poly(self._sequence)
        self._basis_mat = basis_vals[:, :-1]
        self._basis_vec = basis_vals[:, -1:]

    def set_sequence(self, sequence: Array):
        if sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError("sequence must be a 2d array row vector")
        self._sequence = sequence
        self._set_reused_data()
        self._weights = self._compute_weights(self._sequence)
        self._weights_jac = self._compute_weights_jacobian(self._sequence)
        if self._weights.ndim != 2 or self._weights.shape[1] != 1:
            raise ValueError("weights must be a 2d array column vector")
        self._set_coefficients()

    def _set_coefficients(self):
        sqrt_weights = self._bkd.sqrt(self._weights)
        self._coef = self._bkd.lstsq(
            sqrt_weights * self._basis_mat,
            sqrt_weights * self._basis_vec,
        )

    def _values(self, samples: Array) -> Array:
        basis_mat = self._poly(samples)
        basis_mat, new_basis = basis_mat[:, :-1], basis_mat[:, -1:]
        weights = self._compute_weights(samples)
        pvals = basis_mat @ self._coef
        residual = new_basis - pvals
        return -weights * self._bkd.sum(residual**2, axis=1)[:, None]

    def _jacobian(self, sample: Array) -> Array:
        vals = self._poly._derivatives(sample, order=1, return_all=True)
        basis_vals = vals[:, : self._poly.nterms()]
        basis_jac = vals[:, self._poly.nterms() :]
        bvals = basis_vals[:, -1:]
        pvals = basis_vals[:, :-1] @ self._coef
        bderivs = basis_jac[:, -1:]
        pderivs = basis_jac[:, :-1] @ self._coef
        residual = bvals - pvals
        residual_jac = bderivs - pderivs
        weight = self._compute_weights(sample)
        weight_jac = self._compute_weights_jacobian(sample)
        if self._bkd.max(residual) > np.sqrt(np.finfo(float).max / 100):
            return np.inf * self._bkd.full((residual.shape[0],), np.inf)
        jac = self._bkd.sum(
            residual**2 * weight_jac + 2 * weight * residual * residual_jac,
            axis=1,
        )
        return -jac[None, :]

    def plot(self, ax):
        ax.plot(self.sequence()[0], self.sequence()[0] * 0, "o")
        plot_samples = self._bkd.linspace(*self._plot_bounds(), 101)[None, :]
        ax.plot(plot_samples[0], self(plot_samples))

    def _initial_iterates(self) -> Tuple[Array, Array]:
        eps = 1e-6  # must be larger than optimization tolerance
        bounds = self._bounds
        intervals = self._bkd.sort(self._sequence)
        if self._bkd.isfinite(bounds[0]) and (
            self._bkd.min(self._sequence) > bounds[0] + eps
        ):
            intervals = self._bkd.hstack(
                (self._bkd.asarray([[bounds[0]]]), intervals)
            )
        if self._bkd.isfinite(bounds[1]) and (
            self._bkd.max(self._sequence) < bounds[1] - eps
        ):
            intervals = self._bkd.hstack(
                (intervals, self._bkd.asarray([[bounds[1]]]))
            )
        if not self._bkd.isfinite(bounds[0]):
            intervals = self._bkd.hstack(
                (
                    self._bkd.asarray(
                        [[min(1.1 * self._bkd.min(self._sequence), -0.1)]]
                    ),
                    intervals,
                )
            )
        if not self._bkd.isfinite(bounds[1]):
            intervals = self._bkd.hstack(
                (
                    intervals,
                    self._bkd.asarray(
                        [[max(1.1 * self._bkd.max(self._sequence), 0.1)]]
                    ),
                )
            )
        iterates = intervals[:, :-1] + self._bkd.diff(intervals) / 2.0
        # put intervals in form useful for bounding 1d optimization problems
        intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
        if not self._bkd.isfinite(bounds[0]):
            intervals[0] = -np.inf
        if not self._bkd.isfinite(bounds[1]):
            intervals[-1] = np.inf

        bounds = []
        for jj in range(iterates.shape[1]):
            bounds.append(
                self._bkd.array([[intervals[jj], intervals[jj + 1]]])
            )
        return iterates, bounds


class TwoPointLejaObjective(LejaObjective):
    def _nopt_vars(self):
        return 2

    def _set_reused_data(self):
        self._poly.set_nterms(self.nsamples() + self._nopt_vars())
        basis_vals = self._poly(self._sequence)
        self._basis_mat = basis_vals[:, : -self._nopt_vars()]
        self._basis_vec = basis_vals[:, -self._nopt_vars() :]

    def _values(self, samples: Array) -> Array:
        if samples.shape[0] != 2:
            raise ValueError("samples must be a 2d array with two rows")
        nsamples = samples.shape[1]
        # first row of samples is multiple values of first iterate
        # second row of samples is multiple values of second iterate
        flat_samples = self._bkd.reshape(
            samples, (1, nsamples * self._nopt_vars())
        )
        basis_vals = self._poly(flat_samples)
        basis_mat = basis_vals[:, : -self._nopt_vars()]
        new_basis = basis_vals[:, -self._nopt_vars() :]
        sqrt_weights = self._bkd.sqrt(self._compute_weights(flat_samples))
        pvals = basis_mat @ self._coef
        residuals = sqrt_weights * (new_basis - pvals)
        vals = (
            -(
                residuals[:nsamples, 0] * residuals[nsamples:, 1]
                - residuals[:nsamples, 1] * residuals[nsamples:, 0]
            )[:, None]
            ** 2
        )
        return vals

    def _jacobian(self, sample: Array) -> Array:
        flat_sample = self._bkd.reshape(sample, (1, self._nopt_vars()))
        vals = self._poly._derivatives(flat_sample, order=1, return_all=True)
        basis_vals = vals[:, : self._poly.nterms()]
        basis_jac = vals[:, self._poly.nterms() :]
        bvals = basis_vals[:, -self._nopt_vars() :]
        pvals = basis_vals[:, : -self._nopt_vars()] @ self._coef
        bderivs = basis_jac[:, -self._nopt_vars() :]
        pderivs = basis_jac[:, : -self._nopt_vars()] @ self._coef
        sqrt_weights = self._bkd.sqrt(self._compute_weights(flat_sample))
        weights_jac = self._compute_weights_jacobian(flat_sample)
        sqrt_weights_jac = weights_jac / (2 * sqrt_weights[:, 0])
        residuals = sqrt_weights * (bvals - pvals)
        residuals_jac = sqrt_weights * (
            bderivs - pderivs
        ) + sqrt_weights_jac.T * (bvals - pvals)
        determinant_jac = self._bkd.stack(
            (
                residuals_jac[:1, 0] * residuals[1:2, 1]
                - residuals_jac[:1, 1] * residuals[1:2, 0],
                residuals[:1, 0] * residuals_jac[1:2, 1]
                - residuals[:1, 1] * residuals_jac[1:2, 0],
            ),
            axis=1,
        )
        determinant = (
            residuals[:1, 0] * residuals[1:, 1]
            - residuals[:1, 1] * residuals[1:, 0]
        )
        jac = -2 * determinant * determinant_jac
        return jac

    def plot(self, ax):
        ax.plot(self.sequence()[0], self.sequence()[0], "o")
        X, Y, Z = get_meshgrid_function_data(
            self.__call__,
            self._plot_bounds(),
            51,
            bkd=self._bkd,
        )
        ncontour_levels = 20
        im = ax.contourf(
            X,
            Y,
            Z,
            levels=self._bkd.linspace(Z.min(), Z.max(), ncontour_levels),
        )
        plt.colorbar(im, ax=ax)

    def _initial_iterates(self) -> Tuple[Array, Array]:
        iterates_1d, bounds_1d = super()._initial_iterates()
        iterates, bounds = [], []
        for ii in range(iterates_1d.shape[1]):
            for jj in range(ii + 1, iterates_1d.shape[1]):
                iterates.append(
                    self._bkd.vstack((iterates_1d[:, ii], iterates_1d[:, jj]))
                )
                bounds.append(self._bkd.vstack((bounds_1d[ii], bounds_1d[jj])))
        return self._bkd.hstack(iterates), bounds


class PDFLejaObjectiveMixin:
    def _set_marginal(self, marginal):
        custom_marginal = _custom_marginal_from_scipy_marginal(
            marginal, self._bkd
        )
        if not isinstance(custom_marginal, Marginal):
            raise ValueError("marginal must be an instance of Marginal")
        self._marginal = marginal
        self._custom_marginal = custom_marginal

    def _compute_weights(self, samples: Array) -> Array:
        return self._custom_marginal.pdf(samples[0])[:, None]

    def _compute_weights_jacobian(self, samples: Array) -> Array:
        return self._custom_marginal.pdf_jacobian(samples[0])


class ChristoffelLejaObjectiveMixin:
    def _set_marginal(self, marginal):
        self._marginal = marginal

    def _christoffel_fun(self, basis_mat: Array) -> Array:
        return self._bkd.sum(basis_mat**2, axis=1)[:, None] / self.nsamples()

    def _compute_weights(self, samples: Array) -> Array:
        basis_mat = self._poly(samples)[
            :, : self._poly.nterms()  # - self._nopt_vars() + 1
        ]
        return 1.0 / self._christoffel_fun(basis_mat)

    def _compute_weights_jacobian(self, sample: Array) -> Array:
        vals = self._poly._derivatives(sample, order=1, return_all=True)
        # basis_mat.shape = (nsamples, len(sequence))
        basis_mat = vals[:, : self._poly.nterms()][:, : self._poly.nterms()]
        # basis_mat.shape = (nsamples, 2)
        basis_jac = vals[:, self._poly.nterms() :][:, : self._poly.nterms()]
        christoffel_jac = (
            2
            / self.nsamples()
            * self._bkd.sum(basis_mat * basis_jac, axis=1)[None, :]
        )
        # chain rule g(x) = christoffel_fun(x)
        # d/dx 1/g(x) = -g'(x)/(g(x))
        return -1.0 / self._christoffel_fun(basis_mat).T ** 2 * christoffel_jac


class OnePointPDFLejaObjective(PDFLejaObjectiveMixin, LejaObjective):
    pass


class TwoPointPDFLejaObjective(PDFLejaObjectiveMixin, TwoPointLejaObjective):
    pass


class OnePointChristoffelLejaObjective(
    ChristoffelLejaObjectiveMixin, LejaObjective
):
    pass


class TwoPointChristoffelLejaObjective(
    ChristoffelLejaObjectiveMixin, TwoPointLejaObjective
):
    pass


class LejaSequence:
    def __init__(self, objective: LejaObjective, optimizer: Optimizer):
        if not isinstance(objective, LejaObjective):
            raise ValueError("objective must be an instance of LejaObjective")
        self._obj = objective
        if not isinstance(optimizer, Optimizer):
            raise ValueError(
                "optimizer {0} must be instance of Optimizer".format(optimizer)
            )
        self._optimizer = optimizer
        self._bkd = objective._bkd
        self._optimizer.set_objective_function(self._obj)

    def _step(self):
        iterates, bounds = self._obj._initial_iterates()
        results = []
        for jj in range(iterates.shape[1]):
            self._optimizer.set_bounds(bounds[jj])
            # self._obj.plot(plt.figure().gca())
            # plt.show()
            results.append(self._optimizer.minimize(iterates[:, jj : jj + 1]))
        best_idx = self._bkd.argmin(
            self._bkd.array([res.fun for res in results])
        )
        chosen_samples = self._bkd.reshape(
            results[best_idx].x, (1, results[best_idx].x.shape[0])
        )
        sequence = self._bkd.hstack((self._obj.sequence(), chosen_samples))
        self._obj.set_sequence(sequence)

    def step(self, nsamples: int):
        if nsamples <= self.nsamples():
            raise ValueError(
                "nsamples {0} must be >= size of current sequence {1}".format(
                    nsamples, self.nsamples()
                )
            )
        if ((nsamples - self.nsamples()) % self._obj._nopt_vars()) != 0:
            raise ValueError(
                "extra samples requested must be divisisible by {0}".format(
                    self._obj._nopt_vars()
                )
            )
        while self.nsamples() < nsamples:
            self._step()

    def sequence(self) -> Array:
        return self._bkd.copy(self._obj.sequence())

    def nsamples(self) -> int:
        return self._obj.nsamples()

    def plot(self, ax):
        return self._obj.plot(ax)

    def quadrature_weights(self, sequence: Array) -> Array:
        sqrt_weights = self._bkd.sqrt(self._obj._compute_weights(sequence))
        # ignore last basis which exists for when new points
        # are added to sequence
        basis_mat = self._obj._poly(sequence)[:, : sequence.shape[1]]
        basis_mat_inv = self._bkd.inv(sqrt_weights * basis_mat)
        # make sure to adjust weights to account for preconditioning
        quad_weights = (basis_mat_inv[0, :] * sqrt_weights[:, 0])[:, None]
        return quad_weights

    def __repr__(self) -> str:
        return "{0}(nsamples={1})".format(
            self.__class__.__name__, self.nsamples()
        )


def setup_univariate_leja_sequence(
    marginal,
    objective_class,
    optimizer: Optimizer = None,
    init_sequence: Array = None,
    backend: BackendMixin = NumpyMixin,
):
    marginal = parse_marginal(marginal, backend)
    if optimizer is None:
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(gtol=1e-6, maxiter=1000, method="trust-constr")
    if init_sequence is None:
        init_sequence = backend.array([[marginal.mean()]])
    poly = setup_univariate_orthogonal_polynomial_from_marginal(
        marginal, backend=backend, transform_enforce_bounds=True
    )
    obj = objective_class(marginal, poly)
    obj.set_sequence(init_sequence)
    return LejaSequence(obj, optimizer)


class LejaQuadratureRule(UnivariateQuadratureRule):
    def __init__(
        self,
        marginal,
        optimizer: Optimizer = None,
        init_sequence: Array = None,
        backend: BackendMixin = NumpyMixin,
        store: bool = False,
    ):
        self._leja_seq = setup_univariate_leja_sequence(
            marginal,
            self._leja_objective_class(),
            optimizer=optimizer,
            init_sequence=init_sequence,
            backend=backend,
        )
        super().__init__(backend, store)

    @abstractmethod
    def _leja_objective_class(self) -> LejaObjective:
        raise NotImplementedError

    def _quad_rule(self, nnodes: int) -> Tuple[Array, Array]:
        if self._leja_seq.nsamples() < nnodes:
            self._leja_seq.step(nnodes)
        return (
            self._leja_seq.sequence()[:, :nnodes],
            self._leja_seq.quadrature_weights(
                self._leja_seq.sequence()[:, :nnodes]
            ),
        )

    def __repr__(self) -> str:
        return "{0}(poly={1}, bkd={2})".format(
            self.__class__.__name__,
            self._seq._objective._poly,
            self._bkd.__name__,
        )


class OnePointChristoffelLejaQuadratureRule(LejaQuadratureRule):
    def _leja_objective_class(self) -> LejaObjective:
        return OnePointChristoffelLejaObjective


class TwoPointChristoffelLejaQuadratureRule(LejaQuadratureRule):
    def _leja_objective_class(self) -> LejaObjective:
        return TwoPointChristoffelLejaObjective


class OnePointPDFLejaQuadratureRule(LejaQuadratureRule):
    def _leja_objective_class(self) -> LejaObjective:
        return OnePointPDFLejaObjective


class TwoPointPDFLejaQuadratureRule(LejaQuadratureRule):
    def _leja_objective_class(self) -> LejaObjective:
        return TwoPointPDFLejaObjective

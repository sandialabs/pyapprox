from abc import ABC, abstractmethod

import numpy as np
from scipy import stats, special

from pyapprox.interface.model import Model
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.marginals import get_pdf, get_distribution_info
from pyapprox.optimization.pya_minimize import Optimizer


class Marginal(ABC):
    def __init__(self, backend):
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend

    @abstractmethod
    def _pdf(self, samples):
        raise NotImplementedError

    def pdf(self, samples):
        self._check_samples(samples)
        vals = self._pdf(samples)
        if vals.ndim != 2 or vals.shape[1] != 1:
            raise ValueError("vals must be a 2d column vector")
        return vals

    def _check_samples(self, samples):
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be 2d row vector")

    def _pdf_jacobian(self, samples):
        raise NotImplementedError

    def pdf_jacobian(self, samples):
        self._check_samples(samples)
        jac = self._pdf_jacobian(samples)
        if jac.shape != (1, samples.shape[1]):
            raise ValueError("jacobian must be a 2D row vector")
        return jac


class ScipyMarginal(Marginal):
    def __init__(self, marginal, backend=None):
        super().__init__(backend)
        self._marginal = marginal
        self._marginal_pdf = get_pdf(marginal)
        self._name, self._scales, self._shapes = get_distribution_info(
            marginal
        )

    def _pdf(self, samples):
        return self._bkd.asarray(self._marginal_pdf(samples[0]))[:, None]


class GaussianMarginal(ScipyMarginal):
    def __init__(self, marginal, backend=None):
        super().__init__(marginal, backend)
        if self._name != "norm":
            raise ValueError("marginal must be stats.norm")

    def _pdf_jacobian(self, samples):
        mu = self._scales["loc"][0]
        sigma = self._scales["scale"][0]
        return (
            self._marginal_pdf(samples[0]) * (mu - samples[0]) / sigma**2
        )[:, None]


class BetaMarginal(ScipyMarginal):
    def __init__(self, marginal, backend=None):
        super().__init__(marginal, backend)
        self._lb, self._ub = self._marginal.interval(1)
        if self._name != "beta":
            raise ValueError("marginal must be stats.beta")
        self._alpha, self._beta = self._shapes["a"], self._shapes["b"]
        self._const = 1.0 / special.beta(self._alpha, self._beta)

    def _pdf_01(self, samples):
        return (
            samples.T ** (self._alpha - 1)
            * (1 - samples.T) ** (self._beta - 1)
        )*self._const

    def _pdf(self, samples):
        denom = self._ub-self._lb
        return self._pdf_01((samples - self._lb) / denom) / denom

    def _pdf_jacobian_01(self, sample):
        deriv = self._bkd.zeros(sample.shape)
        if self._alpha > 1:
            deriv += (self._alpha - 1) * (
                sample ** (self._alpha - 2) * (1 - sample) ** (self._beta - 1)
            )
        if self._beta > 1:
            deriv -= (self._beta - 1) * (
                sample ** (self._alpha - 1) * (1 - sample) ** (self._beta - 2)
            )
        return (deriv * self._const)

    def _pdf_jacobian(self, sample):
        denom = self._ub-self._lb
        return self._pdf_jacobian_01((sample - self._lb) / denom) / denom**2


class UniformMarginal(BetaMarginal):
    def __init__(self, marginal, backend=None):
        name = get_distribution_info(marginal)[0]
        if name != "uniform":
            raise ValueError("marginal must be stats.uniform")
        lb, ub = marginal.interval(1)
        super().__init__(
            stats.beta(1, 1, loc=lb, scale=ub - lb), backend=backend
        )


def canonical_pdf_jacobian_from_marginal(marginal):
    name, scales, shapes = get_distribution_info(marginal)
    marginals = {
        "uniform": UniformMarginal,
        "beta": BetaMarginal,
        "gaussian": GaussianMarginal,
    }
    if name not in marginals:
        raise ValueError("{0} not supported".format(name))
    return marginals[name]()


class LejaObjective(Model):
    def __init__(self, poly):
        super().__init__(backend=poly._bkd)
        self._poly = poly
        self._coef = None
        self._jacobian_implemented = True
        self._sequence = None

    def nqoi(self):
        return 1

    def nsamples(self):
        return self._sequence.shape[1]

    def sequence(self):
        return self._bkd.copy(self._sequence)

    @abstractmethod
    def _compute_weights(self, samples):
        raise NotImplementedError

    @abstractmethod
    def _compute_weights_jacobian(self, samples):
        raise NotImplementedError

    @abstractmethod
    def _plot_bounds(self, samples):
        raise NotImplementedError

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    def set_sequence(self, sequence):
        if sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError("sequence must be a 2d array row vector")
        self._sequence = sequence
        self._weights = self._compute_weights(self._sequence)
        self._weights_jac = self._compute_weights_jacobian(self._sequence)
        if self._weights.ndim != 2 or self._weights.shape[1] != 1:
            raise ValueError("weights must be a 2d array column vector")
        self._poly.set_nterms(self.nsamples() + 1)
        basis_vals = self._poly(self._sequence)
        self._basis_mat = basis_vals[:, :-1]
        self._basis_vec = basis_vals[:, -1:]
        self._set_coefficients()

    def _set_coefficients(self):
        self._coef = self._bkd.lstsq(
            self._weights * self._basis_mat,
            self._weights * self._basis_vec,
        )

    def _values(self, samples):
        basis_mat = self._poly(samples)
        basis_mat, new_basis = basis_mat[:, :-1], basis_mat[:, -1:]
        weights = self._compute_weights(samples)
        pvals = basis_mat @ self._coef
        residual = new_basis - pvals
        return -weights * self._bkd.sum(residual**2, axis=1)[:, None]

    def _jacobian(self, sample):
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
        return -jac


class PDFLejaObjective(LejaObjective):
    def __init__(self, marginal, poly):
        super().__init__(poly)
        if not isinstance(marginal, Marginal):
            raise ValueError("marginal must be an instance of Marginal")
        self._marginal = marginal
        self._bounds = marginal._marginal.interval(1)

    def _compute_weights(self, samples):
        return self._marginal.pdf(samples)

    def _compute_weights_jacobian(self, samples):
        return self._marginal.pdf_jacobian(samples)

    def _plot_bounds(self):
        return self._marginal._marginal.interval(1-1e-3)


class LejaSequence:
    def __init__(self, objective, optimizer):
        if not isinstance(objective, LejaObjective):
            raise ValueError("objective must be an instance of LejaObjective")
        self._obj = objective
        if not isinstance(optimizer, Optimizer):
            raise ValueError(
                "optimizer {0} must be instance of Optimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer
        self._bkd = objective._bkd
        self._optimizer.set_objective_function(self._obj)

    def _step(self):
        iterates, intervals = self._initial_iterates()
        results = []
        for jj in range(iterates.shape[1]):
            bounds = self._bkd.array([[intervals[jj], intervals[jj+1]]])
            self._optimizer.set_bounds(bounds)
            results.append(self._optimizer.minimize(iterates[:, jj:jj+1]))
        best_idx = self._bkd.argmin(
            self._bkd.array([res.fun for res in results])
        )
        best_sample = results[best_idx].x
        sequence = self._bkd.hstack((self._obj.sequence(), best_sample))
        self._obj.set_sequence(sequence)

    def step(self, nsamples):
        if nsamples <= self.nsamples():
            raise ValueError(
                "nsamples {0} must be >= size of current sequence {1}".format(
                    nsamples, self.nsamples())
            )
        while self.nsamples() < nsamples:
            self._step()

    def _initial_iterates(self):
        eps = 1e-6  # must be larger than optimization tolerance
        bounds = self._obj._bounds
        intervals = np.sort(self._obj._sequence)
        if (
                np.isfinite(bounds[0])
                and (self._bkd.min(self._obj._sequence) > bounds[0]+eps)
        ):
            intervals = np.hstack(([[bounds[0]]], intervals))
        if (
                np.isfinite(bounds[1])
                and (self._bkd.max(self._obj._sequence) < bounds[1]-eps)
        ):
            intervals = np.hstack((intervals, [[bounds[1]]]))
        if not np.isfinite(bounds[0]):
            intervals = np.hstack(
                (
                    [[min(1.1*self._bkd.min(self._obj._sequence), -0.1)]],
                    intervals)
            )
        if not np.isfinite(bounds[1]):
            intervals = np.hstack(
                (
                    intervals,
                    [[max(1.1*self._bkd.max(self._obj._sequence), 0.1)]]
                )
            )
        initial_guesses = intervals[:, :-1]+np.diff(intervals)/2.0
        # put intervals in form useful for bounding 1d optimization problems
        intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
        if not np.isfinite(bounds[0]):
            intervals[0] = -np.inf
        if not np.isfinite(bounds[1]):
            intervals[-1] = np.inf
        return initial_guesses, intervals

    def sequence(self):
        return self._bkd.copy(self._obj.sequence())

    def nsamples(self):
        return self._obj.nsamples()

    def plot(self, ax):
        ax.plot(self.sequence()[0], self.sequence()[0]*0, 'o')
        plot_samples = self._bkd.linspace(
            *self._obj._plot_bounds(), 101
        )[None, :]
        ax.plot(plot_samples[0], self._obj(plot_samples))

    def quadrature_weights(self, sequence):
        sqrt_weights = self._bkd.sqrt(self._obj._compute_weights(sequence))
        # ignore last basis which exists for when new points
        # are added to sequence
        basis_mat = self._obj._poly(sequence)[:, :-1]
        basis_mat_inv = self._bkd.inv(sqrt_weights*basis_mat)
        # make sure to adjust weights to account for preconditioning
        quad_weights = (basis_mat_inv[0, :]*sqrt_weights[:, 0])[:, None]
        return quad_weights

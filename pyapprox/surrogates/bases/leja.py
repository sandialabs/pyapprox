from abc import ABC, abstractmethod

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

from pyapprox.interface.model import Model
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.marginals import get_pdf, get_distribution_info
from pyapprox.optimization.pya_minimize import (
    Optimizer, ScipyConstrainedOptimizer
)
from pyapprox.surrogates.bases.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal
)
from pyapprox.util.visualization import get_meshgrid_function_data


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
        )[None, :]


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


def _custom_marginal_from_scipy_marginal(scipy_marginal, backend=None):
    name = scipy_marginal.dist.name
    marginals = {
        "uniform": UniformMarginal,
        "beta": BetaMarginal,
        "norm": GaussianMarginal,
    }
    if name not in marginals:
        raise ValueError("{0} not supported".format(name))
    return marginals[name](scipy_marginal, backend)


class LejaObjective(Model):
    def __init__(self, marginal, poly):
        super().__init__(backend=poly._bkd)
        self._poly = poly
        self._coef = None
        self._jacobian_implemented = True
        self._sequence = None
        self._marginal = None
        self._set_marginal(marginal)
        self._bounds = self._marginal.interval(1)

    def nqoi(self):
        return 1

    def _nopt_vars(self):
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

    def _plot_bounds(self):
        return self._marginal.interval(1-1e-3)

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    def _set_reused_data(self):
        self._poly.set_nterms(self.nsamples() + 1)
        basis_vals = self._poly(self._sequence)
        self._basis_mat = basis_vals[:, :-1]
        self._basis_vec = basis_vals[:, -1:]

    def set_sequence(self, sequence):
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
        return -jac[None, :]

    def plot(self, ax):
        ax.plot(self.sequence()[0], self.sequence()[0]*0, 'o')
        plot_samples = self._bkd.linspace(*self._plot_bounds(), 101)[None, :]
        ax.plot(plot_samples[0], self(plot_samples))

    def _initial_iterates(self):
        eps = 1e-6  # must be larger than optimization tolerance
        bounds = self._bounds
        intervals = np.sort(self._sequence)
        if (
                np.isfinite(bounds[0])
                and (self._bkd.min(self._sequence) > bounds[0]+eps)
        ):
            intervals = np.hstack(([[bounds[0]]], intervals))
        if (
                np.isfinite(bounds[1])
                and (self._bkd.max(self._sequence) < bounds[1]-eps)
        ):
            intervals = np.hstack((intervals, [[bounds[1]]]))
        if not np.isfinite(bounds[0]):
            intervals = np.hstack(
                (
                    [[min(1.1*self._bkd.min(self._sequence), -0.1)]],
                    intervals)
            )
        if not np.isfinite(bounds[1]):
            intervals = np.hstack(
                (
                    intervals,
                    [[max(1.1*self._bkd.max(self._sequence), 0.1)]]
                )
            )
        iterates = intervals[:, :-1]+np.diff(intervals)/2.0
        # put intervals in form useful for bounding 1d optimization problems
        intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
        if not np.isfinite(bounds[0]):
            intervals[0] = -np.inf
        if not np.isfinite(bounds[1]):
            intervals[-1] = np.inf

        bounds = []
        for jj in range(iterates.shape[1]):
            bounds.append(self._bkd.array([[intervals[jj], intervals[jj+1]]]))
        return iterates, bounds


class TwoPointLejaObjective(LejaObjective):
    def _nopt_vars(self):
        return 2

    def _set_reused_data(self):
        self._poly.set_nterms(self.nsamples() + self._nopt_vars())
        basis_vals = self._poly(self._sequence)
        self._basis_mat = basis_vals[:, :-self._nopt_vars()]
        self._basis_vec = basis_vals[:, -self._nopt_vars():]

    def _values(self, samples):
        if samples.shape[0] != 2:
            raise ValueError("samples must be a 2d array with two rows")
        nsamples = samples.shape[1]
        # first row of samples is multiple values of first iterate
        # second row of samples is multiple values of second iterate
        flat_samples = self._bkd.reshape(
            samples, (1, nsamples*self._nopt_vars())
        )
        basis_vals = self._poly(flat_samples)
        basis_mat = basis_vals[:, :-self._nopt_vars()]
        new_basis = basis_vals[:, -self._nopt_vars():]
        sqrt_weights = self._bkd.sqrt(self._compute_weights(flat_samples))
        pvals = basis_mat @ self._coef
        residuals = sqrt_weights*(new_basis - pvals)
        return -(
            residuals[:nsamples, 0]*residuals[nsamples:, 1]
            - residuals[:nsamples, 1]*residuals[nsamples:, 0]
        )[:, None]**2

    def _jacobian(self, sample):
        flat_sample = self._bkd.reshape(sample, (1, self._nopt_vars()))
        vals = self._poly._derivatives(flat_sample, order=1, return_all=True)
        basis_vals = vals[:, : self._poly.nterms()]
        basis_jac = vals[:, self._poly.nterms() :]
        bvals = basis_vals[:, -self._nopt_vars():]
        pvals = basis_vals[:, :-self._nopt_vars()] @ self._coef
        bderivs = basis_jac[:, -self._nopt_vars():]
        pderivs = basis_jac[:, :-self._nopt_vars()] @ self._coef
        sqrt_weights = self._bkd.sqrt(self._compute_weights(flat_sample))
        weights_jac = self._compute_weights_jacobian(flat_sample)
        sqrt_weights_jac = weights_jac/(2*sqrt_weights[:, 0])
        residuals = sqrt_weights*(bvals - pvals)
        residuals_jac = (
            sqrt_weights*(bderivs - pderivs)
            + sqrt_weights_jac.T*(bvals - pvals)
        )
        determinant_jac = self._bkd.stack(
            (residuals_jac[:1, 0]*residuals[1:2, 1]
             - residuals_jac[:1, 1]*residuals[1:2, 0],
             residuals[:1, 0]*residuals_jac[1:2, 1]
             - residuals[:1, 1]*residuals_jac[1:2, 0],
             ),
            axis=1,
        )
        determinant = (
            residuals[:1, 0]*residuals[1:, 1]
            - residuals[:1, 1]*residuals[1:, 0])
        return -2*determinant*determinant_jac

    def plot(self, ax):
        ax.plot(self.sequence()[0], self.sequence()[0], 'o')
        X, Y, Z = get_meshgrid_function_data(
            self.__call__,
            self._plot_bounds()+self._plot_bounds(),
            51,
            bkd=self._bkd
        )
        ncontour_levels = 20
        im = ax.contourf(
            X, Y, Z,
            levels=self._bkd.linspace(Z.min(), Z.max(), ncontour_levels))
        plt.colorbar(im, ax=ax)

    def _initial_iterates(self):
        iterates_1d, bounds_1d = super()._initial_iterates()
        iterates, bounds = [], []
        for ii in range(iterates_1d.shape[1]):
            for jj in range(ii+1, iterates_1d.shape[1]):
                iterates.append(
                    np.vstack((iterates_1d[:, ii], iterates_1d[:, jj]))
                )
                bounds.append(np.vstack((bounds_1d[ii], bounds_1d[jj])))
        return np.hstack(iterates), bounds


class PDFLejaObjectiveMixin:
    def _set_marginal(self, marginal):
        custom_marginal = _custom_marginal_from_scipy_marginal(marginal)
        if not isinstance(custom_marginal, Marginal):
            raise ValueError("marginal must be an instance of Marginal")
        self._marginal = marginal
        self._custom_marginal = custom_marginal

    def _compute_weights(self, samples):
        return self._custom_marginal.pdf(samples)

    def _compute_weights_jacobian(self, samples):
        return self._custom_marginal.pdf_jacobian(samples)


class ChristoffelLejaObjectiveMixin:
    def _set_marginal(self, marginal):
        self._marginal = marginal

    def _christoffel_fun(self, basis_mat):
        return self._bkd.sum(basis_mat**2, axis=1)[:, None]/self.nsamples()

    def _compute_weights(self, samples):
        basis_mat = self._poly(samples)[:, :-self._nopt_vars()]
        return 1/self._christoffel_fun(basis_mat)

    def _compute_weights_jacobian(self, sample):
        vals = self._poly._derivatives(sample, order=1, return_all=True)
        # basis_mat.shape = (nsamples, len(sequence))
        basis_mat = vals[:, : self._poly.nterms()][:, :-self._nopt_vars()]
        # basis_mat.shape = (nsamples, 2)
        basis_jac = vals[:, self._poly.nterms() :][:, :-self._nopt_vars()]
        christoffel_jac = 2/self.nsamples()*self._bkd.sum(
            basis_mat*basis_jac, axis=1
        )[None, :]
        # chain rule g(x) = christoffel_fun(x)
        # d/dx 1/g(x) = -g'(x)/(g(x))
        return (-1/self._christoffel_fun(basis_mat).T**2*christoffel_jac)


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
        iterates, bounds = self._obj._initial_iterates()
        results = []
        for jj in range(iterates.shape[1]):
            self._optimizer.set_bounds(bounds[jj])
            results.append(self._optimizer.minimize(iterates[:, jj:jj+1]))
        best_idx = self._bkd.argmin(
            self._bkd.array([res.fun for res in results])
        )
        chosen_samples = self._bkd.reshape(
            results[best_idx].x, (1, results[best_idx].x.shape[0]))
        sequence = self._bkd.hstack((self._obj.sequence(), chosen_samples))
        self._obj.set_sequence(sequence)

    def step(self, nsamples):
        if nsamples <= self.nsamples():
            raise ValueError(
                "nsamples {0} must be >= size of current sequence {1}".format(
                    nsamples, self.nsamples())
            )
        if ((nsamples-self.nsamples()) % self._obj._nopt_vars()) != 0:
            raise ValueError(
                "extra samples requested must be divisisible by {0}".format(
                    self._obj._nopt_vars())
            )
        while self.nsamples() < nsamples:
            self._step()

    def sequence(self):
        return self._bkd.copy(self._obj.sequence())

    def nsamples(self):
        return self._obj.nsamples()

    def plot(self, ax):
        return self._obj.plot(ax)

    def quadrature_weights(self, sequence):
        sqrt_weights = self._bkd.sqrt(self._obj._compute_weights(sequence))
        # ignore last basis which exists for when new points
        # are added to sequence
        basis_mat = self._obj._poly(sequence)[:, :-self._obj._nopt_vars()]
        basis_mat_inv = self._bkd.inv(sqrt_weights*basis_mat)
        # make sure to adjust weights to account for preconditioning
        quad_weights = (basis_mat_inv[0, :]*sqrt_weights[:, 0])[:, None]
        return quad_weights

    def __repr__(self):
        return "{0}(nsamples={1})".format(
            self.__class__.__name__, self.nsamples()
        )


def setup_univariate_leja_sequence(
        marginal, objective_class, optimizer=None, init_sequence=None,
        backend=NumpyLinAlgMixin):
    if optimizer is None:
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, maxiter=1000, method="trust-constr"
        )
    if init_sequence is None:
        init_sequence = backend.array([[marginal.mean()]])
    poly = setup_univariate_orthogonal_polynomial_from_marginal(
        marginal, backend=backend)
    obj = objective_class(marginal, poly)
    obj.set_sequence(init_sequence)
    return LejaSequence(obj, optimizer)

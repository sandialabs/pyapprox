"""
Classes and functions specific to optimizing  average value at risk

Notes
-----
Sample Average Projection based smoothe versions of AVaR
SampleAverageSmoothedAverageValueAtRisk and
SampleAverageSmoothedAverageValueAtDeviaiton are found in
sampleaverage.py because unlike functions below they do not require
optimization of the value at risk.
"""

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.optimization.sampleaverage import (
    SampleAverageStat,
    SampleAverageConstraint,
)
from pyapprox.interface.model import Model, SingleSampleModel
from pyapprox.optimization.minimize import (
    SlackBasedConstraintFromModel,
    SlackBasedAdjustedConstraint,
    ConstrainedOptimizer,
    SlackBasedOptimizer,
    SmoothLogBasedMaxFunction,
)


class ObjectiveWithAVaRConstraints(Model):
    """
    When optimizing for AVaR additional variables t are introduced.
    This class wraps a function that does not take variables t
    and returns a jacobian that includes derivatives with respect to the
    variables t (they will be zero).

    Assumes samples consist of vstack(random_vars, t)
    """

    def __init__(
        self, model: Model, ncvar_constraints: int, ndesign_vars: int
    ):
        super().__init__(backend=model._bkd)
        self._model = model
        self._ndesign_vars = ndesign_vars
        if model.nqoi() != 1:
            raise ValueError("objective can only have one QoI")
        self._ncvar_constraints = ncvar_constraints
        for attr in [
            "apply_jacobian_implemented",
            "jacobian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))
        # until sampleaveragecvar.hessian is implemented turn
        # off objective hessian
        # self._hessian_implemented = self._model.hessian_implemented()

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._ndesign_vars + self._ncvar_constraints

    def _values(self, design_samples: Array) -> Array:
        return self._model(design_samples[: -self._ncvar_constraints])

    def _apply_jacobian(self, design_sample: Array, vec: Array) -> Array:
        return self._model.apply_jacobian(
            design_sample[: -self._ncvar_constraints],
            vec[: -self._ncvar_constraints],
        )

    def _jacobian(self, design_sample: Array) -> Array:
        jac = self._model.jacobian(design_sample[: -self._ncvar_constraints])
        return self._bkd.hstack(
            (jac, self._bkd.zeros((jac.shape[0], self._ncvar_constraints)))
        )

    def _hessian(self, design_sample: Array) -> Array:
        model_hess = self._model.hessian(
            design_sample[: -self._ncvar_constraints]
        )
        nvars = model_hess.shape[-1]
        hess = self._bkd.zeros(
            (
                self.nqoi(),
                nvars + self._ncvar_constraints,
                nvars + self._ncvar_constraints,
            )
        )
        idx = np.ix_(self._bkd.arange(nvars), self._bkd.arange(nvars))
        hess[:, idx[0], idx[1]] = model_hess
        # hess[:, :nvars, :nvars] = model_hess
        return hess


class SampleAverageAverageValueAtRisk(SampleAverageStat):
    def __init__(
        self,
        alpha: float,
        backend: BackendMixin,
        eps: float = 1e-2,
    ):
        super().__init__(backend)
        alpha = self._bkd.atleast1d(self._bkd.asarray(alpha))
        self._alpha = alpha
        self._max = SmoothLogBasedMaxFunction(1, eps, backend=self._bkd)
        self._t = None

    def jacobian_implemented() -> bool:
        return True

    def set_value_at_risk(self, t: float):
        t = self._bkd.atleast1d(t)
        if t.shape[0] != self._alpha.shape[0]:
            msg = "VaR shape {0} and alpha shape {1} are inconsitent".format(
                t.shape, self._alpha.shape
            )
            raise ValueError(msg)
        if t.ndim != 1:
            raise ValueError("t must be a 1D array")
        self._t = t

    def _values(self, values: Array, weights: Array) -> Array:
        if values.shape[1] != self._t.shape[0]:
            raise ValueError("must specify a VaR for each QoI")
        return self._t + (self._max(values - self._t).T @ weights).T / (
            1 - self._alpha
        )

    def jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        # must overwride jacobian so shape checks are not called from base class
        # the use of t here will violate those constraints
        # grad withe respect to parameters of x
        max_jac = self._max.first_derivative(values - self._t)[..., None]
        param_jac = self._bkd.einsum(
            "ijk,i->jk", (max_jac * jac_values), weights[:, 0]
        ) / (1 - self._alpha[:, None])
        t_jac = 1 - self._bkd.einsum(
            "ij,i->j", max_jac[..., 0], weights[:, 0]
        ) / (1 - self._alpha)
        return self._bkd.hstack((param_jac, self._bkd.diag(t_jac)))


class AVaRSampleAverageConstraint(SampleAverageConstraint):
    def __init__(
        self,
        model: Model,
        samples: Array,
        weights: Array,
        stat: SampleAverageStat,
        design_bounds: Array,
        nvars: int,
        design_indices: Array,
        backend: BackendMixin,
        keep_feasible: bool = False,
    ):
        if not isinstance(stat, SampleAverageAverageValueAtRisk):
            msg = "stat not instance of SampleAverageAverageValueAtRisk"
            raise ValueError(msg)
        self._nconstraints = stat._alpha.shape[0]
        super().__init__(
            model,
            samples,
            weights,
            stat,
            design_bounds,
            nvars,
            design_indices,
            backend,
            keep_feasible,
        )

    def nvars(self) -> int:
        # optimizers obtain nvars from here so must be size
        # of design variables
        return self._design_indices.shape[0] + self._nconstraints

    def apply_jacobian_implemented(self) -> bool:
        # even if model has apply jacobian sample_average_constraint does not
        # so turn off. If it is True then use of jacobian to compute
        # jacobian apply will fail due to passing around VaR
        return False

    def __call__(self, design_sample: Array) -> Array:
        # have to ovewrite call instead of just defining values
        # to avoid error check that will not work here
        # assumes avar variable t is at the end of design_sample
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        return super()._values(design_sample[: -self._nconstraints])

    def _jacobian(self, design_sample: Array) -> Array:
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        jac = super()._jacobian(design_sample[: -self._nconstraints])
        return jac


class AVaRObjective(SingleSampleModel):
    def __init__(self, nmodel_vars: int, backend: BackendMixin):
        self._nmodel_vars = nmodel_vars
        super().__init__(backend)

    def set_beta(self, beta: float):
        self._beta = beta

    def set_quadrature_weights(self, quadw: Array):
        if quadw.ndim != 1:
            raise ValueError("quadw has the wrong shape")
        self._quadw = quadw

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nslack(self) -> int:
        return 1 + self._quadw.shape[0]

    def nvars(self) -> int:
        return self.nslack() + self._nmodel_vars

    def _evaluate(self, sample: Array) -> Array:
        t_slack = sample[:1]
        gamma_slack = sample[1 : self.nslack()]
        return t_slack + 1.0 / (1.0 - self._beta) * self._quadw @ gamma_slack

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.hstack(
            (
                self._bkd.ones((1,)),
                self._quadw / (1.0 - self._beta),
                self._bkd.zeros((sample.shape[0] - self.nslack(),)),
            ),
        )[None, :]

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._bkd.zeros((sample.shape[0], vec.shape[1]))


class AVaRConstraintFromModel(SlackBasedConstraintFromModel):
    def nslack(self) -> int:
        return 1 + self._model.nqoi()

    def _values(self, sample: Array) -> Array:
        return (
            sample[:1].T
            + sample[1 : self.nslack()].T
            - self._model(sample[self.nslack() :])
        )

    def _jacobian(self, sample: Array) -> Array:
        model_jac = self._model.jacobian(sample[self.nslack() :])
        jac = self._bkd.hstack(
            (
                self._bkd.ones((model_jac.shape[0], 1)),
                self._bkd.eye(model_jac.shape[0]),
                -model_jac,
            )
        )
        return jac


class AVaRAdjustedConstraint(SlackBasedAdjustedConstraint):
    def nslack(self) -> int:
        return 1


class AVaRSlackBasedOptimizer(SlackBasedOptimizer):
    """
    AVaR optimization with only slack variables arising from replacing the
    objective. Cannot be used with constraints that require additional
    slack variables.
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        beta: float,
        quadrature_weights: Array,
        backend: BackendMixin,
    ):
        self._beta = beta
        self.set_quadrature_weights(quadrature_weights)
        super().__init__(optimizer, 1 + self._quadw.shape[0], backend)
        self.set_slack_bounds(
            self._bkd.vstack(
                (
                    self._bkd.array([[-np.inf, np.inf]]),
                    self._bkd.tile(
                        self._bkd.array([0, np.inf]), (self.nslack() - 1, 1)
                    ),
                ),
            )
        )

    def set_quadrature_weights(self, quadw: Array):
        if quadw.ndim != 1:
            raise ValueError("quadw has the wrong shape")
        self._quadw = quadw

    def _convert_objective_function(
        self, model: Model
    ) -> AVaRConstraintFromModel:
        return AVaRConstraintFromModel(model, keep_feasible=False)

    def _set_objective(self):
        objective = AVaRObjective(
            self._constraint_from_objective._model.nvars(), backend=self._bkd
        )
        objective.set_beta(self._beta)
        objective.set_quadrature_weights(self._quadw)
        self._optimizer.set_objective_function(objective)


class _AVaRDummyModel(Model):
    """
    Model with no parameters to be used to compute AVaR from a set of samples.
    Only to be used by EmpiricalAVaRSlackBasedOptimizer
    """

    def __init__(self, samples: Array, backend: BackendMixin):
        super().__init__(backend)
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be 2D row vector")
        self._samples = samples

    def jacobian_implemented(self) -> bool:
        return True

    def weighted_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return self._samples.shape[1]

    def nvars(self) -> int:
        return 0

    def _values(self, samples) -> Array:
        return self._samples

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.zeros((self.nqoi(), self.nvars()))

    def _weighted_hessian(self, sample: Array, weights) -> Array:
        return self._bkd.zeros((self.nvars(), self.nvars()))


class EmpiricalAVaRSlackBasedOptimizer(AVaRSlackBasedOptimizer):
    """
    Compute AVaR from a set of samples using the optimization based formulation
    Only intended for testing and tutorials. If one wants to solve with
    optimization one should solve the equivalent linear program.
    The use of nonlinear optimization here is just to check important
    components of non-linear constrained avar minimization without adding
    the complexity of a model
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        beta: float,
        samples: Array,
        quadrature_weights: Array,
        backend: BackendMixin,
    ):
        super().__init__(optimizer, beta, quadrature_weights, backend=backend)
        self.set_objective_function(_AVaRDummyModel(samples, backend=backend))
        self.set_constraints([])

"""InexactWrapper — composition wrapper for inexact evaluation.

Wraps a model + statistic + strategy to provide tolerance-dependent
value and jacobian evaluation. Satisfies ``NonlinearConstraintProtocol``
(and ``ObjectiveProtocol`` when ``nqoi=1``), plus ``InexactEvaluable``
and ``InexactDifferentiable``.
"""

from typing import Generic, List, Optional, cast

from pyapprox.expdesign.protocols.statistics import (
    DifferentiableSampleStatisticProtocol,
    SampleStatisticProtocol,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactGradientStrategyProtocol,
)
from pyapprox.optimization.minimize.utils import assemble_full_samples
from pyapprox.util.backends.protocols import Array, Backend


class InexactWrapper(Generic[Array]):
    """Wraps a model + stat + strategy to provide inexact value/jacobian.

    Given a model ``f(random, design)``, a ``SampleStatistic`` (e.g. mean),
    and an ``InexactGradientStrategy``, evaluates:

    - ``__call__(sample)`` / ``inexact_value(sample, tol)``:
      ``stat(f(random, design), weights)`` using strategy-determined samples
    - ``jacobian(sample)`` / ``inexact_jacobian(sample, tol)``:
      ``stat.jacobian(...)`` using strategy-determined samples

    Parameters
    ----------
    model : FunctionProtocol[Array]
        A function satisfying ``FunctionProtocol``. Must have ``nvars()``,
        ``nqoi()``, ``__call__(samples)``. May also have ``jacobian(sample)``.
    stat : SampleStatisticProtocol[Array]
        A ``SampleStatisticProtocol`` with ``__call__(values, weights)``
        and optionally ``jacobian(values, jac_values, weights)``.
    strategy : InexactGradientStrategyProtocol[Array]
        An ``InexactGradientStrategyProtocol`` with
        ``samples_and_weights(tol)`` returning ``(samples, weights)``.
    design_indices : List[int]
        Indices of design variables within the full model input.
    bkd : Backend[Array]
        Computational backend.
    constraint_lb : Array, optional
        Lower bounds for the constraint. Shape ``(nqoi,)``.
    constraint_ub : Array, optional
        Upper bounds for the constraint. Shape ``(nqoi,)``.
    """

    def __init__(
        self,
        model: FunctionProtocol[Array],
        stat: SampleStatisticProtocol[Array],
        strategy: InexactGradientStrategyProtocol[Array],
        design_indices: List[int],
        bkd: Backend[Array],
        constraint_lb: Optional[Array] = None,
        constraint_ub: Optional[Array] = None,
    ) -> None:
        if not isinstance(model, FunctionProtocol):
            raise TypeError(
                f"model must satisfy FunctionProtocol, "
                f"got {type(model).__name__}"
            )
        if not isinstance(stat, SampleStatisticProtocol):
            raise TypeError(
                f"stat must satisfy SampleStatisticProtocol, "
                f"got {type(stat).__name__}"
            )
        self._model = model
        self._stat = stat
        self._strategy = strategy
        self._design_indices = design_indices
        self._bkd = bkd
        self._constraint_lb = constraint_lb
        self._constraint_ub = constraint_ub

        self._nvars_full: int = model.nvars()
        self._nqoi: int = model.nqoi()

        # Compute random variable indices (complement of design)
        all_indices = set(range(self._nvars_full))
        self._random_indices = sorted(all_indices - set(design_indices))

        # Dynamic binding of jacobian methods
        if (
            isinstance(model, FunctionWithJacobianProtocol)
            and isinstance(stat, DifferentiableSampleStatisticProtocol)
            and stat.jacobian_implemented()
        ):
            self.jacobian = self._jacobian
            self.inexact_jacobian = self._inexact_jacobian

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of design variables."""
        return len(self._design_indices)

    def nqoi(self) -> int:
        """Return the number of constraint/objective outputs."""
        return self._nqoi

    def lb(self) -> Array:
        """Return constraint lower bounds. Shape ``(nqoi,)``."""
        if self._constraint_lb is None:
            raise AttributeError("No lower bounds set")
        return self._constraint_lb

    def ub(self) -> Array:
        """Return constraint upper bounds. Shape ``(nqoi,)``."""
        if self._constraint_ub is None:
            raise AttributeError("No upper bounds set")
        return self._constraint_ub

    def _differentiable_model(self) -> FunctionWithJacobianProtocol[Array]:
        """Typed access to model as differentiable.

        Safe — only called from ``_jacobian_with_samples`` which is only
        reachable when ``isinstance(model, FunctionWithJacobianProtocol)``.
        """
        return cast(FunctionWithJacobianProtocol[Array], self._model)

    def _differentiable_stat(self) -> DifferentiableSampleStatisticProtocol[Array]:
        """Typed access to stat as differentiable.

        Safe — only called from ``_jacobian_with_samples`` which is only
        reachable when ``isinstance(stat, DifferentiableSampleStatisticProtocol)``.
        """
        return cast(DifferentiableSampleStatisticProtocol[Array], self._stat)

    def _evaluate_with_samples(
        self, design_sample: Array, quad_samples: Array, quad_weights: Array,
    ) -> Array:
        """Evaluate stat(model(random, design), weights).

        Parameters
        ----------
        design_sample : Array
            Shape ``(n_design, 1)``.
        quad_samples : Array
            Shape ``(n_random_vars, n_quad_pts)``.
        quad_weights : Array
            Shape ``(n_quad_pts,)``.

        Returns
        -------
        Array
            Shape ``(nqoi, 1)``.
        """
        bkd = self._bkd
        full_samples = assemble_full_samples(
            design_sample,
            quad_samples,
            self._design_indices,
            self._random_indices,
            self._nvars_full,
            bkd,
        )
        model_values = self._model(full_samples)
        weights_2d = bkd.reshape(quad_weights, (1, -1))
        return self._stat(model_values, weights_2d)

    def _jacobian_with_samples(
        self, design_sample: Array, quad_samples: Array, quad_weights: Array,
    ) -> Array:
        """Compute stat jacobian w.r.t. design variables.

        Parameters
        ----------
        design_sample : Array
            Shape ``(n_design, 1)``.
        quad_samples : Array
            Shape ``(n_random_vars, n_quad_pts)``.
        quad_weights : Array
            Shape ``(n_quad_pts,)``.

        Returns
        -------
        Array
            Shape ``(nqoi, n_design)``.
        """
        bkd = self._bkd
        diff_model = self._differentiable_model()
        diff_stat = self._differentiable_stat()
        full_samples = assemble_full_samples(
            design_sample,
            quad_samples,
            self._design_indices,
            self._random_indices,
            self._nvars_full,
            bkd,
        )
        model_values = diff_model(full_samples)

        n_design = len(self._design_indices)
        n_quad = quad_samples.shape[1]
        nqoi = self._nqoi

        # Collect model jacobians at each quad point
        jac_values = bkd.zeros((nqoi, n_quad, n_design))
        for qq in range(n_quad):
            single_sample = full_samples[:, qq : qq + 1]
            jac_full = diff_model.jacobian(single_sample)
            jac_values[:, qq, :] = jac_full[:, self._design_indices]

        weights_2d = bkd.reshape(quad_weights, (1, -1))
        return diff_stat.jacobian(model_values, jac_values, weights_2d)

    def __call__(self, sample: Array) -> Array:
        """Evaluate using all available samples (exact, tol=0).

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.

        Returns
        -------
        Array
            Shape ``(nqoi, 1)``.
        """
        return self.inexact_value(sample, 0.0)

    def inexact_value(self, sample: Array, tol: float) -> Array:
        """Evaluate with tolerance-dependent accuracy.

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        Array
            Shape ``(nqoi, 1)``.
        """
        quad_samples, quad_weights = self._strategy.samples_and_weights(tol)
        return self._evaluate_with_samples(sample, quad_samples, quad_weights)

    def _jacobian(self, sample: Array) -> Array:
        """Jacobian using all available samples (exact, tol=0).

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.

        Returns
        -------
        Array
            Shape ``(nqoi, n_design)``.
        """
        return self._inexact_jacobian(sample, 0.0)

    def _inexact_jacobian(self, sample: Array, tol: float) -> Array:
        """Compute jacobian with tolerance-dependent accuracy.

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        Array
            Shape ``(nqoi, n_design)``.
        """
        quad_samples, quad_weights = self._strategy.samples_and_weights(tol)
        return self._jacobian_with_samples(
            sample, quad_samples, quad_weights,
        )

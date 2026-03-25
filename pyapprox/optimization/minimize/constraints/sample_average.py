"""Sample-average constraint for design under uncertainty.

Wraps a model with quadrature-based statistical constraints. Given a model
``f(random, design)`` and a quadrature rule over the random variables, evaluates
a user-supplied ``SampleStatistic`` (e.g. mean, mean + k*stdev) on the model
outputs and presents the result as a ``NonlinearConstraintProtocol``.

TODO: Check if this can share infrastructure with ``probability/risk/``
stat classes in a future consolidation.
"""

from typing import Generic, List

from pyapprox.optimization.minimize.utils import assemble_full_samples
from pyapprox.util.backends.protocols import Array, Backend


class SampleAverageConstraint(Generic[Array]):
    """Statistical constraint: stat(f(random, design)) in [lb, ub].

    Evaluates a model at quadrature points (integrating out random
    variables), applies a ``SampleStatistic`` to the model outputs, and
    returns the result as a constraint value. Satisfies
    ``NonlinearConstraintProtocol``.

    Jacobian support is dynamically bound when the wrapped model has
    ``jacobian()`` and the statistic has ``jacobian_implemented() == True``.

    Parameters
    ----------
    model : object
        A function satisfying FunctionProtocol. Must have ``nvars()``,
        ``nqoi()``, ``__call__(samples)``. May also have ``jacobian(sample)``.
    quad_samples : Array
        Quadrature points for random variables.
        Shape ``(n_random_vars, n_quad_pts)``.
    quad_weights : Array
        Quadrature weights. Shape ``(n_quad_pts,)`` (1D) — will be
        reshaped internally to ``(1, n_quad_pts)`` for the stat call.
    stat : object
        A ``SampleStatisticProtocol`` with ``__call__(values, weights)``
        and optionally ``jacobian(values, jac_values, weights)``.
    design_indices : List[int]
        Indices of design variables within the full model input.
    constraint_lb : Array
        Lower bounds for the constraint. Shape ``(nqoi,)``.
    constraint_ub : Array
        Upper bounds for the constraint. Shape ``(nqoi,)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        model: object,
        quad_samples: Array,
        quad_weights: Array,
        stat: object,
        design_indices: List[int],
        constraint_lb: Array,
        constraint_ub: Array,
        bkd: Backend[Array],
    ) -> None:
        self._model = model
        self._quad_samples = quad_samples
        # Ensure weights are (1, n_quad_pts) for stat call
        if quad_weights.ndim == 1:
            self._quad_weights = bkd.reshape(quad_weights, (1, -1))
        else:
            self._quad_weights = quad_weights
        self._stat = stat
        self._design_indices = design_indices
        self._constraint_lb = constraint_lb
        self._constraint_ub = constraint_ub
        self._bkd = bkd

        self._nvars_full: int = model.nvars()
        self._nqoi: int = model.nqoi()
        self._n_quad_pts: int = quad_samples.shape[1]

        # Compute random variable indices (complement of design)
        all_indices = set(range(self._nvars_full))
        self._random_indices = sorted(all_indices - set(design_indices))

        # Dynamic binding of jacobian
        if (
            hasattr(model, "jacobian")
            and hasattr(stat, "jacobian_implemented")
            and stat.jacobian_implemented()
        ):
            self.jacobian = self._jacobian

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of design variables."""
        return len(self._design_indices)

    def nqoi(self) -> int:
        """Return the number of constraint outputs."""
        return self._nqoi

    def lb(self) -> Array:
        """Return constraint lower bounds. Shape ``(nqoi,)``."""
        return self._constraint_lb

    def ub(self) -> Array:
        """Return constraint upper bounds. Shape ``(nqoi,)``."""
        return self._constraint_ub

    def _assemble_full_samples(self, design_sample: Array) -> Array:
        """Build full-dimensional samples by combining design + quad points.

        Parameters
        ----------
        design_sample : Array
            Shape ``(n_design, 1)``.

        Returns
        -------
        Array
            Shape ``(nvars_full, n_quad_pts)``.
        """
        return assemble_full_samples(
            design_sample,
            self._quad_samples,
            self._design_indices,
            self._random_indices,
            self._nvars_full,
            self._bkd,
        )

    def __call__(self, sample: Array) -> Array:
        """Evaluate the statistical constraint.

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.

        Returns
        -------
        Array
            Constraint values. Shape ``(nqoi, 1)``.
        """
        full_samples = self._assemble_full_samples(sample)
        # Model output: (nqoi, n_quad_pts)
        model_values = self._model(full_samples)  # type: ignore[operator]
        # Apply statistic: (nqoi, n_quad_pts), (1, n_quad_pts) -> (nqoi, 1)
        return self._stat(model_values, self._quad_weights)  # type: ignore[operator]

    def _jacobian(self, sample: Array) -> Array:
        """Jacobian of the statistical constraint w.r.t. design variables.

        Parameters
        ----------
        sample : Array
            Design variable values. Shape ``(n_design, 1)``.

        Returns
        -------
        Array
            Jacobian. Shape ``(nqoi, n_design)``.
        """
        bkd = self._bkd
        full_samples = self._assemble_full_samples(sample)

        # Model output: (nqoi, n_quad_pts)
        model_values = self._model(full_samples)  # type: ignore[operator]

        # Collect model jacobians at each quad point
        n_design = len(self._design_indices)
        n_quad = self._n_quad_pts
        nqoi = self._nqoi

        # jac_values: (nqoi, n_quad_pts, n_design)
        jac_values = bkd.zeros((nqoi, n_quad, n_design))
        for qq in range(n_quad):
            single_sample = full_samples[:, qq : qq + 1]
            # Full jacobian: (nqoi, nvars_full)
            jac_full = self._model.jacobian(single_sample)
            # Extract design columns: (nqoi, n_design)
            jac_values[:, qq, :] = jac_full[:, self._design_indices]

        # Apply stat jacobian: (nqoi, n_quad, n_design), -> (nqoi, n_design)
        return self._stat.jacobian(
            model_values, jac_values, self._quad_weights
        )

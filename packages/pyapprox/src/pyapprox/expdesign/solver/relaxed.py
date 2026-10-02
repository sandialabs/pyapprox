"""
Relaxed (continuous) solver for OED problems.

The relaxed solver treats design weights as continuous variables and
searches the feasible set given by a design space, using trust-region
constrained optimization. By default the set is ``[weight_floor, 1]`` per
weight with a sum-to-one constraint.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Tuple

from pyapprox.expdesign.design_space import BoxBudgetDesignSpace
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.expdesign.protocols.design_space import DesignSpaceProtocol
from pyapprox.expdesign.protocols.objective import OEDObjectiveProtocol
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.protocols import Array, Backend


@dataclass
class RelaxedOEDConfig:
    """Configuration for relaxed OED solver.

    Parameters
    ----------
    verbosity : int
        Optimizer verbosity level. Default 0.
    maxiter : int, optional
        Maximum optimizer iterations. None uses SciPy default.
    gtol : float, optional
        Gradient tolerance. None uses SciPy default.
    xtol : float, optional
        Step tolerance. None uses SciPy default.
    weight_floor : float
        Lower bound on every design weight. MC OED objectives divide
        by the weights (effective noise variance ``sigma^2 / w_i``),
        so the objective is singular at ``w_i = 0``; a positive floor
        enforces iterates stay in the domain of definition instead of
        relying on the optimizer keeping strictly interior iterates.
        The optimum shifts by at most ``nobs * weight_floor`` in
        probability mass (value change well below MC noise at the
        default). Set to 0.0 to recover the closed simplex. Used only
        by the default design space; ignored when the solver is given
        a ``design_space``.
    """

    verbosity: int = 0
    maxiter: Optional[int] = None
    gtol: Optional[float] = None
    xtol: Optional[float] = None
    weight_floor: float = 1e-6


class RelaxedOEDSolver(Generic[Array]):
    """Relaxed (continuous) solver for any OED objective.

    Solves the continuous relaxation of the OED problem:
        min objective(w)
        s.t. w in the design space

    The default design space is
        sum(w) = 1
        weight_floor <= w_i <= 1

    Parameters
    ----------
    objective : OEDObjectiveProtocol[Array]
        Any OED objective function satisfying the protocol.
    config : RelaxedOEDConfig, optional
        Solver configuration for the default optimizer. Uses defaults
        if None. Ignored when ``optimizer`` is provided.
    optimizer : BindableOptimizerProtocol, optional
        Configured unbound optimizer (e.g. ``ScipyTrustConstrOptimizer``).
        Cloned during ``solve()`` to avoid shared state. If None, a
        ``ScipyTrustConstrOptimizer`` is built from ``config``.
    design_space : DesignSpaceProtocol, optional
        Feasible set of the weights, supplying bounds, constraints and
        the default starting point. If None, weights lie in
        ``[config.weight_floor, 1]`` and sum to one.
    """

    def __init__(
        self,
        objective: OEDObjectiveProtocol[Array],
        config: Optional[RelaxedOEDConfig] = None,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        design_space: Optional[DesignSpaceProtocol[Array]] = None,
    ) -> None:
        self._objective = objective
        self._bkd = objective.bkd()
        self._config = config or RelaxedOEDConfig()
        self._optimizer = optimizer
        self._nobs = objective.nvars()
        if design_space is None:
            design_space = BoxBudgetDesignSpace(
                self._nobs,
                1.0,
                self._bkd,
                lower=self._config.weight_floor,
                upper=1.0,
            )
        if not isinstance(design_space, DesignSpaceProtocol):
            raise TypeError(
                "design_space must satisfy DesignSpaceProtocol, got "
                f"{type(design_space).__name__}"
            )
        if design_space.nvars() != self._nobs:
            raise ValueError(
                f"design_space has {design_space.nvars()} weights but the "
                f"objective has {self._nobs}"
            )
        self._design_space = design_space

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def design_space(self) -> DesignSpaceProtocol[Array]:
        """Feasible set of the weights."""
        return self._design_space

    def solve(self, init_weights: Optional[Array] = None) -> Tuple[Array, float]:
        """Solve the relaxed OED problem.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nobs, 1)
            If None, uses the design space's starting point.

        Returns
        -------
        optimal_weights : Array
            Optimal design weights. Shape: (nobs, 1)
        optimal_value : float
            Objective value at optimal design.
        """
        if init_weights is None:
            init_weights = self._design_space.initial()

        optimizer: BindableOptimizerProtocol[Array]
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            # Explicit type application: the constructor takes no
            # Array-typed arguments, so inference cannot bind Array
            optimizer = ScipyTrustConstrOptimizer[Array](
                verbosity=self._config.verbosity,
                maxiter=self._config.maxiter,
                gtol=self._config.gtol,
                xtol=self._config.xtol,
            )
        optimizer.bind(
            self._objective,
            self._design_space.bounds(),
            self._design_space.constraints(),
        )

        # Run optimization
        result = optimizer.minimize(init_weights)

        # Extract optimal weights (optima() returns (nvars, 1))
        optimal_weights = result.optima()

        # Compute objective value at optimal
        optimal_value = float(
            self._bkd.to_numpy(self._objective(optimal_weights))[0, 0]
        )

        return optimal_weights, optimal_value


class RelaxedKLOEDSolver(RelaxedOEDSolver[Array]):
    """Relaxed (continuous) solver for KL-OED.

    Specializes RelaxedOEDSolver for KL-OED objectives, adding
    expected_information_gain to the return values.

    Solves the continuous relaxation of the OED problem:
        min -EIG(w)
        s.t. w in the design space (by default sum(w) = 1 and
             weight_floor <= w_i <= 1)

    Parameters
    ----------
    objective : KLOEDObjective[Array]
        The KL-OED objective function.
    config : RelaxedOEDConfig, optional
        Solver configuration for the default optimizer. Uses defaults
        if None. Ignored when ``optimizer`` is provided.
    optimizer : BindableOptimizerProtocol, optional
        Configured unbound optimizer, cloned during ``solve()``.
    design_space : DesignSpaceProtocol, optional
        Feasible set of the weights. See ``RelaxedOEDSolver``.
    """

    def __init__(
        self,
        objective: KLOEDObjective[Array],
        config: Optional[RelaxedOEDConfig] = None,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        design_space: Optional[DesignSpaceProtocol[Array]] = None,
    ) -> None:
        super().__init__(objective, config, optimizer, design_space)
        self._kl_objective = objective

    def solve(self, init_weights: Optional[Array] = None) -> Tuple[Array, float]:
        """Solve the relaxed KL-OED problem.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nobs, 1)
            If None, uses the design space's starting point.

        Returns
        -------
        optimal_weights : Array
            Optimal design weights. Shape: (nobs, 1)
        optimal_eig : float
            Expected information gain at optimal design.
        """
        optimal_weights, _ = super().solve(init_weights)

        # Compute EIG at optimal
        optimal_eig = self._kl_objective.expected_information_gain(optimal_weights)

        return optimal_weights, optimal_eig

"""
Trajectory quadrature: a stepper's scheme-implied rule applied to
stored trajectories.

Time-integrated functionals approximate :math:`\\int_0^T q(y(t))\\, dt`
as :math:`\\sum_j w_j\\, q(\\hat{y}_j)` where the sample states
:math:`\\hat{y} = S y` are affine combinations of the stored trajectory
columns. The pair :math:`(w, S)` is implied by the time-integration
scheme, so each stepper constructs its own rule object via
``trajectory_quadrature(times)`` — consumers obtain instances from the
stepper of an actual solve and never build weights by hand, which
would silently break the order-consistency between scheme and
quadrature. A new stepper with a new sample layout implements
:class:`TrajectoryQuadratureProtocol` (or reuses a constructor below)
and returns it; nothing else changes.

A quadrature integrates a QoI. A functional that needs the scheme's
summed step residuals themselves (a time-integrated boundary flux) asks
the quadrature for its :class:`StepResidualRuleProtocol`: the states,
times and weights at which the scheme evaluates ``F``, and how its mass
terms telescope. For a scheme that evaluates ``F`` at this quadrature's
samples the two coincide (:class:`LinearStepResidualRule`); a scheme
whose stages are not reproduced by any rule here returns None.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MassTelescopeProtocol(Protocol):
    """How a scheme's mass terms combine when its steps are summed.

    A scheme fact, supplied by the stepper to the quadrature it builds;
    see :meth:`StepResidualRuleProtocol.mass_telescope_coefficients`.
    """

    def coefficients(self, times: Array, bkd: Backend[Array]) -> Array:
        """Return :math:`c`, shape ``(ntimes,)``, for these step times."""
        ...


class OneStepMassTelescope:
    """A one-step scheme: the mass terms :math:`M(y_n - y_{n-1})`
    telescope to :math:`M(y_N - y_0)`, so :math:`c = e_N - e_0`."""

    def coefficients(self, times: Array, bkd: Backend[Array]) -> Array:
        ntimes = int(times.shape[0])
        coefficients = bkd.copy(bkd.zeros((ntimes,)))
        coefficients[0] = -1.0
        coefficients[ntimes - 1] = 1.0
        return coefficients

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class StepResidualDeclaration(Generic[Array]):
    """What a stepper declares about its step residuals.

    Both facts come from the stepper together, so a functional built on
    the summed step residuals cannot pair a telescope with a mass the
    scheme did not use.

    Attributes
    ----------
    mass_telescope : MassTelescopeProtocol
        How the scheme's mass terms combine over its summed steps.
    mass : MassMatrixProtocol
        The mass the scheme steps with: the stepper's own residual's
        ``mass_matrix()``, lumped or consistent as the scheme was built.
    """

    mass_telescope: MassTelescopeProtocol
    mass: MassMatrixProtocol[Array]


@runtime_checkable
class TrajectoryQuadratureProtocol(Protocol, Generic[Array]):
    """Quadrature over a stored trajectory: Q = sum_j w_j q((S y)_j)."""

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def ntimes(self) -> int:
        """Return the number of stored trajectory columns."""
        ...

    def nsamples(self) -> int:
        """Return the number of quadrature samples."""
        ...

    def weights(self) -> Array:
        """Return quadrature weights. Shape: ``(nsamples,)``."""
        ...

    def sample(self, traj: Array) -> Array:
        """Map a trajectory to sample states :math:`S y`.

        Parameters
        ----------
        traj : Array
            Trajectory. Shape: ``(nstates, ntimes)``.

        Returns
        -------
        Array
            Sample states. Shape: ``(nstates, nsamples)``.
        """
        ...

    def accumulate(self, sampled: Array) -> Array:
        """Map sample-space values back by :math:`S^\\top`.

        The adjoint of :meth:`sample`; chain rule for trajectory
        derivatives: :math:`dQ/dy = S^\\top (w \\odot q'(\\hat{y}))`.

        Parameters
        ----------
        sampled : Array
            Sample-space values. Shape: ``(nstates, nsamples)``.

        Returns
        -------
        Array
            Trajectory-space values. Shape: ``(nstates, ntimes)``.
        """
        ...

    def is_time_diagonal(self) -> bool:
        """Return True when every sample involves exactly one column.

        Time-diagonal rules make a time-integrated functional's second
        derivative block-diagonal in time — the structure the per-step
        HVP machinery assumes.
        """
        ...

    def nodal_weights(self) -> Array:
        """Return dense per-column weights :math:`S^\\top w`.

        Shape: ``(ntimes,)``. Raises for time-coupled rules, where a
        single per-column weight cannot represent the quadrature.
        """
        ...

    def sample_times(self) -> Array:
        """Return the time of each sample state. Shape: ``(nsamples,)``.

        A functional whose integrand depends on time evaluates it at
        these times: the node times for nodal rules, the interval
        midpoints for the midpoint rule.
        """
        ...

    def step_residual_rule(self) -> Optional["StepResidualRuleProtocol[Array]"]:
        """Return the scheme's step-residual structure, or None.

        This quadrature integrates a QoI; the step-residual rule
        reproduces the scheme's summed step residuals, which a
        time-integrated boundary flux needs. The two coincide when the
        scheme evaluates ``F`` exactly at this rule's samples. They
        differ for Heun, whose trapezoid rule integrates a QoI correctly
        but whose step evaluates ``F`` at an unstored predictor state;
        its rule returns None until a rule that reconstructs the
        predictor exists.
        """
        ...


@runtime_checkable
class StepResidualRuleProtocol(Protocol, Generic[Array]):
    """The states, times and weights at which a scheme evaluates ``F``.

    With :math:`Y` the trajectory, the scheme's step residuals summed
    over all steps are

    .. math::

        M Y c - \\sum_j w_j F(\\hat{y}_j, t_j),

    with :math:`c` the mass telescope and :math:`\\hat{y} = S(Y)` the
    stage states. Exact, not an approximation: a functional built on it
    (a time-integrated boundary flux) is discretely conservative.

    The stage map takes the whole trajectory, and its adjoint action the
    trajectory too, so that a scheme whose stages are nonlinear in the
    stored states (Heun's predictor) fits the same interface; such a
    rule reports ``is_linear() is False`` and additionally supplies its
    parameter and curvature actions.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def ntimes(self) -> int:
        """Return the number of stored trajectory columns."""
        ...

    def nsamples(self) -> int:
        """Return the number of stage states."""
        ...

    def weights(self) -> Array:
        """Return the weight of each stage state. Shape: ``(nsamples,)``."""
        ...

    def sample_times(self) -> Array:
        """Return the time of each stage state. Shape: ``(nsamples,)``."""
        ...

    def sample(self, traj: Array) -> Array:
        """Return the stage states :math:`S(Y)`.

        Shape: ``(nstates, nsamples)`` from ``traj`` ``(nstates,
        ntimes)``.
        """
        ...

    def accumulate(self, traj: Array, sampled: Array) -> Array:
        """Apply :math:`(dS/dY)^\\top` at ``traj`` to ``sampled``.

        Shape: ``(nstates, ntimes)`` from ``sampled`` ``(nstates,
        nsamples)``. For a linear rule, independent of ``traj``.
        """
        ...

    def mass_telescope_coefficients(self) -> Array:
        """Return :math:`c`, shape ``(ntimes,)``; :math:`e_N - e_0` for
        one-step schemes."""
        ...

    def is_time_diagonal(self) -> bool:
        """Return True when every stage state involves one column."""
        ...

    def is_linear(self) -> bool:
        """Return True when the stage states are linear in the stored
        trajectory and independent of the parameters, so the stage map
        contributes no parameter derivative and no curvature."""
        ...

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the mass :math:`M` the scheme steps with."""
        ...


class NodalTrajectoryQuadrature(Generic[Array]):
    """Quadrature whose samples are stored trajectory columns.

    Covers the rules of backward Euler (right endpoints), forward
    Euler (left endpoints), and Crank-Nicolson/Heun (all nodes,
    trapezoidal weights).

    Parameters
    ----------
    node_indices : Array
        Column index of each sample, all distinct. Shape:
        ``(nsamples,)``, integer dtype.
    weights : Array
        Quadrature weights. Shape: ``(nsamples,)``.
    times : Array
        Time of each stored trajectory column. Shape: ``(ntimes,)``.
    bkd : Backend
        Backend for array operations.
    step_residual : StepResidualDeclaration, optional
        The scheme's mass telescope and mass. Given only by a stepper
        whose step evaluates ``F`` exactly at this rule's samples;
        :meth:`step_residual_rule` then returns this rule as the
        scheme's step-residual rule. Default None: only the stepper that
        owns the rule declares it.
    """

    def __init__(
        self,
        node_indices: Array,
        weights: Array,
        times: Array,
        bkd: Backend[Array],
        step_residual: Optional[StepResidualDeclaration[Array]] = None,
    ) -> None:
        if node_indices.ndim != 1 or weights.ndim != 1 or times.ndim != 1:
            raise ValueError("node_indices, weights and times must be 1D")
        if node_indices.shape[0] != weights.shape[0]:
            raise ValueError(
                f"node_indices has {node_indices.shape[0]} entries but "
                f"weights has {weights.shape[0]}"
            )
        self._node_indices = node_indices
        self._weights = weights
        self._times = times
        self._ntimes = int(times.shape[0])
        self._bkd = bkd
        self._step_residual = _declared_step_residual(step_residual, times, bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ntimes(self) -> int:
        return self._ntimes

    def nsamples(self) -> int:
        return int(self._weights.shape[0])

    def weights(self) -> Array:
        return self._weights

    def sample(self, traj: Array) -> Array:
        return traj[:, self._node_indices]

    def accumulate(self, sampled: Array) -> Array:
        out = self._bkd.copy(self._bkd.zeros((sampled.shape[0], self._ntimes)))
        out[:, self._node_indices] = sampled
        return out

    def is_time_diagonal(self) -> bool:
        return True

    def nodal_weights(self) -> Array:
        out = self._bkd.copy(self._bkd.zeros((self._ntimes,)))
        out[self._node_indices] = self._weights
        return out

    def sample_times(self) -> Array:
        return self._times[self._node_indices]

    def step_residual_rule(self) -> Optional["LinearStepResidualRule[Array]"]:
        if self._step_residual is None:
            return None
        coefficients, mass = self._step_residual
        return LinearStepResidualRule(self, coefficients, mass)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nsamples={self.nsamples()}, ntimes={self._ntimes})"
        )


class MidpointTrajectoryQuadrature(Generic[Array]):
    """Quadrature sampling interval midpoints (implicit midpoint rule).

    Sample states are averages of adjacent columns,
    :math:`\\hat{y}_j = (y_j + y_{j+1})/2` — the states the implicit
    midpoint scheme itself steps through. Not time-diagonal: a
    functional's second derivative couples adjacent columns, so
    per-step HVP support requires the block-tridiagonal extension of
    the adjoint machinery (value and gradient are exact).

    Parameters
    ----------
    weights : Array
        Interval weights. Shape: ``(ntimes - 1,)``.
    times : Array
        Time of each stored trajectory column. Shape: ``(ntimes,)``.
    bkd : Backend
        Backend for array operations.
    step_residual : StepResidualDeclaration, optional
        The scheme's mass telescope and mass. Given only by a stepper
        whose step evaluates ``F`` exactly at this rule's samples;
        :meth:`step_residual_rule` then returns this rule as the
        scheme's step-residual rule. Default None: only the stepper that
        owns the rule declares it.
    """

    def __init__(
        self,
        weights: Array,
        times: Array,
        bkd: Backend[Array],
        step_residual: Optional[StepResidualDeclaration[Array]] = None,
    ) -> None:
        if weights.ndim != 1 or times.ndim != 1:
            raise ValueError("weights and times must be 1D")
        ntimes = int(times.shape[0])
        if weights.shape[0] != ntimes - 1:
            raise ValueError(
                f"midpoint rule needs ntimes - 1 = {ntimes - 1} weights, "
                f"got {weights.shape[0]}"
            )
        self._weights = weights
        self._times = times
        self._ntimes = ntimes
        self._bkd = bkd
        self._step_residual = _declared_step_residual(step_residual, times, bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ntimes(self) -> int:
        return self._ntimes

    def nsamples(self) -> int:
        return self._ntimes - 1

    def weights(self) -> Array:
        return self._weights

    def sample(self, traj: Array) -> Array:
        return 0.5 * (traj[:, :-1] + traj[:, 1:])

    def accumulate(self, sampled: Array) -> Array:
        pad = self._bkd.zeros((sampled.shape[0], 1))
        return 0.5 * (
            self._bkd.hstack([sampled, pad]) + self._bkd.hstack([pad, sampled])
        )

    def is_time_diagonal(self) -> bool:
        return False

    def nodal_weights(self) -> Array:
        raise ValueError(
            "the midpoint rule couples adjacent trajectory columns; no "
            "per-column weight vector represents it. Use sample()/"
            "accumulate(), or a nodal-rule scheme for per-step HVPs."
        )

    def sample_times(self) -> Array:
        return 0.5 * (self._times[:-1] + self._times[1:])

    def step_residual_rule(self) -> Optional["LinearStepResidualRule[Array]"]:
        if self._step_residual is None:
            return None
        coefficients, mass = self._step_residual
        return LinearStepResidualRule(self, coefficients, mass)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nsamples={self.nsamples()}, ntimes={self._ntimes})"
        )


class LinearStepResidualRule(Generic[Array]):
    """A step-residual rule whose stage states are the quadrature's
    samples: linear in the stored trajectory, independent of the
    parameters.

    The rule of every scheme whose step evaluates ``F`` at stored
    columns or affine combinations of them (backward and forward Euler,
    Crank-Nicolson, implicit midpoint). Built by the quadrature, from
    the declaration its stepper made.

    Parameters
    ----------
    quadrature : TrajectoryQuadratureProtocol
        The scheme's quadrature; its samples, times and weights are the
        stage states, times and weights.
    mass_telescope : Array
        The mass telescope coefficients. Shape: ``(ntimes,)``.
    mass : MassMatrixProtocol
        The mass the scheme steps with.
    """

    def __init__(
        self,
        quadrature: TrajectoryQuadratureProtocol[Array],
        mass_telescope: Array,
        mass: MassMatrixProtocol[Array],
    ) -> None:
        self._quadrature = quadrature
        self._mass_telescope = mass_telescope
        self._mass = mass

    def bkd(self) -> Backend[Array]:
        return self._quadrature.bkd()

    def ntimes(self) -> int:
        return self._quadrature.ntimes()

    def nsamples(self) -> int:
        return self._quadrature.nsamples()

    def weights(self) -> Array:
        return self._quadrature.weights()

    def sample_times(self) -> Array:
        return self._quadrature.sample_times()

    def sample(self, traj: Array) -> Array:
        return self._quadrature.sample(traj)

    def accumulate(self, traj: Array, sampled: Array) -> Array:
        return self._quadrature.accumulate(sampled)

    def mass_telescope_coefficients(self) -> Array:
        return self._mass_telescope

    def is_time_diagonal(self) -> bool:
        return self._quadrature.is_time_diagonal()

    def is_linear(self) -> bool:
        return True

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        return self._mass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._quadrature!r})"


def _validate_times(times: Array) -> int:
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError(
            f"times must be 1D with at least 2 entries, got shape {times.shape}"
        )
    return int(times.shape[0])


def _indices(start: int, stop: int, bkd: Backend[Array]) -> Array:
    return bkd.asarray(list(range(start, stop)), dtype=int)


def _declared_step_residual(
    step_residual: Optional[StepResidualDeclaration[Array]],
    times: Array,
    bkd: Backend[Array],
) -> Optional[Tuple[Array, MassMatrixProtocol[Array]]]:
    """The scheme's telescope coefficients for these times and its mass,
    or None if undeclared."""
    if step_residual is None:
        return None
    if not isinstance(step_residual, StepResidualDeclaration):
        raise TypeError(
            "step_residual must be a StepResidualDeclaration, got "
            f"{type(step_residual).__name__}"
        )
    mass_telescope = step_residual.mass_telescope
    if not isinstance(mass_telescope, MassTelescopeProtocol):
        raise TypeError(
            "mass_telescope must satisfy MassTelescopeProtocol, got "
            f"{type(mass_telescope).__name__}"
        )
    if not isinstance(step_residual.mass, MassMatrixProtocol):
        raise TypeError(
            "mass must satisfy MassMatrixProtocol, got "
            f"{type(step_residual.mass).__name__}"
        )
    coefficients = mass_telescope.coefficients(times, bkd)
    ntimes = int(times.shape[0])
    if tuple(coefficients.shape) != (ntimes,):
        raise ValueError(
            f"{type(mass_telescope).__name__} returned coefficients of "
            f"shape {tuple(coefficients.shape)}; expected ({ntimes},)"
        )
    return coefficients, step_residual.mass


def right_rectangle_quadrature(
    times: Array,
    bkd: Backend[Array],
    step_residual: Optional[StepResidualDeclaration[Array]] = None,
) -> NodalTrajectoryQuadrature[Array]:
    """Right endpoints, interval widths — backward Euler's rule.

    ``step_residual`` is declared only by the stepper that owns the
    rule (see :class:`StepResidualDeclaration`).
    """
    ntimes = _validate_times(times)
    return NodalTrajectoryQuadrature(
        _indices(1, ntimes, bkd),
        bkd.diff(times),
        times,
        bkd,
        step_residual,
    )


def left_rectangle_quadrature(
    times: Array,
    bkd: Backend[Array],
    step_residual: Optional[StepResidualDeclaration[Array]] = None,
) -> NodalTrajectoryQuadrature[Array]:
    """Left endpoints, interval widths — forward Euler's rule.

    ``step_residual`` is declared only by the stepper that owns the
    rule (see :class:`StepResidualDeclaration`).
    """
    ntimes = _validate_times(times)
    return NodalTrajectoryQuadrature(
        _indices(0, ntimes - 1, bkd),
        bkd.diff(times),
        times,
        bkd,
        step_residual,
    )


def trapezoidal_quadrature(
    times: Array,
    bkd: Backend[Array],
    step_residual: Optional[StepResidualDeclaration[Array]] = None,
) -> NodalTrajectoryQuadrature[Array]:
    """All nodes, trapezoidal weights — Crank-Nicolson/Heun's rule.

    ``step_residual`` is declared only by the stepper that owns the
    rule (see :class:`StepResidualDeclaration`).
    """
    ntimes = _validate_times(times)
    deltas = bkd.diff(times)
    pad = bkd.zeros((1,))
    weights = 0.5 * (bkd.hstack([deltas, pad]) + bkd.hstack([pad, deltas]))
    return NodalTrajectoryQuadrature(
        _indices(0, ntimes, bkd),
        weights,
        times,
        bkd,
        step_residual,
    )


def midpoint_quadrature(
    times: Array,
    bkd: Backend[Array],
    step_residual: Optional[StepResidualDeclaration[Array]] = None,
) -> MidpointTrajectoryQuadrature[Array]:
    """Interval midpoints, interval widths — implicit midpoint's rule.

    ``step_residual`` is declared only by the stepper that owns the
    rule (see :class:`StepResidualDeclaration`).
    """
    _validate_times(times)
    return MidpointTrajectoryQuadrature(bkd.diff(times), times, bkd, step_residual)

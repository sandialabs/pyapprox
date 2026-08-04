"""Temporal modulation of a separable parameterized field.

A field map is a time-free spatial leaf: it maps parameters to a field
over space. A *modulation* is the other factor of a separable control,

.. math::

    \\text{field}(x, t) = \\sum_k p_k\\, b_k(t)\\, s_k(x),

where :math:`s_k` are the field map's columns and :math:`b_k` are the
temporal profiles declared here. Keeping the two apart is what makes the
Hessian-vector product cheap: the map stays linear in :math:`p`, so its
second derivative is exactly zero and only the per-column scaling
changes with time.

The modulation multiplies the field map's jacobian COLUMN BY COLUMN. Any
other placement is wrong by :math:`O(1)`, not by a small factor --
:math:`b_k` scales each parameter's contribution individually, so a
scalar or row-wise application does not reproduce it. The single place
that applies it is ``_FieldParameterizationTerm._modulated_jacobian``,
so the tangent, the adjoint transpose, and the HVP path cannot disagree.

Implementations are module-level classes rather than closures so that
parameterizations holding them stay picklable (parallel sampling ships
them to workers).
"""

from typing import Generic, List, Protocol, Sequence, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TimeModulationProtocol(Protocol, Generic[Array]):
    """Per-mode temporal profiles :math:`b_k(t)` of a separable field.

    ``is_time_dependent`` is a declaration, not an inference: nothing can
    detect reliably whether a profile consults time, and a silently
    dropped time produces wrong numbers rather than an error. The same
    rule governs coefficient suppliers
    (:mod:`pyapprox.pde.constitutive.coefficient_functions`).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nmodes(self) -> int:
        """Number of profiles; must match the field map's ``nvars()``."""
        ...

    def values(self, time: float) -> Array:
        """Evaluate every profile. Shape: (nmodes,)."""
        ...

    def is_time_dependent(self) -> bool:
        """Whether the profiles actually vary with ``time``."""
        ...


@runtime_checkable
class NonNegativeModulationProtocol(TimeModulationProtocol[Array], Protocol):
    """A modulation that declares :math:`b_k(t) \\ge 0` for all t.

    Sign matters for controls whose admissibility argument rests on a
    sign. An extraction rate :math:`r = \\sum_k p_k b_k(t) q_k(x)` with
    :math:`p_k \\ge 0` preserves the maximum principle only while
    :math:`b_k \\ge 0`; a sign-changing basis (Fourier, Legendre) turns
    extraction into injection at some times and reinstates negative
    concentrations. Consumers that need the guarantee require this
    protocol rather than testing samples of ``values``, which could not
    establish nonnegativity at untested times anyway.
    """

    def is_non_negative(self) -> bool:
        """Whether every profile is nonnegative at every time."""
        ...


class ConstantModulation(Generic[Array]):
    """Unit profiles: :math:`b_k(t) = 1`, the unmodulated case.

    Useful as an explicit "no modulation" that still satisfies the
    protocol, and as the control case in tests that must show a
    non-constant profile changes the answer.
    """

    def __init__(self, bkd: Backend[Array], nmodes: int) -> None:
        if nmodes <= 0:
            raise ValueError(f"nmodes must be positive, got {nmodes}")
        self._bkd = bkd
        self._nmodes = nmodes
        self._ones = bkd.full((nmodes,), 1.0)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nmodes(self) -> int:
        return self._nmodes

    def values(self, time: float) -> Array:
        return self._ones

    def is_time_dependent(self) -> bool:
        """Constant profiles; the time is accepted and ignored."""
        return False

    def is_non_negative(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ConstantModulation(nmodes={self._nmodes})"


class _KnotModulationBase(Generic[Array]):
    """Shared knot handling for the interpolation bases below.

    Subclasses differ ONLY in how a time between knots is turned into
    profile values -- which is exactly the choice of temporal
    interpolation, and the reason it belongs to the modulation rather
    than to the time integrator. The integrator decides WHICH times to
    ask about (forward Euler asks at the step's left end, backward Euler
    at its right, Crank-Nicolson at both, implicit midpoint at the
    middle); the modulation answers what the control is at the time it
    is asked. Building the integrator's rule into the modulation would
    apply that rule twice and break as soon as the stepper changed.
    """

    def __init__(self, bkd: Backend[Array], knots: Sequence[float]) -> None:
        if len(knots) < 2:
            raise ValueError(
                f"knots must have at least 2 entries, got {len(knots)}"
            )
        knot_list = [float(knot) for knot in knots]
        for lower, upper in zip(knot_list, knot_list[1:]):
            if not upper > lower:
                raise ValueError(
                    f"knots must be strictly increasing, got {knot_list}"
                )
        self._bkd = bkd
        self._knots = knot_list

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def knots(self) -> List[float]:
        """The knot sequence defining the profiles."""
        return list(self._knots)

    def is_time_dependent(self) -> bool:
        return True

    def is_non_negative(self) -> bool:
        """Both bases below are nonnegative partitions of unity."""
        return True


class PiecewiseConstantModulation(_KnotModulationBase[Array]):
    """A control held constant on each interval between knots.

    ``nmodes() == len(knots) - 1``: mode ``k`` is one on interval ``k``
    and zero elsewhere. The coarsest useful time dependence, and the one
    whose optimum reads directly as "how hard to act during each window".

    DISCONTINUOUS by construction, which has a consequence worth stating
    because it is easy to meet by accident. Schemes sample a step at
    different instants, so when a knot falls strictly inside a step the
    schemes disagree about which interval that step belongs to -- not an
    implementation defect but the control genuinely jumping where the
    quadrature assumes smoothness, which costs the scheme its order.
    Place knots on time-step boundaries, or use
    :class:`PiecewiseLinearModulation`, whose continuity removes the
    question entirely.

    A time exactly on an interior knot belongs to the interval ENDING
    there, so a step's right endpoint lands in that step's own interval
    rather than the next one.
    """

    def nmodes(self) -> int:
        return len(self._knots) - 1

    def values(self, time: float) -> Array:
        indicators = [0.0] * self.nmodes()
        for index in range(self.nmodes()):
            lower = self._knots[index]
            upper = self._knots[index + 1]
            inside = (
                lower <= time <= upper
                if index == 0
                else lower < time <= upper
            )
            if inside:
                indicators[index] = 1.0
                break
        return self._bkd.asarray(indicators)

    def __repr__(self) -> str:
        return (
            f"PiecewiseConstantModulation(nmodes={self.nmodes()}, "
            f"knots=[{self._knots[0]}, ..., {self._knots[-1]}])"
        )


class PiecewiseLinearModulation(_KnotModulationBase[Array]):
    """A control interpolated linearly between knot values (hat basis).

    ``nmodes() == len(knots)``: mode ``k`` is the hat peaking at knot
    ``k``, so the parameters ARE the control's values at the knots.

    CONTINUOUS, and a partition of unity summing to one at every time in
    range, so a control's total amplitude is preserved and the
    nonnegativity that a sign-based admissibility argument needs is kept.
    Continuity is the practical advantage over the piecewise-constant
    basis: every scheme samples a smooth function, so Crank-Nicolson and
    implicit midpoint agree to their own order instead of straddling a
    jump, and knots need not align with time steps.
    """

    def nmodes(self) -> int:
        return len(self._knots)

    def values(self, time: float) -> Array:
        hats = [0.0] * self.nmodes()
        for index, center in enumerate(self._knots):
            if time == center:
                hats[index] = 1.0
                continue
            if index > 0 and self._knots[index - 1] < time < center:
                lower = self._knots[index - 1]
                hats[index] = (time - lower) / (center - lower)
            elif (
                index < self.nmodes() - 1
                and center < time < self._knots[index + 1]
            ):
                upper = self._knots[index + 1]
                hats[index] = (upper - time) / (upper - center)
        return self._bkd.asarray(hats)

    def __repr__(self) -> str:
        return (
            f"PiecewiseLinearModulation(nmodes={self.nmodes()}, "
            f"knots=[{self._knots[0]}, ..., {self._knots[-1]}])"
        )

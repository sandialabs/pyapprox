r"""Coefficients of linear multistep methods.

A :math:`k`-step method relates the new state to the :math:`k` stored
states before it,

.. math::

    \sum_{l=0}^{k} \alpha_l\, y_{n+1-l}
    = \Delta t \sum_{l=0}^{k} \beta_l\, f(y_{n+1-l}, t_{n+1-l}),

with :math:`\alpha_0 \ne 0` and :math:`\Delta t = t_{n+1} - t_n`; lag
:math:`l` is the stored state :math:`y_{n+1-l}`, lag 0 the new state.
The formula is explicit when :math:`\beta_0 = 0`. The stepper reads only
the coefficients, through :class:`MultistepFormulaProtocol`: a method
gives the formula of any step from that step's times, one coefficient
pair per lag.

Steps may differ in size. :class:`BackwardDifferentiation`,
:class:`AdamsBashforth` and :class:`AdamsMoulton` compute their formulas
from the times, from the Lagrange polynomial through the step's nodes
(differentiated for BDF, integrated for Adams), so a shortened last step
or an adaptive controller's steps keep the method's order, and a startup
step, reading fewer states, is the family's lower-order member. The
coefficients depend only on the times, so the backward sweeps recompute
them exactly from the recorded grid. :class:`MultistepMethod` holds a
fixed table, derived for equal steps, with its startup formulas, and
refuses a step whose earlier steps had another size rather than use the
table there.
"""

from dataclasses import dataclass
from typing import (
    FrozenSet,
    List,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from numpy.polynomial import Polynomial


@dataclass(frozen=True)
class MultistepCoefficients:
    r"""The coefficients :math:`(\alpha_l, \beta_l)_{l=0}^{k}` of one
    linear multistep formula.

    Parameters
    ----------
    alpha, beta : tuple of float
        Indexed by lag, of equal length :math:`k + 1`;
        :math:`\alpha_0 \ne 0`.

    Raises
    ------
    ValueError
        If the lengths differ, there is no stored state (:math:`k = 0`),
        or :math:`\alpha_0 = 0`.
    """

    alpha: Tuple[float, ...]
    beta: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.alpha) != len(self.beta):
            raise ValueError("alpha and beta must have one entry per lag")
        if len(self.alpha) < 2:
            raise ValueError("a multistep formula reads at least one stored state")
        if self.alpha[0] == 0.0:
            raise ValueError("alpha_0 must be nonzero: it scales the new state")

    def depth(self) -> int:
        """The number of stored states the formula reads, :math:`k`."""
        return len(self.alpha) - 1

    def is_explicit(self) -> bool:
        r"""Whether :math:`\beta_0 = 0`: the new state needs a mass solve
        only."""
        return self.beta[0] == 0.0


@runtime_checkable
class MultistepFormulaProtocol(Protocol):
    """A linear multistep method as the stepper reads it: the formula of
    any step, from that step's times."""

    @property
    def name(self) -> str:
        """For messages and ``repr``."""
        ...

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        ...

    def is_explicit(self) -> bool:
        r"""Whether every formula has :math:`\beta_0 = 0`."""
        ...

    def curved_lags(self) -> FrozenSet[int]:
        r"""The lags whose :math:`\beta_l` can be nonzero in some step: only
        they carry second derivatives."""
        ...

    def coefficients_at(self, times: Sequence[float]) -> MultistepCoefficients:
        r"""The formula of a step with times :math:`(t_{n+1}, t_n,
        \dots)`, one per lag, so ``len(times) - 1`` stored states: exactly
        one coefficient pair per lag. Raises ``ValueError`` when it has
        none for those times."""
        ...


@dataclass(frozen=True)
class MultistepMethod:
    """A linear multistep method and its startup.

    Parameters
    ----------
    coefficients : MultistepCoefficients
        The method, reading ``depth`` stored states.
    startup : tuple of MultistepCoefficients
        ``startup[j]`` reads :math:`j + 1` stored states and takes step
        :math:`j`, for :math:`j < k - 1`; empty for a one-step method.
    order : int
        The global order of accuracy.
    name : str
        For messages and ``repr``.

    Raises
    ------
    ValueError
        If a startup formula does not read exactly the states there are
        at its step, or the formulas disagree on being explicit.
    """

    coefficients: MultistepCoefficients
    startup: Tuple[MultistepCoefficients, ...]
    order: int
    name: str

    def __post_init__(self) -> None:
        depth = self.coefficients.depth()
        if len(self.startup) != depth - 1:
            raise ValueError(
                f"{self.name}: a {depth}-step method needs {depth - 1} startup "
                f"formulas, got {len(self.startup)}"
            )
        for j, formula in enumerate(self.startup):
            if formula.depth() != j + 1:
                raise ValueError(
                    f"{self.name}: startup formula {j} must read {j + 1} states, "
                    f"reads {formula.depth()}"
                )
            if formula.is_explicit() != self.coefficients.is_explicit():
                raise ValueError(
                    f"{self.name}: startup formula {j} and the method must be "
                    "both explicit or both implicit"
                )

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        return self.coefficients.depth()

    def is_explicit(self) -> bool:
        r"""Whether every formula has :math:`\beta_0 = 0`."""
        return self.coefficients.is_explicit()

    def formulas(self) -> Tuple[MultistepCoefficients, ...]:
        """The startup formulas, then the method's."""
        return self.startup + (self.coefficients,)

    def curved_lags(self) -> FrozenSet[int]:
        r"""The lags whose :math:`\beta_l` is nonzero in some formula."""
        return frozenset(
            lag
            for formula in self.formulas()
            for lag, beta in enumerate(formula.beta)
            if beta != 0.0
        )

    def at(self, nstates: int) -> MultistepCoefficients:
        """The formula of a step reading ``nstates`` stored states.

        Raises
        ------
        ValueError
            If ``nstates`` is not between 1 and the depth.
        """
        if not 1 <= nstates <= self.depth():
            raise ValueError(
                f"{self.name} reads between 1 and {self.depth()} states, got {nstates}"
            )
        return self.formulas()[nstates - 1]

    def coefficients_at(self, times: Sequence[float]) -> MultistepCoefficients:
        r"""The formula of a step with these times, one per lag:
        :math:`(t_{n+1}, t_n, \dots)`, so ``len(times) - 1`` stored states.
        The result has exactly one coefficient pair per lag.

        These coefficients are fixed, derived for equal steps, so every
        step the formula spans must have the same size (to a relative
        :math:`10^{-10}`, since sizes computed from times differ by
        rounding); a one-state formula spans one step and takes any size.

        Raises
        ------
        ValueError
            If the number of states is not between 1 and the depth, or the
            steps the formula spans differ in size.
        """
        formula = self.at(len(times) - 1)
        sizes = [later - earlier for later, earlier in zip(times, times[1:])]
        if any(abs(size - sizes[0]) > 1e-10 * abs(sizes[0]) for size in sizes):
            raise ValueError(
                f"{self.name} has fixed coefficients, which assume equal steps; "
                f"got step sizes {tuple(sizes)}"
            )
        return formula

    def __repr__(self) -> str:
        return f"MultistepMethod({self.name}, {self.depth()} steps)"


def _states_read(name: str, times: Sequence[float], depth: int) -> int:
    """The number of stored states a step with ``times`` (one per lag)
    reads, checked against the method's depth.

    Raises
    ------
    ValueError
        If it is not between 1 and ``depth``.
    """
    nstates = len(times) - 1
    if not 1 <= nstates <= depth:
        raise ValueError(f"{name} reads between 1 and {depth} states, got {nstates}")
    return nstates


def _lagrange_basis(times: Sequence[float], lags: Sequence[int]) -> List[Polynomial]:
    r"""The Lagrange basis on the nodes of ``lags``, in the step's own
    variable :math:`s = (t - t_n)/(t_{n+1} - t_n)`, so the step runs over
    :math:`s \in [0, 1]` whatever its size and the earlier nodes sit at
    minus the ratios of the earlier steps to this one.

    Parameters
    ----------
    times : sequence of float
        :math:`(t_{n+1}, t_n, \dots)`, one per lag.
    lags : sequence of int
        The lags whose times are the nodes.

    Returns
    -------
    list of Polynomial
        :math:`L_l(s)` for each lag in ``lags``, in that order.
    """
    deltat = times[0] - times[1]
    nodes = [(times[lag] - times[1]) / deltat for lag in lags]
    basis = []
    for i, node in enumerate(nodes):
        others = [other for j, other in enumerate(nodes) if j != i]
        scale = 1.0
        for other in others:
            scale *= node - other
        # One node alone has the constant basis 1.
        numerator = Polynomial.fromroots(others) if others else Polynomial([1.0])
        basis.append(numerator / scale)
    return basis


@dataclass(frozen=True)
class BackwardDifferentiation:
    r"""The backward differentiation formula of ``steps`` steps, for any
    step sizes.

    A step reading :math:`k` stored states differentiates the polynomial
    interpolating :math:`y` at :math:`t_{n+1}, \dots, t_{n+1-k}` and asks
    its derivative at :math:`t_{n+1}` to equal :math:`f(y_{n+1},
    t_{n+1})`: in the step's variable :math:`s`,

    .. math::

        \sum_{l=0}^{k} L_l'(1)\, y_{n+1-l} = \Delta t\, f_{n+1},

    scaled so that :math:`\alpha_0 = 1`. Order :math:`k`; at equal steps
    these are the classical BDF coefficients, and a startup step, reading
    fewer states, is the lower-order member of the family. Variable-step
    BDF2 is zero-stable for step ratios :math:`h_n/h_{n-1} < 1 + \sqrt 2`.

    Parameters
    ----------
    steps : int
        The number of stored states a full step reads, at least 1.
    name : str
        For messages and ``repr``.

    Raises
    ------
    ValueError
        If ``steps`` is below 1.
    """

    steps: int
    name: str

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"{self.name}: steps must be at least 1, got {self.steps}")

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        return self.steps

    def is_explicit(self) -> bool:
        """BDF is implicit."""
        return False

    def curved_lags(self) -> FrozenSet[int]:
        """Only the new state is passed through :math:`f`."""
        return frozenset({0})

    def coefficients_at(self, times: Sequence[float]) -> MultistepCoefficients:
        r"""The formula of a step with these times, one per lag:
        :math:`(t_{n+1}, t_n, \dots)`.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, times, self.steps)
        lags = range(nstates + 1)
        slopes = [float(basis.deriv()(1.0)) for basis in _lagrange_basis(times, lags)]
        lead = slopes[0]
        return MultistepCoefficients(
            alpha=tuple(slope / lead for slope in slopes),
            beta=(1.0 / lead,) + (0.0,) * nstates,
        )


def _adams_weights(times: Sequence[float], lags: Sequence[int]) -> List[float]:
    r""":math:`\int_0^1 L_l(s)\, ds` for each lag in ``lags``: the weights
    of :math:`f` at those lags in :math:`y_{n+1} - y_n = \int_{t_n}^{t_{n+1}}
    f\, dt`, in units of the step size."""
    weights = []
    for basis in _lagrange_basis(times, lags):
        antiderivative = basis.integ()
        weights.append(float(antiderivative(1.0) - antiderivative(0.0)))
    return weights


@dataclass(frozen=True)
class AdamsBashforth:
    r"""The explicit Adams method of ``steps`` steps, for any step sizes.

    A step reading :math:`k` stored states integrates, over
    :math:`[t_n, t_{n+1}]`, the polynomial interpolating :math:`f` at
    :math:`t_n, \dots, t_{n+1-k}`:

    .. math::

        y_{n+1} - y_n = \Delta t \sum_{l=1}^{k} \beta_l f_{n+1-l},
        \qquad \beta_l = \int_0^1 L_l(s)\, ds.

    Order :math:`k`; a startup step is the lower-order member of the
    family (forward Euler for one state).

    Parameters
    ----------
    steps : int
        The number of stored states a full step reads, at least 1.
    name : str
        For messages and ``repr``.

    Raises
    ------
    ValueError
        If ``steps`` is below 1.
    """

    steps: int
    name: str

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"{self.name}: steps must be at least 1, got {self.steps}")

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        return self.steps

    def is_explicit(self) -> bool:
        r""":math:`\beta_0 = 0`."""
        return True

    def curved_lags(self) -> FrozenSet[int]:
        """Every stored state is passed through :math:`f`; the new one is
        not."""
        return frozenset(range(1, self.steps + 1))

    def coefficients_at(self, times: Sequence[float]) -> MultistepCoefficients:
        r"""The formula of a step with these times, one per lag:
        :math:`(t_{n+1}, t_n, \dots)`.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, times, self.steps)
        weights = _adams_weights(times, range(1, nstates + 1))
        return MultistepCoefficients(
            alpha=(1.0, -1.0) + (0.0,) * (nstates - 1),
            beta=(0.0, *weights),
        )


@dataclass(frozen=True)
class AdamsMoulton:
    r"""The implicit Adams method of ``steps`` steps, for any step sizes.

    A step reading :math:`k` stored states integrates, over
    :math:`[t_n, t_{n+1}]`, the polynomial interpolating :math:`f` at
    :math:`t_{n+1}, \dots, t_{n+1-k}`:

    .. math::

        y_{n+1} - y_n = \Delta t \sum_{l=0}^{k} \beta_l f_{n+1-l},
        \qquad \beta_l = \int_0^1 L_l(s)\, ds.

    Order :math:`k + 1`; a startup step is the lower-order member of the
    family (the trapezoidal rule for one state).

    Parameters
    ----------
    steps : int
        The number of stored states a full step reads, at least 1.
    name : str
        For messages and ``repr``.

    Raises
    ------
    ValueError
        If ``steps`` is below 1.
    """

    steps: int
    name: str

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"{self.name}: steps must be at least 1, got {self.steps}")

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        return self.steps

    def is_explicit(self) -> bool:
        r""":math:`\beta_0 \ne 0`: implicit."""
        return False

    def curved_lags(self) -> FrozenSet[int]:
        """Every state, new and stored, is passed through :math:`f`."""
        return frozenset(range(self.steps + 1))

    def coefficients_at(self, times: Sequence[float]) -> MultistepCoefficients:
        r"""The formula of a step with these times, one per lag:
        :math:`(t_{n+1}, t_n, \dots)`.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, times, self.steps)
        weights = _adams_weights(times, range(nstates + 1))
        return MultistepCoefficients(
            alpha=(1.0, -1.0) + (0.0,) * (nstates - 1),
            beta=tuple(weights),
        )


def bdf2() -> BackwardDifferentiation:
    r"""The two-step backward differentiation formula, order 2; at equal
    steps :math:`y_{n+1} - \tfrac43 y_n + \tfrac13 y_{n-1} = \tfrac23
    \Delta t f_{n+1}`, and with :math:`\omega = h_n/h_{n-1}`,
    :math:`\tfrac{1+2\omega}{1+\omega} y_{n+1} - (1+\omega) y_n +
    \tfrac{\omega^2}{1+\omega} y_{n-1} = h_n f_{n+1}` (scaled to
    :math:`\alpha_0 = 1`); backward Euler takes the first step."""
    return BackwardDifferentiation(2, "BDF2")


def adams_bashforth2() -> AdamsBashforth:
    r"""The two-step Adams-Bashforth method, explicit, order 2; at equal
    steps :math:`y_{n+1} - y_n = \Delta t (\tfrac32 f_n - \tfrac12
    f_{n-1})`, and with :math:`\omega = h_n/h_{n-1}`, :math:`y_{n+1} - y_n
    = h_n ((1 + \tfrac\omega2) f_n - \tfrac\omega2 f_{n-1})`; forward Euler
    takes the first step."""
    return AdamsBashforth(2, "Adams-Bashforth 2")


def adams_moulton3() -> AdamsMoulton:
    r"""The two-step Adams-Moulton method, implicit, order 3; at equal
    steps :math:`y_{n+1} - y_n = \Delta t (\tfrac{5}{12} f_{n+1} +
    \tfrac{8}{12} f_n - \tfrac{1}{12} f_{n-1})`; the trapezoidal rule takes
    the first step."""
    return AdamsMoulton(2, "Adams-Moulton 3")

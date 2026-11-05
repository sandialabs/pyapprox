r"""Coefficients of linear multistep methods.

A :math:`k`-step method relates the new state to the :math:`k` stored
states before it,

.. math::

    \sum_{l=0}^{k} \alpha_l\, y_{n+1-l}
    = \Delta t \sum_{l=0}^{k} \beta_l\, f(y_{n+1-l}, t_{n+1-l}),

with :math:`\alpha_0 \ne 0` and :math:`\Delta t = h_n`, the step's own
size; lag :math:`l` is the stored state :math:`y_{n+1-l}`, lag 0 the new
state. The formula is explicit when :math:`\beta_0 = 0`. The stepper
reads only the coefficients, through :class:`MultistepFormulaProtocol`:
a method gives the formula of any step from that step's span (see
:mod:`pyapprox.ode.step_span`), one coefficient pair per lag.

Steps may differ in size. :class:`BackwardDifferentiation`,
:class:`AdamsBashforth` and :class:`AdamsMoulton` compute their formulas
from the step sizes, from the Lagrange polynomial through the step's
nodes (differentiated for BDF, integrated for Adams), so a shortened last
step or an adaptive controller's steps keep the method's order, and a
startup step, reading fewer states, is the family's lower-order member.
The nodes come from the sizes alone, never from differences of times,
and exactly equal sizes give the equal-step nodes exactly, so steps of
one size share one formula bit for bit and the backward sweeps recompute
it exactly from the recorded sizes. :class:`MultistepMethod` holds a
fixed table, derived for equal steps, with its startup formulas, and
refuses a step whose earlier steps had another size rather than use the
table there.

The families also give their continuous extension
(:class:`MultistepContinuousCoefficients`): the method's own polynomial
evaluated inside the step, the interpolant of :math:`y` for BDF and the
partial integral of the slopes' interpolant for Adams, so it makes the
smoothness assumption the step makes and no other. A fixed table has
none.
"""

from dataclasses import dataclass
from typing import (
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from numpy.polynomial import Polynomial

from pyapprox.ode.step_span import StepSpan


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


@dataclass(frozen=True)
class MultistepContinuousCoefficients:
    r"""A multistep method's continuous extension over one step: the
    value at :math:`t_n + \theta \Delta t` from the step's states and
    slopes,

    .. math::

        M y(t) = M \sum_{l=0}^{k} c_l\, y_{n+1-l}
        + \Delta t \sum_{l=0}^{k} d_l\, f(y_{n+1-l}, t_{n+1-l}),

    lag 0 being the new state. It is the method's own polynomial: the
    interpolant of :math:`y` for BDF, the integral of the slopes'
    interpolant for Adams.

    Parameters
    ----------
    states, slopes : tuple of float
        :math:`c_l` and :math:`d_l`, indexed by lag, of equal length.

    Raises
    ------
    ValueError
        If the lengths differ or there is no stored state.
    """

    states: Tuple[float, ...]
    slopes: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.states) != len(self.slopes):
            raise ValueError("states and slopes must have one entry per lag")
        if len(self.states) < 2:
            raise ValueError("a multistep extension reads at least one stored state")


def _fraction(theta: float) -> float:
    """``theta``, checked to lie in ``[0, 1]``."""
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must lie in [0, 1], got {theta}")
    return theta


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

    def coefficients_at(self, span: StepSpan) -> MultistepCoefficients:
        r"""The formula of a step over ``span``, reading one stored state
        per size: exactly one coefficient pair per lag. Raises
        ``ValueError`` when it has none for those sizes."""
        ...

    def continuous_coefficients_at(
        self, span: StepSpan, theta: float
    ) -> Optional[MultistepContinuousCoefficients]:
        r"""The continuous extension over ``span`` at the fraction
        ``theta`` of the step, one coefficient pair per lag; it gives the
        step's new state at ``theta = 1``. None for a method without one.
        Raises ``ValueError`` for ``theta`` outside ``[0, 1]``."""
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

    def coefficients_at(self, span: StepSpan) -> MultistepCoefficients:
        r"""The formula of a step over ``span``, reading one stored state
        per size. The result has exactly one coefficient pair per lag.

        These coefficients are fixed, derived for equal steps, so every
        step the formula spans must have exactly the same size (sizes are
        given, not computed, so equal steps are equal); a one-state
        formula spans one step and takes any size. A plan landing exactly
        on its end (``FixedStep``) gives its last step a size of its own,
        which this refuses: give equal sizes, or use a variable-step
        family.

        Raises
        ------
        ValueError
            If the number of states is not between 1 and the depth, or the
            steps the formula spans differ in size.
        """
        formula = self.at(len(span.sizes))
        if len(set(span.sizes)) != 1:
            raise ValueError(
                f"{self.name} has fixed coefficients, which assume equal steps; "
                f"got step sizes {span.sizes}"
            )
        return formula

    def continuous_coefficients_at(
        self, span: StepSpan, theta: float
    ) -> Optional[MultistepContinuousCoefficients]:
        """None: a fixed table carries no polynomial to extend; a family
        class (:class:`BackwardDifferentiation`, :class:`AdamsBashforth`,
        :class:`AdamsMoulton`) does."""
        _fraction(theta)
        return None

    def __repr__(self) -> str:
        return f"MultistepMethod({self.name}, {self.depth()} steps)"


def _states_read(name: str, span: StepSpan, depth: int) -> int:
    """The number of stored states a step over ``span`` reads (one per
    size), checked against the method's depth.

    Raises
    ------
    ValueError
        If it is not between 1 and ``depth``.
    """
    nstates = len(span.sizes)
    if not 1 <= nstates <= depth:
        raise ValueError(f"{name} reads between 1 and {depth} states, got {nstates}")
    return nstates


def _node(span: StepSpan, lag: int) -> float:
    r"""Lag ``lag``'s node in the step's variable :math:`s = (t - t_n)/h_n`:
    1 for the new state, 0 for :math:`y_n`, and minus the earlier steps'
    sizes summed, over :math:`h_n`, before that. From the sizes alone;
    when they are all equal, the equal-step node :math:`1 - l` exactly
    (a sum such as :math:`2h + h` need not round to :math:`3h`)."""
    if lag <= 1:
        return 1.0 - lag
    sizes = span.sizes
    if len(set(sizes)) == 1:
        return 1.0 - lag
    earlier = 0.0
    # h_{n-1}, ..., h_{n+1-lag}: the sizes before the step's own.
    for size in reversed(sizes[len(sizes) - lag : len(sizes) - 1]):
        earlier = earlier + size
    return -earlier / sizes[-1]


def _lagrange_basis(span: StepSpan, lags: Sequence[int]) -> List[Polynomial]:
    r"""The Lagrange basis on the nodes of ``lags``, in the step's own
    variable :math:`s = (t - t_n)/h_n`, so the step runs over
    :math:`s \in [0, 1]` whatever its size and the earlier nodes sit at
    minus the ratios of the earlier steps to this one.

    Parameters
    ----------
    span : StepSpan
        The step, one size per stored state read.
    lags : sequence of int
        The lags whose states are the nodes.

    Returns
    -------
    list of Polynomial
        :math:`L_l(s)` for each lag in ``lags``, in that order.
    """
    nodes = [_node(span, lag) for lag in lags]
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

    def coefficients_at(self, span: StepSpan) -> MultistepCoefficients:
        r"""The formula of a step over ``span``.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, span, self.steps)
        lags = range(nstates + 1)
        slopes = [float(basis.deriv()(1.0)) for basis in _lagrange_basis(span, lags)]
        lead = slopes[0]
        return MultistepCoefficients(
            alpha=tuple(slope / lead for slope in slopes),
            beta=(1.0 / lead,) + (0.0,) * nstates,
        )

    def continuous_coefficients_at(
        self, span: StepSpan, theta: float
    ) -> MultistepContinuousCoefficients:
        r"""The interpolant of :math:`y` the step differentiates, at
        ``theta``: :math:`c_l = L_l(\theta)` over the new state and the
        stored states, no slopes.

        Raises
        ------
        ValueError
            If ``theta`` is outside ``[0, 1]`` or the step reads more
            states than the method's depth, or none.
        """
        theta = _fraction(theta)
        nstates = _states_read(self.name, span, self.steps)
        basis = _lagrange_basis(span, range(nstates + 1))
        return MultistepContinuousCoefficients(
            states=tuple(float(polynomial(theta)) for polynomial in basis),
            slopes=(0.0,) * (nstates + 1),
        )


def _adams_weights(
    span: StepSpan, lags: Sequence[int], upper: float = 1.0
) -> List[float]:
    r""":math:`\int_0^{\theta} L_l(s)\, ds` for each lag in ``lags``,
    :math:`\theta` = ``upper``: the weights of :math:`f` at those lags in
    :math:`y(t_n + \theta \Delta t) - y_n = \int_{t_n}^{t} f\, dt`, in units
    of the step size; the step's own at :math:`\theta = 1`."""
    weights = []
    for basis in _lagrange_basis(span, lags):
        antiderivative = basis.integ()
        weights.append(float(antiderivative(upper) - antiderivative(0.0)))
    return weights


def _adams_extension(
    span: StepSpan, nstates: int, first_lag: int, theta: float
) -> MultistepContinuousCoefficients:
    r"""An Adams method's extension at ``theta``: :math:`y_n` plus the
    integral to :math:`\theta` of the slopes' interpolant over lags
    ``first_lag`` (0 implicit, 1 explicit) to ``nstates``."""
    weights = _adams_weights(span, range(first_lag, nstates + 1), theta)
    return MultistepContinuousCoefficients(
        states=(0.0, 1.0) + (0.0,) * (nstates - 1),
        slopes=(0.0,) * first_lag + tuple(weights),
    )


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

    def coefficients_at(self, span: StepSpan) -> MultistepCoefficients:
        r"""The formula of a step over ``span``.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, span, self.steps)
        weights = _adams_weights(span, range(1, nstates + 1))
        return MultistepCoefficients(
            alpha=(1.0, -1.0) + (0.0,) * (nstates - 1),
            beta=(0.0, *weights),
        )

    def continuous_coefficients_at(
        self, span: StepSpan, theta: float
    ) -> MultistepContinuousCoefficients:
        r""":math:`y_n` plus the integral to ``theta`` of the stored
        slopes' interpolant, :math:`d_l = \int_0^\theta L_l(s)\, ds`.

        Raises
        ------
        ValueError
            If ``theta`` is outside ``[0, 1]`` or the step reads more
            states than the method's depth, or none.
        """
        theta = _fraction(theta)
        nstates = _states_read(self.name, span, self.steps)
        return _adams_extension(span, nstates, 1, theta)


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

    def coefficients_at(self, span: StepSpan) -> MultistepCoefficients:
        r"""The formula of a step over ``span``.

        Raises
        ------
        ValueError
            If the step reads more states than the method's depth, or none.
        """
        nstates = _states_read(self.name, span, self.steps)
        weights = _adams_weights(span, range(nstates + 1))
        return MultistepCoefficients(
            alpha=(1.0, -1.0) + (0.0,) * (nstates - 1),
            beta=tuple(weights),
        )

    def continuous_coefficients_at(
        self, span: StepSpan, theta: float
    ) -> MultistepContinuousCoefficients:
        r""":math:`y_n` plus the integral to ``theta`` of the slopes'
        interpolant, the new state's included, :math:`d_l =
        \int_0^\theta L_l(s)\, ds`.

        Raises
        ------
        ValueError
            If ``theta`` is outside ``[0, 1]`` or the step reads more
            states than the method's depth, or none.
        """
        theta = _fraction(theta)
        nstates = _states_read(self.name, span, self.steps)
        return _adams_extension(span, nstates, 0, theta)


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

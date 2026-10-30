r"""Coefficients of linear multistep methods.

A :math:`k`-step method relates the new state to the :math:`k` stored
states before it,

.. math::

    \sum_{l=0}^{k} \alpha_l\, y_{n+1-l}
    = \Delta t \sum_{l=0}^{k} \beta_l\, f(y_{n+1-l}, t_{n+1-l}),

with :math:`\alpha_0 \ne 0`; lag :math:`l` is the stored state
:math:`y_{n+1-l}`, lag 0 the new state. The method is explicit when
:math:`\beta_0 = 0`. The stepper reads only the coefficients, so a new
method is new coefficients, never new code.

The first steps have fewer stored states than the method reads, so a
method carries its startup: the coefficients for a step reading one
state, two states, and so on up to its own depth, normally lower-order
members of the same family. A startup step of order one less than the
method, taken a fixed number of times, keeps the method's global order.

A step is given the times of all its lags, and :class:`MultistepMethod`
holds fixed coefficients, derived for equal steps, so it refuses a step
whose earlier steps had another size rather than use them there.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple


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


_BDF1 = MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0, 0.0))
_AB1 = MultistepCoefficients(alpha=(1.0, -1.0), beta=(0.0, 1.0))
_AM2 = MultistepCoefficients(alpha=(1.0, -1.0), beta=(0.5, 0.5))


def bdf2() -> MultistepMethod:
    r"""The two-step backward differentiation formula, order 2,
    :math:`y_{n+1} - \tfrac43 y_n + \tfrac13 y_{n-1} = \tfrac23 \Delta t
    f_{n+1}`; backward Euler takes the first step."""
    return MultistepMethod(
        coefficients=MultistepCoefficients(
            alpha=(1.0, -4.0 / 3.0, 1.0 / 3.0), beta=(2.0 / 3.0, 0.0, 0.0)
        ),
        startup=(_BDF1,),
        order=2,
        name="BDF2",
    )


def adams_bashforth2() -> MultistepMethod:
    r"""The two-step Adams-Bashforth method, explicit, order 2,
    :math:`y_{n+1} - y_n = \Delta t (\tfrac32 f_n - \tfrac12 f_{n-1})`;
    forward Euler takes the first step."""
    return MultistepMethod(
        coefficients=MultistepCoefficients(
            alpha=(1.0, -1.0, 0.0), beta=(0.0, 1.5, -0.5)
        ),
        startup=(_AB1,),
        order=2,
        name="Adams-Bashforth 2",
    )


def adams_moulton3() -> MultistepMethod:
    r"""The two-step Adams-Moulton method, implicit, order 3,
    :math:`y_{n+1} - y_n = \Delta t (\tfrac{5}{12} f_{n+1} + \tfrac{8}{12}
    f_n - \tfrac{1}{12} f_{n-1})`; the trapezoidal rule takes the first
    step."""
    return MultistepMethod(
        coefficients=MultistepCoefficients(
            alpha=(1.0, -1.0, 0.0), beta=(5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)
        ),
        startup=(_AM2,),
        order=3,
        name="Adams-Moulton 3",
    )

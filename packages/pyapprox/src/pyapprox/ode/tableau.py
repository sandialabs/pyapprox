r"""Butcher tableaux of Runge-Kutta methods.

A tableau :math:`(A, b, c)` defines an :math:`s`-stage method: stage
:math:`i` is :math:`Y_i = y_n + \Delta t \sum_j a_{ij} K_j` with
:math:`K_j` the slope at :math:`(Y_j, t_n + c_j \Delta t)`, and
:math:`y_{n+1} = y_n + \Delta t \sum_i b_i K_i`. The stepper reads only
the coefficients, so a new method is a new tableau, never new code.

Only lower-triangular tableaux (:math:`a_{ij} = 0` for :math:`j > i`) are
accepted: explicit methods (a zero diagonal) and diagonally implicit ones
(DIRK, SDIRK, ESDIRK), whose stages are solved one after another. A
fully implicit tableau (Gauss, Radau IIA) couples all its stages into
one block solve and is not supported yet. The functions below return the
methods the library uses; any other tableau is built the same way, with
no registration.
"""

from dataclasses import dataclass
from typing import Tuple

_TOL = 1e-14


@dataclass(frozen=True)
class ButcherTableau:
    """The coefficients of a Runge-Kutta method with a lower-triangular
    :math:`A`: explicit or diagonally implicit.

    Parameters
    ----------
    a : tuple of tuples of float
        The :math:`s \\times s` matrix :math:`A`, lower triangular.
    b : tuple of float
        The :math:`s` weights.
    c : tuple of float
        The :math:`s` nodes; each equals its row sum of :math:`A`.
    name : str
        For messages and ``repr``.

    Raises
    ------
    ValueError
        If the shapes disagree, :math:`A` has an entry above its diagonal,
        or a node is not its row sum.
    """

    a: Tuple[Tuple[float, ...], ...]
    b: Tuple[float, ...]
    c: Tuple[float, ...]
    name: str

    def __post_init__(self) -> None:
        s = len(self.b)
        if s == 0 or len(self.c) != s or len(self.a) != s:
            raise ValueError(
                f"{self.name}: a, b and c must describe the same number of stages"
            )
        for i, row in enumerate(self.a):
            if len(row) != s:
                raise ValueError(f"{self.name}: row {i} of a has {len(row)} entries")
            if any(row[j] != 0.0 for j in range(i + 1, s)):
                raise ValueError(
                    f"{self.name}: a has an entry above its diagonal in row {i}; "
                    "fully implicit tableaux, whose stages are solved together, "
                    "are not supported yet"
                )
            if abs(sum(row) - self.c[i]) > _TOL:
                raise ValueError(
                    f"{self.name}: node c[{i}] = {self.c[i]} is not the row sum "
                    f"{sum(row)}"
                )

    def nstages(self) -> int:
        """The number of stages :math:`s`."""
        return len(self.b)

    def is_explicit_stage(self, i: int) -> bool:
        """Whether stage ``i`` does not depend on itself
        (:math:`a_{ii} = 0`): it needs only a mass solve."""
        return self.a[i][i] == 0.0

    def is_explicit(self) -> bool:
        """Whether every stage is explicit."""
        return all(self.is_explicit_stage(i) for i in range(self.nstages()))

    def has_trivial_first_stage(self) -> bool:
        """Whether the first stage is :math:`y_n` itself (its row of
        :math:`A` is zero), so it is not an unknown."""
        return all(entry == 0.0 for entry in self.a[0])

    def is_stiffly_accurate(self) -> bool:
        """Whether the last row of :math:`A` equals :math:`b`, so the new
        state is the last stage and needs no final solve."""
        return all(
            abs(entry - weight) <= _TOL for entry, weight in zip(self.a[-1], self.b)
        )

    def __repr__(self) -> str:
        return f"ButcherTableau({self.name}, {self.nstages()} stages)"


def forward_euler() -> ButcherTableau:
    """Forward Euler: explicit, order 1."""
    return ButcherTableau(a=((0.0,),), b=(1.0,), c=(0.0,), name="forward Euler")


def backward_euler() -> ButcherTableau:
    """Backward Euler: one implicit stage, order 1, stiffly accurate."""
    return ButcherTableau(a=((1.0,),), b=(1.0,), c=(1.0,), name="backward Euler")


def crank_nicolson() -> ButcherTableau:
    """The trapezoidal rule: a trivial explicit first stage and one
    implicit stage, order 2, stiffly accurate."""
    return ButcherTableau(
        a=((0.0, 0.0), (0.5, 0.5)),
        b=(0.5, 0.5),
        c=(0.0, 1.0),
        name="Crank-Nicolson",
    )


def implicit_midpoint() -> ButcherTableau:
    """The implicit midpoint rule: one implicit stage at the midpoint,
    order 2, not stiffly accurate."""
    return ButcherTableau(a=((0.5,),), b=(1.0,), c=(0.5,), name="implicit midpoint")


def heun() -> ButcherTableau:
    """Heun's method: explicit, two stages, order 2."""
    return ButcherTableau(
        a=((0.0, 0.0), (1.0, 0.0)), b=(0.5, 0.5), c=(0.0, 1.0), name="Heun"
    )


def classical_rk4() -> ButcherTableau:
    """The classical fourth-order method: explicit, four stages."""
    return ButcherTableau(
        a=(
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        c=(0.0, 0.5, 0.5, 1.0),
        name="classical RK4",
    )


def alexander_sdirk3() -> ButcherTableau:
    """Alexander's three-stage singly diagonally implicit method: order 3,
    L-stable, stiffly accurate.

    :math:`\\gamma \\approx 0.4358665215` is the root of
    :math:`\\gamma^3 - 3\\gamma^2 + 3\\gamma/2 - 1/6 = 0` in
    :math:`(1/6, 1/2)`.
    """
    gamma = 0.43586652150845899942
    tau = (1.0 + gamma) / 2.0
    b1 = -(6.0 * gamma * gamma - 16.0 * gamma + 1.0) / 4.0
    b2 = (6.0 * gamma * gamma - 20.0 * gamma + 5.0) / 4.0
    return ButcherTableau(
        a=(
            (gamma, 0.0, 0.0),
            (tau - gamma, gamma, 0.0),
            (b1, b2, gamma),
        ),
        b=(b1, b2, gamma),
        c=(gamma, tau, 1.0),
        name="Alexander SDIRK3",
    )

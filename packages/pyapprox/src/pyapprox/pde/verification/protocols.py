"""The seam between a verification study and the stack it verifies.

A stack supplies a discretizer. It names its refinement levels by
requests of a type it defines (:class:`SpaceLevel` for a family of
spatial levels), reports a level's :class:`Sizes` without solving it, and
solves a problem at one request. Sizes come before the solve so a study
can choose its levels from them, e.g. a time step tied to the mesh size.

Each approximation integrates its own errors against the problem's exact
solution, with its own quadrature, elements and geometry (as deal.II's
``VectorTools::integrate_difference`` and MFEM's
``GridFunction::ComputeL2Error`` do), so the study never touches the
stack's meshes or vectors. A distributed stack reduces across processes
before returning.

A stack that computes quantities of interest supplies a
:class:`QuantityDiscretizerProtocol`, whose approximations report each
quantity's value from their own solution. It is a separate protocol, so a
study that needs quantities is typed to need them, and a stack without
them is a type error rather than a failure at run time.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.pde.verification.quantities import QuantityProtocol

P_contra = TypeVar("P_contra", contravariant=True)
R = TypeVar("R")
A_co = TypeVar("A_co", covariant=True)


@dataclass(frozen=True)
class Axis:
    """A way of refining: anything a level can be sized by.

    The set is open. The common axes are declared here (:data:`H`,
    :data:`DEGREE`, :data:`NDOFS`, :data:`DT`, :data:`NPARTS`); a stack
    or a study that refines some other way declares its own axis, e.g.
    ``Axis("aspect")``, with no change here. Axes compare by name, which is
    also how a level's sizes are written to and read from a record.
    """

    name: str


H = Axis("h")
"""The mesh size."""
DEGREE = Axis("p")
"""The polynomial degree, for p-refinement and spectral methods; a
spectral method of N nodes along a direction has degree N - 1 (the highest
over the directions)."""
NDOFS = Axis("ndofs")
"""The number of unknowns."""
DT = Axis("dt")
"""The time step."""
NPARTS = Axis("nparts")
"""The number of parts (subdomains) of a decomposed problem."""


def part(name: str, axis: Axis) -> Axis:
    """Axis ``axis`` of part (subdomain) ``name``, e.g. ``part("left", H)``,
    so a study can refine one subdomain and fit against its own size."""
    return Axis(f"{name}/{axis.name}")


@dataclass(frozen=True)
class Sizes:
    """How fine one level is, along each axis it reports.

    Parameters
    ----------
    values : Mapping[Axis, float]
        The level's size along each axis, e.g. ``{H: 0.1, DEGREE: 2}``.

    Raises
    ------
    ValueError
        If no axis is given, or a size is not positive and finite.
    TypeError
        If a key is not an :class:`Axis`.
    """

    values: Mapping[Axis, float]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("a level needs its size along at least one axis")
        for axis, value in self.values.items():
            if not isinstance(axis, Axis):
                raise TypeError(f"sizes are keyed by Axis, got {axis!r}")
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"sizes must be positive and finite; {axis.name} is {value}"
                )

    def of(self, axis: Axis) -> float:
        """The size along ``axis``.

        Raises
        ------
        ValueError
            If the level does not report ``axis``.
        """
        if axis not in self.values:
            raise ValueError(
                f"the level has no size {axis.name!r}; it reports "
                f"{[a.name for a in self.values]}"
            )
        return float(self.values[axis])


@dataclass(frozen=True)
class SpaceLevel:
    """A request for one level of a stack's family of spatial
    discretizations, counted from the coarsest.

    Raises
    ------
    ValueError
        If ``index`` is negative.
    """

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"a level index is non-negative, got {self.index}")


@runtime_checkable
class ApproximationProtocol(Protocol):
    """One refinement level's solution of a problem."""

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, summed over components."""
        ...

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the stack's nodes and
        components."""
        ...


@runtime_checkable
class DiscretizerProtocol(Protocol[P_contra, R, A_co]):
    """Solves a problem at the refinement levels it is asked for.

    Generic in the problem type it accepts (a stack's discretizer for one
    physics takes that physics' problem class), in the requests that name
    its levels, and in the approximations it returns, so a measure meant
    for another stack's approximations is a type error."""

    def requests(self, problem: P_contra) -> Sequence[R]:
        """Every level of the stack's own family, coarsest first: the
        study's levels when it is given none."""
        ...

    def sizes(self, problem: P_contra, request: R) -> Sizes:
        """The sizes of the level ``request`` names, without solving."""
        ...

    def solve(self, problem: P_contra, request: R) -> A_co:
        """The solution of ``problem`` at the level ``request`` names."""
        ...


@runtime_checkable
class QuantityApproximationProtocol(Protocol):
    """One refinement level's solution, measured by quantities of
    interest."""

    def h(self) -> float:
        """The mesh size."""
        ...

    def quantity(self, quantity: QuantityProtocol) -> float:
        """The value of ``quantity`` at this level's solution."""
        ...


@runtime_checkable
class QuantityDiscretizerProtocol(Protocol[P_contra]):
    """Solves a problem on successively refined discretizations, whose
    solutions report quantities of interest."""

    def approximations(
        self, problem: P_contra
    ) -> Sequence[QuantityApproximationProtocol]:
        """One approximation per refinement level, coarsest first."""
        ...

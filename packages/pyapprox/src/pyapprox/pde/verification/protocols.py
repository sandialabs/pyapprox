"""The seam between a verification study and the stack it verifies.

A stack supplies a discretizer, which turns a problem into a sequence of
approximations, one per refinement level. Each approximation reports its
mesh size and integrates its own errors against the problem's exact
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

from typing import Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.pde.verification.quantities import QuantityProtocol

P_contra = TypeVar("P_contra", contravariant=True)


@runtime_checkable
class ApproximationProtocol(Protocol):
    """One refinement level's solution of a problem."""

    def h(self) -> float:
        """The mesh size."""
        ...

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, summed over components."""
        ...

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the stack's nodes and
        components."""
        ...


@runtime_checkable
class DiscretizerProtocol(Protocol[P_contra]):
    """Solves a problem on successively refined discretizations.

    Generic in the problem type it accepts, so a stack's discretizer for
    one physics takes that physics' problem class."""

    def approximations(self, problem: P_contra) -> Sequence[ApproximationProtocol]:
        """One approximation per refinement level, coarsest first."""
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

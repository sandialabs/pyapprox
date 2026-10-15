"""Time order of a physics, as seen by first-order transient paths.

Every transient path in ``pde`` integrates a first-order system
``M du/dt = F(u, t)``. A physics whose equation is second order in time,
``rho u_tt = div sigma + f``, cannot be integrated that way: dropping the
inertia leaves ``M du/dt = F(u)``, a viscous relaxation toward the
steady state with no density and no wave propagation. Such a physics
declares itself by satisfying ``SecondOrderInTimeProtocol``, and the
first-order transient paths refuse it with the physics' own message.

The declaration lives here, below both discretizations, so Galerkin and
collocation share one protocol and one message.

Why the refusal exists: integrating an elasticity physics as
``M du/dt = F(u, t)`` returns a viscous relaxation that reads as
elastodynamics, with no error. Refusing turns that silent wrong answer
into an error.

Why the guard does not reduce the equation to first order itself: the
faithful first-order form is the doubled state ``(u, v)`` with block
mass ``[I 0; 0 M]``, constraints on both ``u`` and its rate, and an
initial velocity. That is a second-order integration path in its own
right, not a fix inside the guard.

What lifts the refusal: a second-order integration path (Newmark or
generalized-alpha, or the doubled first-order system) admits physics
satisfying ``SecondOrderInTimeProtocol``; first-order paths keep
refusing them. When that path exists, the last sentence of
``second_order_in_time_refusal`` should name its entry point.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SecondOrderInTimeProtocol(Protocol):
    """A physics second order in time, refused by first-order paths.

    The mass matrix of such a physics stays available (steady solvers
    and reduced models read it); only integrating ``M du/dt = F(u, t)``
    with it is refused.
    """

    def first_order_transient_refusal(self) -> str:
        """Return why a first-order transient path refuses this physics."""
        ...


def second_order_in_time_refusal(physics_name: str) -> str:
    """Build the refusal message for a physics second order in time.

    Parameters
    ----------
    physics_name : str
        Name of the refused physics class.

    Returns
    -------
    str
        The message a first-order transient path raises with.
    """
    return (
        f"{physics_name} is second order in time (rho u_tt = div sigma + "
        "f), but this transient path integrates the first-order system "
        "M du/dt = F(u, t): a viscous relaxation with no inertia or "
        "density, not elastodynamics. Solve the steady problem instead; "
        "elastodynamics needs second-order time integration, which no "
        "transient path provides yet."
    )


def refuse_second_order_in_time(physics: object) -> None:
    """Raise if ``physics`` is second order in time.

    Called at the entry of every first-order transient path.

    Raises
    ------
    TypeError
        If ``physics`` satisfies ``SecondOrderInTimeProtocol``; the
        message is the physics' own refusal.
    """
    if isinstance(physics, SecondOrderInTimeProtocol):
        raise TypeError(physics.first_order_transient_refusal())

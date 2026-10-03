"""Which objects a parameterization may write: binding by membership.

A parameterization writes coefficients of its ``targets()`` --- a
physics, or (for BC data) a boundary term. A model or adapter accepts
it when every target belongs to the object it solves: that object
itself, or something it ``owns()`` (``pyapprox.pde.ownership``).
Membership replaces an identity check against one physics, which could
not accept a parameterization of a term inside a composed system, nor a
composite whose parts target different terms.
"""

from pyapprox.pde.ownership import owns
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array


def require_owned_targets(
    parameterization: ParameterizationProtocol[Array], owner: object
) -> None:
    """Raise unless every target of ``parameterization`` is in ``owner``.

    Raises
    ------
    ValueError
        Naming the first target ``owner`` does not hold.
    """
    for target in parameterization.targets():
        if not owns(owner, target):
            raise ValueError(
                f"{type(parameterization).__name__} writes coefficients of "
                f"a {type(target).__name__} that this "
                f"{type(owner).__name__} does not hold; construct the "
                "parameterization on the physics (or term) being solved"
            )

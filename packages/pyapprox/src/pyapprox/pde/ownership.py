"""Ownership: which objects belong to a physics or composed system.

A parameterization writes coefficients of its targets (a physics, or a
boundary term); a model accepts it when every target belongs to what it
solves. ``owns`` answers that by identity, structurally: an object
implementing ``OwnerProtocol.owns`` declares what it holds, and one that
does not still holds itself. Shared by every discretization and by the
models above them, so it lives below both.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class OwnerProtocol(Protocol):
    """An object that can say which objects are part of it."""

    def owns(self, target: object) -> bool:
        """Whether ``target`` is part of this object, by identity.

        Two equal but distinct physics are different targets.
        """
        ...


def owns(owner: object, target: object) -> bool:
    """Whether ``target`` is ``owner`` or something ``owner`` holds."""
    if target is owner:
        return True
    return isinstance(owner, OwnerProtocol) and owner.owns(target)

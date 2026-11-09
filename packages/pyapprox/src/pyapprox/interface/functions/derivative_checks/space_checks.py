r"""Finite-difference and dot tests through vector spaces, on any stack.

The core shared by the derivative checkers that work through a stack's
spaces (``axpby``, ``norm``, ``dot``) rather than on backend arrays:
the finite-difference errors of a declared derivative along a random
direction, with their predicted rounding, and the dot test of a
linear map against its transpose. Random vectors come from a caller's
``sample``, since a vector space provides none.
"""

from typing import Callable, Generic, Sequence, TypeVar

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    FDCurve,
    predicted_rounding,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    SpaceDescriptor,
)

V = TypeVar("V")
W = TypeVar("W")
U = TypeVar("U")
A = TypeVar("A", bound=ArrayProtocol)

DEFAULT_FD_STEPS = tuple(10.0**-k for k in range(1, 14))
"""Finite-difference steps, 1e-1 down to 1e-13."""


def scalar(value: A) -> float:
    """A ``(1, 1)`` reduction (a single column's norm or dot) as a float,
    for a report: no value derived from it flows into a computation."""
    return float(value.item())


class SpaceSampler(Generic[V]):
    """Random single columns of block spaces, component by component,
    each component drawn with the next seed.

    Parameters
    ----------
    sample : callable
        ``(descriptor, seed) ->`` a random single-column vector of the
        component space ``descriptor`` describes.
    seed : int
        The first seed.
    """

    def __init__(self, sample: Callable[[SpaceDescriptor, int], V], seed: int) -> None:
        self._sample, self._next_seed = sample, seed

    def random(self, space: BlockInnerProductSpace[V, A]) -> BlockVector[V]:
        """A random single column of ``space``."""
        blocks = []
        for i in range(space.nblocks()):
            blocks.append(
                self._sample(space.component(i).descriptor(), self._next_seed)
            )
            self._next_seed += 1
        return BlockVector(tuple(blocks))


def space_fd_check(
    point: BlockVector[W],
    space: BlockInnerProductSpace[W, A],
    output: BlockInnerProductSpace[U, A],
    quantity: Callable[[BlockVector[W]], BlockVector[U]],
    action: Callable[[BlockVector[W], BlockVector[W]], BlockVector[U]],
    direction: BlockVector[W],
    fd_steps: Sequence[float],
    central: bool,
) -> FDCurve:
    r"""The finite-difference errors of a declared derivative over the
    steps, along ``direction`` scaled to unit length, with their predicted
    rounding, for the caller to judge: each relative to the action's norm,
    or absolute where the action is zero (a stated zero).

    Parameters
    ----------
    point : BlockVector
        Where the derivative is checked, a single column of ``space``.
    space : BlockInnerProductSpace
        The argument's space.
    output : BlockInnerProductSpace
        The quantity's space.
    quantity : callable
        ``x -> q(x)``. Its result is copied before the next call, so it
        may be storage the quantity reuses.
    action : callable
        ``(x, d) ->`` the declared derivative of ``q`` at ``x`` applied to
        ``d``; copied likewise.
    direction : BlockVector
        The direction, a single column of ``space``.
    fd_steps : sequence of float
        The steps.
    central : bool
        Central rather than forward differences.
    """
    direction = space.scale(1.0 / scalar(space.norm(direction)), direction)
    base = output.copy(quantity(point))
    exact = output.copy(action(point, direction))
    size = scalar(output.norm(exact))
    relative = size > 1e-16
    errors = []
    for h in fd_steps:
        plus = output.copy(quantity(space.axpby(h, direction, 1.0, point)))
        if central:
            minus = quantity(space.axpby(-h, direction, 1.0, point))
            difference = output.axpby(0.5 / h, plus, -0.5 / h, minus)
        else:
            difference = output.axpby(1.0 / h, plus, -1.0 / h, base)
        error = scalar(output.norm(output.axpby(1.0, difference, -1.0, exact)))
        errors.append(error / size if relative else error)
    eps = float(np.finfo(output.component(0).descriptor().dtype).eps)
    rounding = predicted_rounding(
        eps,
        scalar(output.norm(base)),
        size,
        scalar(space.norm(point)),
        fd_steps,
    )
    return FDCurve(tuple(errors), tuple(fd_steps), central, rounding)


def space_dot_test_error(
    apply: Callable[[BlockVector[W]], BlockVector[U]],
    apply_transpose: Callable[[BlockVector[U]], BlockVector[W]],
    v: BlockVector[W],
    w: BlockVector[U],
    domain: BlockInnerProductSpace[W, A],
    range_: BlockInnerProductSpace[U, A],
) -> float:
    r"""The relative difference of :math:`\langle w, A v\rangle` and
    :math:`\langle A^\top w, v\rangle`."""
    av = range_.copy(apply(v))
    atw = apply_transpose(w)
    size = max(
        scalar(range_.norm(w)) * scalar(range_.norm(av)),
        scalar(domain.norm(v)) * scalar(domain.norm(atw)),
    )
    difference = scalar(range_.dot(w, av)) - scalar(domain.dot(atw, v))
    return 0.0 if size == 0.0 else abs(difference) / size

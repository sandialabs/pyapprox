"""
Group-independent joint distribution.

Extends IndependentJoint to accept groups of variables, where each group
is either a univariate MarginalProtocol or a multi-dimensional
IndependentJoint.  Internally flattens to a single IndependentJoint but
tracks group boundaries for slicing.
"""

from typing import Generic, List, Sequence

from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GroupIndependentJoint(Generic[Array]):
    """Joint distribution with grouped independent variables.

    Each group is a :class:`MarginalProtocol` (1-D) or an
    :class:`IndependentJoint` (multi-D).  The composite joint has
    ``nvars = sum of group dimensions``.

    All distribution methods (``rvs``, ``logpdf``, ``marginals``, etc.)
    operate on the flattened ``(nvars, n)`` representation.  Use
    :meth:`group_slice` to recover the row indices for a given group.

    Parameters
    ----------
    groups : Sequence[MarginalProtocol | IndependentJoint]
        One or more groups.  Each MarginalProtocol contributes 1
        dimension; each IndependentJoint contributes ``len(marginals())``
        dimensions.
    bkd : Backend[Array]

    Examples
    --------
    >>> from pyapprox.probability import (
    ...     GaussianMarginal, BetaMarginal, UniformMarginal,
    ...     IndependentJoint, GroupIndependentJoint,
    ... )
    >>> bkd = NumpyBkd()
    >>> source = GaussianMarginal(0, 1, bkd)
    >>> latent = IndependentJoint([
    ...     BetaMarginal(1, 2, bkd, lb=0, ub=6.28),
    ...     UniformMarginal(0, 6.28, bkd),
    ... ], bkd)
    >>> joint = GroupIndependentJoint([source, latent], bkd)
    >>> joint.nvars()
    3
    >>> joint.group_slice(0)
    slice(0, 1, None)
    >>> joint.group_slice(1)
    slice(1, 3, None)
    """

    def __init__(
        self,
        groups: Sequence,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._groups: List = []
        flat: List[MarginalProtocol[Array]] = []
        slices: List[slice] = []
        offset = 0

        for g in groups:
            if isinstance(g, IndependentJoint):
                ms = g.marginals()
                n = len(ms)
                flat.extend(ms)
            elif isinstance(g, MarginalProtocol):
                n = 1
                flat.append(g)
            else:
                raise TypeError(
                    f"Each group must be a MarginalProtocol or "
                    f"IndependentJoint; got {type(g).__name__}"
                )
            self._groups.append(g)
            slices.append(slice(offset, offset + n))
            offset += n

        self._flat_joint = IndependentJoint(flat, bkd)
        self._group_slices = slices

    def nvars(self) -> int:
        """Total number of variables across all groups."""
        return self._flat_joint.nvars()

    def ngroups(self) -> int:
        """Number of groups."""
        return len(self._groups)

    def group_slice(self, i: int) -> slice:
        """Row slice for group *i* in the ``(nvars, n)`` sample array."""
        return self._group_slices[i]

    def rvs(self, nsamples: int) -> Array:
        """Draw samples, shape ``(nvars, nsamples)``."""
        return self._flat_joint.rvs(nsamples)

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Flat list of all marginals across groups."""
        return self._flat_joint.marginals()

    def logpdf(self, samples: Array) -> Array:
        """Joint log-PDF."""
        return self._flat_joint.logpdf(samples)

    def pdf(self, samples: Array) -> Array:
        """Joint PDF."""
        return self._flat_joint.pdf(samples)

    def invcdf(self, probs: Array) -> Array:
        """Inverse CDF (delegates to the flat joint)."""
        return self._flat_joint.invcdf(probs)

    def bkd(self) -> Backend[Array]:
        """Computational backend."""
        return self._bkd

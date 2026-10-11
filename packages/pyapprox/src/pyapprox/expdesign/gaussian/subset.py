r"""The A-criterion of a binary design, without forming posterior covariances.

Choosing a subset :math:`S` of the observations conditions the target on
:math:`y_S` alone, so

.. math::

    \mathrm{tr}(C\Gamma_{t|y_S}C^\top)
    = \mathrm{tr}(C\Gamma_{tt}C^\top) - \mathrm{tr}(\Gamma_{yy,SS}^{-1} M_{SS}),
    \qquad M = \Gamma_{yt}C^\top C\,\Gamma_{ty},

with :math:`\Gamma_{yy} = \Gamma_{gg} + \Gamma_e`. The first term and
:math:`M` do not depend on the design, so after a one-time cost each value
is a solve with the :math:`|S| \times |S|` block, however large the target.
Dropping observations marginalizes them for any noise covariance, so the
values equal those of the blended relaxation at 0/1 weights.
"""

from typing import Generic, List, Optional, Sequence

from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.util.backends.protocols import Array, Backend


def _check_groups(groups: Sequence[Sequence[int]], nobs: int) -> List[List[int]]:
    if len(groups) == 0:
        raise ValueError("groups is empty")
    seen: set[int] = set()
    checked: List[List[int]] = []
    for jj, group in enumerate(groups):
        members = list(group)
        if len(members) == 0:
            raise ValueError(f"group {jj} is empty")
        bad = [ii for ii in members if not 0 <= ii < nobs]
        if bad:
            raise ValueError(
                f"group {jj} has indices {bad} outside the {nobs} observations"
            )
        repeated = seen.intersection(members)
        if repeated or len(set(members)) != len(members):
            raise ValueError(
                f"group {jj} repeats observations {sorted(repeated)}; groups "
                "must be disjoint"
            )
        seen.update(members)
        checked.append(members)
    return checked


class AOptimalSubsetObjective(Generic[Array]):
    r"""The A-criterion :math:`\mathrm{tr}(C\Gamma_{t|y_S}C^\top)` of a chosen subset.

    Satisfies ``SubsetObjectiveProtocol``: a design variable is a group of
    observations, and a subset switches its groups on. Wrap it in
    ``ReevaluatingIncremental`` for greedy and exchange searches; each value
    costs only a solve with the chosen observations' block.

    Parameters
    ----------
    joint : JointGaussian[Array]
        The joint Gaussian of the targets and the noisy observations.
    index : int
        Which target block.
    groups : Sequence[Sequence[int]]
        Observation indices of each design variable, non-empty and
        disjoint. One observation per group selects observations
        individually.
    target_map : Array, optional
        :math:`C`. Shape: (k, n_t). Default the identity.
    """

    def __init__(
        self,
        joint: JointGaussian[Array],
        index: int,
        groups: Sequence[Sequence[int]],
        target_map: Optional[Array] = None,
    ) -> None:
        if not isinstance(joint, JointGaussian):
            raise TypeError(
                f"joint must be a JointGaussian, got {type(joint).__name__}"
            )
        sizes = joint.target_sizes()
        if not 0 <= index < len(sizes):
            raise ValueError(f"index must lie in [0, {len(sizes)}), got {index}")
        self._groups = _check_groups(groups, joint.nobs())
        bkd = joint.bkd()
        blocks = joint.blocks()
        cross = blocks.target_obs_covariance(index)
        target_cov = blocks.target_covariance(index)
        if target_map is not None:
            if target_map.ndim != 2 or target_map.shape[1] != sizes[index]:
                raise ValueError(
                    f"target_map must have shape (*, {sizes[index]}), got "
                    f"{tuple(target_map.shape)}"
                )
            cross = bkd.dot(target_map, cross)
            target_cov = bkd.dot(bkd.dot(target_map, target_cov), target_map.T)
        self._bkd = bkd
        self._prior_value = bkd.to_float(bkd.trace(target_cov))
        self._gain = bkd.dot(cross.T, cross)
        self._obs_cov = joint.obs_covariance()

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def ncandidates(self) -> int:
        """Number of groups a subset is drawn from."""
        return len(self._groups)

    def observations(self, subset: Sequence[int]) -> List[int]:
        """The observations that ``subset`` switches on, in group order."""
        chosen = list(subset)
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"subset {chosen} repeats design variables")
        bad = [jj for jj in chosen if not 0 <= jj < len(self._groups)]
        if bad:
            raise ValueError(
                f"indices {bad} are outside the {len(self._groups)} design "
                "variables"
            )
        return [ii for jj in chosen for ii in self._groups[jj]]

    def value(self, subset: Sequence[int]) -> float:
        """The A-criterion with exactly ``subset`` switched on.

        The empty subset gives the prior value
        :math:`\\mathrm{tr}(C\\Gamma_{tt}C^\\top)`.
        """
        rows = self.observations(subset)
        if len(rows) == 0:
            return self._prior_value
        bkd = self._bkd
        reduction = bkd.solve(
            self._obs_cov[rows][:, rows], self._gain[rows][:, rows]
        )
        return self._prior_value - bkd.to_float(bkd.trace(reduction))

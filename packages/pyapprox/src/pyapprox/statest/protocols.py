"""What the estimators require of the objects handed to them.

These are contracts rather than base classes: an implementation
satisfies one by having the members, not by inheriting anything. That
matters because the alternative -- checking membership of a concrete
class -- ties the estimators to one implementation of a statistic and
demands every member that class happens to declare, whether or not any
estimator reads it.

The member lists here were measured rather than read off the abstract
base class. Driving the Monte Carlo and control variate tolerance paths
with objects that provide only these members prices a campaign
correctly, and objects that refuse everything else are never asked for
anything else; ``tests/statest/test_statistic_contract.py`` is that
measurement, and will fail if a requirement is added here or grows in an
estimator.
"""

from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class EstimableStatistic(Protocol[Array]):
    """A statistic an estimator can compute a covariance from.

    Two things an estimator does, and this spans both. Choosing an
    allocation needs only the shape and covariance members, which is
    why a statistic providing just those can price a campaign; the
    remaining members are reached once model values exist and the
    estimate itself is formed.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend its arrays belong to."""
        ...

    def nmodels(self) -> int:
        """Return how many models the pilot quantities describe."""
        ...

    def nqoi(self) -> int:
        """Return how many quantities of interest each model returns."""
        ...

    def nstats(self) -> int:
        """Return how many statistics are estimated."""
        ...

    def sample_estimate(self, values: Array) -> Array:
        """Return the statistic evaluated on model values.

        Shape ``(nstats,)``. Not reached while an allocation is being
        chosen, there being no values then.
        """
        ...

    def min_nsamples(self) -> int:
        """Return the fewest samples for which the statistic is defined.

        A variance needs an ``n*(n-1)`` denominator, so it is undefined
        below two; a mean is defined at one. An allocator uses this as
        the floor no tolerance may push it beneath.
        """
        ...

    def high_fidelity_estimator_covariance(
        self, nhf_samples: Array
    ) -> Array:
        """Return the covariance from ``nhf_samples`` of model zero alone.

        Shape ``(nstats, nstats)``. Accepts a non-integral count, since
        an allocator solves the continuous relaxation before rounding.
        """
        ...

    def pilot_covariance(self) -> Array:
        """Return the covariance estimated from the pilot sample.

        Shape ``(nmodels * nqoi, nmodels * nqoi)``. Raises if pilot
        quantities have not been supplied.
        """
        ...

    def has_pilot_covariance(self) -> bool:
        """Return whether a pilot covariance has been supplied yet.

        For callers that branch on it rather than requiring it, the
        accessor above being unable to answer without raising.
        """
        ...


@runtime_checkable
class ResamplableStatistic(EstimableStatistic[Array], Protocol[Array]):
    """A statistic that can also be rebuilt from resampled pilot values.

    The bootstrap of a fitted estimator resamples the values it was
    given, recomputes pilot quantities from the resample, and re-forms
    the estimate. That is what these two members are for, and why a
    statistic used only to choose an allocation never reaches them.
    """

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Any, ...]:
        """Return the pilot quantities implied by ``pilot_values``.

        Pure: the arity varies by statistic, which is why callers splat
        the result straight into :meth:`set_pilot_quantities` rather
        than naming the parts.
        """
        ...

    def set_pilot_quantities(self, *args: Any) -> None:
        """Adopt pilot quantities previously computed."""
        ...


@runtime_checkable
class CVDiscrepancyStatistic(ResamplableStatistic[Array], Protocol[Array]):
    """A statistic that can also report control variate discrepancies.

    Everything above plus the one member the control variate covariance
    needs. Kept separate because the Monte Carlo path never asks for it,
    and a combined contract would make a statistic serving only that
    path look incomplete.

    The member is private-named for historical reasons; it is
    nonetheless part of the contract, being what a control variate
    estimator reads off a statistic it did not construct.
    """

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Return ``(CF, cf)`` for the control variate weights.

        ``CF`` has shape ``(nlf, nlf)`` and ``cf`` shape ``(nqoi, nlf)``,
        where ``nlf`` counts the low-fidelity statistics.
        """
        ...


@runtime_checkable
class GroupBlockStatistic(ResamplableStatistic[Array], Protocol[Array]):
    """A statistic a group approximate control variate estimator can use.

    The widest of these contracts, because a group estimator searches
    over model subsets and over which statistics to keep: hence
    :meth:`subset` and :meth:`stat_slot_indices` alongside the sigma
    blocks that give the covariance between two groups.

    The blocks are private-named, and one of them is optional in
    practice -- a statistic that cannot supply derivatives raises
    ``NotImplementedError`` from it, which the optimizer treats as
    "gradients unavailable" rather than as a failure. Declaring it here
    states that a statistic must have the method, not that every call
    must succeed.
    """

    def continuous_dead_threshold(self) -> float:
        """Return the sample count below which a group is inactive."""
        ...

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        """Return which slots hold the named statistic.

        Raises ``ValueError`` for a name the statistic does not report,
        which callers use to discover what it does report.
        """
        ...

    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "GroupBlockStatistic[Array]":
        """Return the same statistic restricted to a subset of models."""
        ...

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Union[int, Array],
        nsamples_subset0: Union[int, Array],
        nsamples_subset1: Union[int, Array],
    ) -> Array:
        """Return the covariance block between two groups."""
        ...

    def _group_acv_sigma_block_derivs(
        self, subset: Array, nsamples: Union[int, Array]
    ) -> Tuple[Array, Array]:
        """Return the first and second derivatives of a sigma block."""
        ...


@runtime_checkable
class ACVDiscrepancyStatistic(ResamplableStatistic[Array], Protocol[Array]):
    """A statistic an approximate control variate estimator can use.

    The same shape as :class:`CVDiscrepancyStatistic` but for a
    different discrepancy: an approximate control variate allocates
    samples across partitions, so its covariances depend on the
    allocation matrix as well as the counts. Neither family reaches the
    other's member, which is why these are two contracts rather than
    one with both.
    """

    def _get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Return ``(CF, cf)`` for an approximate control variate."""
        ...

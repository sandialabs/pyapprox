"""What an estimator actually requires of a statistic.

Every other test in this package hands the estimators a real
``MultiOutputStatistic``, so what they *need* has never been separated
from what that class happens to provide. These tests supply objects that
provide only a stated set of members, which makes the requirement
measurable rather than asserted.

Two directions, and the second is the one that keeps the first honest.
:class:`_MinimalStatistic` implements what is believed necessary, so a
missing member fails the test. :class:`_StrictStatistic` refuses
everything outside that set, so a member quietly *added* to the
requirement fails too -- otherwise an estimator could start depending on
something new and no test would notice until an outside implementation
tried to satisfy the contract and found it had grown.

:class:`_MinimalGroupStatistic` extends the first to the group
estimator, whose check previously named the three concrete statistic
classes. That named every subclass in existence and so restricted
nothing, while still refusing by name an outside statistic supplying
everything the estimator needs. This fake is such a statistic.
"""

from typing import Any, Generic, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from pyapprox.statest import (
    CVEstimator,
    CVToleranceAllocator,
    KnownMean,
    KnownStatistic,
    MCEstimator,
    MCToleranceAllocator,
    MultiOutputMean,
)
from pyapprox.statest.groupacv.variants import GroupACVEstimatorIS
from pyapprox.statest.protocols import (
    EstimableStatistic,
    GroupBlockStatistic,
)
from pyapprox.statest.tolerance import MaxMarginalStandardErrorConstraint
from pyapprox.util.backends.protocols import Array, Backend


def _protocol_members(protocol: type) -> frozenset:
    """The members a Protocol declares, on every supported Python.

    ``__protocol_attrs__`` would say this directly but was added in
    3.12, so reading it makes the test fail outright on 3.11 rather
    than measure anything. Walk the protocol's own bases instead and
    collect annotations plus callables, skipping the dunders and the
    typing machinery that ``Protocol`` mixes in.
    """
    members: set = set()
    for base in protocol.__mro__:
        if base in (object, Generic):
            continue
        if getattr(base, "_is_protocol", False) is False and base is not protocol:
            continue
        members |= set(getattr(base, "__annotations__", {}))
        members |= {
            name
            for name, value in vars(base).items()
            if callable(value) and not name.startswith("__")
        }
    return frozenset(members)


MC_CONTRACT = frozenset(
    {"bkd", "nmodels", "high_fidelity_estimator_covariance", "min_nsamples"}
)
"""What the Monte Carlo *tolerance* path asks of a statistic.

Four members, none private, and deliberately narrower than
:class:`EstimableStatistic`. Choosing an allocation happens before any
model values exist, so the members that consume values -- ``nqoi``,
``nstats``, ``sample_estimate`` -- are declared by the protocol but
never reached here. That gap is the point: it is measured, not assumed,
and a statistic providing only these four prices a campaign correctly.
"""

CV_CONTRACT = MC_CONTRACT | {"_get_cv_discrepancy_covariances"}
"""The control variate tolerance path adds exactly one member.

Still private, and unlike the others it has no public accessor to move
to -- it is a genuine part of the contract wearing a private name.
"""

GROUPACV_CONSTRUCTION_CONTRACT = frozenset({"bkd", "nstats"})
"""What constructing a group estimator asks of a statistic.

Two members, against the sixteen :class:`GroupBlockStatistic` declares.
The subsets, allocation matrix and restriction matrices are all built
from the model count -- which comes from ``costs``, not the statistic --
and the statistic count. Everything else waits: the sigma blocks until a
covariance is formed, :meth:`subset` until a model subset is searched,
``stat_slot_indices`` until known quantities are supplied.
"""


def _pilot_covariance(
    bkd: Backend[Array], nmodels: int, nqoi: int = 1
) -> Array:
    """A covariance that is positive definite and model-correlated."""
    rng = np.random.RandomState(0)
    values = rng.normal(0.0, 2.0, (nmodels * nqoi, 200))
    values[nqoi:] += values[:nqoi] * 0.9
    # atleast_2d: np.cov of a single row returns a 0-d scalar, and a
    # covariance is (n, n) even when n is one.
    return bkd.asarray(np.atleast_2d(np.cov(values)))


class _MinimalStatistic(Generic[Array]):
    """Implements the contract and nothing else.

    Deliberately not a ``MultiOutputStatistic`` subclass: inheriting
    would supply the very members whose necessity is in question.
    """

    def __init__(
        self, bkd: Backend[Array], nmodels: int, nqoi: int = 1
    ) -> None:
        self._bkd = bkd
        self._nmodels = nmodels
        self._nqoi = nqoi
        self._cov = _pilot_covariance(bkd, nmodels, nqoi)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nmodels(self) -> int:
        return self._nmodels

    def min_nsamples(self) -> int:
        return 1

    # Declared by EstimableStatistic and so required to pass its
    # isinstance gate, but never reached while an allocation is being
    # chosen -- which is what the strict fake below demonstrates.

    def nqoi(self) -> int:
        return self._nqoi

    def nstats(self) -> int:
        return self._nqoi

    def sample_estimate(self, values: Array) -> Array:
        return self._bkd.mean(values, axis=1)

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Any, ...]:
        return (self._cov,)

    def set_pilot_quantities(self, *args: Any) -> None:
        self._cov = args[0]

    def pilot_covariance(self) -> Array:
        return self._cov

    def has_pilot_covariance(self) -> bool:
        return self._cov is not None

    def high_fidelity_estimator_covariance(self, nhf_samples: Array) -> Array:
        return self._cov[: self._nqoi, : self._nqoi] / nhf_samples

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        nhf = npartition_samples[0]
        return (
            self._cov[self._nqoi :, self._nqoi :] / nhf,
            self._cov[: self._nqoi, self._nqoi :] / nhf,
        )


class _MinimalGroupStatistic(_MinimalStatistic[Array]):
    """Adds what :class:`GroupBlockStatistic` asks for beyond the rest.

    The sigma block is the covariance between the per-group estimators
    of two model subsets. For a mean under independent sampling that is
    the pilot covariance restricted to the shared models and divided by
    the number of samples the two groups share, which is enough to
    exercise the contract without reproducing the real derivation.
    """

    def continuous_dead_threshold(self) -> float:
        return 0.0

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        if stat_name == "mean":
            return list(range(self._nqoi))
        raise ValueError(f"{stat_name!r} not available on {type(self).__name__}")

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        if isinstance(known, KnownMean):
            return list(range(self._nqoi))
        raise ValueError(
            f"{type(self).__name__} accepts no known {type(known).__name__}"
        )

    def check_known(self, known: Sequence[KnownStatistic[Array]]) -> None:
        for item in known:
            self.known_slots(item)

    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "_MinimalGroupStatistic[Array]":
        if 0 not in model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")
        return _MinimalGroupStatistic(
            self._bkd, len(model_indices), self._nqoi
        )

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Any,
        nsamples_subset0: Any,
        nsamples_subset1: Any,
    ) -> Array:
        bkd = self._bkd
        rows = bkd.to_numpy(subset0).astype(int).tolist()
        cols = bkd.to_numpy(subset1).astype(int).tolist()
        block = bkd.to_numpy(self._cov)[np.ix_(rows, cols)]
        return bkd.asarray(block) * nsamples_intersect / (
            nsamples_subset0 * nsamples_subset1
        )

    def _group_acv_sigma_block_derivs(
        self, subset: Array, nsamples: Any
    ) -> Tuple[Array, Array]:
        raise NotImplementedError(
            "this fake supplies no derivatives, which the optimizer is "
            "expected to treat as gradients being unavailable"
        )


class _StrictStatistic(_MinimalStatistic[Array]):
    """Raises on any access outside the declared contract.

    The guard against silent widening: if an estimator starts reaching
    for a member the contract does not name, this fails immediately
    rather than passing because a real statistic happened to have it.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nmodels: int,
        contract: frozenset,
        nqoi: int = 1,
    ) -> None:
        super().__init__(bkd, nmodels, nqoi)
        object.__setattr__(self, "_contract", contract)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so the contract members
        # and the private fields above never arrive here.
        raise AssertionError(
            f"{name!r} was requested but is not part of the declared "
            f"contract {sorted(self._contract)}. Either the estimator "
            "grew a requirement or the contract is understated."
        )


def _tolerance(bkd: Backend[Array]) -> MaxMarginalStandardErrorConstraint:
    return MaxMarginalStandardErrorConstraint(0.05, bkd)


class TestTheContractIsSmallerThanTheAbstractBaseClass:
    """Four members reach a budget; the ABC declares thirteen."""

    def test_monte_carlo_needs_only_the_declared_members(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        stat = _MinimalStatistic(numpy_bkd, nmodels=1)
        fitted = MCToleranceAllocator(
            MCEstimator(stat, [1.0])
        ).allocate_for_tolerance(_tolerance(numpy_bkd))
        assert fitted.actual_cost() > 0.0

    def test_control_variate_adds_exactly_one(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        stat = _MinimalStatistic(numpy_bkd, nmodels=3)
        fitted = CVToleranceAllocator(
            CVEstimator(stat, [1.0, 0.1, 0.01])
        ).allocate_for_tolerance(_tolerance(numpy_bkd))
        assert fitted.actual_cost() > 0.0

    def test_the_two_contracts_differ_by_one_member(self) -> None:
        assert CV_CONTRACT - MC_CONTRACT == {"_get_cv_discrepancy_covariances"}


class TestNothingOutsideTheContractIsTouched:
    """The guard that keeps the contract from widening unnoticed."""

    def test_monte_carlo_reaches_for_nothing_else(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        stat = _StrictStatistic(numpy_bkd, 1, MC_CONTRACT)
        fitted = MCToleranceAllocator(
            MCEstimator(stat, [1.0])
        ).allocate_for_tolerance(_tolerance(numpy_bkd))
        assert fitted.actual_cost() > 0.0

    def test_control_variate_reaches_for_nothing_else(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        stat = _StrictStatistic(numpy_bkd, 3, CV_CONTRACT)
        fitted = CVToleranceAllocator(
            CVEstimator(stat, [1.0, 0.1, 0.01])
        ).allocate_for_tolerance(_tolerance(numpy_bkd))
        assert fitted.actual_cost() > 0.0

    @pytest.mark.parametrize(
        "member", ["subset", "stat_slot_indices", "_group_acv_sigma_block"]
    )
    def test_the_guard_itself_fires(
        self, numpy_bkd: Backend[Array], member: str
    ) -> None:
        """Without this the two tests above could pass vacuously.

        The members named here are ones the abstract base class declares
        and no estimator path reaches -- exactly the surplus that made
        the requirement look larger than it is.
        """
        stat = _StrictStatistic(numpy_bkd, 1, MC_CONTRACT)
        with pytest.raises(AssertionError, match="not part of the declared"):
            getattr(stat, member)


class TestChoosingAnAllocationTouchesLessThanTheProtocolDeclares:
    """The protocol spans two jobs; pricing needs only one of them.

    ``EstimableStatistic`` declares the members for choosing an
    allocation *and* for forming an estimate, because one object does
    both. But no model values exist while an allocation is being
    chosen, so the value-consuming members cannot be reached then --
    and the strict fakes above prove they are not.
    """

    VALUE_CONSUMING = ("nqoi", "nstats", "sample_estimate")

    @pytest.mark.parametrize("member", VALUE_CONSUMING)
    def test_declared_by_the_protocol(self, member: str) -> None:
        assert hasattr(EstimableStatistic, member)

    @pytest.mark.parametrize("member", VALUE_CONSUMING)
    def test_but_absent_from_the_pricing_contract(self, member: str) -> None:
        assert member not in MC_CONTRACT


class TestTheGroupEstimatorAcceptsAnyConformingStatistic:
    """The check it replaced named the three concrete classes.

    That named every subclass in existence, so it restricted nothing --
    while still rejecting, by name, any outside statistic that supplied
    everything a group estimator needs. These tests are that statistic.
    """

    def test_a_conforming_fake_is_accepted(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        stat = _MinimalGroupStatistic(numpy_bkd, nmodels=3)
        est = GroupACVEstimatorIS(stat, [1.0, 0.1, 0.01])
        assert est.nmodels() == 3

    def test_a_statistic_missing_the_group_members_is_refused(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """The narrower fake satisfies the MC contract but not this one."""
        stat = _MinimalStatistic(numpy_bkd, nmodels=3)
        with pytest.raises(ValueError, match="GroupBlockStatistic"):
            GroupACVEstimatorIS(stat, [1.0, 0.1, 0.01])

    def test_construction_touches_only_two_members(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """Measured, and much narrower than the protocol declares.

        The estimator learns its model count from ``costs`` rather than
        from the statistic, so building the subsets and allocation
        matrix needs only the backend and the statistic count.
        """
        touched: set = set()
        stat = _MinimalGroupStatistic(numpy_bkd, nmodels=3)
        for name in sorted(_protocol_members(GroupBlockStatistic)):
            original = getattr(stat, name)

            def record(*args: Any, _n: str = name, _o: Any = original) -> Any:
                touched.add(_n)
                return _o(*args)

            object.__setattr__(stat, name, record)
        GroupACVEstimatorIS(stat, [1.0, 0.1, 0.01])
        assert touched == set(GROUPACV_CONSTRUCTION_CONTRACT)


class TestTheRealStatisticSatisfiesTheContract:
    """The fakes describe the real class, not a parallel invention."""

    @pytest.mark.parametrize("member", sorted(MC_CONTRACT))
    def test_multi_output_mean_has_every_member(
        self, numpy_bkd: Backend[Array], member: str
    ) -> None:
        assert hasattr(MultiOutputMean(1, numpy_bkd), member)

    def test_costs_agree_with_the_real_statistic(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """Same covariance through both paths must price identically.

        Establishes that the fake stands in for the real class rather
        than merely satisfying the estimator's attribute lookups.
        """
        nmodels, npilot = 1, 200
        rng = np.random.RandomState(0)
        values = [
            numpy_bkd.asarray(rng.normal(0.0, 2.0, (1, npilot)))
            for _ in range(nmodels)
        ]
        real = MultiOutputMean(1, numpy_bkd)
        real.set_pilot_quantities(*real.compute_pilot_quantities(values))
        real_cost = (
            MCToleranceAllocator(MCEstimator(real, [1.0]))
            .allocate_for_tolerance(_tolerance(numpy_bkd))
            .actual_cost()
        )

        fake = _MinimalStatistic(numpy_bkd, nmodels=1)
        fake._cov = real._cov
        fake_cost = (
            MCToleranceAllocator(MCEstimator(fake, [1.0]))
            .allocate_for_tolerance(_tolerance(numpy_bkd))
            .actual_cost()
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([fake_cost]), numpy_bkd.asarray([real_cost])
        )

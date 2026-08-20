"""Bootstrapping pilot uncertainty through allocation and covariance.

Pilot quantities are estimated from a finite pilot sample, so they carry
uncertainty of their own. That uncertainty propagates: into the budget a
tolerance-driven allocation reports, and into the estimator covariance a
budget-driven allocation achieves. Neither is visible in a single solve,
which reports one number as though the pilot were exact.

Two directions, distinguished by what is held fixed:

- :func:`bootstrap_budget_from_pilot` lets the allocation vary. The
  tolerance is fixed and the cost is the output, so pilot noise lands
  directly on the number a caller plans compute around.
- :func:`bootstrap_covariance_from_pilot` holds the allocation fixed.
  The samples are committed, so re-optimizing would describe an
  allocation the caller is not going to draw; what varies is the
  accuracy those committed samples achieve.

Both resample through a :class:`PilotReplicateProtocol`, which is the
only part that knows where replicate pilot values come from. Resampling
the recorded values models pilot-size variability alone; an
implementation that redraws a surrogate and re-evaluates it would add
surrogate error to the same loop without either function changing.

These are deliberately free functions rather than methods. Each
replicate needs a *fresh* statistic and, for the budget direction, a
fresh allocator: reusing one instance would mean mutating the pilot
quantities of an object mid-loop and restoring them afterwards. Taking
factories keeps every replicate independent by construction.
"""

from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from pyapprox.util.backends.protocols import (
    Array,
    Array_co,
    ArrayProtocol,
    Backend,
)

if TYPE_CHECKING:
    from pyapprox.statest.tolerance import ToleranceConstraintProtocol


@runtime_checkable
class PilotStatisticProtocol(Protocol[Array]):
    """The pilot round trip: values in, pilot quantities set."""

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, ...]:
        ...

    def set_pilot_quantities(self, *args: Array) -> None:
        ...


@runtime_checkable
class BudgetSolverProtocol(Protocol[Array]):
    """Solves one replicate for the cost of meeting a requirement.

    Narrower than an allocator: it reports only what the bootstrap
    reads, a cost and the allocation that achieved it. The shipped
    allocators disagree on their return type -- the Monte Carlo and
    control variate ones report cost through a method, the group one
    through an attribute, and they signal failure differently. Rather
    than force one shape on them, which would change a public return
    type as a side effect of adding bootstrapping, each family adapts
    to this protocol through a small wrapper below.

    An implementation signals an unreachable requirement by raising
    ``ValueError`` or ``RuntimeError``; the loop counts that replicate
    as a failure.
    """

    def solve(
        self, statistic: "PilotStatisticProtocol[Array]"
    ) -> Tuple[float, Array]:
        """Return ``(cost, allocation)`` for one replicate."""
        ...


class SampleCountBudgetSolver(Generic[Array]):
    """Solves replicates with an allocator returning per-model counts.

    Selected by the *convention* the allocator follows rather than by
    which family it belongs to: cost reported through the method
    ``actual_cost()``, an allocation of one sample count per model, and
    an unreachable requirement signalled by raising, which the loop
    already counts as a failed replicate. The Monte Carlo and control
    variate allocators follow it today; any later allocator that does
    is served by this class unchanged.

    Nothing here is fitted in the usual sense: no samples are drawn. The
    allocator returns an estimator object because that is its normal
    contract, and this adapter reads the cost and the sample counts off
    it and discards the rest.
    """

    def __init__(
        self,
        allocator_factory: Callable[
            ["PilotStatisticProtocol[Array]"],
            "_SampleCountAllocator[Array]",
        ],
        constraint: "ToleranceConstraintProtocol[Array]",
    ):
        self._allocator_factory = allocator_factory
        self._constraint = constraint

    def solve(
        self, statistic: "PilotStatisticProtocol[Array]"
    ) -> Tuple[float, Array]:
        fitted = self._allocator_factory(statistic).allocate_for_tolerance(
            self._constraint
        )
        return float(fitted.actual_cost()), fitted.nsamples_per_model()


class PartitionBudgetSolver(Generic[Array]):
    """Solves replicates with an allocator returning per-partition counts.

    Selected by the *convention* the allocator follows rather than by
    which family it belongs to: cost reported through the attribute
    ``total_cost``, an allocation of one sample count per partition, and
    failure reported through a ``success`` flag rather than by raising,
    which is converted here into the exception the loop counts. The
    group allocator follows it today; an approximate control variate
    allocator adopting the same convention would be served by this
    class unchanged, there being no tolerance-driven one at present.
    """

    def __init__(
        self,
        allocator_factory: Callable[
            ["PilotStatisticProtocol[Array]"],
            "_PartitionAllocator[Array]",
        ],
        constraint: "ToleranceConstraintProtocol[Array]",
        round_nsamples: bool = True,
    ):
        self._allocator_factory = allocator_factory
        self._constraint = constraint
        self._round_nsamples = round_nsamples

    def solve(
        self, statistic: "PilotStatisticProtocol[Array]"
    ) -> Tuple[float, Array]:
        result = self._allocator_factory(statistic).allocate_for_tolerance(
            self._constraint, round_nsamples=self._round_nsamples
        )
        if not result.success:
            raise RuntimeError(result.message)
        return float(result.total_cost), result.npartition_samples


class _SampleCountResult(Protocol[Array_co]):
    """The parts of a sample-count allocation result the adapter reads.

    Covariant in the array type: this protocol only produces arrays and
    never consumes them, which the shared invariant typevar cannot
    express.
    """

    def actual_cost(self) -> float: ...

    def nsamples_per_model(self) -> Array_co: ...


class _SampleCountAllocator(Protocol[Array]):
    """An allocator whose allocation is a sample count per model."""

    def allocate_for_tolerance(
        self, constraint: "ToleranceConstraintProtocol[Array]"
    ) -> "_SampleCountResult[Array]":
        ...


class _PartitionResult(Protocol[Array_co]):
    """The parts of a partition allocation result the adapter reads.

    Covariant for the same reason as :class:`_SampleCountResult`.
    """

    @property
    def total_cost(self) -> float: ...

    @property
    def npartition_samples(self) -> Array_co: ...

    @property
    def success(self) -> bool: ...

    @property
    def message(self) -> str: ...


class _PartitionAllocator(Protocol[Array]):
    """An allocator whose allocation is a sample count per partition."""

    def allocate_for_tolerance(
        self,
        constraint: "ToleranceConstraintProtocol[Array]",
        round_nsamples: bool = ...,
    ) -> "_PartitionResult[Array]":
        ...


@runtime_checkable
class CovarianceEstimatorProtocol(Protocol[Array]):
    """An estimator that reports its covariance at a given allocation."""

    def covariance_at_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        ...


@runtime_checkable
class PilotReplicateProtocol(Protocol[Array]):
    """Draws one bootstrap replicate of the pilot values.

    The single seam between "where replicate pilot values come from" and
    "what is done with them". Implementations decide what varies between
    replicates; the bootstrap loop is indifferent to it.

    Every replicate must contain :meth:`npilot` samples. A bootstrap
    estimates the sampling distribution of a statistic *at the sample
    size actually held*, and pilot uncertainty in the covariance scales
    roughly as ``1/sqrt(npilot)``. Replicates of differing size would
    blend several hypothetical pilot sizes into one spread, answering
    neither "how uncertain is my pilot" nor "how much would a larger
    pilot help". The latter is a pilot-size sweep: run this bootstrap
    separately at each fixed size and compare.
    """

    def npilot(self) -> int:
        """Samples per replicate. Constant across replicates."""
        ...

    def nmodels(self) -> int:
        """Number of models each replicate supplies values for."""
        ...

    def draw(self) -> List[Array]:
        """One replicate: a list of ``(nqoi, npilot)`` arrays."""
        ...


class ResampledPilotValues(Generic[Array]):
    """Replicates drawn by resampling recorded pilot values.

    The nonparametric bootstrap: draw ``npilot`` column indices with
    replacement and take those columns from every model.

    The index set is **shared across models**. Pilot values are the same
    inputs pushed through different models, and that shared input is
    what produces the cross-model correlations the multi-fidelity method
    exploits. Resampling each model independently would drive the
    off-diagonals of the covariance toward zero, giving a distribution
    for an uncorrelated ensemble -- a problem where multi-fidelity buys
    nothing, and not the one being asked about.

    This models pilot-size variability only. It treats the recorded
    values as exact, which is right when they are direct model
    evaluations.
    """

    def __init__(self, pilot_values: List[Array], bkd: Backend[Array]):
        """
        Parameters
        ----------
        pilot_values : List[Array]
            One ``(nqoi, npilot)`` array per model, all sharing the same
            ``npilot`` and drawn at the same inputs.
        bkd : Backend
            Backend used to build index arrays.
        """
        if not isinstance(pilot_values, list) or len(pilot_values) == 0:
            raise ValueError("pilot_values must be a non-empty list")
        for vals in pilot_values:
            if not isinstance(vals, ArrayProtocol):
                raise ValueError("pilot_values entry must be an ArrayProtocol")
            if vals.ndim != 2:
                raise ValueError(
                    "pilot_values entry must be 2D (nqoi, npilot), got "
                    f"ndim={vals.ndim}"
                )
        npilot = pilot_values[0].shape[1]
        for ii, vals in enumerate(pilot_values):
            if vals.shape[1] != npilot:
                raise ValueError(
                    "every model must supply the same number of pilot "
                    f"samples; model 0 has {npilot}, model {ii} has "
                    f"{vals.shape[1]}. A shared resampling index requires "
                    "values drawn at common inputs."
                )
        self._pilot_values = pilot_values
        self._bkd = bkd
        self._npilot = int(npilot)

    def npilot(self) -> int:
        return self._npilot

    def nmodels(self) -> int:
        return len(self._pilot_values)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def draw(self) -> List[Array]:
        indices = self._bkd.array(
            np.random.choice(
                np.arange(self._npilot, dtype=int),
                size=self._npilot,
                replace=True,
            ),
            dtype=int,
        )
        return [vals[:, indices] for vals in self._pilot_values]


class BootstrapSamples(Generic[Array]):
    """Per-replicate bootstrap results, with the failures counted.

    Holds the raw replicates rather than a summary. The quantities
    bootstrapped here are nonlinear in the pilot covariance, so the mean
    over replicates is not the value obtained at the nominal pilot, and
    a caller planning compute usually wants an upper quantile rather
    than a center.

    Replicates that failed are counted, never silently dropped. A pilot
    covariance drawn with replacement can make a tolerance unreachable
    or an optimization fail; discarding those would bias the reported
    distribution toward the pilots that happened to be easy, which is
    the opposite of what the bootstrap is being asked.
    """

    def __init__(
        self,
        values: Array,
        nfailures: int,
        nbootstraps: int,
        bkd: Backend[Array],
        allocations: Optional[List[Array]] = None,
    ):
        self._values = values
        self._nfailures = int(nfailures)
        self._nbootstraps = int(nbootstraps)
        self._bkd = bkd
        self._allocations = allocations

    def values(self) -> Array:
        """Successful replicates, shape ``(nsuccess, ...)``."""
        return self._values

    def allocations(self) -> Optional[List[Array]]:
        """Allocation of each successful replicate, when recorded.

        Worth inspecting for group estimators: pilot noise can change
        which model subsets are active, which moves the cost
        discontinuously. Quantiles of the cost alone do not show that.
        """
        return self._allocations

    def nfailures(self) -> int:
        """Replicates that did not produce a result."""
        return self._nfailures

    def nsuccesses(self) -> int:
        """Replicates that did."""
        return self._nbootstraps - self._nfailures

    def nbootstraps(self) -> int:
        """Replicates attempted."""
        return self._nbootstraps

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def quantile(self, level: float) -> Array:
        """Quantile of the replicate values at ``level`` in [0, 1]."""
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        if self.nsuccesses() == 0:
            raise ValueError(
                "no successful replicates; "
                f"all {self._nbootstraps} failed"
            )
        return self._bkd.quantile(self._values, level, axis=0)

    def mean(self) -> Array:
        """Mean over replicates.

        Not the value at the nominal pilot: these quantities are
        nonlinear in the pilot covariance, so the two differ.
        """
        if self.nsuccesses() == 0:
            raise ValueError(
                "no successful replicates; "
                f"all {self._nbootstraps} failed"
            )
        return self._bkd.mean(self._values, axis=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nsuccesses={self.nsuccesses()}, "
            f"nfailures={self._nfailures})"
        )


def _bootstrap_pilot_loop(
    pilot: PilotReplicateProtocol[Array],
    statistic_factory: Callable[[], PilotStatisticProtocol[Array]],
    evaluate: Callable[
        [PilotStatisticProtocol[Array]], Tuple[Array, Optional[Array]]
    ],
    nbootstraps: int,
    bkd: Backend[Array],
) -> BootstrapSamples[Array]:
    """Draw replicates, refresh the statistic, evaluate, collect.

    The shared body of both public functions. They differ only in
    ``evaluate``, which returns the quantity of interest and, optionally,
    the allocation that produced it.
    """
    if nbootstraps < 1:
        raise ValueError(f"nbootstraps must be >= 1, got {nbootstraps}")
    npilot = pilot.npilot()
    collected: List[Array] = []
    allocations: List[Array] = []
    nfailures = 0
    for _ in range(nbootstraps):
        replicate = pilot.draw()
        if len(replicate) != pilot.nmodels():
            raise ValueError(
                f"replicate has {len(replicate)} models, expected "
                f"{pilot.nmodels()}"
            )
        for vals in replicate:
            if vals.shape[1] != npilot:
                raise ValueError(
                    f"replicate has {vals.shape[1]} samples, expected "
                    f"{npilot}. Replicates must all be the size the "
                    "bootstrap is estimating the distribution at."
                )
        stat = statistic_factory()
        stat.set_pilot_quantities(*stat.compute_pilot_quantities(replicate))
        try:
            value, allocation = evaluate(stat)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            nfailures += 1
            continue
        if value is None:
            nfailures += 1
            continue
        collected.append(bkd.flatten(value))
        if allocation is not None:
            allocations.append(allocation)
    if collected:
        values = bkd.stack(collected)
    else:
        values = bkd.array([])
    return BootstrapSamples(
        values,
        nfailures,
        nbootstraps,
        bkd,
        allocations if allocations else None,
    )


def bootstrap_budget_from_pilot(
    pilot: PilotReplicateProtocol[Array],
    statistic_factory: Callable[[], PilotStatisticProtocol[Array]],
    solver: BudgetSolverProtocol[Array],
    bkd: Backend[Array],
    nbootstraps: int = 1000,
) -> BootstrapSamples[Array]:
    """Distribution of the cost needed to meet an accuracy requirement.

    Re-solves the tolerance-driven allocation for every pilot replicate,
    so the allocation varies and the cost is the output. This is the
    quantity a caller budgets against, and a single solve reports it
    without any indication of how much the pilot could move it.

    Parameters
    ----------
    pilot : PilotReplicateProtocol
        Supplies replicate pilot values.
    statistic_factory : callable
        Returns a fresh statistic of the right type per replicate.
    allocator_factory : callable
        Maps a statistic to an allocator exposing
        Solves one replicate, carrying the accuracy requirement and the
        allocator it applies to. Use :class:`SampleCountBudgetSolver`
        for the Monte Carlo and control variate allocators and
        :class:`PartitionBudgetSolver` for the group allocator; this
        function does not branch on which.
    bkd : Backend
    nbootstraps : int

    Returns
    -------
    BootstrapSamples
        Per-replicate total cost, with the replicates whose requirement
        proved unreachable counted as failures.
    """

    def _evaluate(
        stat: PilotStatisticProtocol[Array],
    ) -> Tuple[Array, Optional[Array]]:
        cost, allocation = solver.solve(stat)
        return bkd.array([cost]), allocation

    return _bootstrap_pilot_loop(
        pilot, statistic_factory, _evaluate, nbootstraps, bkd
    )


def bootstrap_covariance_from_pilot(
    pilot: PilotReplicateProtocol[Array],
    statistic_factory: Callable[[], PilotStatisticProtocol[Array]],
    estimator_factory: Callable[
        [PilotStatisticProtocol[Array]], CovarianceEstimatorProtocol[Array]
    ],
    npartition_samples: Array,
    bkd: Backend[Array],
    nbootstraps: int = 1000,
) -> BootstrapSamples[Array]:
    """Distribution of the estimator covariance at a fixed allocation.

    The complement of :func:`bootstrap_budget_from_pilot`: here the
    allocation is held fixed and the achieved accuracy varies. That is
    the right question once samples are committed, when re-optimizing
    would describe an allocation the caller is not going to draw.

    Parameters
    ----------
    pilot : PilotReplicateProtocol
        Supplies replicate pilot values.
    statistic_factory : callable
        Returns a fresh statistic of the right type per replicate.
    estimator_factory : callable
        Maps a statistic to an estimator exposing
        ``covariance_at_npartition_samples``.
    npartition_samples : Array
        The committed allocation, shape ``(npartitions,)``, float-typed.
    bkd : Backend
    nbootstraps : int

    Returns
    -------
    BootstrapSamples
        Per-replicate flattened covariance. No allocations are recorded,
        the allocation being fixed by construction.
    """
    nps_float = bkd.asarray(npartition_samples, dtype=bkd.double_dtype())

    def _evaluate(
        stat: PilotStatisticProtocol[Array],
    ) -> Tuple[Array, Optional[Array]]:
        estimator = estimator_factory(stat)
        covariance = estimator.covariance_at_npartition_samples(nps_float)
        return covariance, None

    return _bootstrap_pilot_loop(
        pilot, statistic_factory, _evaluate, nbootstraps, bkd
    )


__all__: Sequence[str] = [
    "BootstrapSamples",
    "BudgetSolverProtocol",
    "CovarianceEstimatorProtocol",
    "PartitionBudgetSolver",
    "PilotReplicateProtocol",
    "PilotStatisticProtocol",
    "ResampledPilotValues",
    "SampleCountBudgetSolver",
    "bootstrap_budget_from_pilot",
    "bootstrap_covariance_from_pilot",
]

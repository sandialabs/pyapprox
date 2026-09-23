"""Golden regression tests pinning adaptive refinement order.

These tests assert the exact sequence of subspace indices promoted from
candidate to selected, as literal tuples. They exist to catch unintended
reordering while the fitter's internals are rewritten: incremental
Smolyak coefficients, delta-based candidate scoring and the separation of
priority from error must all leave the chosen sequence untouched.

A failure here is meaningful only after ruling out a tie. The targets are
Genz oscillatory with decaying coefficients, which makes the per-subspace
errors distinct, so two candidates should not score equally. Confirm any
diff is a genuine tie before updating a literal.

The sequences were recorded from the implementation predating that
rewrite.
"""

from typing import Dict, List, Tuple

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    ClenshawCurtisGrowthRule,
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    L2GlobalSurplusIndicator,
    L2SurplusIndicator,
    VarianceChangeIndicator,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox_benchmarks.quadrature.genz import GenzOscillatoryBenchmark

# Sample budget per run. Large enough that the order is non-trivial,
# small enough that these stay fast tests.
_BUDGET = 60

_INDICATORS = {
    "l2": L2SurplusIndicator,
    "l2global": L2GlobalSurplusIndicator,
    "variance": VarianceChangeIndicator,
}


def _selected_sequence(
    fitter: SingleFidelityAdaptiveSparseGridFitter,
    target,
    budget: int,
) -> List[Tuple[int, ...]]:
    """Refine to a sample budget, returning promotions in order.

    Promotion happens inside ``step_samples``, which moves the chosen
    candidate into the selected set before returning that candidate's new
    samples. Snapshotting around that call — not around ``step_values`` —
    is what makes the newly selected index visible.

    A single ``step_samples`` can promote more than one subspace: when a
    promoted candidate has no admissible neighbors the fitter promotes
    the next one immediately, without asking for evaluations. Appending
    every newly selected key preserves that.
    """
    seen: List[Tuple[int, ...]] = []

    def snapshot() -> List[Tuple[int, ...]]:
        indices = fitter.get_selected_indices()
        return [
            tuple(int(v) for v in indices[:, j])
            for j in range(indices.shape[1])
        ]

    known = set(snapshot())
    while True:
        samples = fitter.step_samples()
        if samples is None:
            break
        for key in snapshot():
            if key not in known:
                known.add(key)
                seen.append(key)
        fitter.step_values(target(samples))
        if fitter.get_samples("all").shape[1] >= budget:
            break
    return seen


def _build_sf_fitter(
    bkd,
    nvars: int,
    basis_type: str,
    indicator_name: str,
):
    """Build a single-fidelity fitter and its Genz oscillatory target."""
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    # Clenshaw-Curtis points are only nested under its own doubling rule;
    # pairing it with a linear rule asks the factory for point counts its
    # nested sequence cannot supply.
    growth = (
        ClenshawCurtisGrowthRule()
        if basis_type == "clenshaw_curtis"
        else LinearGrowthRule(scale=1, shift=1)
    )
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    # Deliberately non-binding: the sample budget stops the loop.
    admis = MaxLevelCriteria(max_level=20, pnorm=1.0, bkd=bkd)
    fitter = SingleFidelityAdaptiveSparseGridFitter(
        bkd,
        tp_factory,
        admis,
        error_indicator=_INDICATORS[indicator_name](bkd),
    )
    benchmark = GenzOscillatoryBenchmark(bkd, nvars, "quadratic")
    return fitter, benchmark.problem().function()


# Recorded sequences, keyed by (basis_type, indicator, nvars).
# Captured from the pre-rewrite implementation; see module docstring.
_GOLDEN: Dict[Tuple[str, str, int], List[Tuple[int, ...]]] = {
    ('gauss', 'l2', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (0, 2), (2, 1), (1,
        2), (4, 0), (3, 1), (0, 3)
    ],
    ('gauss', 'l2', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0),
        (1, 0, 1), (0, 2, 0), (3, 0, 0), (2, 1, 0), (0, 1, 1)
    ],
    ('gauss', 'l2global', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1,
        2), (4, 0), (0, 3)
    ],
    ('gauss', 'l2global', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 0, 1), (1, 1, 0),
        (0, 2, 0), (1, 0, 1), (3, 0, 0), (2, 1, 0), (0, 0, 2)
    ],
    ('gauss', 'variance', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (3, 0), (1,
        2), (4, 0), (3, 1), (2, 2)
    ],
    ('gauss', 'variance', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0),
        (1, 0, 1), (0, 2, 0), (0, 1, 1), (2, 1, 0), (3, 0, 0)
    ],
    ('leja', 'l2', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (3, 0), (1,
        2), (2, 2), (4, 0), (3, 1), (0, 3), (1, 3), (4, 1), (3, 2), (2, 3),
        (5, 0), (0, 4), (4, 2), (1, 4), (6, 0), (5, 1), (2, 4), (3, 3), (5,
        2), (6, 1), (0, 5), (4, 3), (7, 0), (3, 4), (1, 5), (6, 2), (2, 5),
        (7, 1), (0, 6), (4, 4), (5, 3), (8, 0), (1, 6), (7, 2), (2, 6), (3,
        5), (6, 3), (8, 1), (5, 4), (9, 0), (4, 5), (0, 7), (8, 2), (5, 5),
        (6, 4), (3, 6)
    ],
    ('leja', 'l2', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 0, 1), (1, 1, 0),
        (1, 0, 1), (0, 2, 0), (2, 1, 0), (0, 1, 1), (2, 0, 1), (3, 0, 0),
        (0, 0, 2), (1, 2, 0), (1, 1, 1), (1, 0, 2), (2, 2, 0), (0, 2, 1),
        (2, 1, 1), (4, 0, 0), (0, 1, 2), (3, 1, 0), (2, 0, 2), (1, 2, 1),
        (3, 0, 1), (0, 3, 0), (1, 1, 2), (2, 2, 1), (1, 3, 0), (4, 1, 0),
        (3, 2, 0), (0, 2, 2), (2, 1, 2), (0, 0, 3), (4, 0, 1), (3, 1, 1),
        (2, 3, 0), (5, 0, 0), (3, 0, 2), (1, 2, 2), (0, 3, 1), (1, 0, 3),
        (0, 4, 0)
    ],
    ('leja', 'l2global', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (3, 0), (1,
        2), (2, 2), (4, 0), (3, 1), (0, 3), (1, 3), (4, 1), (3, 2), (2, 3),
        (5, 0), (0, 4), (4, 2), (1, 4), (6, 0), (5, 1), (2, 4), (3, 3), (5,
        2), (6, 1), (0, 5), (4, 3), (7, 0), (3, 4), (1, 5), (6, 2), (2, 5),
        (7, 1), (0, 6), (4, 4), (5, 3), (8, 0), (1, 6), (7, 2), (2, 6), (3,
        5), (6, 3), (8, 1), (5, 4), (9, 0), (0, 7), (8, 2), (3, 6), (4, 5),
        (1, 7)
    ],
    ('leja', 'l2global', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 0, 1), (1, 1, 0),
        (1, 0, 1), (0, 2, 0), (2, 1, 0), (0, 1, 1), (2, 0, 1), (3, 0, 0),
        (0, 0, 2), (1, 2, 0), (1, 1, 1), (1, 0, 2), (2, 2, 0), (0, 2, 1),
        (2, 1, 1), (4, 0, 0), (0, 1, 2), (3, 1, 0), (2, 0, 2), (1, 2, 1),
        (3, 0, 1), (0, 3, 0), (1, 1, 2), (2, 2, 1), (1, 3, 0), (4, 1, 0),
        (3, 2, 0), (0, 2, 2), (2, 1, 2), (0, 0, 3), (4, 0, 1), (3, 1, 1),
        (2, 3, 0), (5, 0, 0), (3, 0, 2), (1, 2, 2), (0, 3, 1), (1, 0, 3),
        (0, 4, 0)
    ],
    ('leja', 'variance', 2): [
        (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1,
        2), (4, 0), (5, 0), (2, 2), (0, 3), (1, 3), (2, 3), (6, 0), (7, 0),
        (3, 1), (4, 1), (3, 2), (4, 2), (5, 1), (5, 2), (6, 1), (0, 4), (1,
        4), (3, 3), (4, 3), (0, 5), (2, 4), (3, 4), (8, 0), (9, 0), (6, 2),
        (5, 3), (6, 3), (7, 1), (7, 2), (8, 1), (1, 5), (2, 5), (4, 4), (5,
        4), (3, 5), (4, 5), (8, 2), (0, 6), (1, 6), (9, 1), (9, 2), (10, 0),
        (11, 0), (12, 0)
    ],
    ('leja', 'variance', 3): [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0),
        (2, 1, 0), (0, 2, 0), (1, 2, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1),
        (0, 0, 2), (4, 0, 0), (5, 0, 0), (2, 2, 0), (1, 0, 2), (0, 3, 0),
        (0, 1, 1), (0, 2, 1), (2, 0, 2), (0, 1, 2), (1, 1, 1), (2, 1, 1),
        (3, 0, 1), (4, 0, 1), (3, 0, 2), (1, 2, 1), (2, 2, 1), (0, 0, 3),
        (1, 3, 0), (2, 3, 0), (1, 1, 2), (2, 1, 2), (3, 1, 0), (4, 1, 0),
        (3, 2, 0), (3, 1, 1), (6, 0, 0), (7, 0, 0), (0, 2, 2), (1, 2, 2),
        (4, 2, 0), (5, 1, 0), (5, 2, 0), (6, 1, 0)
    ],
    ('clenshaw_curtis', 'l2', 2): [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (0, 2), (1, 2), (3,
        0), (3, 1), (2, 2), (0, 3)
    ],
    ('clenshaw_curtis', 'l2', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
        (2, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 2, 0), (2, 0, 1)
    ],
    ('clenshaw_curtis', 'l2global', 2): [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (0, 2), (1, 2), (3,
        0), (3, 1), (2, 2), (0, 3)
    ],
    ('clenshaw_curtis', 'l2global', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
        (2, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (2, 0, 1), (0, 2, 0)
    ],
    ('clenshaw_curtis', 'variance', 2): [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (3, 0), (1,
        2), (2, 2), (3, 1), (0, 3)
    ],
    ('clenshaw_curtis', 'variance', 3): [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 0, 1), (1, 1, 0),
        (1, 0, 1), (0, 2, 0), (2, 1, 0), (0, 1, 1), (3, 0, 0)
    ],
}


class TestGoldenRefinementOrder:
    """Pin the promotion order for each rule/indicator combination."""

    @pytest.mark.parametrize("basis_type", ["gauss", "leja", "clenshaw_curtis"])
    @pytest.mark.parametrize(
        "indicator_name", ["l2", "l2global", "variance"]
    )
    @pytest.mark.parametrize("nvars", [2, 3])
    def test_selected_sequence(
        self, numpy_bkd, basis_type: str, indicator_name: str, nvars: int
    ) -> None:
        """The promotion order matches the recorded sequence."""
        key = (basis_type, indicator_name, nvars)
        if key not in _GOLDEN:
            pytest.skip(f"no recorded sequence for {key}")
        fitter, target = _build_sf_fitter(
            numpy_bkd, nvars, basis_type, indicator_name
        )
        sequence = _selected_sequence(fitter, target, _BUDGET)
        assert sequence == _GOLDEN[key]

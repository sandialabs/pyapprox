"""Golden regression tests pinning adaptive refinement order.

These tests assert the exact sequence of subspace indices promoted from
candidate to selected, as literal tuples. They exist to catch unintended
reordering while the fitter's internals are rewritten: incremental
Smolyak coefficients, delta-based candidate scoring and the separation of
priority from error must all leave the chosen sequence untouched.

A failure here is meaningful only after ruling out a tie. The targets are
Genz oscillatory with decaying coefficients, which keeps the
per-subspace errors distinct while the target is still being resolved.
Confirm any diff is a genuine tie before updating a literal.

Only the first ``_NCOMPARED`` promotions are asserted; the recorded
sequences run longer so the cut can be moved without re-recording.

The sequences were recorded from the implementation predating that
rewrite.
"""

from typing import Dict, List, Tuple

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    ClenshawCurtisGrowthRule,
    CompositeCriteria,
    LinearGrowthRule,
    Max1DLevelsCriteria,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.cost_model import (
    ExponentialConfigCostModel,
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

# Promotions to compare. Beyond roughly 40 subspaces the 2D Leja cases
# resolve the target to machine precision, and every remaining
# candidate's surplus is a few multiples of eps. Candidates then tie
# exactly --- two were observed with bit-identical priorities of
# 5.55e-16 --- so which one the queue returns is decided by summation
# order rather than by refinement. Comparing only the prefix keeps these
# tests measuring the choice of subspace instead of floating-point
# associativity.
_NCOMPARED = 40

_INDICATORS = {
    "l2": L2SurplusIndicator,
    "l2global": L2GlobalSurplusIndicator,
    "variance": VarianceChangeIndicator,
}


def _nsamples(fitter) -> int:
    """Unique samples held, for either fitter flavour.

    The single-fidelity wrapper returns an array; the multi-fidelity
    fitter returns one array per config, and the budget counts every
    evaluation regardless of which fidelity paid for it.
    """
    samples = fitter.get_samples("all")
    if isinstance(samples, dict):
        return sum(s.shape[1] for s in samples.values())
    return samples.shape[1]


def _selected_sequence(
    fitter,
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
        if _nsamples(fitter) >= budget:
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


def _build_mf_fitter(bkd, indicator_name: str):
    """Build a 2D-physical, 1-config-var fitter and its model hierarchy.

    Fidelity alpha perturbs the Genz oscillatory target by a phase shift
    that shrinks with alpha, so the coarse model is cheap but biased and
    the fine model is exact. With ``ExponentialConfigCostModel`` the
    cost-weighted priority has to trade physical refinement against
    buying a better fidelity, which is the interaction the config
    dimension exists to exercise.
    """
    nvars = 2
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, "gauss")
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    max_config_level = 1
    # One entry per index dimension: the physical caps are deliberately
    # loose, since the sample budget stops the loop, while the trailing
    # config cap keeps the hierarchy to two fidelities.
    max_1d = bkd.asarray(
        [20] * nvars + [max_config_level], dtype=bkd.int64_dtype()
    )
    admis = CompositeCriteria(
        MaxLevelCriteria(max_level=20, pnorm=1.0, bkd=bkd),
        Max1DLevelsCriteria(max_1d, bkd),
    )
    fitter = MultiFidelityAdaptiveSparseGridFitter(
        bkd,
        tp_factory,
        admis,
        nconfig_vars=1,
        error_indicator=_INDICATORS[indicator_name](bkd),
        cost_model=ExponentialConfigCostModel(base=10.0),
    )

    exact = GenzOscillatoryBenchmark(bkd, nvars, "quadratic")
    exact_fn = exact.problem().function()

    def _model_for(alpha: int):
        # alpha = 1 is the exact target; alpha = 0 is phase-shifted.
        bias = 0.0 if alpha == max_config_level else 0.15

        def model(samples):
            return exact_fn(samples) + bias * bkd.cos(samples[:1, :])

        return model

    models = {
        (alpha,): _model_for(alpha)
        for alpha in range(max_config_level + 1)
    }

    def target(sample_dict):
        return {cfg: models[cfg](s) for cfg, s in sample_dict.items()}

    return fitter, target


# Recorded multi-fidelity sequences, keyed by indicator. The full index
# is [physical_0, physical_1, config]; a trailing 1 is a fidelity
# purchase rather than a physical refinement.
_GOLDEN_MF: Dict[str, List[Tuple[int, ...]]] = {
    "l2": [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
        (1, 1, 0), (1, 0, 1), (3, 0, 0), (0, 2, 0), (2, 1, 0),
        (2, 0, 1), (1, 2, 0), (3, 0, 1), (3, 1, 0), (0, 3, 0),
    ],
    "l2global": [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
        (1, 1, 0), (1, 0, 1), (3, 0, 0), (0, 2, 0), (2, 1, 0),
        (2, 0, 1), (1, 2, 0), (3, 0, 1), (0, 3, 0),
    ],
    "variance": [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
        (1, 0, 1), (1, 1, 0), (2, 0, 1), (0, 2, 0), (3, 0, 0),
        (2, 1, 0), (1, 2, 0), (3, 0, 1), (4, 0, 0),
    ],
}


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
        fitter, target = _build_sf_fitter(
            numpy_bkd, nvars, basis_type, indicator_name
        )
        sequence = _selected_sequence(fitter, target, _BUDGET)
        expected = _GOLDEN[key]
        assert sequence[:_NCOMPARED] == expected[:_NCOMPARED]


class TestGoldenMultiFidelityRefinementOrder:
    """Pin the promotion order when a config dimension is present.

    The config dimension is where the cost model enters the priority, so
    it exercises a path the single-fidelity cases cannot: the fitter has
    to weigh refining the physical grid against buying a higher fidelity.
    Phase 3 of the rewrite changes how coefficients are tracked for full
    indices that include config dims, which is exactly what this pins.
    """

    @pytest.mark.parametrize(
        "indicator_name", ["l2", "l2global", "variance"]
    )
    def test_selected_sequence(
        self, numpy_bkd, indicator_name: str
    ) -> None:
        """The promotion order matches the recorded sequence."""
        fitter, target = _build_mf_fitter(numpy_bkd, indicator_name)
        sequence = _selected_sequence(fitter, target, _BUDGET)
        expected = _GOLDEN_MF[indicator_name]
        assert sequence[:_NCOMPARED] == expected[:_NCOMPARED]

    def test_config_dimension_is_exercised(self, numpy_bkd) -> None:
        """The recorded runs actually buy fidelity, not just refine.

        A sequence that never raised the config level would pin nothing
        about multi-fidelity behavior, so assert the premise rather than
        trusting the literals to keep covering it.
        """
        for sequence in _GOLDEN_MF.values():
            assert any(key[-1] > 0 for key in sequence)
            assert any(key[-1] == 0 and sum(key) > 0 for key in sequence)

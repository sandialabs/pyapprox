"""The fitter's incremental coefficients match inclusion-exclusion.

The adaptive fitter maintains Smolyak coefficients one promotion at a
time. These tests check that result against
``compute_smolyak_coefficients`` on the same index set, after every
step, so a divergence is caught at the step that caused it rather than
at the end of a run.
"""

from typing import List, Tuple

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
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
from pyapprox.surrogates.sparsegrids.smolyak import (
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox_benchmarks.quadrature.genz import GenzOscillatoryBenchmark


def _expected(keys: List[Tuple[int, ...]], bkd) -> List[int]:
    """Coefficients from inclusion-exclusion, aligned with ``keys``."""
    nvars = len(keys[0])
    indices = bkd.asarray([[k[d] for k in keys] for d in range(nvars)])
    return [round(float(c)) for c in compute_smolyak_coefficients(indices, bkd)]


def _make_sf(bkd, nvars: int, max_level: int = 20):
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, "leja")
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )
    admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
    fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)
    target = GenzOscillatoryBenchmark(bkd, nvars, "quadratic")
    return fitter, target.problem().function()


def _make_mf(bkd):
    nvars = 2
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, "gauss")
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )
    max_1d = bkd.asarray([20] * nvars + [1], dtype=bkd.int64_dtype())
    admis = CompositeCriteria(
        MaxLevelCriteria(max_level=20, pnorm=1.0, bkd=bkd),
        Max1DLevelsCriteria(max_1d, bkd),
    )
    fitter = MultiFidelityAdaptiveSparseGridFitter(
        bkd, tp_factory, admis, nconfig_vars=1
    )
    exact = GenzOscillatoryBenchmark(bkd, nvars, "quadratic")
    exact_fn = exact.problem().function()

    def target(sample_dict):
        return {
            cfg: exact_fn(s) + (0.0 if cfg[-1] == 1 else 0.15)
            for cfg, s in sample_dict.items()
        }

    return fitter, target


class TestCoefficientsMatchFromScratch:
    """Selected coefficients agree after every step."""

    @pytest.mark.parametrize("nvars", [2, 3])
    def test_single_fidelity(self, numpy_bkd, nvars: int) -> None:
        fitter, target = _make_sf(numpy_bkd, nvars)
        inner = fitter._fitter
        for _ in range(12):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
            keys = inner._smolyak.keys()
            assert inner._smolyak.coefficient_list(keys) == _expected(
                keys, numpy_bkd
            )
            assert inner._smolyak.nterms() == fitter.nselected()

    def test_multi_fidelity(self, numpy_bkd) -> None:
        """Full keys including a config dimension stay consistent."""
        fitter, target = _make_mf(numpy_bkd)
        for _ in range(12):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
            keys = fitter._smolyak.keys()
            assert fitter._smolyak.coefficient_list(keys) == _expected(
                keys, numpy_bkd
            )
            assert fitter._smolyak.nterms() == fitter.nselected()

    @pytest.mark.parametrize("nvars", [2, 3])
    def test_with_candidates(self, numpy_bkd, nvars: int) -> None:
        """Selected plus candidates also matches inclusion-exclusion."""
        fitter, target = _make_sf(numpy_bkd, nvars)
        for _ in range(8):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
            result = fitter.result(include_candidates=True)
            keys = [
                tuple(int(v) for v in result.indices[:, j])
                for j in range(result.indices.shape[1])
            ]
            got = [round(float(c)) for c in result.coefficients]
            assert got == _expected(keys, numpy_bkd)

    def test_multi_promotion_step_under_level_cap(self, numpy_bkd) -> None:
        """A tight cap makes a step promote several subspaces at once.

        When a promoted candidate has no admissible neighbors the fitter
        promotes the next one without asking for values, so one step can
        add several keys. The coefficients must still agree.
        """
        fitter, target = _make_sf(numpy_bkd, 2, max_level=4)
        inner = fitter._fitter
        promotions = []
        for _ in range(30):
            before = fitter.nselected()
            samples = fitter.step_samples()
            if samples is None:
                break
            promotions.append(fitter.nselected() - before)
            fitter.step_values(target(samples))
            keys = inner._smolyak.keys()
            assert inner._smolyak.coefficient_list(keys) == _expected(
                keys, numpy_bkd
            )
        # The cap must actually have forced a multi-promotion step,
        # otherwise this test is not exercising the continue path.
        assert max(promotions) > 1


class TestCoefficientSanity:
    """Properties that hold for any downward-closed selected set."""

    def test_coefficients_sum_to_one(self, numpy_bkd) -> None:
        fitter, target = _make_sf(numpy_bkd, 2)
        inner = fitter._fitter
        for _ in range(10):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
            keys = inner._smolyak.keys()
            assert sum(inner._smolyak.coefficient_list(keys)) == 1

    def test_every_selected_key_has_a_subspace(self, numpy_bkd) -> None:
        """Coefficient keys and the subspace lookup stay in step."""
        fitter, target = _make_sf(numpy_bkd, 3)
        inner = fitter._fitter
        for _ in range(10):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(target(samples))
            for key in inner._smolyak.keys():
                assert key in inner._subspace_by_key

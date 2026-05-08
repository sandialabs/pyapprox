"""Tests for candidate generator."""

import pytest

from pyapprox.surrogates.gaussianprocess.adaptive.candidate_generator import (
    HybridSobolRandomCandidateGenerator,
)
from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    CandidateGeneratorProtocol,
)


class TestCandidateGenerator:
    """Base tests for HybridSobolRandomCandidateGenerator."""

    def test_protocol_compliance(self, bkd) -> None:
        gen = HybridSobolRandomCandidateGenerator(2, bkd)
        assert isinstance(gen, CandidateGeneratorProtocol)

    def test_shape(self, bkd) -> None:
        nvars, ncandidates = 3, 100
        gen = HybridSobolRandomCandidateGenerator(nvars, bkd)
        candidates = gen.generate(ncandidates)
        assert candidates.shape == (nvars, ncandidates)

    def test_default_bounds(self, bkd) -> None:
        gen = HybridSobolRandomCandidateGenerator(2, bkd)
        candidates = gen.generate(200)
        assert bkd.all_bool(candidates >= 0.0)
        assert bkd.all_bool(candidates <= 1.0)

    def test_custom_bounds(self, bkd) -> None:
        bounds = bkd.asarray([[-1.0, 1.0], [0.0, 2.0]])
        gen = HybridSobolRandomCandidateGenerator(2, bkd, scaled_bounds=bounds)
        candidates = gen.generate(200)
        assert bkd.all_bool(candidates[0, :] >= -1.0)
        assert bkd.all_bool(candidates[0, :] <= 1.0)
        assert bkd.all_bool(candidates[1, :] >= 0.0)
        assert bkd.all_bool(candidates[1, :] <= 2.0)

    def test_reproducibility(self, bkd) -> None:
        gen1 = HybridSobolRandomCandidateGenerator(2, bkd, seed=7)
        gen2 = HybridSobolRandomCandidateGenerator(2, bkd, seed=7)
        c1 = gen1.generate(50)
        c2 = gen2.generate(50)
        bkd.assert_allclose(c1, c2)

    def test_invalid_bounds_shape(self, bkd) -> None:
        bounds = bkd.asarray([[0.0, 1.0]])
        with pytest.raises(ValueError):
            HybridSobolRandomCandidateGenerator(2, bkd, scaled_bounds=bounds)

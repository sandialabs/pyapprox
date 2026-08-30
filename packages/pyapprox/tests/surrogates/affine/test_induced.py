"""Tests for induced sampling.

The gate is test_weighted_gram_converges_at_monte_carlo_rate: it is the
only test here that fails if the sampling weight and its reciprocal are
swapped. A fit to noiseless in-span data is recovered exactly under any
positive diagonal weighting, so recovery tests cannot detect that error.

The convergence tests assert a *rate* rather than a threshold. An
unbiased estimator's error falls like 1/sqrt(M); a biased one plateaus.
A loose fixed bound passes for both, so it would not have caught the
inverted weight that motivated this design.
"""

from typing import Any

import numpy as np
import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.surrogates.affine.basis.orthonormal_poly import (
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.induced import (
    InducedSampler,
    MonteCarloSampler,
    WeightedSample,
    christoffel_function,
)
from pyapprox.surrogates.affine.univariate.factory import create_bases_1d
from pyapprox.surrogates.operatorlearning import (
    gram_condition_number,
    sample_complexity,
    weighted_gram,
)
from pyapprox.util.backends.protocols import Backend


def _setup(bkd: Backend, nvars: int, max_level: int) -> Any:
    """Return (orthonormal basis, reference measure) on [-1, 1]^nvars."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    basis = OrthonormalPolynomialBasis(create_bases_1d(marginals, bkd), bkd)
    basis.set_indices(compute_hyperbolic_indices(nvars, max_level, 1.0, bkd))
    return basis, IndependentJoint(marginals, bkd)


class TestMonteCarloSampler:
    def test_shapes(self, bkd: Backend) -> None:
        _, rho = _setup(bkd, 2, 2)
        sample = MonteCarloSampler(rho, bkd)(50)
        assert sample.samples.shape == (2, 50)
        assert sample.weights.shape == (50,)

    def test_weights_are_one(self, bkd: Backend) -> None:
        """The sampling measure is the reference measure, so drho/dmu = 1."""
        _, rho = _setup(bkd, 2, 2)
        sample = MonteCarloSampler(rho, bkd)(30)
        bkd.assert_allclose(sample.weights, bkd.ones((30,)))

    def test_samples_lie_in_domain(self, bkd: Backend) -> None:
        _, rho = _setup(bkd, 2, 2)
        coefs = MonteCarloSampler(rho, bkd)(200).samples
        assert bool(bkd.min(coefs) >= -1.0)
        assert bool(bkd.max(coefs) <= 1.0)

    def test_returns_weighted_sample(self, bkd: Backend) -> None:
        _, rho = _setup(bkd, 2, 2)
        assert isinstance(MonteCarloSampler(rho, bkd)(5), WeightedSample)

    def test_rejects_nonpositive_nsamples(self, bkd: Backend) -> None:
        _, rho = _setup(bkd, 2, 2)
        with pytest.raises(ValueError, match="must be positive"):
            MonteCarloSampler(rho, bkd)(0)

    def test_rejects_non_joint(self, bkd: Backend) -> None:
        with pytest.raises(TypeError, match="IndependentJoint"):
            MonteCarloSampler("not_a_measure", bkd)


class TestInducedSampler:
    def test_shapes(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 3)
        sample = InducedSampler(basis, rho, bkd, nquad=50)(60)
        assert sample.samples.shape == (2, 60)
        assert sample.weights.shape == (60,)

    def test_weight_is_reciprocal_christoffel(self, bkd: Backend) -> None:
        """The weight is drho/dmu = 1 / k_Lambda, not k_Lambda."""
        basis, rho = _setup(bkd, 2, 3)
        sample = InducedSampler(basis, rho, bkd, nquad=50)(40)
        expected = 1.0 / christoffel_function(basis, sample.samples, bkd)
        bkd.assert_allclose(sample.weights, expected, rtol=1e-13)

    def test_samples_lie_in_domain(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 3)
        coefs = InducedSampler(basis, rho, bkd, nquad=50)(200).samples
        assert bool(bkd.min(coefs) >= -1.0)
        assert bool(bkd.max(coefs) <= 1.0)

    def test_weights_are_positive(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 3)
        weights = InducedSampler(basis, rho, bkd, nquad=50)(100).weights
        assert bool(bkd.min(weights) > 0.0)

    @staticmethod
    def _gram_deviation(
        basis: Any, sampler: Any, nsamples: int, bkd: Backend, invert: bool
    ) -> float:
        """Return ||G - I||_inf for a fresh sample of the given size."""
        sample = sampler(nsamples)
        weights = 1.0 / sample.weights if invert else sample.weights
        gram = weighted_gram(basis(sample.samples), weights, bkd)
        identity = bkd.eye(basis.nterms())
        return float(bkd.max(bkd.abs(gram - identity)))

    def test_weighted_gram_converges_at_monte_carlo_rate(
        self, numpy_bkd: Backend
    ) -> None:
        """T4: (1/M) sum_i w_i p_lambda p_lambda' -> delta.

        The unbiasedness identity, and the gate for this phase.
        Asserts the Monte Carlo *rate* rather than a fixed threshold:
        an unbiased estimator's error falls like 1/sqrt(M), so
        quadrupling the samples halves it. A biased estimator plateaus
        instead, which a loose fixed bound would not detect.
        """
        np.random.seed(0)
        basis, rho = _setup(numpy_bkd, 2, 3)
        sampler = InducedSampler(basis, rho, numpy_bkd, nquad=300)

        coarse = self._gram_deviation(basis, sampler, 4000, numpy_bkd, False)
        fine = self._gram_deviation(basis, sampler, 64000, numpy_bkd, False)

        # 16x the samples is 4x the accuracy for an unbiased estimator.
        # Allow a factor of two of slack for sampling noise in the
        # ratio itself, so this still fails hard on a plateau.
        assert fine < coarse / 2.0
        assert fine < 0.05

    def test_reciprocal_weights_do_not_converge(
        self, numpy_bkd: Backend
    ) -> None:
        """The inverted convention is biased, so it plateaus.

        Pins the bug this design exists to prevent: passing christoffel
        where its reciprocal belongs. The error does not shrink with
        more samples, which is what distinguishes bias from noise.
        """
        np.random.seed(0)
        basis, rho = _setup(numpy_bkd, 2, 3)
        sampler = InducedSampler(basis, rho, numpy_bkd, nquad=300)

        coarse = self._gram_deviation(basis, sampler, 4000, numpy_bkd, True)
        fine = self._gram_deviation(basis, sampler, 64000, numpy_bkd, True)

        # Bias does not average away: 16x the samples buys almost
        # nothing, and the error stays far from zero.
        assert fine > coarse / 2.0
        assert fine > 0.5

    @staticmethod
    def _sweep_conditioning(bkd: Backend, levels: range) -> Any:
        """Return (nterms, induced cond, MC cond) at each level.

        Each level is sampled at the count the theory prescribes for
        that basis size, so the sweep tests the claim the bound makes
        rather than an arbitrary sample count.
        """
        rows = []
        for level in levels:
            basis, rho = _setup(bkd, 2, level)
            nterms = basis.nterms()
            nsamples = sample_complexity(nterms, 0.5, 0.5)

            induced = InducedSampler(basis, rho, bkd, nquad=300)(nsamples)
            mc = MonteCarloSampler(rho, bkd)(nsamples)
            rows.append(
                (
                    nterms,
                    gram_condition_number(
                        basis(induced.samples), induced.weights, bkd
                    ),
                    gram_condition_number(basis(mc.samples), mc.weights, bkd),
                )
            )
        return rows

    def test_induced_conditioning_stays_bounded(
        self, numpy_bkd: Backend
    ) -> None:
        """T8: conditioning does not grow with the basis.

        Sampling at the prescribed count keeps the weighted Gram close
        to the identity however large the basis, which is the property
        the whole method rests on. Measured across a twelvefold growth
        in basis size the condition number stays near two, so a bound
        of three leaves room for sampling noise without being vacuous.
        """
        np.random.seed(0)
        rows = self._sweep_conditioning(numpy_bkd, range(1, 8))
        for nterms, induced_cond, _ in rows:
            assert induced_cond < 3.0, (
                f"induced conditioning {induced_cond:.3f} exceeded the "
                f"bound at nterms={nterms}"
            )

    def test_induced_conditioning_does_not_trend_upward(
        self, numpy_bkd: Backend
    ) -> None:
        """T8: the bound holds because conditioning is flat, not slow.

        A method that degraded gently would satisfy a fixed bound over
        a short sweep while still failing at scale, so the largest
        basis must be no worse than the smallest.
        """
        np.random.seed(0)
        rows = self._sweep_conditioning(numpy_bkd, range(1, 8))
        assert rows[-1][1] < 1.5 * rows[0][1]

    def test_monte_carlo_degrades_as_the_basis_grows(
        self, numpy_bkd: Backend
    ) -> None:
        """T8: the comparison induced sampling exists to win.

        Monte Carlo is competitive for a small basis — it is slightly
        better at the smallest size here — and degrades from there, so
        the claim is about growth rather than about being worse
        everywhere.
        """
        np.random.seed(0)
        rows = self._sweep_conditioning(numpy_bkd, range(1, 8))
        assert rows[-1][2] > 3.0 * rows[0][2]
        assert rows[-1][2] > 2.0 * rows[-1][1]

    @staticmethod
    def _cdf_deviation(sampler: Any, nsamples: int, bkd: Backend) -> float:
        """Return the Kolmogorov distance to the degree-one induced CDF.

        For Legendre degree one on [-1, 1], p_1(x)^2 rho(x) = 3x^2/2,
        so the CDF is (x^3 + 1) / 2.
        """
        coefs = sampler(nsamples).samples
        samples = np.sort(bkd.to_numpy(coefs)[0])
        empirical = np.arange(1, samples.size + 1) / samples.size
        analytic = (samples**3 + 1.0) / 2.0
        return float(np.max(np.abs(empirical - analytic)))

    def test_marginal_converges_to_analytic_cdf(
        self, numpy_bkd: Backend
    ) -> None:
        """T3: the 1-D conditional draw follows int p_k^2 d_rho.

        Asserts the rate rather than a threshold: the Kolmogorov
        distance of an empirical CDF to the true one falls like
        1/sqrt(M), so sampling from the wrong density — which converges
        to a nonzero distance — fails even though a loose fixed bound
        might not.
        """
        np.random.seed(0)
        marginals = [UniformMarginal(-1.0, 1.0, numpy_bkd)]
        basis = OrthonormalPolynomialBasis(
            create_bases_1d(marginals, numpy_bkd), numpy_bkd
        )
        # Index set {1} alone, so every draw uses the degree-one density.
        basis.set_indices(
            numpy_bkd.asarray([[1]], dtype=numpy_bkd.int64_dtype())
        )
        rho = IndependentJoint(marginals, numpy_bkd)
        sampler = InducedSampler(basis, rho, numpy_bkd, nquad=400)

        coarse = self._cdf_deviation(sampler, 2000, numpy_bkd)
        fine = self._cdf_deviation(sampler, 32000, numpy_bkd)

        # 16x the samples is 4x the accuracy; allow slack of two.
        assert fine < coarse / 2.0
        assert fine < 0.02

    def test_returns_weighted_sample(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 2)
        sampler = InducedSampler(basis, rho, bkd, nquad=50)
        assert isinstance(sampler(5), WeightedSample)

    def test_follows_index_set_changes(self, bkd: Backend) -> None:
        """Weights track Lambda, so an adaptive refit stays consistent."""
        basis, rho = _setup(bkd, 2, 2)
        sampler = InducedSampler(basis, rho, bkd, nquad=50)
        basis.set_indices(compute_hyperbolic_indices(2, 3, 1.0, bkd))
        sample = sampler(20)
        expected = 1.0 / christoffel_function(basis, sample.samples, bkd)
        bkd.assert_allclose(sample.weights, expected, rtol=1e-13)

    def test_rejects_non_basis(self, bkd: Backend) -> None:
        _, rho = _setup(bkd, 2, 2)
        with pytest.raises(
            TypeError, match="SampleableMultiIndexBasisProtocol"
        ):
            InducedSampler("not_a_basis", rho, bkd)

    def test_rejects_non_joint(self, bkd: Backend) -> None:
        basis, _ = _setup(bkd, 2, 2)
        with pytest.raises(TypeError, match="IndependentJoint"):
            InducedSampler(basis, "not_a_measure", bkd)

    def test_rejects_small_nquad(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 2)
        with pytest.raises(ValueError, match="at least two"):
            InducedSampler(basis, rho, bkd, nquad=1)

    def test_rejects_nonpositive_nsamples(self, bkd: Backend) -> None:
        basis, rho = _setup(bkd, 2, 2)
        with pytest.raises(ValueError, match="must be positive"):
            InducedSampler(basis, rho, bkd, nquad=50)(0)

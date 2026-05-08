"""Tests for ProbabilityMeasureQuadratureFactory."""

import numpy as np

from pyapprox.interface.functions.marginalize import (
    QuadratureFactoryProtocol,
)
from pyapprox.surrogates.quadrature.probability_measure_factory import (
    ProbabilityMeasureQuadratureFactory,
)


class TestProbabilityMeasureQuadratureFactory:
    def test_weights_sum_to_one_uniform(self, bkd) -> None:
        """Weights from uniform marginals should sum to 1."""
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(0.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [5, 5], bkd)
        rule = factory([0, 1])
        _, weights = rule()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_weights_sum_to_one_beta(self, bkd) -> None:
        """Weights from Beta marginals should sum to 1."""
        from pyapprox.probability.univariate.beta import BetaMarginal

        marginals = [
            BetaMarginal(2.0, 5.0, bkd),
            BetaMarginal(1.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [8, 8], bkd)
        rule = factory([0, 1])
        _, weights = rule()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_integration_exact_for_polynomials(self, bkd) -> None:
        """E[x^k] matches analytical moments for Uniform[-1,1]."""
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )

        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [5], bkd)
        rule = factory([0])
        samples, weights = rule()

        # E[x] = 0 for Uniform[-1,1]
        ex = bkd.sum(weights * samples[0])
        bkd.assert_allclose(bkd.asarray([ex]), bkd.asarray([0.0]), atol=1e-14)

        # E[x^2] = 1/3 for Uniform[-1,1]
        ex2 = bkd.sum(weights * samples[0] ** 2)
        bkd.assert_allclose(bkd.asarray([ex2]), bkd.asarray([1.0 / 3.0]), rtol=1e-12)

        # E[x^4] = 1/5 for Uniform[-1,1]
        ex4 = bkd.sum(weights * samples[0] ** 4)
        bkd.assert_allclose(bkd.asarray([ex4]), bkd.asarray([1.0 / 5.0]), rtol=1e-12)

    def test_integration_2d_product(self, bkd) -> None:
        """E[x*y] = E[x]*E[y] for independent Uniform variables."""
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )

        # Uniform[0,2]: E[x] = 1, Uniform[1,3]: E[y] = 2
        marginals = [
            UniformMarginal(0.0, 2.0, bkd),
            UniformMarginal(1.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [5, 5], bkd)
        rule = factory([0, 1])
        samples, weights = rule()

        # E[x*y] = 1 * 2 = 2
        exy = bkd.sum(weights * samples[0] * samples[1])
        bkd.assert_allclose(bkd.asarray([exy]), bkd.asarray([2.0]), rtol=1e-12)

    def test_factory_subset_selection(self, bkd) -> None:
        """Correct variables are selected from a 3-variable factory."""
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(0.0, 2.0, bkd),
            UniformMarginal(5.0, 10.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [3, 5, 7], bkd)

        # Select only variable 1
        rule = factory([1])
        samples, weights = rule()
        assert samples.shape[0] == 1
        assert samples.shape[1] == 5
        # Points in [0, 2]
        s_np = bkd.to_numpy(samples)
        assert np.all(s_np >= 0.0 - 1e-14)
        assert np.all(s_np <= 2.0 + 1e-14)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

        # Select variables 0 and 2
        rule02 = factory([0, 2])
        samples02, weights02 = rule02()
        assert samples02.shape[0] == 2
        assert samples02.shape[1] == 3 * 7
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights02)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_satisfies_protocol(self, bkd) -> None:
        """Factory satisfies QuadratureFactoryProtocol."""
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )

        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        factory = ProbabilityMeasureQuadratureFactory(marginals, [5], bkd)
        assert isinstance(factory, QuadratureFactoryProtocol)

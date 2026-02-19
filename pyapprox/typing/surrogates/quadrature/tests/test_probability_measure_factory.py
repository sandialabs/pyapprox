"""Tests for ProbabilityMeasureQuadratureFactory."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.quadrature.probability_measure_factory import (
    ProbabilityMeasureQuadratureFactory,
)
from pyapprox.typing.interface.functions.marginalize import (
    QuadratureFactoryProtocol,
)


class TestProbabilityMeasureQuadratureFactory(
    Generic[Array], unittest.TestCase
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_weights_sum_to_one_uniform(self) -> None:
        """Weights from uniform marginals should sum to 1."""
        from pyapprox.typing.probability.univariate.uniform import (
            UniformMarginal,
        )

        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(0.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [5, 5], bkd
        )
        rule = factory([0, 1])
        _, weights = rule()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_weights_sum_to_one_beta(self) -> None:
        """Weights from Beta marginals should sum to 1."""
        from pyapprox.typing.probability.univariate.beta import BetaMarginal

        bkd = self._bkd
        marginals = [
            BetaMarginal(2.0, 5.0, bkd),
            BetaMarginal(1.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [8, 8], bkd
        )
        rule = factory([0, 1])
        _, weights = rule()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_integration_exact_for_polynomials(self) -> None:
        """E[x^k] matches analytical moments for Uniform[-1,1]."""
        from pyapprox.typing.probability.univariate.uniform import (
            UniformMarginal,
        )

        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [5], bkd
        )
        rule = factory([0])
        samples, weights = rule()

        # E[x] = 0 for Uniform[-1,1]
        ex = bkd.sum(weights * samples[0])
        bkd.assert_allclose(
            bkd.asarray([ex]), bkd.asarray([0.0]), atol=1e-14
        )

        # E[x^2] = 1/3 for Uniform[-1,1]
        ex2 = bkd.sum(weights * samples[0] ** 2)
        bkd.assert_allclose(
            bkd.asarray([ex2]), bkd.asarray([1.0 / 3.0]), rtol=1e-12
        )

        # E[x^4] = 1/5 for Uniform[-1,1]
        ex4 = bkd.sum(weights * samples[0] ** 4)
        bkd.assert_allclose(
            bkd.asarray([ex4]), bkd.asarray([1.0 / 5.0]), rtol=1e-12
        )

    def test_integration_2d_product(self) -> None:
        """E[x*y] = E[x]*E[y] for independent Uniform variables."""
        from pyapprox.typing.probability.univariate.uniform import (
            UniformMarginal,
        )

        bkd = self._bkd
        # Uniform[0,2]: E[x] = 1, Uniform[1,3]: E[y] = 2
        marginals = [
            UniformMarginal(0.0, 2.0, bkd),
            UniformMarginal(1.0, 3.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [5, 5], bkd
        )
        rule = factory([0, 1])
        samples, weights = rule()

        # E[x*y] = 1 * 2 = 2
        exy = bkd.sum(weights * samples[0] * samples[1])
        bkd.assert_allclose(
            bkd.asarray([exy]), bkd.asarray([2.0]), rtol=1e-12
        )

    def test_factory_subset_selection(self) -> None:
        """Correct variables are selected from a 3-variable factory."""
        from pyapprox.typing.probability.univariate.uniform import (
            UniformMarginal,
        )

        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(0.0, 2.0, bkd),
            UniformMarginal(5.0, 10.0, bkd),
        ]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [3, 5, 7], bkd
        )

        # Select only variable 1
        rule = factory([1])
        samples, weights = rule()
        self.assertEqual(samples.shape[0], 1)
        self.assertEqual(samples.shape[1], 5)
        # Points in [0, 2]
        s_np = bkd.to_numpy(samples)
        self.assertTrue(np.all(s_np >= 0.0 - 1e-14))
        self.assertTrue(np.all(s_np <= 2.0 + 1e-14))
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([1.0]), rtol=1e-12
        )

        # Select variables 0 and 2
        rule02 = factory([0, 2])
        samples02, weights02 = rule02()
        self.assertEqual(samples02.shape[0], 2)
        self.assertEqual(samples02.shape[1], 3 * 7)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights02)]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_satisfies_protocol(self) -> None:
        """Factory satisfies QuadratureFactoryProtocol."""
        from pyapprox.typing.probability.univariate.uniform import (
            UniformMarginal,
        )

        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        factory = ProbabilityMeasureQuadratureFactory(
            marginals, [5], bkd
        )
        self.assertIsInstance(factory, QuadratureFactoryProtocol)


class TestProbabilityMeasureQuadratureFactoryNumpy(
    TestProbabilityMeasureQuadratureFactory[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestProbabilityMeasureQuadratureFactoryTorch(
    TestProbabilityMeasureQuadratureFactory[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

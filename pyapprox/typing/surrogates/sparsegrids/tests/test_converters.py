"""Dual-backend tests for sparse grid to PCE converter.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.sparsegrids import (
    SparseGridToPCEConverter,
    TensorProductSubspaceToPCEConverter,
    TensorProductSubspace,
    IsotropicSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
    create_basis_factories,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
)
from pyapprox.typing.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.typing.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
)
from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.surrogates.affine.indices import (
    IsotropicSparseGridBasisIndexGenerator,
    LinearGrowthRule,
    MaxLevelCriteria,
)


class TestSparseGridToPCEConverter(Generic[Array], unittest.TestCase):
    """Tests for SparseGridToPCEConverter - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _build_isotropic(self, nvars, level, growth):
        """Build an isotropic sparse grid fitter and fit a function."""
        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(m, self._bkd) for m in marginals
        ]
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(
            self._bkd, tp_factory, level
        )
        return fitter, marginals

    def test_simple_polynomial(self) -> None:
        """Test conversion for a simple polynomial."""
        nvars = 2
        level = 3
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = self._bkd.asarray([[-0.5, 0.0, 0.5], [0.3, 0.0, -0.3]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_pce_mean_variance(self) -> None:
        """Test PCE statistics are correct."""
        nvars = 2
        level = 4
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        # E[f] = E[x^2] = 1/3
        # Var[f] = 13/15
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Check mean and variance
        pce_mean = pce.mean()
        pce_var = pce.variance()

        exact_mean = 1.0 / 3.0
        exact_var = 13.0 / 15.0

        self._bkd.assert_allclose(
            self._bkd.asarray([float(pce_mean[0])]),
            self._bkd.asarray([exact_mean]),
            rtol=1e-10,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([float(pce_var[0])]),
            self._bkd.asarray([exact_var]),
            rtol=1e-10,
        )

    def test_sobol_indices(self) -> None:
        """Test PCE Sobol indices are correct."""
        nvars = 2
        level = 4
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Check Sobol indices
        total_sobol = pce.total_sobol_indices()
        main_sobol = pce.main_effect_sobol_indices()

        # Exact values
        exact_total_x = 8.0 / 13.0
        exact_total_y = 35.0 / 39.0
        exact_main_x = 4.0 / 39.0
        exact_main_y = 5.0 / 13.0

        self._bkd.assert_allclose(
            self._bkd.asarray([float(main_sobol[0, 0])]),
            self._bkd.asarray([exact_main_x]),
            rtol=1e-6,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([float(main_sobol[1, 0])]),
            self._bkd.asarray([exact_main_y]),
            rtol=1e-6,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([float(total_sobol[0, 0])]),
            self._bkd.asarray([exact_total_x]),
            rtol=1e-6,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([float(total_sobol[1, 0])]),
            self._bkd.asarray([exact_total_y]),
            rtol=1e-6,
        )

    def test_3d_conversion(self) -> None:
        """Test conversion for 3D sparse grid."""
        nvars = 3
        level = 2
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth)
        samples = fitter.get_samples()

        # f(x, y, z) = x + y + z
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
        )
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = self._bkd.asarray([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

        # Mean should be 0 for linear function
        pce_mean = pce.mean()
        self._bkd.assert_allclose(
            self._bkd.asarray([float(pce_mean[0])]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_non_canonical_domain(self) -> None:
        """Test conversion with non-canonical domain [0, 1]."""
        nvars = 2
        level = 3

        marginals = [UniformMarginal(0.0, 1.0, self._bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(m, self._bkd) for m in marginals
        ]
        growth = LinearGrowthRule(scale=2, shift=1)

        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        fitter = IsotropicSparseGridFitter(
            self._bkd, tp_factory, level
        )
        samples = fitter.get_samples()

        # Verify samples are in [0, 1] domain
        lb_check = samples >= self._bkd.asarray(0.0)
        ub_check = samples <= self._bkd.asarray(1.0)
        assert not isinstance(lb_check, bool)  # for mypy
        assert not isinstance(ub_check, bool)  # for mypy
        self.assertTrue(self._bkd.all_bool(lb_check))
        self.assertTrue(self._bkd.all_bool(ub_check))

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches at points in [0, 1] domain
        test_pts = self._bkd.asarray([[0.25, 0.5, 0.75], [0.3, 0.5, 0.7]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        self._bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_pce_indices_match_sparse_grid_index_generator(self) -> None:
        """Test that converted PCE index set matches IsotropicSparseGridBasisIndexGenerator."""
        for nvars, level in [(2, 3), (3, 2)]:
            growth = LinearGrowthRule(scale=2, shift=1)
            marginals = [
                UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)
            ]
            factories: List[BasisFactoryProtocol[Array]] = [
                GaussLagrangeFactory(m, self._bkd) for m in marginals
            ]

            tp_factory = TensorProductSubspaceFactory(
                self._bkd, factories, growth
            )
            fitter = IsotropicSparseGridFitter(
                self._bkd, tp_factory, level
            )
            samples = fitter.get_samples()

            # Use a simple polynomial so all subspace conversions are valid
            values = self._bkd.reshape(
                self._bkd.sum(samples, axis=0), (1, -1)
            )
            result = fitter.fit(values)

            pce_bases_1d = create_bases_1d(marginals, self._bkd)
            converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
            pce = converter.convert(result.surrogate)

            # Get PCE index set as set of tuples
            pce_indices = pce.get_indices()
            pce_set = set()
            for j in range(pce_indices.shape[1]):
                pce_set.add(
                    tuple(
                        int(self._bkd.to_numpy(pce_indices[i, j]))
                        for i in range(nvars)
                    )
                )

            # Get index set from IsotropicSparseGridBasisIndexGenerator
            gen = IsotropicSparseGridBasisIndexGenerator(
                nvars, level, self._bkd, growth_rules=growth,
            )
            gen_indices = gen.get_indices()
            gen_set = set()
            for j in range(gen_indices.shape[1]):
                gen_set.add(
                    tuple(
                        int(self._bkd.to_numpy(gen_indices[i, j]))
                        for i in range(nvars)
                    )
                )

            # The PCE index set should match the generator's set
            self.assertEqual(
                pce_set, gen_set,
                f"nvars={nvars}, level={level}: index sets differ. "
                f"In gen but not PCE: {gen_set - pce_set}. "
                f"In PCE but not gen: {pce_set - gen_set}",
            )

    def test_multi_qoi_conversion(self) -> None:
        """Test conversion with multiple quantities of interest."""
        nvars = 2
        level = 3
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth)
        samples = fitter.get_samples()

        # Two QoIs: f1 = x, f2 = y - shape (nqoi, nsamples) = (2, nsamples)
        values = self._bkd.stack([samples[0, :], samples[1, :]], axis=0)
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        self.assertEqual(pce_vals.shape[0], 2)  # nqoi=2 is first dimension
        self._bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)


class TestTensorProductSubspaceToPCEConverter(Generic[Array], unittest.TestCase):
    """Tests for TensorProductSubspaceToPCEConverter - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_subspace_conversion(self) -> None:
        """Test conversion of single tensor product subspace."""
        nvars = 2
        marginals = [UniformMarginal(-1.0, 1.0, self._bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(m, self._bkd) for m in marginals
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        # Create a subspace
        index = self._bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            self._bkd, index, factories, growth
        )

        # f(x, y) = x^2 + y
        samples = subspace.get_samples()
        values = self._bkd.reshape(samples[0, :] ** 2 + samples[1, :], (1, -1))
        subspace.set_values(values)

        # Convert to PCE coefficients
        pce_bases_1d = create_bases_1d(marginals, self._bkd)
        converter = TensorProductSubspaceToPCEConverter(self._bkd, pce_bases_1d)
        indices, coefficients = converter.convert_subspace(subspace)

        # Verify shapes - coefficients is (nqoi, ncoefs)
        self.assertEqual(indices.shape[0], nvars)
        self.assertEqual(coefficients.shape[0], 1)  # nqoi is first dimension
        self.assertEqual(indices.shape[1], coefficients.shape[1])


class TestAdaptiveSGToPCEConverter(Generic[Array], unittest.TestCase):
    """Tests for converting adaptive sparse grids to PCE - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_adaptive_sg_to_pce_evaluation(self) -> None:
        """Test adaptive SG -> PCE conversion preserves evaluation."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        # Convert to PCE
        pce_bases_1d = create_bases_1d(joint.marginals(), self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        # Verify evaluation matches
        np.random.seed(123)
        test_pts = joint.rvs(20)
        self._bkd.assert_allclose(
            pce(test_pts), ada_result.surrogate(test_pts), rtol=1e-10
        )

    def test_adaptive_sg_to_pce_mean(self) -> None:
        """Test adaptive SG -> PCE conversion preserves mean."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        pce_bases_1d = create_bases_1d(joint.marginals(), self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        self._bkd.assert_allclose(
            pce.mean(), ada_result.surrogate.mean(), rtol=1e-10
        )

    def test_adaptive_sg_to_pce_variance(self) -> None:
        """Test adaptive SG -> PCE conversion preserves variance."""
        joint = create_test_joint("2d_uniform", self._bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=self._bkd)

        factories = create_basis_factories(
            joint.marginals(), self._bkd, "gauss"
        )
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(
            self._bkd, factories, growth
        )
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        pce_bases_1d = create_bases_1d(joint.marginals(), self._bkd)
        converter = SparseGridToPCEConverter(self._bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        self._bkd.assert_allclose(
            pce.variance(), ada_result.surrogate.variance(), rtol=1e-10
        )


# NumPy backend tests
class TestSparseGridToPCEConverterNumpy(
    TestSparseGridToPCEConverter[NDArray[Any]]
):
    """NumPy backend tests for SparseGridToPCEConverter."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductSubspaceToPCEConverterNumpy(
    TestTensorProductSubspaceToPCEConverter[NDArray[Any]]
):
    """NumPy backend tests for TensorProductSubspaceToPCEConverter."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveSGToPCEConverterNumpy(
    TestAdaptiveSGToPCEConverter[NDArray[Any]]
):
    """NumPy backend tests for adaptive SG -> PCE converter."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestSparseGridToPCEConverterTorch(
    TestSparseGridToPCEConverter[torch.Tensor]
):
    """PyTorch backend tests for SparseGridToPCEConverter."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestTensorProductSubspaceToPCEConverterTorch(
    TestTensorProductSubspaceToPCEConverter[torch.Tensor]
):
    """PyTorch backend tests for TensorProductSubspaceToPCEConverter."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestAdaptiveSGToPCEConverterTorch(
    TestAdaptiveSGToPCEConverter[torch.Tensor]
):
    """PyTorch backend tests for adaptive SG -> PCE converter."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()

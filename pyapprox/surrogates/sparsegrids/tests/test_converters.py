"""Dual-backend tests for sparse grid to PCE converter.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

from typing import List

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    IsotropicSparseGridBasisIndexGenerator,
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.sparsegrids import (
    IsotropicSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
    SparseGridToPCEConverter,
    TensorProductSubspace,
    TensorProductSubspaceToPCEConverter,
    create_basis_factories,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.surrogates.sparsegrids.tests.test_helpers import (
    create_test_joint,
    create_test_pce,
)
from pyapprox.util.test_utils import slow_test


class TestSparseGridToPCEConverter:
    """Tests for SparseGridToPCEConverter - dual backend."""

    def _build_isotropic(self, nvars, level, growth, bkd):
        """Build an isotropic sparse grid fitter and fit a function."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol] = [
            GaussLagrangeFactory(m, bkd) for m in marginals
        ]
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
        return fitter, marginals

    def test_simple_polynomial(self, bkd) -> None:
        """Test conversion for a simple polynomial."""
        nvars = 2
        level = 3
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth, bkd)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = bkd.asarray([[-0.5, 0.0, 0.5], [0.3, 0.0, -0.3]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_pce_mean_variance(self, bkd) -> None:
        """Test PCE statistics are correct."""
        nvars = 2
        level = 4
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth, bkd)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        # E[f] = E[x^2] = 1/3
        # Var[f] = 13/15
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Check mean and variance
        pce_mean = pce.mean()
        pce_var = pce.variance()

        exact_mean = 1.0 / 3.0
        exact_var = 13.0 / 15.0

        bkd.assert_allclose(
            bkd.asarray([float(pce_mean[0])]),
            bkd.asarray([exact_mean]),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            bkd.asarray([float(pce_var[0])]),
            bkd.asarray([exact_var]),
            rtol=1e-10,
        )

    def test_sobol_indices(self, bkd) -> None:
        """Test PCE Sobol indices are correct."""
        nvars = 2
        level = 4
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth, bkd)
        samples = fitter.get_samples()

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Check Sobol indices
        total_sobol = pce.total_sobol_indices()
        main_sobol = pce.main_effect_sobol_indices()

        # Exact values
        exact_total_x = 8.0 / 13.0
        exact_total_y = 35.0 / 39.0
        exact_main_x = 4.0 / 39.0
        exact_main_y = 5.0 / 13.0

        bkd.assert_allclose(
            bkd.asarray([float(main_sobol[0, 0])]),
            bkd.asarray([exact_main_x]),
            rtol=1e-6,
        )
        bkd.assert_allclose(
            bkd.asarray([float(main_sobol[1, 0])]),
            bkd.asarray([exact_main_y]),
            rtol=1e-6,
        )
        bkd.assert_allclose(
            bkd.asarray([float(total_sobol[0, 0])]),
            bkd.asarray([exact_total_x]),
            rtol=1e-6,
        )
        bkd.assert_allclose(
            bkd.asarray([float(total_sobol[1, 0])]),
            bkd.asarray([exact_total_y]),
            rtol=1e-6,
        )

    @pytest.mark.slow_on("TorchBkd")
    def test_3d_conversion(self, bkd) -> None:
        """Test conversion for 3D sparse grid."""
        nvars = 3
        level = 2
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth, bkd)
        samples = fitter.get_samples()

        # f(x, y, z) = x + y + z
        values = bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
        )
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = bkd.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

        # Mean should be 0 for linear function
        pce_mean = pce.mean()
        bkd.assert_allclose(
            bkd.asarray([float(pce_mean[0])]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_non_canonical_domain(self, bkd) -> None:
        """Test conversion with non-canonical domain [0, 1]."""
        nvars = 2
        level = 3

        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol] = [
            GaussLagrangeFactory(m, bkd) for m in marginals
        ]
        growth = LinearGrowthRule(scale=2, shift=1)

        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
        samples = fitter.get_samples()

        # Verify samples are in [0, 1] domain
        lb_check = samples >= bkd.asarray(0.0)
        ub_check = samples <= bkd.asarray(1.0)
        assert not isinstance(lb_check, bool)  # for mypy
        assert not isinstance(ub_check, bool)  # for mypy
        assert bkd.all_bool(lb_check)
        assert bkd.all_bool(ub_check)

        # f(x, y) = x^2 + 2*x*y + y
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2 * x * y + y, (1, -1))
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches at points in [0, 1] domain
        test_pts = bkd.asarray([[0.25, 0.5, 0.75], [0.3, 0.5, 0.7]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)

    def test_pce_indices_match_sparse_grid_index_generator(self, bkd) -> None:
        """Test that converted PCE index set matches
        IsotropicSparseGridBasisIndexGenerator."""
        for nvars, level in [(2, 3), (3, 2)]:
            growth = LinearGrowthRule(scale=2, shift=1)
            marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
            factories: List[BasisFactoryProtocol] = [
                GaussLagrangeFactory(m, bkd) for m in marginals
            ]

            tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
            fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
            samples = fitter.get_samples()

            # Use a simple polynomial so all subspace conversions are valid
            values = bkd.reshape(bkd.sum(samples, axis=0), (1, -1))
            result = fitter.fit(values)

            pce_bases_1d = create_bases_1d(marginals, bkd)
            converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
            pce = converter.convert(result.surrogate)

            # Get PCE index set as set of tuples
            pce_indices = pce.get_indices()
            pce_set = set()
            for j in range(pce_indices.shape[1]):
                pce_set.add(
                    tuple(
                        int(bkd.to_numpy(pce_indices[i, j])) for i in range(nvars)
                    )
                )

            # Get index set from IsotropicSparseGridBasisIndexGenerator
            gen = IsotropicSparseGridBasisIndexGenerator(
                nvars,
                level,
                bkd,
                growth_rules=growth,
            )
            gen_indices = gen.get_indices()
            gen_set = set()
            for j in range(gen_indices.shape[1]):
                gen_set.add(
                    tuple(
                        int(bkd.to_numpy(gen_indices[i, j])) for i in range(nvars)
                    )
                )

            # The PCE index set should match the generator's set
            assert pce_set == gen_set, (
                f"nvars={nvars}, level={level}: index sets differ. "
                f"In gen but not PCE: {gen_set - pce_set}. "
                f"In PCE but not gen: {pce_set - gen_set}"
            )

    def test_multi_qoi_conversion(self, bkd) -> None:
        """Test conversion with multiple quantities of interest."""
        nvars = 2
        level = 3
        growth = LinearGrowthRule(scale=2, shift=1)

        fitter, marginals = self._build_isotropic(nvars, level, growth, bkd)
        samples = fitter.get_samples()

        # Two QoIs: f1 = x, f2 = y - shape (nqoi, nsamples) = (2, nsamples)
        values = bkd.stack([samples[0, :], samples[1, :]], axis=0)
        result = fitter.fit(values)

        # Convert to PCE
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(result.surrogate)

        # Test evaluation matches
        test_pts = bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        sg_vals = result.surrogate(test_pts)
        pce_vals = pce(test_pts)

        assert pce_vals.shape[0] == 2  # nqoi=2 is first dimension
        bkd.assert_allclose(sg_vals, pce_vals, rtol=1e-10, atol=1e-14)


class TestTensorProductSubspaceToPCEConverter:
    """Tests for TensorProductSubspaceToPCEConverter - dual backend."""

    def test_subspace_conversion(self, bkd) -> None:
        """Test conversion of single tensor product subspace."""
        nvars = 2
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        factories: List[BasisFactoryProtocol] = [
            GaussLagrangeFactory(m, bkd) for m in marginals
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        # Create a subspace
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, factories, growth)

        # f(x, y) = x^2 + y
        samples = subspace.get_samples()
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :], (1, -1))
        subspace.set_values(values)

        # Convert to PCE coefficients
        pce_bases_1d = create_bases_1d(marginals, bkd)
        converter = TensorProductSubspaceToPCEConverter(bkd, pce_bases_1d)
        indices, coefficients = converter.convert_subspace(subspace)

        # Verify shapes - coefficients is (nqoi, ncoefs)
        assert indices.shape[0] == nvars
        assert coefficients.shape[0] == 1  # nqoi is first dimension
        assert indices.shape[1] == coefficients.shape[1]


class TestAdaptiveSGToPCEConverter:
    """Tests for converting adaptive sparse grids to PCE - dual backend."""

    def test_adaptive_sg_to_pce_evaluation(self, bkd) -> None:
        """Test adaptive SG -> PCE conversion preserves evaluation."""
        joint = create_test_joint("2d_uniform", bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        # Convert to PCE
        pce_bases_1d = create_bases_1d(joint.marginals(), bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        # Verify evaluation matches
        np.random.seed(123)
        test_pts = joint.rvs(20)
        bkd.assert_allclose(
            pce(test_pts), ada_result.surrogate(test_pts), rtol=1e-10
        )

    def test_adaptive_sg_to_pce_mean(self, bkd) -> None:
        """Test adaptive SG -> PCE conversion preserves mean."""
        joint = create_test_joint("2d_uniform", bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        pce_bases_1d = create_bases_1d(joint.marginals(), bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        bkd.assert_allclose(pce.mean(), ada_result.surrogate.mean(), rtol=1e-10)

    def test_adaptive_sg_to_pce_variance(self, bkd) -> None:
        """Test adaptive SG -> PCE conversion preserves variance."""
        joint = create_test_joint("2d_uniform", bkd)
        pce_target = create_test_pce(joint, level=3, nqoi=1, bkd=bkd)

        factories = create_basis_factories(joint.marginals(), bkd, "gauss")
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        ada_result = fitter.refine_to_tolerance(
            lambda s: pce_target(s), tol=1e-12, max_steps=50
        )

        pce_bases_1d = create_bases_1d(joint.marginals(), bkd)
        converter = SparseGridToPCEConverter(bkd, pce_bases_1d)
        pce = converter.convert(ada_result.surrogate)

        bkd.assert_allclose(
            pce.variance(), ada_result.surrogate.variance(), rtol=1e-10
        )

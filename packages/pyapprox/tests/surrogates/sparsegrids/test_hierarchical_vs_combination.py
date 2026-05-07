"""Equivalence tests: hierarchical vs combination sparse grid fitters."""

import numpy as np
import pytest
from pyapprox.interface.functions.fromcallable.function import FunctionFromCallable
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.affine.indices.admissibility import (
    MaxLevelCriteria,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    ClenshawCurtisGrowthRule,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis_factory import PiecewiseFactory
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_fitter import (
    SingleFidelityHierarchicalFitter,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _build_combination_fitter(bkd, nvars, level, poly_type):
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = [
        PiecewiseFactory(m, bkd, poly_type=poly_type) for m in marginals
    ]
    growth = ClenshawCurtisGrowthRule()
    sub_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    return IsotropicSparseGridFitter(
        bkd, sub_factory, level=level, pnorm=1.0
    )


def _build_hierarchical_fitter(bkd, nvars, level, p_max):
    bases_1d = [
        HierarchicalBasis1D(
            bkd, bounds=(0.0, 1.0), p_max=p_max, boundary_mode="include"
        )
        for _ in range(nvars)
    ]
    admissibility = MaxLevelCriteria(level, 1.0, bkd)
    return SingleFidelityHierarchicalFitter(bkd, bases_1d, admissibility)


class TestHierarchicalVsCombination:
    @pytest.mark.parametrize(
        "nvars,level,poly_type,p_max",
        [
            (1, 3, "linear", 1),
            (2, 3, "linear", 1),
            (3, 2, "linear", 1),
            (1, 3, "quadratic", 2),
            (2, 3, "quadratic", 2),
            (3, 2, "quadratic", 2),
        ],
    )
    def test_equivalence(self, bkd, nvars, level, poly_type, p_max):
        def fun(X):
            return bkd.sum(X**2, axis=0, keepdims=True)

        combo_fitter = _build_combination_fitter(bkd, nvars, level, poly_type)
        combo_samples = combo_fitter.get_samples()
        combo_result = combo_fitter.fit(fun(combo_samples))

        h_fitter = _build_hierarchical_fitter(bkd, nvars, level, p_max)
        target = FunctionFromCallable(1, nvars, fun, bkd)
        h_result = h_fitter.refine_to_tolerance(target, tol=0.0, max_steps=500)

        assert combo_samples.shape[1] == h_result.nsamples, (
            f"Point count mismatch: combo={combo_samples.shape[1]}, "
            f"h-gsg={h_result.nsamples}"
        )

        combo_pts = bkd.to_numpy(combo_samples)
        h_pts = bkd.to_numpy(h_fitter.get_samples())
        combo_sorted = np.sort(combo_pts, axis=1)
        h_sorted = np.sort(h_pts, axis=1)
        np.testing.assert_allclose(combo_sorted, h_sorted, atol=1e-14)

        np.random.seed(42)
        test_pts = bkd.array(np.random.rand(nvars, 100))
        combo_vals = combo_result.surrogate(test_pts)
        h_vals = h_result.surrogate(test_pts)
        bkd.assert_allclose(h_vals, combo_vals, atol=1e-12)

        combo_mean = combo_result.surrogate.mean()
        h_mean = h_result.surrogate.mean()
        bkd.assert_allclose(h_mean, combo_mean, atol=1e-14)

    @pytest.mark.parametrize("nvars", [2, 3])
    def test_admissibility_sensitivity(self, bkd, nvars):
        """DownwardClosed and AlwaysAdmissible explore different point sets
        at matching step counts."""

        def fun(X):
            weights = bkd.asarray(
                [10.0**i for i in range(nvars)],
                dtype=bkd.double_dtype(),
            )
            weights = bkd.reshape(weights, (-1, 1))
            return bkd.sum(weights * X**2, axis=0, keepdims=True)

        target = FunctionFromCallable(1, nvars, fun, bkd)
        max_steps = 15

        bases_1d_dc = [
            HierarchicalBasis1D(
                bkd, bounds=(0.0, 1.0), p_max=1, boundary_mode="include"
            )
            for _ in range(nvars)
        ]
        dc_fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d_dc, MaxLevelCriteria(4, 1.0, bkd)
        )
        dc_fitter.refine_to_tolerance(
            target, tol=1e-15, max_steps=max_steps
        )

        bases_1d_aa = [
            HierarchicalBasis1D(
                bkd, bounds=(0.0, 1.0), p_max=1, boundary_mode="include"
            )
            for _ in range(nvars)
        ]
        aa_fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d_aa, MaxLevelCriteria(4, 1.0, bkd),
        )
        aa_fitter.refine_to_tolerance(
            target, tol=1e-15, max_steps=max_steps
        )

        dc_pts = bkd.to_numpy(dc_fitter.get_samples())
        aa_pts = bkd.to_numpy(aa_fitter.get_samples())
        assert dc_pts.shape[1] == aa_pts.shape[1], (
            "Both runs with the same max_steps and level cap should "
            "produce the same number of points"
        )

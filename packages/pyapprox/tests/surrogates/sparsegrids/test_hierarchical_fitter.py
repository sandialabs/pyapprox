"""Tests for MultiFidelityHierarchicalFitter and SingleFidelityHierarchicalFitter."""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.surrogates.affine.indices.admissibility import (
    AlwaysAdmissible,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_fitter import (
    MultiFidelityHierarchicalFitter,
    SingleFidelityHierarchicalFitter,
)
from pyapprox.surrogates.sparsegrids.model_factory import DictModelFactory


class TestSingleFidelityHierarchicalFitter:
    def test_zero_function(self, bkd):
        """All surpluses zero except root for f=0."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)

        def f_zero(x):
            return bkd.zeros((1, x.shape[1]), dtype=bkd.double_dtype())

        result = fitter.refine_to_tolerance(f_zero, tol=1e-15, max_steps=50)
        surrogate = result.surrogate
        x_test = bkd.asarray(
            np.linspace(0, 1, 11).reshape(1, -1), dtype=bkd.double_dtype()
        )
        vals = surrogate(x_test)
        bkd.assert_allclose(
            vals, bkd.zeros((1, 11), dtype=bkd.double_dtype()), atol=1e-14
        )

    def test_1d_linear_exact(self, bkd):
        """f(x) = 3x + 1 is reproduced exactly."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)

        def f_linear(x):
            return 3 * x + 1

        result = fitter.refine_to_tolerance(f_linear, tol=1e-15, max_steps=50)
        x_test = bkd.asarray(
            np.linspace(0, 1, 51).reshape(1, -1), dtype=bkd.double_dtype()
        )
        vals = result.surrogate(x_test)
        expected = 3 * x_test + 1
        bkd.assert_allclose(vals, expected, atol=1e-13)

    def test_step_by_step(self, bkd):
        """Manual step_samples/step_values loop."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)

        def f(x):
            return bkd.sin(x * 6.28)

        steps = 0
        while True:
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(f(samples))
            steps += 1
            if steps > 100:
                raise RuntimeError("Too many steps")

        result = fitter.result(converged=True)
        assert result.nsamples > 0
        assert result.nsteps > 0

    def test_1d_adaptive_cubic(self, bkd):
        """Piecewise function: linear left of 0.75, cubic right.

        The linear part is captured exactly (zero surpluses), so all
        refinement concentrates on the cubic region. Non-uniform
        curvature (f''=18(x-0.75)) gives varying surpluses, so
        refinement interleaves across levels.
        """
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1
        )

        def f(x):
            x_np = bkd.to_numpy(x)
            vals = np.where(
                x_np < 0.75,
                2.0 * x_np,
                1.5 + 3.0 * (x_np - 0.75) ** 3,
            )
            return bkd.asarray(vals, dtype=bkd.double_dtype())

        x_test = bkd.asarray(
            np.linspace(0, 1, 1001).reshape(1, -1), dtype=bkd.double_dtype()
        )
        exact = bkd.to_numpy(f(x_test))
        dx = 1.0 / 1000

        prev_l2 = float("inf")
        for step in range(15):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(f(samples))
            vals = bkd.to_numpy(fitter.result(converged=False).surrogate(x_test))
            l2 = np.sqrt(np.sum((vals - exact) ** 2) * dx)
            assert l2 <= prev_l2 + 1e-14
            prev_l2 = l2

        assert prev_l2 < 1e-4

    def test_mean_of_linear(self, bkd):
        """Mean of f(x)=x over [0,1] should be 0.5."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)

        def f(x):
            return x

        result = fitter.refine_to_tolerance(f, tol=1e-15, max_steps=50)
        mean = result.surrogate.mean()
        bkd.assert_allclose(mean, bkd.asarray([0.5]), atol=1e-14)

    def test_always_admissible_no_deferred(self, bkd):
        """With AlwaysAdmissible, deferred registry should stay empty."""
        bases_1d = [
            HierarchicalBasis1D(bkd, boundary_mode="include"),
            HierarchicalBasis1D(bkd, boundary_mode="include"),
        ]
        admis = AlwaysAdmissible(bkd)
        fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1
        )

        def f(x):
            return x[0:1, :] + x[1:2, :]

        # Run a few steps
        for _ in range(10):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(f(samples))

        assert fitter._fitter._deferred.empty()

    def test_batch_size(self, bkd):
        """batch_size > 1 should still produce a correct result."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)

        def f(x):
            return bkd.sin(x * 5.0 + 1.0)

        # batch_size=1
        fitter1 = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1
        )
        r1 = fitter1.refine_to_tolerance(f, tol=1e-15, max_steps=100)

        # batch_size=3
        fitter3 = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=3
        )
        r3 = fitter3.refine_to_tolerance(f, tol=1e-15, max_steps=100)

        # Both should interpolate correctly
        x_test = bkd.asarray(
            np.linspace(0, 1, 21).reshape(1, -1), dtype=bkd.double_dtype()
        )
        v1 = r1.surrogate(x_test)
        v3 = r3.surrogate(x_test)
        exact = bkd.sin(x_test * 5.0 + 1.0)
        bkd.assert_allclose(v1, exact, atol=0.01)
        bkd.assert_allclose(v3, exact, atol=0.01)


def _run_2d_adaptive(
    bkd,
    admissibility,
    max_pts: int,
) -> Tuple[object, object, object]:
    """Run 2D adaptive sparse grid and return (fitter, basis_nd, mf)."""
    bases_1d = [
        HierarchicalBasis1D(bkd, boundary_mode="include"),
        HierarchicalBasis1D(bkd, boundary_mode="include"),
    ]
    basis_nd = HierarchicalBasisND(bkd, bases_1d)
    fitter = SingleFidelityHierarchicalFitter(
        bkd, bases_1d, admissibility, batch_size=1,
    )

    def f(x):
        x_np = bkd.to_numpy(x)
        v = (
            10.0 * x_np[0] * x_np[1]
            + 30.0
            * np.maximum(x_np[0] - 0.3, 0) ** 2
            * np.maximum(x_np[1] - 0.3, 0) ** 2
        )
        return bkd.asarray(v.reshape(1, -1), dtype=bkd.double_dtype())

    total_pts = 0
    for _ in range(500):
        samples = fitter.step_samples()
        if samples is None:
            break
        total_pts += samples.shape[1]
        fitter.step_values(f(samples))
        if total_pts >= max_pts:
            break

    return fitter, basis_nd, fitter._fitter


def _check_interpolation_property(bkd, fitter, basis_nd):
    """Assert surrogate interpolates exactly at all grid nodes."""
    mf = fitter._fitter
    surr = fitter.result(converged=False).surrogate

    def f(x):
        x_np = bkd.to_numpy(x)
        v = (
            10.0 * x_np[0] * x_np[1]
            + 30.0
            * np.maximum(x_np[0] - 0.3, 0) ** 2
            * np.maximum(x_np[1] - 0.3, 0) ** 2
        )
        return bkd.asarray(v.reshape(1, -1), dtype=bkd.double_dtype())

    for pid in range(mf._point_mgr.n_points()):
        if not mf._point_mgr.is_evaluated(pid):
            continue
        key = mf._point_mgr.get_key(pid)
        node = basis_nd.node(*key)
        fval = f(node)
        sval = surr(node)
        bkd.assert_allclose(sval, fval, atol=1e-12)


class TestHierarchicalFitterConvergence:
    """Convergence and interpolation tests on the 2D test function

        f(x,y) = 10*x*y + 30*max(x-0.3,0)^2 * max(y-0.3,0)^2

    using both DownwardClosed and AlwaysAdmissible admissibility.
    """

    @pytest.mark.parametrize("mode", ["DownwardClosed", "AlwaysAdmissible"])
    def test_interpolation_property(self, bkd, mode):
        """Surrogate must interpolate f exactly at every grid node."""
        if mode == "DownwardClosed":
            admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        else:
            admis = AlwaysAdmissible(bkd)

        fitter, basis_nd, _ = _run_2d_adaptive(bkd, admis, max_pts=100)
        _check_interpolation_property(bkd, fitter, basis_nd)

    def test_downward_closed_convergence(self, bkd):
        admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        fitter, basis_nd, _ = _run_2d_adaptive(bkd, admis, max_pts=200)

        surr = fitter.result(converged=False).surrogate
        xx, yy = np.meshgrid(
            np.linspace(0, 1, 201), np.linspace(0, 1, 201)
        )
        test_pts = bkd.asarray(
            np.vstack([xx.ravel(), yy.ravel()]),
            dtype=bkd.double_dtype(),
        )
        x_np = bkd.to_numpy(test_pts)
        exact = bkd.asarray(
            (
                10.0 * x_np[0] * x_np[1]
                + 30.0
                * np.maximum(x_np[0] - 0.3, 0) ** 2
                * np.maximum(x_np[1] - 0.3, 0) ** 2
            ).reshape(1, -1),
            dtype=bkd.double_dtype(),
        )
        err = bkd.to_numpy(surr(test_pts) - exact)
        l2 = float(np.sqrt(np.mean(err**2)))
        linf = float(np.max(np.abs(err)))
        assert l2 == pytest.approx(0.004855, abs=1e-3)
        assert linf == pytest.approx(0.023934, abs=5e-3)

        _check_interpolation_property(bkd, fitter, basis_nd)

    def test_always_admissible_convergence(self, bkd):
        admis = AlwaysAdmissible(bkd)
        fitter, basis_nd, _ = _run_2d_adaptive(bkd, admis, max_pts=200)

        surr = fitter.result(converged=False).surrogate
        xx, yy = np.meshgrid(
            np.linspace(0, 1, 201), np.linspace(0, 1, 201)
        )
        test_pts = bkd.asarray(
            np.vstack([xx.ravel(), yy.ravel()]),
            dtype=bkd.double_dtype(),
        )
        x_np = bkd.to_numpy(test_pts)
        exact = bkd.asarray(
            (
                10.0 * x_np[0] * x_np[1]
                + 30.0
                * np.maximum(x_np[0] - 0.3, 0) ** 2
                * np.maximum(x_np[1] - 0.3, 0) ** 2
            ).reshape(1, -1),
            dtype=bkd.double_dtype(),
        )
        err = bkd.to_numpy(surr(test_pts) - exact)
        l2 = float(np.sqrt(np.mean(err**2)))
        linf = float(np.max(np.abs(err)))
        assert l2 == pytest.approx(0.001224, abs=1e-4)
        assert linf == pytest.approx(0.006647, abs=1e-3)

        _check_interpolation_property(bkd, fitter, basis_nd)


    def test_downward_closed_3d_anisotropy(self, bkd):
        """DC refinement respects anisotropy: linear dim 3 stays at level ≤ 2.

        f(x,y,z) = g(x,y) + 5z.  The hierarchical basis reproduces
        linears exactly at level 1, so level-2 surpluses in dim 3 are
        zero.  Those zero-surplus points are never popped, so no
        subspace with l3 >= 3 should be selected.
        """

        def f_3d(x):
            x_np = bkd.to_numpy(x)
            v = (
                10.0 * x_np[0] * x_np[1]
                + 30.0
                * np.maximum(x_np[0] - 0.3, 0) ** 2
                * np.maximum(x_np[1] - 0.3, 0) ** 2
                + 5.0 * x_np[2]
            )
            return bkd.asarray(v.reshape(1, -1), dtype=bkd.double_dtype())

        bases_1d = [
            HierarchicalBasis1D(bkd, boundary_mode="include"),
            HierarchicalBasis1D(bkd, boundary_mode="include"),
            HierarchicalBasis1D(bkd, boundary_mode="include"),
        ]
        basis_nd = HierarchicalBasisND(bkd, bases_1d)
        admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1,
        )
        mf = fitter._fitter

        total_pts = 0
        for _ in range(500):
            samples = fitter.step_samples()
            if samples is None:
                break
            total_pts += samples.shape[1]
            fitter.step_values(f_3d(samples))
            if total_pts >= 200:
                break

        for pid in range(mf._point_mgr.n_points()):
            if not mf._point_mgr.is_evaluated(pid):
                continue
            key = mf._point_mgr.get_key(pid)
            assert key[0][2] <= 2, (
                f"Point in subspace {key[0]} has l3={key[0][2]} >= 3; "
                "deferred promotion should prevent this for a linear dim"
            )
            if key[0][2] >= 2:
                bkd.assert_allclose(
                    mf._point_mgr.get_surplus(pid),
                    bkd.zeros_like(mf._point_mgr.get_surplus(pid)),
                    atol=1e-13,
                )

    def test_4d_constant_dims_no_wasted_points(self, bkd):
        """Deferred promotion avoids allocating points in constant dims.

        f(x) = g(x0,x1) with 4 inputs.  Dims 2,3 are constant so
        level-1 surpluses are zero.  With deferred promotion, those
        subspaces never promote and no points beyond level 0 appear
        in dims 2,3.
        """

        def f_const_4d(x):
            x_np = bkd.to_numpy(x)
            v = (
                10.0 * x_np[0] * x_np[1]
                + 30.0
                * np.maximum(x_np[0] - 0.3, 0) ** 2
                * np.maximum(x_np[1] - 0.3, 0) ** 2
            )
            return bkd.asarray(v.reshape(1, -1), dtype=bkd.double_dtype())

        bases_1d = [
            HierarchicalBasis1D(bkd, boundary_mode="include")
            for _ in range(4)
        ]
        admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1,
        )
        mf = fitter._fitter

        np.random.seed(42)
        test_pts = bkd.asarray(
            np.random.rand(4, 10000), dtype=bkd.double_dtype()
        )
        exact = f_const_4d(test_pts)

        total_pts = 0
        for _ in range(2000):
            samples = fitter.step_samples()
            if samples is None:
                break
            total_pts += samples.shape[1]
            fitter.step_values(f_const_4d(samples))
            if total_pts >= 200:
                break

        n_l3_ge1 = 0
        for pid in range(mf._point_mgr.n_points()):
            if not mf._point_mgr.is_evaluated(pid):
                continue
            key = mf._point_mgr.get_key(pid)
            assert key[0][2] <= 1 and key[0][3] <= 1, (
                f"Point in subspace {key[0]} has l3={key[0][2]} or "
                f"l4={key[0][3]} >= 2; deferred promotion should "
                "prevent this for constant dims"
            )
            if key[0][2] >= 1 or key[0][3] >= 1:
                n_l3_ge1 += 1
        assert n_l3_ge1 <= 4, (
            f"Expected at most 4 points in constant dims (level-1 "
            f"boundary), got {n_l3_ge1}"
        )

        surr = fitter.result(converged=False).surrogate
        err = bkd.to_numpy(surr(test_pts) - exact)
        l2 = float(np.sqrt(np.mean(err**2)))
        assert l2 < 0.01

    def test_4d_weak_quadratic_fewer_points_in_weak_dims(self, bkd):
        """Deferred promotion allocates fewer points to weak dimensions.

        f(x) = g(x0,x1) + 0.5*x2^2 + 0.3*x3^2.  Dims 2,3 have small
        nonzero surpluses, so deferred promotion eventually explores
        them but allocates far fewer points than eager promotion.
        """

        def f_quad_4d(x):
            x_np = bkd.to_numpy(x)
            v = (
                10.0 * x_np[0] * x_np[1]
                + 30.0
                * np.maximum(x_np[0] - 0.3, 0) ** 2
                * np.maximum(x_np[1] - 0.3, 0) ** 2
                + 0.5 * x_np[2] ** 2
                + 0.3 * x_np[3] ** 2
            )
            return bkd.asarray(v.reshape(1, -1), dtype=bkd.double_dtype())

        bases_1d = [
            HierarchicalBasis1D(bkd, boundary_mode="include")
            for _ in range(4)
        ]
        admis = MaxLevelCriteria(max_level=10, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(
            bkd, bases_1d, admis, batch_size=1,
        )
        mf = fitter._fitter

        np.random.seed(42)
        test_pts = bkd.asarray(
            np.random.rand(4, 10000), dtype=bkd.double_dtype()
        )
        exact = f_quad_4d(test_pts)

        total_pts = 0
        for _ in range(2000):
            samples = fitter.step_samples()
            if samples is None:
                break
            total_pts += samples.shape[1]
            fitter.step_values(f_quad_4d(samples))
            if total_pts >= 300:
                break

        n_weak_dim_pts = 0
        for pid in range(mf._point_mgr.n_points()):
            if not mf._point_mgr.is_evaluated(pid):
                continue
            key = mf._point_mgr.get_key(pid)
            if key[0][2] >= 1 or key[0][3] >= 1:
                n_weak_dim_pts += 1

        assert n_weak_dim_pts < 120, (
            f"Deferred promotion should allocate < 120 points to weak "
            f"dims, got {n_weak_dim_pts}"
        )

        surr = fitter.result(converged=False).surrogate
        err = bkd.to_numpy(surr(test_pts) - exact)
        l2 = float(np.sqrt(np.mean(err**2)))
        assert l2 < 0.005


class TestMultiFidelityHierarchicalFitter:
    def test_nconfig_vars_1(self, bkd):
        """Multi-fidelity with 1 config variable."""
        bases_1d = [HierarchicalBasis1D(bkd, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=bkd)

        fitter = MultiFidelityHierarchicalFitter(
            bkd, bases_1d, admis, nconfig_vars=1
        )

        def model_0(x):
            return x

        def model_1(x):
            return x ** 2

        factory = DictModelFactory({(0,): model_0, (1,): model_1})
        result = fitter.refine_to_tolerance(factory, tol=1e-15, max_steps=50)
        assert result.nsamples > 0

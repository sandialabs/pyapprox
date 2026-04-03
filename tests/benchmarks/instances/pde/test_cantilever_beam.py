"""Integration tests for the cantilever beam benchmark instances."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.benchmarks.instances.pde.cantilever_beam import (
    MESH_PATHS,
    CompositeBeam1DForwardModel,
    cantilever_beam_1d,
    cantilever_beam_1d_spde,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_linear_spde,
    cantilever_beam_2d_neohookean,
    cantilever_beam_2d_neohookean_spde,
)
from pyapprox.benchmarks.protocols import BenchmarkWithPriorProtocol
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.markers import slower_test  # noqa: F401

_TEST_MESH = MESH_PATHS[2]  # h=2 for fast tests


# =========================================================================
# 1D Euler-Bernoulli beam tests
# =========================================================================


class TestCantileverBeam1D:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm = cantilever_beam_1d(cls._bkd, nx=20, num_kle_terms=2)
        cls._bm3 = cantilever_beam_1d(cls._bkd, num_kle_terms=3)
        cls._bm1 = cantilever_beam_1d(cls._bkd, nx=20, num_kle_terms=1)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm.function()
        assert fwd.nvars() == 2
        assert fwd.nqoi() == 3
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (3, 1)
        # QoI 0: tip deflection should be positive
        assert float(result[0, 0]) > 0.0
        # QoI 1: integrated stress should be positive
        assert float(result[1, 0]) > 0.0
        # QoI 2: max curvature should be positive
        assert float(result[2, 0]) > 0.0

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm.function()
        r1 = fwd(bkd.zeros((2, 1)))
        r2 = fwd(bkd.array([[1.0], [0.0]]))
        assert not np.allclose(
            bkd.to_numpy(r1), bkd.to_numpy(r2)
        ), "Different KLE params should give different results"

    def test_batch_evaluation(self):
        bkd = self._bkd
        fwd = self._bm.function()
        samples = bkd.array([[0.0, 0.5, -0.5], [0.0, 0.3, -0.3]])
        result = fwd(samples)
        assert result.shape == (3, 3)

    def test_prior_and_domain(self):
        bm = self._bm3
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        assert samples.shape == (3, 5)
        bounds = bm.domain().bounds()
        assert bounds.shape == (3, 2)

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_1d", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (3, 1)

    def test_one_kle_term_recovers_constant(self):
        """With 1 KLE term at params=0, result is deterministic baseline."""
        bkd = self._bkd
        fwd = self._bm1.function()
        r1 = fwd(bkd.zeros((1, 1)))
        r2 = fwd(bkd.zeros((1, 1)))
        bkd.assert_allclose(r1, r2)


# =========================================================================
# 2D linear elastic beam tests
# =========================================================================


class TestCantileverBeam2DLinear:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_linear(
            cls._bkd,
            num_kle_terms=1,
            mesh_path=_TEST_MESH,
        )
        cls._bm3 = cantilever_beam_2d_linear(
            cls._bkd,
            num_kle_terms=3,
            mesh_path=_TEST_MESH,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        assert fwd.nqoi() == 2
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)
        # QoI 0: Tip deflection should be negative (downward)
        assert float(result[0, 0]) < 0.0
        # QoI 1: Total von Mises stress should be positive
        assert float(result[1, 0]) > 0.0

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        nvars = fwd.nvars()
        r1 = fwd(bkd.zeros((nvars, 1)))
        r2 = fwd(bkd.ones((nvars, 1)))
        assert not np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2))

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = cantilever_beam_2d_linear(bkd, mesh_path=_TEST_MESH)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)

    def test_nvars_matches_subdomains_times_kle(self):
        fwd = self._bm3.function()
        # Should be 3 subdomains * 3 KLE terms = 9
        assert fwd.nvars() == 9


# =========================================================================
# 2D Neo-Hookean beam tests
# =========================================================================


class TestCantileverBeam2DNeoHookean:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_neohookean(
            cls._bkd,
            num_kle_terms=1,
            mesh_path=_TEST_MESH,
        )
        cls._bm_lin_small = cantilever_beam_2d_linear(
            cls._bkd,
            num_kle_terms=1,
            q0=0.1,
            mesh_path=_TEST_MESH,
        )
        cls._bm_neo_small = cantilever_beam_2d_neohookean(
            cls._bkd,
            num_kle_terms=1,
            q0=0.1,
            mesh_path=_TEST_MESH,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        assert fwd.nqoi() == 2
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)
        assert float(result[0, 0]) < 0.0
        assert float(result[1, 0]) > 0.0

    def test_close_to_linear_at_small_load(self):
        """Neo-Hookean and linear should give similar results for small load."""
        bkd = self._bkd
        sample = bkd.zeros((self._bm_lin_small.function().nvars(), 1))
        r_lin = float(self._bm_lin_small.function()(sample)[0, 0])
        r_neo = float(self._bm_neo_small.function()(sample)[0, 0])
        rel_diff = abs(r_lin - r_neo) / abs(r_lin)
        assert rel_diff < 0.05

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = cantilever_beam_2d_neohookean(bkd, mesh_path=_TEST_MESH)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)


# =========================================================================
# SPDE-based 1D Euler-Bernoulli beam tests
# =========================================================================


class TestCantileverBeam1DSPDE:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm = cantilever_beam_1d_spde(cls._bkd, nx=20, num_kle_terms=2)
        cls._bm3 = cantilever_beam_1d_spde(cls._bkd, num_kle_terms=3)
        cls._bm1 = cantilever_beam_1d_spde(cls._bkd, nx=20, num_kle_terms=1)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm.function()
        assert fwd.nvars() == 2
        assert fwd.nqoi() == 3
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (3, 1)
        assert float(result[0, 0]) > 0.0
        assert float(result[1, 0]) > 0.0
        assert float(result[2, 0]) > 0.0

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm.function()
        r1 = fwd(bkd.zeros((2, 1)))
        r2 = fwd(bkd.array([[1.0], [0.0]]))
        assert not np.allclose(
            bkd.to_numpy(r1), bkd.to_numpy(r2)
        ), "Different KLE params should give different results"

    def test_batch_evaluation(self):
        bkd = self._bkd
        fwd = self._bm.function()
        samples = bkd.array([[0.0, 0.5, -0.5], [0.0, 0.3, -0.3]])
        result = fwd(samples)
        assert result.shape == (3, 3)

    def test_prior_and_domain(self):
        bm = self._bm3
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        assert samples.shape == (3, 5)
        bounds = bm.domain().bounds()
        assert bounds.shape == (3, 2)

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_1d_spde", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (3, 1)

    def test_one_kle_term_recovers_constant(self):
        """With 1 KLE term at params=0, result is deterministic baseline."""
        bkd = self._bkd
        fwd = self._bm1.function()
        r1 = fwd(bkd.zeros((1, 1)))
        r2 = fwd(bkd.zeros((1, 1)))
        bkd.assert_allclose(r1, r2)


# =========================================================================
# SPDE-based 2D linear elastic beam tests
# =========================================================================


class TestCantileverBeam2DLinearSPDE:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_linear_spde(
            cls._bkd,
            num_kle_terms=1,
            mesh_path=_TEST_MESH,
        )
        cls._bm3 = cantilever_beam_2d_linear_spde(
            cls._bkd,
            num_kle_terms=3,
            mesh_path=_TEST_MESH,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        assert fwd.nqoi() == 2
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)
        # QoI 0: Tip deflection should be negative (downward)
        assert float(result[0, 0]) < 0.0
        # QoI 1: Total von Mises stress should be positive
        assert float(result[1, 0]) > 0.0

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        nvars = fwd.nvars()
        r1 = fwd(bkd.zeros((nvars, 1)))
        r2 = fwd(bkd.ones((nvars, 1)))
        assert not np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2))

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = cantilever_beam_2d_linear_spde(bkd, mesh_path=_TEST_MESH)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)

    def test_nvars_matches_subdomains_times_kle(self):
        fwd = self._bm3.function()
        # Should be 3 subdomains * 3 KLE terms = 9
        assert fwd.nvars() == 9


# =========================================================================
# SPDE-based 2D Neo-Hookean beam tests
# =========================================================================


class TestCantileverBeam2DNeoHookeanSPDE:
    @classmethod
    def setup_class(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_neohookean_spde(
            cls._bkd,
            num_kle_terms=1,
            mesh_path=_TEST_MESH,
        )
        cls._bm_lin_small = cantilever_beam_2d_linear_spde(
            cls._bkd,
            num_kle_terms=1,
            q0=0.1,
            mesh_path=_TEST_MESH,
        )
        cls._bm_neo_small = cantilever_beam_2d_neohookean_spde(
            cls._bkd,
            num_kle_terms=1,
            q0=0.1,
            mesh_path=_TEST_MESH,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        assert fwd.nqoi() == 2
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)
        assert float(result[0, 0]) < 0.0
        assert float(result[1, 0]) > 0.0

    def test_close_to_linear_at_small_load(self):
        """Neo-Hookean and linear should give similar results for small load."""
        bkd = self._bkd
        sample = bkd.zeros((self._bm_lin_small.function().nvars(), 1))
        r_lin = float(self._bm_lin_small.function()(sample)[0, 0])
        r_neo = float(self._bm_neo_small.function()(sample)[0, 0])
        rel_diff = abs(r_lin - r_neo) / abs(r_lin)
        assert rel_diff < 0.05

    def test_protocol_compliance(self):
        assert isinstance(self._bm1, BenchmarkWithPriorProtocol)
        assert isinstance(self._bm1.function(), FunctionProtocol)

    def test_registry_access(self):
        bkd = self._bkd
        bm = cantilever_beam_2d_neohookean_spde(bkd, mesh_path=_TEST_MESH)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        assert result.shape == (2, 1)


# =========================================================================
# FEM vs Analytical verification tests
# =========================================================================


class TestFEMvsAnalytical:
    """Verify FEM results match analytical closed-form solutions."""

    def test_uniform_EI_matches_analytical(self):
        """CompositeBeam1DForwardModel vs CantileverBeam1DAnalytical."""
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam import (
            CantileverBeam1DAnalytical,
        )

        bkd = NumpyBkd()
        L, H, q0, skin_t = 100.0, 30.0, 10.0, 5.0

        fem_model = CompositeBeam1DForwardModel(
            nx=100,
            length=L,
            height=H,
            skin_thickness=skin_t,
            load_func=lambda x: q0 * x / L,
            bkd=bkd,
        )
        analytical_model = CantileverBeam1DAnalytical(
            length=L,
            height=H,
            skin_thickness=skin_t,
            q0=q0,
            bkd=bkd,
        )

        np.random.seed(42)
        E1_vals = np.random.uniform(18000, 22000, 5)
        E2_vals = np.random.uniform(4500, 5500, 5)
        samples = bkd.asarray(np.vstack([E1_vals, E2_vals]))

        fem_result = fem_model(samples)
        ana_result = analytical_model(samples)

        # Tip deflection: FEM should match analytical closely with nx=100
        bkd.assert_allclose(fem_result[0:1, :], ana_result[0:1, :], rtol=1e-4)
        # Integrated stress: constant for uniform EI (independent of E1, E2)
        bkd.assert_allclose(fem_result[1:2, :], ana_result[1:2, :], rtol=1e-4)
        # Max curvature: FEM finite differences are less accurate
        bkd.assert_allclose(fem_result[2:3, :], ana_result[2:3, :], rtol=1e-2)

    def test_analytical_jacobian(self):
        """Verify analytical Jacobian via DerivativeChecker."""
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam import (
            CantileverBeam1DAnalytical,
        )
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = NumpyBkd()
        model = CantileverBeam1DAnalytical(
            length=100.0,
            height=30.0,
            skin_thickness=5.0,
            q0=10.0,
            bkd=bkd,
        )

        sample = bkd.asarray([[20000.0], [5000.0]])
        jac = model.jacobian(sample)
        assert jac.shape == (3, 2)

        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    def test_analytical_registry(self):
        """Verify analytical benchmark is accessible via registry."""
        bkd = NumpyBkd()
        bm = BenchmarkRegistry.get("cantilever_beam_1d_analytical", bkd)
        fwd = bm.function()
        assert fwd.nvars() == 2
        assert fwd.nqoi() == 3
        result = fwd(bkd.asarray([[20000.0], [5000.0]]))
        assert result.shape == (3, 1)
        assert float(result[0, 0]) > 0.0
        assert float(result[1, 0]) > 0.0
        assert float(result[2, 0]) > 0.0

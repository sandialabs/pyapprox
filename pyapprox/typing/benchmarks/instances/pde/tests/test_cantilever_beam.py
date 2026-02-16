"""Integration tests for the cantilever beam benchmark instances."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests, slower_test  # noqa: F401
from pyapprox.typing.interface.functions.protocols import FunctionProtocol
from pyapprox.typing.benchmarks.protocols import BenchmarkWithPriorProtocol
from pyapprox.typing.benchmarks.instances.pde.cantilever_beam import (
    cantilever_beam_1d,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
    cantilever_beam_1d_spde,
    cantilever_beam_2d_linear_spde,
    cantilever_beam_2d_neohookean_spde,
)
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry


# =========================================================================
# 1D Euler-Bernoulli beam tests
# =========================================================================


class TestCantileverBeam1D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm = cantilever_beam_1d(cls._bkd, nx=20, num_kle_terms=2)
        cls._bm3 = cantilever_beam_1d(cls._bkd, num_kle_terms=3)
        cls._bm1 = cantilever_beam_1d(cls._bkd, nx=20, num_kle_terms=1)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm.function()
        self.assertEqual(fwd.nvars(), 2)
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm.function()
        r1 = fwd(bkd.zeros((2, 1)))
        r2 = fwd(bkd.array([[1.0], [0.0]]))
        self.assertFalse(
            np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2)),
            "Different KLE params should give different tip deflection",
        )

    def test_batch_evaluation(self):
        bkd = self._bkd
        fwd = self._bm.function()
        samples = bkd.array([[0.0, 0.5, -0.5], [0.0, 0.3, -0.3]])
        result = fwd(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_prior_and_domain(self):
        bkd = self._bkd
        bm = self._bm3
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        self.assertEqual(samples.shape, (3, 5))
        bounds = bm.domain().bounds()
        self.assertEqual(bounds.shape, (3, 2))

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_1d", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))

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


class TestCantileverBeam2DLinear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_linear(cls._bkd, num_kle_terms=1)
        cls._bm3 = cantilever_beam_2d_linear(cls._bkd, num_kle_terms=3)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))
        # Tip deflection should be negative (downward)
        self.assertLess(float(result[0, 0]), 0.0)

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        nvars = fwd.nvars()
        r1 = fwd(bkd.zeros((nvars, 1)))
        r2 = fwd(bkd.ones((nvars, 1)))
        self.assertFalse(
            np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2)),
        )

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_2d_linear", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))

    def test_nvars_matches_subdomains_times_kle(self):
        fwd = self._bm3.function()
        # Should be 3 subdomains * 3 KLE terms = 9
        self.assertEqual(fwd.nvars(), 9)


# =========================================================================
# 2D Neo-Hookean beam tests
# =========================================================================


class TestCantileverBeam2DNeoHookean(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_neohookean(cls._bkd, num_kle_terms=1)
        cls._bm_lin_small = cantilever_beam_2d_linear(
            cls._bkd, num_kle_terms=1, q0=0.1,
        )
        cls._bm_neo_small = cantilever_beam_2d_neohookean(
            cls._bkd, num_kle_terms=1, q0=0.1,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertLess(float(result[0, 0]), 0.0)

    def test_close_to_linear_at_small_load(self):
        """Neo-Hookean and linear should give similar results for small load."""
        bkd = self._bkd
        sample = bkd.zeros((self._bm_lin_small.function().nvars(), 1))
        r_lin = float(self._bm_lin_small.function()(sample)[0, 0])
        r_neo = float(self._bm_neo_small.function()(sample)[0, 0])
        rel_diff = abs(r_lin - r_neo) / abs(r_lin)
        self.assertLess(rel_diff, 0.05)

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_2d_neohookean", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))


# =========================================================================
# SPDE-based 1D Euler-Bernoulli beam tests
# =========================================================================


class TestCantileverBeam1DSPDE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm = cantilever_beam_1d_spde(cls._bkd, nx=20, num_kle_terms=2)
        cls._bm3 = cantilever_beam_1d_spde(cls._bkd, num_kle_terms=3)
        cls._bm1 = cantilever_beam_1d_spde(cls._bkd, nx=20, num_kle_terms=1)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm.function()
        self.assertEqual(fwd.nvars(), 2)
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm.function()
        r1 = fwd(bkd.zeros((2, 1)))
        r2 = fwd(bkd.array([[1.0], [0.0]]))
        self.assertFalse(
            np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2)),
            "Different KLE params should give different tip deflection",
        )

    def test_batch_evaluation(self):
        bkd = self._bkd
        fwd = self._bm.function()
        samples = bkd.array([[0.0, 0.5, -0.5], [0.0, 0.3, -0.3]])
        result = fwd(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_prior_and_domain(self):
        bkd = self._bkd
        bm = self._bm3
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        self.assertEqual(samples.shape, (3, 5))
        bounds = bm.domain().bounds()
        self.assertEqual(bounds.shape, (3, 2))

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_1d_spde", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))

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


class TestCantileverBeam2DLinearSPDE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_linear_spde(cls._bkd, num_kle_terms=1)
        cls._bm3 = cantilever_beam_2d_linear_spde(cls._bkd, num_kle_terms=3)

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))
        # Tip deflection should be negative (downward)
        self.assertLess(float(result[0, 0]), 0.0)

    def test_different_params_give_different_output(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        nvars = fwd.nvars()
        r1 = fwd(bkd.zeros((nvars, 1)))
        r2 = fwd(bkd.ones((nvars, 1)))
        self.assertFalse(
            np.allclose(bkd.to_numpy(r1), bkd.to_numpy(r2)),
        )

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_2d_linear_spde", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))

    def test_nvars_matches_subdomains_times_kle(self):
        fwd = self._bm3.function()
        # Should be 3 subdomains * 3 KLE terms = 9
        self.assertEqual(fwd.nvars(), 9)


# =========================================================================
# SPDE-based 2D Neo-Hookean beam tests
# =========================================================================


class TestCantileverBeam2DNeoHookeanSPDE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._bkd = NumpyBkd()
        cls._bm1 = cantilever_beam_2d_neohookean_spde(
            cls._bkd, num_kle_terms=1,
        )
        cls._bm_lin_small = cantilever_beam_2d_linear_spde(
            cls._bkd, num_kle_terms=1, q0=0.1,
        )
        cls._bm_neo_small = cantilever_beam_2d_neohookean_spde(
            cls._bkd, num_kle_terms=1, q0=0.1,
        )

    def test_evaluate_at_zero(self):
        bkd = self._bkd
        fwd = self._bm1.function()
        self.assertEqual(fwd.nqoi(), 1)
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertLess(float(result[0, 0]), 0.0)

    def test_close_to_linear_at_small_load(self):
        """Neo-Hookean and linear should give similar results for small load."""
        bkd = self._bkd
        sample = bkd.zeros((self._bm_lin_small.function().nvars(), 1))
        r_lin = float(self._bm_lin_small.function()(sample)[0, 0])
        r_neo = float(self._bm_neo_small.function()(sample)[0, 0])
        rel_diff = abs(r_lin - r_neo) / abs(r_lin)
        self.assertLess(rel_diff, 0.05)

    def test_protocol_compliance(self):
        self.assertTrue(isinstance(self._bm1, BenchmarkWithPriorProtocol))
        self.assertTrue(isinstance(self._bm1.function(), FunctionProtocol))

    def test_registry_access(self):
        bkd = self._bkd
        bm = BenchmarkRegistry.get("cantilever_beam_2d_neohookean_spde", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((fwd.nvars(), 1)))
        self.assertEqual(result.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()

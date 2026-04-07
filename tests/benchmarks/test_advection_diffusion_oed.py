"""Tests for ObstructedAdvectionDiffusionOEDBenchmark.

NumPy only (Galerkin uses skfem which is NumPy-based).
Forward evaluations are slow, so model evaluation tests use @slow_test.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox_benchmarks.instances.oed.advection_diffusion import (
    ObstructedAdvectionDiffusionOEDBenchmark,
    _build_obstructed_mesh,
    _solve_stokes,
)
from tests._helpers.markers import slow_test


class TestAdvectionDiffusionOEDBenchmark:
    """Tests for ObstructedAdvectionDiffusionOEDBenchmark (NumPy only)."""

    def _setup_data(self, bkd):
        # Use minimal refinement and few KLE terms for speed
        self._nstokes_refine = 1
        self._nadvec_diff_refine = 1
        self._nkle_terms = 3
        self._nsensors = 5

    def _create_benchmark(self, bkd):
        return ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
        )

    def test_construction(self, bkd):
        """Test benchmark can be constructed."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench is not None

    def test_nparams(self, bkd):
        """Test correct number of parameters: nkle + 3."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nparams() == self._nkle_terms + 3

    def test_prior_nvars(self, bkd):
        """Test prior has correct number of variables."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.prior().nvars() == self._nkle_terms + 3

    def test_prior_rvs_shape(self, bkd):
        """Test prior sampling returns correct shape."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        assert samples.shape == (self._nkle_terms + 3, 3)

    def test_observation_model_nqoi(self, bkd):
        """Test observation model has nsensors QoI."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.obs_map().nqoi() == self._nsensors

    def test_prediction_model_nqoi(self, bkd):
        """Test prediction model has 1 QoI."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.qoi_map().nqoi() == 1

    def test_design_conditions_shape(self, bkd):
        """Test design conditions shape is (2, nsensors)."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        locs = bench.design_conditions()
        assert locs.shape == (2, self._nsensors)

    def test_nobservations(self, bkd):
        """Test nobservations accessor."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nobservations() == self._nsensors

    def test_design_conditions_in_domain(self, bkd):
        """Test sensor locations are within [0,1]^2."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        locs_np = bkd.to_numpy(bench.design_conditions())
        assert np.all(locs_np >= -1e-12)
        assert np.all(locs_np <= 1.0 + 1e-12)

    def test_velocity_component_ordering(self, numpy_bkd):
        """Verify interleaved DOF extraction gives physically correct velocity.

        For left-to-right channel flow, the x-component should have
        positive mean and the y-component should have near-zero mean.
        A bug in DOF ordering (block vs interleaved) would swap or mix
        components.
        """
        bkd = numpy_bkd
        mesh = _build_obstructed_mesh(bkd, nrefine=1)
        sol, stokes, vel_basis, pres_basis = _solve_stokes(
            mesh, bkd, reynolds_num=10.0, vel_shape_params=[2.0, 2.0],
        )
        vel_ndofs = stokes.vel_ndofs()
        vel_state = bkd.to_numpy(sol[:vel_ndofs])

        # Interleaved extraction
        vel_x = vel_state[0::2]
        vel_y = vel_state[1::2]

        # For channel flow from left to right:
        # mean(vx) should be positive and much larger than |mean(vy)|
        assert vel_x.mean() > 0.05, (
            f"mean(vx)={vel_x.mean():.4f} should be positive for L-to-R flow"
        )
        assert abs(vel_y.mean()) < vel_x.mean(), (
            f"|mean(vy)|={abs(vel_y.mean()):.4f} should be smaller than "
            f"mean(vx)={vel_x.mean():.4f}"
        )

    @slow_test
    def test_solve_for_plotting(self, numpy_bkd):
        """Test solve_for_plotting returns correct data."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        data = bench.solve_for_plotting(sample)

        assert data["vel_x"].ndim == 1
        assert data["vel_y"].shape == data["vel_x"].shape
        assert data["vel_magnitude"].shape == data["vel_x"].shape
        assert data["concentration"].ndim == 1
        assert np.all(np.isfinite(data["vel_magnitude"]))
        assert np.all(np.isfinite(data["concentration"]))
        assert np.all(data["vel_magnitude"] >= 0)

    @slow_test
    def test_observation_model_evaluation(self, numpy_bkd):
        """Test observation model returns correct shape (NumPy only: sparse solve)."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        obs = bench.obs_map()(sample)
        assert obs.shape == (self._nsensors, 1)
        obs_np = bkd.to_numpy(obs)
        assert np.all(np.isfinite(obs_np))

    @slow_test
    def test_prediction_model_evaluation(self, numpy_bkd):
        """Test prediction model returns correct shape (NumPy only: sparse solve)."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        pred = bench.qoi_map()(sample)
        assert pred.shape == (1, 1)
        pred_np = bkd.to_numpy(pred)
        assert np.all(np.isfinite(pred_np))

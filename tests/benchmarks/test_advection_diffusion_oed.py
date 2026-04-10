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

    def test_construction(self, numpy_bkd):
        """Test benchmark can be constructed."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench is not None

    def test_nparams(self, numpy_bkd):
        """Test correct number of parameters: nkle + 3."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nparams() == self._nkle_terms + 3

    def test_prior_nvars(self, numpy_bkd):
        """Test prior has correct number of variables."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.prior().nvars() == self._nkle_terms + 3

    def test_prior_rvs_shape(self, numpy_bkd):
        """Test prior sampling returns correct shape."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        assert samples.shape == (self._nkle_terms + 3, 3)

    def test_observation_model_nqoi(self, numpy_bkd):
        """Test observation model has nsensors QoI."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.obs_map().nqoi() == self._nsensors

    def test_prediction_model_nqoi(self, numpy_bkd):
        """Test prediction model has 1 QoI."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.qoi_map().nqoi() == 1

    def test_design_conditions_shape(self, numpy_bkd):
        """Test design conditions shape is (2, nsensors)."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        locs = bench.design_conditions()
        assert locs.shape == (2, self._nsensors)

    def test_nobservations(self, numpy_bkd):
        """Test nobservations accessor."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nobservations() == self._nsensors

    def test_design_conditions_in_domain(self, numpy_bkd):
        """Test sensor locations are within [0,1]^2."""
        bkd = numpy_bkd
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

    def test_kle_forcing_zero_outside_subdomain(self, numpy_bkd):
        """Forcing is exactly zero at every ADR node outside the subdomain."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        subdomain = (0.0, 0.25, 0.0, 1.0)
        bench = ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            kle_subdomain=subdomain,
            kle_correlation_length=0.1,
        )
        rng = np.random.default_rng(0)
        kle_params = bkd.asarray(rng.standard_normal(self._nkle_terms))
        forcing = bkd.to_numpy(bench._kle_map(kle_params))

        nodes = bkd.to_numpy(bench._adr_mesh.nodes())
        xmin, xmax, ymin, ymax = subdomain
        strictly_outside = (
            (nodes[0] > xmax + 1e-12)
            | (nodes[1] > ymax + 1e-12)
            | (nodes[0] < xmin - 1e-12)
            | (nodes[1] < ymin - 1e-12)
        )
        assert np.all(forcing[strictly_outside] == 0.0)
        # Must have at least one strictly-positive interior value
        # (lognormal > 0 at subdomain nodes).
        assert np.any(forcing > 0.0)

    def test_full_domain_kle_subdomain_none(self, numpy_bkd):
        """Passing kle_subdomain=None recovers a full-domain lognormal KLE."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            kle_subdomain=None,
        )
        rng = np.random.default_rng(1)
        kle_params = bkd.asarray(rng.standard_normal(self._nkle_terms))
        forcing = bkd.to_numpy(bench._kle_map(kle_params))
        # Lognormal is strictly positive everywhere on the full mesh.
        assert np.all(forcing > 0.0)

    def test_kle_subdomain_too_small_raises(self, numpy_bkd):
        """Requesting more KLE terms than subdomain nodes raises ValueError."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        # A sliver rectangle that barely contains any nodes, combined
        # with many KLE terms, triggers the guard.
        with pytest.raises(ValueError, match="KLE subdomain"):
            ObstructedAdvectionDiffusionOEDBenchmark(
                bkd,
                nstokes_refine=self._nstokes_refine,
                nadvec_diff_refine=self._nadvec_diff_refine,
                nkle_terms=1000,
                nsensors=self._nsensors,
                kle_subdomain=(0.0, 0.25, 0.0, 1.0),
            )

    def test_kle_subdomain_outside_domain_raises(self, numpy_bkd):
        """Subdomain outside the base domain raises ValueError."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        with pytest.raises(ValueError, match="must lie inside the base"):
            ObstructedAdvectionDiffusionOEDBenchmark(
                bkd,
                nstokes_refine=self._nstokes_refine,
                nadvec_diff_refine=self._nadvec_diff_refine,
                nkle_terms=self._nkle_terms,
                nsensors=self._nsensors,
                kle_subdomain=(2.0, 3.0, 2.0, 3.0),
            )

    def test_construction_accepts_new_kle_kwargs(self, numpy_bkd):
        """Constructor accepts kle_correlation_length and kle_sigma."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        bench = ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            kle_correlation_length=0.05,
            kle_sigma=0.5,
        )
        assert bench is not None

    def test_invalid_source_mode_raises(self, numpy_bkd):
        """source_mode must be 'forcing' or 'initial_condition'."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        with pytest.raises(ValueError, match="source_mode"):
            ObstructedAdvectionDiffusionOEDBenchmark(
                bkd,
                nstokes_refine=self._nstokes_refine,
                nadvec_diff_refine=self._nadvec_diff_refine,
                nkle_terms=self._nkle_terms,
                nsensors=self._nsensors,
                source_mode="bogus",  # type: ignore[arg-type]
            )

    def test_non_numpy_backend_raises(self, torch_bkd):
        """Non-NumPy backends must be rejected at construction time.

        skfem is NumPy-only, so silently accepting a torch backend
        would produce a benchmark whose ``obs_map()`` / ``qoi_map()``
        jacobians return zeros through ``torch.autograd``. The
        constructor raises ``TypeError`` early so this cannot happen.
        """
        self._setup_data(torch_bkd)
        with pytest.raises(TypeError, match="NumpyBkd"):
            ObstructedAdvectionDiffusionOEDBenchmark(
                torch_bkd,
                nstokes_refine=self._nstokes_refine,
                nadvec_diff_refine=self._nadvec_diff_refine,
                nkle_terms=self._nkle_terms,
                nsensors=self._nsensors,
            )

    @slow_test
    def test_source_mode_initial_condition(self, numpy_bkd):
        """Initial-condition mode: IC equals KLE field, forward is finite.

        Under IC mode the ADR equation has zero forcing and ``u(x, 0)``
        equals the KLE nodal field, so the concentration at the first
        time step must match the KLE field exactly. Under forcing mode
        the initial concentration is zero. This locks in the two-mode
        contract.
        """
        bkd = numpy_bkd
        self._setup_data(bkd)

        bench_ic = ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            source_mode="initial_condition",
        )
        bench_fc = ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            source_mode="forcing",
        )

        rng = np.random.default_rng(0)
        sample_np = np.zeros(bench_ic.nparams())
        sample_np[: self._nkle_terms] = rng.standard_normal(
            self._nkle_terms,
        )
        sample_np[-3] = 2.5
        sample_np[-2] = 2.5
        sample_np[-1] = 12.5

        kle_field = bkd.to_numpy(
            bench_ic._kle_map(bkd.asarray(sample_np[: self._nkle_terms]))
        )

        ic_solutions, _ = bench_ic._solve_forward(sample_np)
        fc_solutions, _ = bench_fc._solve_forward(sample_np)

        ic_np = bkd.to_numpy(ic_solutions)
        fc_np = bkd.to_numpy(fc_solutions)

        # Initial time step: IC mode == KLE field; forcing mode == 0.
        np.testing.assert_allclose(ic_np[:, 0], kle_field, rtol=1e-12)
        np.testing.assert_allclose(
            fc_np[:, 0], np.zeros_like(kle_field), atol=1e-12,
        )
        # Finite everywhere under both modes.
        assert np.all(np.isfinite(ic_np))
        assert np.all(np.isfinite(fc_np))
        # Non-trivial difference at the final time.
        assert not np.allclose(ic_np[:, -1], fc_np[:, -1])

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

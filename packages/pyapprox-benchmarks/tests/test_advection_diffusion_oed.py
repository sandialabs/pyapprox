"""Tests for ObstructedAdvectionDiffusionOEDBenchmark.

NumPy only (Galerkin uses skfem which is NumPy-based).
Forward evaluations are slow, so model evaluation tests use @slow_test.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import pickle

import numpy as np
from pyapprox_benchmarks.expdesign.advection_diffusion import (
    FixedVelocityObstructedAdvectionDiffusionOEDBenchmark,
    build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark,
    build_obstructed_advection_diffusion_oed_benchmark,
)
from pyapprox_benchmarks.problems.oed.advection_diffusion import (
    AdvectionDiffusionOEDProblem,
    FixedVelocityAdvectionDiffusionOEDProblem,
)
from pyapprox_benchmarks.problems.oed.advection_diffusion._mesh import (
    _build_obstructed_mesh,
)
from pyapprox_benchmarks.problems.oed.advection_diffusion._stokes import (
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
        return build_obstructed_advection_diffusion_oed_benchmark(
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
        bench = build_obstructed_advection_diffusion_oed_benchmark(
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
        forcing = bkd.to_numpy(bench.problem()._kle_map(kle_params))

        nodes = bkd.to_numpy(bench.problem()._adr_mesh.nodes())
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
        bench = build_obstructed_advection_diffusion_oed_benchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            kle_subdomain=None,
        )
        rng = np.random.default_rng(1)
        kle_params = bkd.asarray(rng.standard_normal(self._nkle_terms))
        forcing = bkd.to_numpy(bench.problem()._kle_map(kle_params))
        # Lognormal is strictly positive everywhere on the full mesh.
        assert np.all(forcing > 0.0)

    def test_kle_subdomain_too_small_raises(self, numpy_bkd):
        """Requesting more KLE terms than subdomain nodes raises ValueError."""
        bkd = numpy_bkd
        self._setup_data(bkd)
        # A sliver rectangle that barely contains any nodes, combined
        # with many KLE terms, triggers the guard.
        with pytest.raises(ValueError, match="KLE subdomain"):
            build_obstructed_advection_diffusion_oed_benchmark(
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
            build_obstructed_advection_diffusion_oed_benchmark(
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
        bench = build_obstructed_advection_diffusion_oed_benchmark(
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
            build_obstructed_advection_diffusion_oed_benchmark(
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
            build_obstructed_advection_diffusion_oed_benchmark(
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

        bench_ic = build_obstructed_advection_diffusion_oed_benchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
            source_mode="initial_condition",
        )
        bench_fc = build_obstructed_advection_diffusion_oed_benchmark(
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
            bench_ic.problem()._kle_map(
                bkd.asarray(sample_np[: self._nkle_terms])
            )
        )

        ic_solutions, _ = bench_ic.problem()._solve_forward(sample_np)
        fc_solutions, _ = bench_fc.problem()._solve_forward(sample_np)

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


class TestPickleRegression:
    """Pickle regression tests for the upstream closure fixes.

    These tests guard against re-introducing the four closure sites
    that previously broke ``pickle.dumps`` on the advdiff benchmark:
    the ``kle_factory`` exp lambdas, the ``_solve_stokes`` BC
    closures, the ``_make_vel_value_func`` nested closures inside
    :class:`StokesPhysics`, and the ``obs_callable`` / ``pred_callable``
    closures formerly built inside the benchmark ``__init__``.
    """

    def _build(self, bkd):
        return build_obstructed_advection_diffusion_oed_benchmark(
            bkd,
            nstokes_refine=1,
            nadvec_diff_refine=1,
            nkle_terms=3,
            nsensors=4,
        )

    def test_benchmark_pickle_roundtrip(self, numpy_bkd):
        """Benchmark shell roundtrips and ``evaluate_nodal`` matches."""
        bench = self._build(numpy_bkd)
        bench2 = pickle.loads(pickle.dumps(bench))
        rng = np.random.default_rng(0)
        samples = numpy_bkd.asarray(rng.standard_normal((6, 2)))
        out1 = numpy_bkd.to_numpy(bench.evaluate_nodal(samples))
        out2 = numpy_bkd.to_numpy(bench2.evaluate_nodal(samples))
        np.testing.assert_allclose(out1, out2, rtol=1e-12)

    def test_problem_pickle_roundtrip(self, numpy_bkd):
        """Problem object pickles on its own (joblib worker contract).

        The problem must pickle without carrying the benchmark shell
        so that parallel workers only ship the compute substrate.
        """
        bench = self._build(numpy_bkd)
        problem = bench.problem()
        problem2 = pickle.loads(pickle.dumps(problem))
        rng = np.random.default_rng(1)
        samples = numpy_bkd.asarray(rng.standard_normal((6, 2)))
        out1 = numpy_bkd.to_numpy(problem.evaluate_nodal(samples))
        out2 = numpy_bkd.to_numpy(problem2.evaluate_nodal(samples))
        np.testing.assert_allclose(out1, out2, rtol=1e-12)

    def test_stokes_result_pickle_roundtrip(self, numpy_bkd):
        """Stokes result tuple pickles and still drives ADR transient."""
        bench = self._build(numpy_bkd)
        problem = bench.problem()
        stokes_result = problem._compute_stokes_result(2.5, 2.5, 12.5)
        stokes_result2 = pickle.loads(pickle.dumps(stokes_result))
        rng = np.random.default_rng(2)
        kle_params = numpy_bkd.asarray(rng.standard_normal(3))
        solutions, times = problem._solve_adr_transient(
            kle_params, stokes_result2,
        )
        sol_np = numpy_bkd.to_numpy(solutions)
        assert sol_np.ndim == 2
        assert sol_np.shape[1] == times.shape[0]
        assert np.all(np.isfinite(sol_np))


class TestFixedVelocityObstructedAdvectionDiffusionOEDBenchmark:
    """Tests for the Pattern B only fixed-velocity advdiff benchmark.

    Every assertion goes through ``bench.problem()`` to double-document
    the benchmark's surface contract: the shell has no Pattern A
    forwarders and the pinned velocity parameters live on the
    problem.
    """

    def _setup_data(self):
        self._nstokes_refine = 1
        self._nadvec_diff_refine = 1
        self._nkle_terms = 3
        self._nsensors = 4

    def _build_fixed(self, bkd, **overrides):
        self._setup_data()
        kwargs = dict(
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
        )
        kwargs.update(overrides)
        return build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark(
            bkd, **kwargs,
        )

    def _build_random(self, bkd):
        self._setup_data()
        return build_obstructed_advection_diffusion_oed_benchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
        )

    def test_construction_reduces_nparams(self, numpy_bkd):
        bench = self._build_fixed(numpy_bkd)
        problem = bench.problem()
        assert problem.nparams() == self._nkle_terms
        assert problem.prior().nvars() == self._nkle_terms
        assert problem.obs_map().nvars() == self._nkle_terms
        assert problem.qoi_map().nvars() == self._nkle_terms

    def test_reduced_prior_is_gaussian_only(self, numpy_bkd):
        from pyapprox.probability.univariate.gaussian import GaussianMarginal
        bench = self._build_fixed(numpy_bkd)
        marginals = bench.problem().prior().marginals()
        assert len(marginals) == self._nkle_terms
        for m in marginals:
            assert isinstance(m, GaussianMarginal)

    def test_problem_composition(self, numpy_bkd):
        bench = self._build_fixed(numpy_bkd)
        problem = bench.problem()
        assert isinstance(problem, FixedVelocityAdvectionDiffusionOEDProblem)
        assert isinstance(problem, AdvectionDiffusionOEDProblem)
        # design_conditions shape must match the random-velocity
        # sibling built with the same substrate kwargs.
        random_bench = self._build_random(numpy_bkd)
        assert (
            numpy_bkd.to_numpy(problem.design_conditions()).shape
            == numpy_bkd.to_numpy(
                random_bench.problem().design_conditions()
            ).shape
        )

    @slow_test
    def test_evaluate_observation_matches_padded(self, numpy_bkd):
        """Reduced obs_map equals random-velocity obs_map at padded sample.

        Fixed-velocity ``obs_map`` on a KLE-only sample must equal the
        random-velocity ``obs_map`` on the same KLE sample padded with
        the pinned velocity triple — the two code paths must agree to
        machine precision.
        """
        vel_a, vel_b, re = 2.5, 2.5, 12.5
        fixed = self._build_fixed(
            numpy_bkd, vel_shape_a=vel_a, vel_shape_b=vel_b, reynolds_num=re,
        )
        random_bench = self._build_random(numpy_bkd)
        rng = np.random.default_rng(3)
        kle = rng.standard_normal((self._nkle_terms, 1))
        padded = np.vstack(
            [kle, np.array([[vel_a], [vel_b], [re]])],
        )
        fixed_out = numpy_bkd.to_numpy(
            fixed.problem().obs_map()(numpy_bkd.asarray(kle))
        )
        random_out = numpy_bkd.to_numpy(
            random_bench.problem().obs_map()(numpy_bkd.asarray(padded))
        )
        np.testing.assert_allclose(fixed_out, random_out, rtol=1e-12)

    @slow_test
    def test_evaluate_prediction_matches_padded(self, numpy_bkd):
        """Reduced qoi_map equals random qoi_map at padded sample."""
        vel_a, vel_b, re = 2.5, 2.5, 12.5
        fixed = self._build_fixed(
            numpy_bkd, vel_shape_a=vel_a, vel_shape_b=vel_b, reynolds_num=re,
        )
        random_bench = self._build_random(numpy_bkd)
        rng = np.random.default_rng(4)
        kle = rng.standard_normal((self._nkle_terms, 1))
        padded = np.vstack(
            [kle, np.array([[vel_a], [vel_b], [re]])],
        )
        fixed_out = numpy_bkd.to_numpy(
            fixed.problem().qoi_map()(numpy_bkd.asarray(kle))
        )
        random_out = numpy_bkd.to_numpy(
            random_bench.problem().qoi_map()(numpy_bkd.asarray(padded))
        )
        np.testing.assert_allclose(fixed_out, random_out, rtol=1e-12)

    @slow_test
    def test_stokes_solved_exactly_once(self, numpy_bkd, monkeypatch):
        """Fixed-velocity problem caches Stokes and reuses it across evals.

        Monkeypatch ``_compute_stokes_result`` at the **class level**
        with a counter. After construction (which calls it once) the
        counter stays fixed across a 3-sample ``evaluate_nodal`` call.

        NOTE: patches class, not instance — would interfere with
        concurrently-running advdiff tests in the same process. Uses
        pytest ``monkeypatch`` (test-scoped, auto-cleaned).
        """
        counter = {"n": 0}
        orig = AdvectionDiffusionOEDProblem._compute_stokes_result

        def counting(self, a, b, re):
            counter["n"] += 1
            return orig(self, a, b, re)

        monkeypatch.setattr(
            AdvectionDiffusionOEDProblem,
            "_compute_stokes_result",
            counting,
        )
        bench = self._build_fixed(numpy_bkd)
        # Construction should have triggered exactly one Stokes solve.
        n_after_ctor = counter["n"]
        assert n_after_ctor == 1
        rng = np.random.default_rng(5)
        samples = numpy_bkd.asarray(
            rng.standard_normal((self._nkle_terms, 3)),
        )
        bench.problem().evaluate_nodal(samples)
        # No additional Stokes solves during forward evaluation.
        assert counter["n"] == n_after_ctor

    def test_pickle_roundtrip(self, numpy_bkd):
        bench = self._build_fixed(numpy_bkd)
        bench2 = pickle.loads(pickle.dumps(bench))
        rng = np.random.default_rng(6)
        samples = numpy_bkd.asarray(
            rng.standard_normal((self._nkle_terms, 2)),
        )
        out1 = numpy_bkd.to_numpy(
            bench.problem().evaluate_nodal(samples),
        )
        out2 = numpy_bkd.to_numpy(
            bench2.problem().evaluate_nodal(samples),
        )
        np.testing.assert_allclose(out1, out2, rtol=1e-12)

    @slow_test
    def test_solve_for_plotting_accepts_reduced_sample(self, numpy_bkd):
        bench = self._build_fixed(numpy_bkd)
        problem = bench.problem()
        np.random.seed(7)
        sample = problem.prior().rvs(1)
        data = problem.solve_for_plotting(sample)
        assert data["vel_x"].ndim == 1
        assert data["concentration"].ndim == 1
        assert np.all(np.isfinite(data["vel_magnitude"]))
        assert np.all(np.isfinite(data["concentration"]))

    def test_build_fixed_returns_shell(self, numpy_bkd):
        """Build function returns correct type."""
        bench = self._build_fixed(numpy_bkd)
        assert isinstance(
            bench, FixedVelocityObstructedAdvectionDiffusionOEDBenchmark,
        )
        assert bench.problem().nparams() == self._nkle_terms

    def test_pattern_b_only_no_forwarders(self, numpy_bkd):
        """Fixed-velocity benchmark exposes no Pattern A forwarders.

        Guards against accidentally re-adding forwarders. Pinned
        velocity parameters must live on the problem, not the shell.
        """
        bench = self._build_fixed(numpy_bkd)
        pattern_a_names = [
            "prior", "obs_map", "qoi_map", "design_conditions",
            "nparams", "nobservations", "evaluate_nodal",
            "evaluate_both", "solve_for_plotting", "mesh_nodes",
            "nnodes", "vel_shape_a", "vel_shape_b", "reynolds_num",
        ]
        for name in pattern_a_names:
            assert not hasattr(bench, name), (
                f"Pattern B only: benchmark must not expose '{name}' — "
                f"go through bench.problem().{name}()"
            )

    def test_pinned_velocity_on_problem(self, numpy_bkd):
        """Pinned velocity accessors live on the problem object."""
        bench = self._build_fixed(
            numpy_bkd,
            vel_shape_a=3.0,
            vel_shape_b=2.5,
            reynolds_num=15.0,
        )
        problem = bench.problem()
        assert problem.vel_shape_a() == 3.0
        assert problem.vel_shape_b() == 2.5
        assert problem.reynolds_num() == 15.0

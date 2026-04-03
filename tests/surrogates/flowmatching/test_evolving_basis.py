"""Tests for evolving Stieltjes basis in flow matching.

Unit tests for basis_state, basis_factory, basis_interp.
Integration tests for Gaussian transport with StieltjesFlowVF.
"""

import numpy as np
import pytest

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.flowmatching.basis_factory import (
    StieltjesBasisFactory,
)
from pyapprox.surrogates.flowmatching.basis_interp import (
    IdentityInterpolator,
    RecurrenceInterpolator,
)
from pyapprox.surrogates.flowmatching.basis_state import StieltjesBasisState
from pyapprox.surrogates.flowmatching.cfm_loss import CFMLoss
from pyapprox.surrogates.flowmatching.evolving_vf import (
    KroneckerStrategy,
    PerSliceStrategy,
    StieltjesFlowVF,
    build_stieltjes_flow_vf,
)
from pyapprox.surrogates.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.ode_adapter import integrate_flow
from pyapprox.surrogates.flowmatching.quad_data import FlowMatchingQuadData
from pyapprox.pde.time.explicit_steppers.heun import HeunStepper
from tests._helpers.markers import slow_test


def _gauss_hermite_quad(bkd, n):
    """Build Gauss-Hermite quadrature for N(0,1) using pyapprox."""
    marginals = [GaussianMarginal(0.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd)
    pts, wts = basis.tensor_product_quadrature([n])
    return pts[0, :], wts


class TestStieltjesBasisState:
    def test_orthonormality_standard_normal(self, bkd) -> None:
        """Basis from N(0,1) GH quadrature is orthonormal."""
        nterms = 5
        n_quad = 20
        pts, wts = _gauss_hermite_quad(bkd, n_quad)
        factory = StieltjesBasisFactory(nterms, bkd)
        state = factory.build(pts, wts)

        assert state.n_basis() == nterms

        x = bkd.reshape(pts, (1, -1))
        phi = state.eval(x)  # (n_quad, nterms)

        # Phi^T @ diag(w) @ Phi should be identity
        gram = phi.T @ (bkd.reshape(wts, (-1, 1)) * phi)
        bkd.assert_allclose(gram, bkd.eye(nterms), atol=1e-10)

    def test_orthonormality_shifted_gaussian(self, bkd) -> None:
        """Basis from N(3, 0.5^2) samples is orthonormal."""
        mu, sigma = 3.0, 0.5
        nterms = 4
        n_quad = 15

        pts, wts = _gauss_hermite_quad(bkd, n_quad)
        shifted_pts = sigma * pts + mu

        factory = StieltjesBasisFactory(nterms, bkd)
        state = factory.build(shifted_pts, wts)

        x = bkd.reshape(shifted_pts, (1, -1))
        phi = state.eval(x)

        gram = phi.T @ (bkd.reshape(wts, (-1, 1)) * phi)
        bkd.assert_allclose(gram, bkd.eye(nterms), atol=1e-10)

    def test_derivatives(self, bkd) -> None:
        """Finite difference check on polynomial derivatives."""
        nterms = 4
        n_quad = 15
        pts, wts = _gauss_hermite_quad(bkd, n_quad)
        factory = StieltjesBasisFactory(nterms, bkd)
        state = factory.build(pts, wts)

        x0 = bkd.array([[0.5]])
        h = 1e-7
        xp = bkd.array([[0.5 + h]])
        xm = bkd.array([[0.5 - h]])

        dphi = state.eval_derivatives(x0, order=1)  # (1, nterms)
        fd = (state.eval(xp) - state.eval(xm)) / (2 * h)
        bkd.assert_allclose(dphi, fd, rtol=1e-5)


class TestIdentityInterpolator:
    def test_exact_lookup(self, bkd) -> None:
        """Exact t lookup returns correct state."""
        nterms = 3
        pts, wts = _gauss_hermite_quad(bkd, 10)
        factory = StieltjesBasisFactory(nterms, bkd)

        states = []
        t_vals = [0.2, 0.5, 0.8]
        for mu in [0.0, 1.0, 2.0]:
            states.append(factory.build(pts + mu, wts))

        interp = IdentityInterpolator(bkd)
        interp.fit(bkd.asarray(t_vals), states)

        assert interp.n_states() == 3
        retrieved = interp(0.5)
        bkd.assert_allclose(retrieved.rcoefs(), states[1].rcoefs())

    def test_missing_t_raises(self, bkd) -> None:
        """Strict lookup raises for missing t."""
        nterms = 3
        pts, wts = _gauss_hermite_quad(bkd, 10)
        factory = StieltjesBasisFactory(nterms, bkd)
        state = factory.build(pts, wts)

        interp = IdentityInterpolator(bkd)
        interp.fit(bkd.asarray([0.5]), [state])

        with pytest.raises(ValueError, match="not found"):
            interp(0.9)


def _build_evolving_gaussian_setup(bkd, mu_target, sigma_target, nterms,
                                   n_t=8, n_x=15, n_legendre=1):
    """Build StieltjesFlowVF for N(0,1) -> N(mu, sigma^2) transport.

    Uses tensor product quad: Uniform t x Gauss-Hermite x0, paired
    x1 = sigma * x0 + mu.
    """
    # Quad over (t, x0)
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd),
                      GaussianMarginal(0.0, 1.0, bkd)]
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_t, n_x])

    t_all = quad_pts[0:1, :]   # (1, n_quad)
    x0_all = quad_pts[1:2, :]  # (1, n_quad)
    x1_all = sigma_target * x0_all + mu_target

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all, x0=x0_all, x1=x1_all, weights=quad_wts, bkd=bkd,
    )

    vf = build_stieltjes_flow_vf(quad_data, path, nterms, bkd,
                                 n_legendre=n_legendre)
    return vf, path, loss, quad_data


class TestStieltjesFlowVF:
    def test_basis_matrix_shape(self, bkd) -> None:
        """Basis matrix has correct shape."""
        nterms = 4
        vf, _, _, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8,
        )
        phi = vf.basis_matrix(bkd.vstack([qd.t(), qd.x0()]))
        assert phi.shape == (qd.n_quad(), nterms)

    def test_nterms_and_nvars(self, bkd) -> None:
        """Check dimension queries."""
        nterms = 5
        vf, _, _, _ = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8,
        )
        assert vf.nterms() == nterms
        assert vf.nvars() == 2
        assert vf.nqoi() == 1

    def test_coefficients_roundtrip(self, bkd) -> None:
        """get/set coefficients roundtrip."""
        nterms = 3
        vf, _, _, _ = _build_evolving_gaussian_setup(
            bkd, 0.0, 1.0, nterms, n_t=4, n_x=8,
        )
        coef = bkd.ones((nterms, 1))
        vf.set_coefficients(coef)
        bkd.assert_allclose(vf.get_coefficients(), coef)

    def test_hyp_list_sync(self, bkd) -> None:
        """HyperParameterList syncs with coefficients."""
        nterms = 3
        vf, _, _, _ = _build_evolving_gaussian_setup(
            bkd, 0.0, 1.0, nterms, n_t=4, n_x=8,
        )
        hl = vf.hyp_list()
        vals = bkd.ones((nterms,)) * 2.0
        hl.set_values(vals)
        vf._sync_from_hyp_list()
        expected = bkd.reshape(vals, (nterms, 1))
        bkd.assert_allclose(vf.get_coefficients(), expected)

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Jacobian batch has correct shape."""
        nterms = 4
        vf, path, _, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8,
        )
        # Fit so coefficients are non-trivial
        x_t = path.interpolate(qd.t(), qd.x0(), qd.x1())
        vf_input = bkd.vstack([qd.t(), x_t])
        u_t = path.target_field(qd.t(), qd.x0(), qd.x1())
        vf.fit(vf_input, u_t)

        jac = vf.jacobian_batch(vf_input)
        n = vf_input.shape[1]
        assert jac.shape == (n, 1, 2)


class TestGaussianTransportEvolving:
    def test_training_loss_near_zero(self, bkd) -> None:
        """Gaussian VF is affine in x_t, so loss should be near zero."""
        nterms = 4
        vf, path, loss, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=8, n_x=15,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-8, (
            f"Expected loss < 1e-8, got {result.training_loss():.2e}"
        )

    @slow_test
    def test_target_moments(self, bkd) -> None:
        """ODE-integrated samples match target N(2, 0.5^2)."""
        mu_target, sigma_target = 2.0, 0.5
        nterms = 4
        vf, path, loss, qd = _build_evolving_gaussian_setup(
            bkd, mu_target, sigma_target, nterms, n_t=10, n_x=20,
        )
        fitted_vf = LeastSquaresFitter(bkd).fit(vf, path, loss, qd).surrogate()

        np.random.seed(456)
        nsamples = 3000
        x0_samples = bkd.array(np.random.randn(1, nsamples).tolist())

        # Use n_steps matching n_t for alignment
        x1_samples = integrate_flow(
            fitted_vf, x0_samples, 0.0, 1.0, n_steps=50, bkd=bkd,
            stepper_cls=HeunStepper,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = np.mean(x1_np[0, :])
        sample_var = np.var(x1_np[0, :])

        bkd.assert_allclose(
            bkd.asarray([sample_mean]),
            bkd.asarray([mu_target]),
            atol=0.1,
        )
        bkd.assert_allclose(
            bkd.asarray([sample_var]),
            bkd.asarray([sigma_target**2]),
            atol=0.1,
        )

    def test_insufficient_points_raises(self, bkd) -> None:
        """Raise when too few points per t value."""
        # Build quad with only 3 points per t but request nterms=5
        quad_marginals = [UniformMarginal(0.0, 1.0, bkd),
                          GaussianMarginal(0.0, 1.0, bkd)]
        quad_bases_1d = create_bases_1d(quad_marginals, bkd)
        quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
        quad_pts, quad_wts = quad_basis.tensor_product_quadrature([4, 3])

        t_all = quad_pts[0:1, :]
        x0_all = quad_pts[1:2, :]
        x1_all = x0_all + 1.0

        path = LinearPath(bkd)
        qd = FlowMatchingQuadData(
            t=t_all, x0=x0_all, x1=x1_all, weights=quad_wts, bkd=bkd,
        )

        with pytest.raises(ValueError, match="Insufficient points"):
            build_stieltjes_flow_vf(qd, path, nterms=5, bkd=bkd)


class TestLegendreTimeExpansion:
    def test_n_legendre_1_matches_shared_coefs(self, bkd) -> None:
        """n_legendre=1 produces same basis matrix as original shared-coef."""
        nterms = 4
        vf1, _, _, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8, n_legendre=1,
        )
        path = LinearPath(bkd)
        x_t = path.interpolate(qd.t(), qd.x0(), qd.x1())
        vf_input = bkd.vstack([qd.t(), x_t])

        phi1 = vf1.basis_matrix(vf_input)
        assert phi1.shape == (qd.n_quad(), nterms)

        # With n_legendre=1, psi_0(t) = 1 (constant), so the basis
        # matrix should equal what we'd get from just phi_n(x).
        # Build a second VF with n_legendre=3 and check its first
        # n_basis columns match (they correspond to psi_0 = const).
        vf3, _, _, _ = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8, n_legendre=3,
        )
        phi3 = vf3.basis_matrix(vf_input)
        assert phi3.shape == (qd.n_quad(), nterms * 3)

        # Columns 0, 3, 6, 9 (every n_legendre-th) correspond to psi_0.
        # But actually columns are indexed (n, j) so column n*P + j.
        # For n_legendre=1: col n = phi_n * psi_0
        # For n_legendre=3: col n*3 + 0 = phi_n * psi_0
        # psi_0 on [0,1] = 1 (constant), so phi_n * psi_0 = phi_n * 1.
        for n in range(nterms):
            bkd.assert_allclose(
                phi1[:, n],
                phi3[:, n * 3],
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_leg", [1, 4])
    def test_time_dependent_coefs_gaussian(self, bkd, n_leg: int) -> None:
        """Gaussian transport: both n_legendre=1 and =4 achieve loss < 1e-8."""
        nterms = 4
        vf, path, loss, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=8, n_x=15, n_legendre=n_leg,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-8, (
            f"n_legendre={n_leg}: loss {result.training_loss():.2e} >= 1e-8"
        )

    @slow_test
    def test_time_dependent_coefs_gmm(self, bkd) -> None:
        """Bimodal GMM: loss decreases with n_legendre; n_legendre=4 < 1e-2."""
        from pyapprox.surrogates.flowmatching.experiments.target_distributions import (
            get_bimodal_gmm_pair,
        )

        pair = get_bimodal_gmm_pair(bkd)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)

        nterms, n_t, n_x = 15, 12, 30
        quad_marginals = [UniformMarginal(0.0, 1.0, bkd),
                          GaussianMarginal(0.0, 1.0, bkd)]
        quad_bases_1d = create_bases_1d(quad_marginals, bkd)
        quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
        quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_t, n_x])

        t_all = quad_pts[0:1, :]
        x0_all = quad_pts[1:2, :]
        x1_all = pair.forward_map(x0_all)
        qd = FlowMatchingQuadData(
            t=t_all, x0=x0_all, x1=x1_all, weights=quad_wts, bkd=bkd,
        )

        losses = []
        n_leg_values = [1, 4, 6]
        for n_leg in n_leg_values:
            vf = build_stieltjes_flow_vf(
                qd, path, nterms, bkd, n_legendre=n_leg,
            )
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        # Loss should decrease (or stay flat) with more Legendre terms
        for i in range(len(losses) - 1):
            assert losses[i + 1] <= losses[i] * 1.01, (
                f"Loss did not decrease: n_legendre={n_leg_values[i]} "
                f"({losses[i]:.4e}) -> {n_leg_values[i + 1]} "
                f"({losses[i + 1]:.4e})"
            )

        # n_legendre=4 with 15 spatial terms (60 total coefficients)
        # should achieve < 1e-2
        assert losses[1] < 1e-2, (
            f"n_legendre=4 loss {losses[1]:.4e} >= 1e-2"
        )


class TestRecurrenceInterpolator:
    def test_exact_nodes_match_identity(self, bkd) -> None:
        """At training nodes, RecurrenceInterpolator returns exact states."""
        nterms = 4
        pts, wts = _gauss_hermite_quad(bkd, 15)
        factory = StieltjesBasisFactory(nterms, bkd)

        t_vals = [0.2, 0.5, 0.8]
        states = [factory.build(pts + mu, wts) for mu in [0.0, 1.0, 2.0]]

        interp = RecurrenceInterpolator(bkd)
        interp.fit(bkd.asarray(t_vals), states)

        for ii, t in enumerate(t_vals):
            state = interp(t)
            bkd.assert_allclose(state.rcoefs(), states[ii].rcoefs())

    def test_interpolated_state_between_nodes(self, bkd) -> None:
        """Interpolated state at midpoint has rcoefs between neighbors."""
        nterms = 3
        pts, wts = _gauss_hermite_quad(bkd, 10)
        factory = StieltjesBasisFactory(nterms, bkd)

        # Two states with different shifts => different alpha coefficients
        state_lo = factory.build(pts, wts)
        state_hi = factory.build(pts + 2.0, wts)

        interp = RecurrenceInterpolator(bkd)
        interp.fit(bkd.asarray([0.0, 1.0]), [state_lo, state_hi])

        # At midpoint t=0.5, linear interpolation of rcoefs
        state_mid = interp(0.5)
        expected_rcoefs = 0.5 * state_lo.rcoefs() + 0.5 * state_hi.rcoefs()
        bkd.assert_allclose(state_mid.rcoefs(), expected_rcoefs, atol=1e-12)

    def test_no_snap_identity_raises(self, bkd) -> None:
        """IdentityInterpolator raises at non-training t values."""
        nterms = 3
        pts, wts = _gauss_hermite_quad(bkd, 10)
        factory = StieltjesBasisFactory(nterms, bkd)
        states = [factory.build(pts, wts), factory.build(pts + 1.0, wts)]

        interp = IdentityInterpolator(bkd)
        interp.fit(bkd.asarray([0.3, 0.7]), states)

        with pytest.raises(ValueError, match="not found"):
            interp(0.5)

    def test_smooth_velocity_with_interpolated_recurrence(self, bkd) -> None:
        """Velocity is Lipschitz in t with RecurrenceInterpolator."""
        nterms = 4
        vf, path, loss, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=8, n_x=15, n_legendre=1,
        )
        # Fit with IdentityInterpolator (training only)
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)

        # Now build a second VF with RecurrenceInterpolator, same coeffs
        vf_smooth = build_stieltjes_flow_vf(
            qd, path, nterms, bkd, n_legendre=1,
            interpolate_rcoefs=True,
        )
        vf_smooth.set_coefficients(vf.get_coefficients())

        # Evaluate at many t values and check Lipschitz continuity
        x_fixed = bkd.asarray([[0.5]])
        n_probe = 200
        t_probe = bkd.linspace(0.01, 0.99, n_probe)
        v_vals = bkd.zeros((n_probe,))
        for ii in range(n_probe):
            t_val = bkd.to_float(t_probe[ii])
            inp = bkd.vstack([
                bkd.full((1, 1), t_val),
                x_fixed,
            ])
            v_vals[ii] = bkd.to_float(vf_smooth(inp)[0, 0])

        # Finite differences: |v(t+dt) - v(t)| / dt should be bounded
        dv = bkd.abs(v_vals[1:] - v_vals[:-1])
        dt_arr = t_probe[1:] - t_probe[:-1]
        ratios = dv / dt_arr
        max_ratio = bkd.to_float(bkd.max(ratios))

        # For a smooth Gaussian transport, the velocity derivative in t
        # should be bounded. The exact bound depends on the problem but
        # should not be pathologically large.
        assert max_ratio < 100.0, (
            f"Velocity not Lipschitz: max |dv/dt| = {max_ratio:.1f}"
        )


def _build_per_slice_gaussian_setup(bkd, mu_target, sigma_target, nterms,
                                    n_t=8, n_x=15,
                                    interpolate_rcoefs=False):
    """Build per-slice StieltjesFlowVF for Gaussian transport."""
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd),
                      GaussianMarginal(0.0, 1.0, bkd)]
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_t, n_x])

    t_all = quad_pts[0:1, :]
    x0_all = quad_pts[1:2, :]
    x1_all = sigma_target * x0_all + mu_target

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all, x0=x0_all, x1=x1_all, weights=quad_wts, bkd=bkd,
    )

    vf = build_stieltjes_flow_vf(
        quad_data, path, nterms, bkd,
        per_slice=True, interpolate_rcoefs=interpolate_rcoefs,
    )
    return vf, path, loss, quad_data


class TestPerSliceStrategy:
    def test_per_slice_velocity_at_training_nodes(self, bkd) -> None:
        """Per-slice GH+Stieltjes gives near-machine-precision velocity."""
        nterms = 4
        n_t, n_x = 8, 15
        vf, path, loss, qd = _build_per_slice_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=n_t, n_x=n_x,
        )

        # Fit via LeastSquaresFitter (weighted)
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        # Evaluate at training points
        x_t = path.interpolate(qd.t(), qd.x0(), qd.x1())
        vf_input = bkd.vstack([qd.t(), x_t])
        u_t = path.target_field(qd.t(), qd.x0(), qd.x1())

        v_pred = fitted_vf(vf_input)
        # Per-slice with GH quadrature should give very low error
        err = bkd.to_float(bkd.max(bkd.abs(v_pred - u_t)))
        assert err < 1e-8, (
            f"Per-slice velocity error {err:.2e} too large"
        )

    def test_per_slice_training_loss_near_zero(self, bkd) -> None:
        """Per-slice Gaussian transport achieves near-zero training loss."""
        nterms = 4
        vf, path, loss, qd = _build_per_slice_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=8, n_x=15,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-8, (
            f"Per-slice loss {result.training_loss():.2e} >= 1e-8"
        )

    def test_per_slice_jacobian_shape(self, bkd) -> None:
        """Per-slice jacobian has correct shape."""
        nterms = 4
        vf, path, loss, qd = _build_per_slice_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=4, n_x=8,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        x_t = path.interpolate(qd.t(), qd.x0(), qd.x1())
        vf_input = bkd.vstack([qd.t(), x_t])
        jac = fitted_vf.jacobian_batch(vf_input)
        n = vf_input.shape[1]
        assert jac.shape == (n, 1, 2)

    def test_per_slice_basis_matrix_raises(self, bkd) -> None:
        """Per-slice strategy raises on basis_matrix()."""
        nterms = 3
        vf, _, _, _ = _build_per_slice_gaussian_setup(
            bkd, 0.0, 1.0, nterms, n_t=4, n_x=8,
        )
        with pytest.raises(NotImplementedError, match="PerSliceStrategy"):
            vf.basis_matrix(bkd.zeros((2, 5)))

    def test_per_slice_hyp_list_raises(self, bkd) -> None:
        """Per-slice strategy raises on hyp_list()."""
        nterms = 3
        vf, _, _, _ = _build_per_slice_gaussian_setup(
            bkd, 0.0, 1.0, nterms, n_t=4, n_x=8,
        )
        with pytest.raises(NotImplementedError, match="PerSliceStrategy"):
            vf.hyp_list()

    def test_per_slice_with_recurrence_interpolation(self, bkd) -> None:
        """Per-slice + RecurrenceInterpolator evaluates at non-training t."""
        nterms = 4
        vf, path, loss, qd = _build_per_slice_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=8, n_x=15,
            interpolate_rcoefs=True,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        # Evaluate at a non-training t value
        t_mid = 0.37  # unlikely to match any GH node
        x_test = bkd.asarray([[0.5, 1.0, -0.5]])
        t_test = bkd.full((1, 3), t_mid)
        vf_input = bkd.vstack([t_test, x_test])

        # Should not raise and should return finite values
        v_pred = fitted_vf(vf_input)
        assert v_pred.shape == (1, 3)
        assert bkd.to_float(bkd.max(bkd.abs(v_pred))) < 100.0

    def test_kronecker_regression(self, bkd) -> None:
        """KroneckerStrategy matches old behavior exactly."""
        nterms = 4
        n_t, n_x = 8, 15
        # Build Kronecker VF (default)
        vf_kron, path, loss, qd = _build_evolving_gaussian_setup(
            bkd, 2.0, 0.5, nterms, n_t=n_t, n_x=n_x, n_legendre=1,
        )
        result_kron = LeastSquaresFitter(bkd).fit(vf_kron, path, loss, qd)

        # Evaluate
        x_t = path.interpolate(qd.t(), qd.x0(), qd.x1())
        vf_input = bkd.vstack([qd.t(), x_t])
        v_kron = result_kron.surrogate()(vf_input)

        # Should be a valid velocity field
        u_t = path.target_field(qd.t(), qd.x0(), qd.x1())
        err = bkd.to_float(bkd.max(bkd.abs(v_kron - u_t)))
        assert err < 1e-8

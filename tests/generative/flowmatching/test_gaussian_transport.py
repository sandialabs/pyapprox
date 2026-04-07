"""Integration test: unconditional Gaussian transport.

Source N(0, I_d) -> Target N(mu, Sigma) via paired coupling x1 = L*x0 + mu.

Because the coupling is linear, the map x0 -> x_t is invertible for all
t in (0,1), so the optimal VF is a deterministic function of (t, x_t)
with zero conditional variance. The VF is rational in t; a polynomial
approximation converges rapidly to zero training loss.

``TestIndependentQuadrature`` verifies that independent (x0, x1) quadrature
— where x0 and x1 are drawn from separate GH rules and paired via tensor
product — recovers the correct marginal velocity for the independent
coupling. For Gaussians, this velocity is still affine in x_t (but with
a different slope than the pushforward coupling), so a low-degree
polynomial basis recovers it exactly.

The marginal velocity depends on the coupling:

* **Pushforward** (x1 = sigma*x0 + mu): u(x_t,t) is affine with slope
  (sigma-1) / ((1-t)+t*sigma).

* **Independent** (x0 perp x1): u(x_t,t) is affine with slope
  (t*sigma^2 - (1-t)) / ((1-t)^2 + t^2*sigma^2).

Both transport N(0,1) to N(mu,sigma^2), but via different flows.

The training loss with independent pairs does NOT go to zero — it has an
irreducible conditional-variance floor. The correct diagnostic is the
velocity RMS against the analytical marginal velocity, and the KL
divergence of the ODE-transported density against the true target.
"""

import numpy as np
import pytest

from pyapprox.ode.explicit_steppers.heun import HeunStepper
from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.generative.flowmatching.cfm_loss import CFMLoss
from pyapprox.generative.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.generative.flowmatching.fitters.optimizer import (
    OptimizerFitter,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.ode_adapter import (
    integrate_flow,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
    build_independent_quad_data,
    pushforward_pair_rule,
    gauss_hermite_rule,
    gauss_legendre_rule,
    tensor_product_pair_rule,
)
from tests._helpers.markers import slow_test


def _gaussian_params(d):
    """Generate fixed Gaussian transport parameters for dimension d."""
    np.random.seed(42)
    mu_np = np.random.randn(d, 1) * 0.5
    A_np = np.random.randn(d, d) * 0.2
    Sigma_np = A_np @ A_np.T + 0.5 * np.eye(d)
    L_np = np.linalg.cholesky(Sigma_np)
    return mu_np, L_np


def _build_transport_setup(bkd, d, degree, n_per_dim=6):
    """Build VF, path, loss, quad_data for N(0,I) -> N(mu, Sigma).

    Uses paired coupling x1 = L @ x0 + mu with Gauss quadrature
    over (t, x0).

    Returns vf, path, loss, quad_data, mu, L.
    """
    mu_np, L_np = _gaussian_params(d)
    mu = bkd.array(mu_np.tolist())
    L = bkd.array(L_np.tolist())

    vf_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    vf_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    vf_bases_1d = create_bases_1d(vf_marginals, bkd)
    indices = compute_hyperbolic_indices(1 + d, degree, 1.0, bkd)
    vf_basis = OrthonormalPolynomialBasis(vf_bases_1d, bkd, indices)
    vf = BasisExpansion(vf_basis, bkd, nqoi=d)

    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_per_dim] * (1 + d))

    t_all = quad_pts[0:1, :]
    z0_all = quad_pts[1 : 1 + d, :]
    x1_all = L @ z0_all + mu

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all,
        x0=z0_all,
        x1=x1_all,
        weights=quad_wts,
        bkd=bkd,
    )
    return vf, path, loss, quad_data, mu, L


class TestGaussianTransport:
    @pytest.mark.parametrize("d", [1, 2])
    @slow_test
    def test_loss_decreases_with_degree(self, bkd, d: int) -> None:
        """Training loss decreases monotonically; degree-4 loss < 1e-4."""
        degrees = [1, 2, 3, 4]
        losses = []
        for deg in degrees:
            vf, path, loss, qd, _, _ = _build_transport_setup(bkd, d, deg)
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1], (
                f"Loss did not decrease from degree {degrees[i]} "
                f"({losses[i]:.2e}) to {degrees[i + 1]} "
                f"({losses[i + 1]:.2e})"
            )
        assert losses[-1] < 1e-4

    @pytest.mark.parametrize("d", [1])
    def test_fitters_agree(self, bkd, d: int) -> None:
        """Both fitters achieve similar loss at each degree."""
        for deg in [2, 4]:
            vf, path, loss, qd, _, _ = _build_transport_setup(bkd, d, deg)
            lstsq_loss = LeastSquaresFitter(bkd).fit(vf, path, loss, qd).training_loss()
            opt_loss = OptimizerFitter(bkd).fit(vf, path, loss, qd).training_loss()
            assert opt_loss < max(lstsq_loss * 100, 1e-4)

    @pytest.mark.parametrize("d", [1, 2])
    def test_target_moments(self, bkd, d: int) -> None:
        """ODE-integrated samples match target mean and cov."""
        vf, path, loss, qd, mu, L = _build_transport_setup(bkd, d, degree=4)
        fitted_vf = LeastSquaresFitter(bkd).fit(vf, path, loss, qd).surrogate()

        np.random.seed(123)
        nsamples = 5000
        x0_samples = bkd.array(np.random.randn(d, nsamples).tolist())
        x1_samples = integrate_flow(
            fitted_vf,
            x0_samples,
            0.0,
            1.0,
            n_steps=50,
            bkd=bkd,
            stepper_cls=HeunStepper,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = np.mean(x1_np, axis=1, keepdims=True)
        bkd.assert_allclose(
            bkd.array(sample_mean.tolist()),
            mu,
            atol=0.05,
        )

        L_np = bkd.to_numpy(L)
        Sigma_np = L_np @ L_np.T
        sample_cov = np.cov(x1_np)
        if d == 1:
            sample_cov = np.array([[sample_cov]])
        bkd.assert_allclose(
            bkd.array(sample_cov.tolist()),
            bkd.array(Sigma_np.tolist()),
            atol=0.1,
        )


# ------------------------------------------------------------------ #
#  Helpers for independent quadrature tests                           #
# ------------------------------------------------------------------ #

def _make_vf_for_independent(bkd, d, degree):
    """Create a BasisExpansion VF with input_dim = 1+d, nqoi = d."""
    marginals = [UniformMarginal(0.0, 1.0, bkd)]
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1 + d, degree, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=d)


def _gaussian_forward_map(L, mu, bkd):
    """Return forward_map: x1 = L @ x0 + mu."""

    def forward_map(x0):
        return bkd.array((L @ bkd.to_numpy(x0) + mu).tolist())

    return forward_map


def _gaussian_analytical_velocity_1d(mu_1, sigma_1, x_t, t,
                                     coupling="pushforward"):
    """Analytical marginal velocity for 1D Gaussian transport.

    Source N(0,1) -> Target N(mu_1, sigma_1^2) with linear path.
    The marginal velocity E[x1-x0 | x_t] depends on the coupling.

    Parameters
    ----------
    coupling : str
        ``"pushforward"`` — x1 = sigma_1 * x0 + mu_1 (OT map).
        ``"independent"`` — x0 ~ N(0,1), x1 ~ N(mu_1, sigma_1^2),
        drawn independently.

    Pushforward coupling
    ~~~~~~~~~~~~~~~~~~~~~~
    x_t = ((1-t) + t*sigma)*x0 + t*mu, so x0 is a function of x_t.
    u(x_t, t) = mu + (sigma-1)/((1-t)+t*sigma) * (x_t - t*mu)

    Independent coupling
    ~~~~~~~~~~~~~~~~~~~~
    (x0, x1, x_t) jointly Gaussian. Conditional expectations via
    Cov[x0,x_t] = (1-t), Cov[x1,x_t] = t*sigma^2,
    Var[x_t] = (1-t)^2 + t^2*sigma^2.
    u(x_t,t) = mu + (t*sigma^2 - (1-t))/Var[x_t] * (x_t - t*mu)
    """
    if coupling == "pushforward":
        sigma_t = (1 - t) + t * sigma_1
        return mu_1 + (sigma_1 - 1) / sigma_t * (x_t - t * mu_1)
    elif coupling == "independent":
        var_xt = (1 - t) ** 2 + t ** 2 * sigma_1 ** 2
        slope = (t * sigma_1 ** 2 - (1 - t)) / var_xt
        return mu_1 + slope * (x_t - t * mu_1)
    else:
        raise ValueError(f"Unknown coupling: {coupling!r}")


def _mc_velocity_rms_1d(fitted_vf, mu_1, sigma_1, bkd,
                        coupling="pushforward",
                        n_mc=5000, n_t_eval=50, seed=99):
    """RMS velocity error on MC test points for 1D Gaussian transport.

    Test points x_t are drawn from the marginal p_t induced by the
    specified coupling, and the analytical velocity is computed for
    that same coupling.
    """
    rng = np.random.RandomState(seed)
    x0_test = rng.randn(n_mc)
    if coupling == "pushforward":
        x1_test = sigma_1 * x0_test + mu_1
    elif coupling == "independent":
        x1_test = mu_1 + sigma_1 * rng.randn(n_mc)
    else:
        raise ValueError(f"Unknown coupling: {coupling!r}")

    t_eval = np.linspace(0.01, 0.99, n_t_eval)
    vel_errs = []
    for t_val in t_eval:
        x_t = (1 - t_val) * x0_test + t_val * x1_test
        u_true = _gaussian_analytical_velocity_1d(
            mu_1, sigma_1, x_t, t_val, coupling=coupling,
        )
        inp = bkd.vstack([
            bkd.asarray(np.full((1, n_mc), t_val)),
            bkd.asarray(x_t.reshape(1, -1)),
        ])
        v_pred = bkd.to_numpy(fitted_vf(inp)).flatten()
        vel_errs.append(np.mean((v_pred - u_true) ** 2))

    return np.sqrt(np.mean(vel_errs))


class TestIndependentQuadrature:
    """Verify independent (x0, x1) quadrature recovers Gaussian VF.

    For Gaussian transport N(0,1) -> N(mu, sigma^2), the marginal
    velocity E[x1-x0 | x_t] is affine in x_t for both pushforward
    and independent couplings. A polynomial basis of sufficient degree
    recovers it exactly.

    The training loss with independent pairs does NOT go to zero — it
    has an irreducible conditional-variance floor. The correct diagnostics
    are velocity RMS against the analytical marginal velocity (which
    depends on the coupling) and KL divergence of the ODE-transported
    density against the true Gaussian target.
    """

    def _fit_with_quad_data(self, bkd, d, degree, quad_data):
        """Fit a VF using LeastSquaresFitter with given quad data."""
        vf = _make_vf_for_independent(bkd, d, degree)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, quad_data)
        return result

    @pytest.mark.parametrize("d", [1])
    def test_pushforward_velocity_recovery(self, bkd, d: int) -> None:
        """Pushforward GH quadrature recovers the OT velocity."""
        mu_np, L_np = _gaussian_params(1)
        mu_1 = float(mu_np[0, 0])
        sigma_1 = float(L_np[0, 0])
        forward_map = _gaussian_forward_map(L_np, mu_np, bkd)

        det_rule = pushforward_pair_rule(
            gauss_hermite_rule(bkd), forward_map, bkd,
        )
        qd = build_independent_quad_data(
            gauss_legendre_rule(bkd), det_rule, 6, 10, bkd,
        )
        result = self._fit_with_quad_data(bkd, 1, 4, qd)

        rms = _mc_velocity_rms_1d(
            result.surrogate(), mu_1, sigma_1, bkd,
            coupling="pushforward",
        )
        assert rms < 1e-3, f"Pushforward velocity RMS {rms:.2e}"

    @pytest.mark.parametrize("d", [1])
    def test_independent_velocity_recovery(self, bkd, d: int) -> None:
        """Independent GH tensor-product quadrature recovers the
        independent-coupling velocity.

        The independent-coupling velocity is u(x_t,t) = mu + slope(t)*(x_t-t*mu)
        where slope(t) is rational in t (not polynomial). A degree-6
        polynomial basis approximates it well but not exactly.
        """
        mu_np, L_np = _gaussian_params(1)
        mu_1 = float(mu_np[0, 0])
        sigma_1 = float(L_np[0, 0])
        forward_map = _gaussian_forward_map(L_np, mu_np, bkd)

        ind_rule = tensor_product_pair_rule(
            gauss_hermite_rule(bkd), gauss_hermite_rule(bkd),
            forward_map, bkd,
        )
        qd = build_independent_quad_data(
            gauss_legendre_rule(bkd), ind_rule, 8, 10, bkd,
        )
        result = self._fit_with_quad_data(bkd, 1, 6, qd)

        rms = _mc_velocity_rms_1d(
            result.surrogate(), mu_1, sigma_1, bkd,
            coupling="independent",
        )
        assert rms < 0.01, f"Independent velocity RMS {rms:.2e}"

    @pytest.mark.parametrize("d", [1])
    def test_independent_velocity_rms_decreases_with_degree(
        self, bkd, d: int
    ) -> None:
        """Velocity RMS with independent quadrature decreases with degree.

        The independent-coupling velocity has a rational dependence on t,
        so velocity RMS decreases with degree but does not reach zero.
        """
        mu_np, L_np = _gaussian_params(d)
        mu_1 = float(mu_np[0, 0])
        sigma_1 = float(L_np[0, 0])
        forward_map = _gaussian_forward_map(L_np, mu_np, bkd)

        ind_rule = tensor_product_pair_rule(
            gauss_hermite_rule(bkd), gauss_hermite_rule(bkd),
            forward_map, bkd,
        )

        rms_values = []
        for degree in [1, 2, 3, 4, 5, 6]:
            qd = build_independent_quad_data(
                gauss_legendre_rule(bkd), ind_rule, 8, 10, bkd,
            )
            result = self._fit_with_quad_data(bkd, d, degree, qd)
            rms = _mc_velocity_rms_1d(
                result.surrogate(), mu_1, sigma_1, bkd,
                coupling="independent",
            )
            rms_values.append(rms)

        for i in range(len(rms_values) - 1):
            assert rms_values[i] >= rms_values[i + 1], (
                f"Velocity RMS did not decrease from degree {i + 1} "
                f"({rms_values[i]:.2e}) to degree {i + 2} "
                f"({rms_values[i + 1]:.2e})"
            )
        assert rms_values[-1] < 0.01

    @pytest.mark.parametrize("d", [1])
    def test_independent_target_moments_and_kl(self, bkd, d: int) -> None:
        """VF fitted with independent quadrature produces correct density.

        Verifies sample moments and KL divergence between the
        ODE-transported density and the true Gaussian target.
        """
        from pyapprox.generative.flowmatching.density import (
            compute_kl_divergence,
        )
        from pyapprox.surrogates.quadrature import gauss_quadrature_rule

        mu_np, L_np = _gaussian_params(d)
        mu = bkd.array(mu_np.tolist())
        forward_map = _gaussian_forward_map(L_np, mu_np, bkd)

        ind_rule = tensor_product_pair_rule(
            gauss_hermite_rule(bkd), gauss_hermite_rule(bkd),
            forward_map, bkd,
        )
        qd = build_independent_quad_data(
            gauss_legendre_rule(bkd), ind_rule, 8, 12, bkd,
        )
        result = self._fit_with_quad_data(bkd, d, 6, qd)
        fitted_vf = result.surrogate()

        # --- Sample moments ---
        np.random.seed(123)
        nsamples = 5000
        x0_samples = bkd.array(np.random.randn(d, nsamples).tolist())
        x1_samples = integrate_flow(
            fitted_vf,
            x0_samples,
            0.0,
            1.0,
            n_steps=50,
            bkd=bkd,
            stepper_cls=HeunStepper,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = np.mean(x1_np, axis=1, keepdims=True)
        bkd.assert_allclose(
            bkd.array(sample_mean.tolist()),
            mu,
            atol=0.05,
        )

        Sigma_np = L_np @ L_np.T
        sample_cov = np.cov(x1_np)
        if d == 1:
            sample_cov = np.array([[sample_cov]])
        bkd.assert_allclose(
            bkd.array(sample_cov.tolist()),
            bkd.array(Sigma_np.tolist()),
            atol=0.1,
        )

        # --- KL divergence ---
        sigma_target = float(np.sqrt(Sigma_np[0, 0]))
        mu_target = float(mu_np[0, 0])
        target_marginal = GaussianMarginal(mu_target, sigma_target, bkd)
        pts, wts = gauss_quadrature_rule(target_marginal, 50, bkd)

        def target_pdf(x):
            return target_marginal.pdf(x)

        kl = compute_kl_divergence(
            fitted_vf, target_pdf, pts, wts, bkd,
            n_steps=200, scheme="heun",
        )
        assert abs(kl) < 0.05, f"Expected KL near 0, got {kl}"


# ------------------------------------------------------------------ #
#  Sample-coupling Gaussian identities                                #
# ------------------------------------------------------------------ #

def _sample_coupling_marginal_variance(sigma_0, sigma_1, t):
    r"""Variance of x_t = (1-t)*x_0 + t*x_1 under sample coupling.

    With x_0 ~ N(0, sigma_0^2), x_1 ~ N(mu_1, sigma_1^2) independent:
        Var(x_t) = (1-t)^2 * sigma_0^2 + t^2 * sigma_1^2
    """
    return (1 - t) ** 2 * sigma_0 ** 2 + t ** 2 * sigma_1 ** 2


def _sample_coupling_conditional_cov(sigma_0, sigma_1, t):
    r"""Conditional (co)variances of (x_0, x_1) given x_t.

    Returns (Var(x_0|x_t), Var(x_1|x_t), Cov(x_0,x_1|x_t)).

    Under sample coupling (x_0 independent of x_1):
        Var(x_0 | x_t) = t^2 * sigma_0^2 * sigma_1^2 / sigma_t^2
        Var(x_1 | x_t) = (1-t)^2 * sigma_0^2 * sigma_1^2 / sigma_t^2
        Cov(x_0, x_1 | x_t) = -t*(1-t) * sigma_0^2 * sigma_1^2 / sigma_t^2
    """
    var_t = _sample_coupling_marginal_variance(sigma_0, sigma_1, t)
    s0s1 = sigma_0 ** 2 * sigma_1 ** 2
    var_x0 = t ** 2 * s0s1 / var_t
    var_x1 = (1 - t) ** 2 * s0s1 / var_t
    cov_01 = -t * (1 - t) * s0s1 / var_t
    return var_x0, var_x1, cov_01


def _sample_coupling_minimum_loss(sigma_0, sigma_1):
    r"""Minimum CFM loss for sample coupling: L* = pi * sigma_0 * sigma_1 / 2.

    This is Var(x_1 - x_0 | x_t) = sigma_0^2 * sigma_1^2 / sigma_t^2
    integrated over t in [0, 1]:
        L* = sigma_0^2 * sigma_1^2 * int_0^1 dt / sigma_t^2
           = pi * sigma_0 * sigma_1 / 2
    """
    return np.pi * sigma_0 * sigma_1 / 2.0


class TestSampleCouplingGaussianIdentities:
    """Verify analytical formulas for sample-coupling Gaussian transport.

    Source N(0, sigma_0^2) -> Target N(mu_1, sigma_1^2) with x_0 and
    x_1 drawn independently (sample coupling).

    Tests:
    1. Marginal p_t variance: Var(x_t) = (1-t)^2 sigma_0^2 + t^2 sigma_1^2
    2. Conditional variances Var(x_0|x_t), Var(x_1|x_t), Cov(x_0,x_1|x_t)
    3. Conditional velocity variance is constant: Var(x_1-x_0|x_t) = sigma_0^2 sigma_1^2 / sigma_t^2
    4. Minimum loss integral: L* = pi * sigma_0 * sigma_1 / 2
    """

    @pytest.mark.parametrize(
        "sigma_0,sigma_1,mu_1",
        [(1.0, 1.5, 0.0), (1.0, 1.5, 2.0), (0.5, 3.0, -1.0)],
    )
    def test_marginal_variance(
        self, numpy_bkd, sigma_0, sigma_1, mu_1,
    ) -> None:
        """MC estimate of Var(x_t) matches analytical formula."""
        rng = np.random.RandomState(0)
        n = 200_000
        x0 = sigma_0 * rng.randn(n)
        x1 = mu_1 + sigma_1 * rng.randn(n)

        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x_t = (1 - t) * x0 + t * x1
            var_mc = np.var(x_t)
            var_exact = _sample_coupling_marginal_variance(sigma_0, sigma_1, t)
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([var_mc]),
                numpy_bkd.asarray([var_exact]),
                rtol=5e-2,
            )

    @pytest.mark.parametrize(
        "sigma_0,sigma_1,mu_1",
        [(1.0, 1.5, 0.0), (1.0, 1.5, 2.0), (0.5, 3.0, -1.0)],
    )
    def test_marginal_mean(
        self, numpy_bkd, sigma_0, sigma_1, mu_1,
    ) -> None:
        """MC estimate of E[x_t] matches mu_t = t * mu_1."""
        rng = np.random.RandomState(0)
        n = 200_000
        x0 = sigma_0 * rng.randn(n)
        x1 = mu_1 + sigma_1 * rng.randn(n)

        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x_t = (1 - t) * x0 + t * x1
            mean_mc = np.mean(x_t)
            mean_exact = t * mu_1
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([mean_mc]),
                numpy_bkd.asarray([mean_exact]),
                atol=5e-2,
            )

    @pytest.mark.parametrize(
        "sigma_0,sigma_1,mu_1",
        [(1.0, 1.5, 0.0), (0.5, 3.0, -1.0)],
    )
    def test_conditional_variances(
        self, numpy_bkd, sigma_0, sigma_1, mu_1,
    ) -> None:
        """Residual variance after regressing on x_t matches Var(x_i|x_t).

        For jointly Gaussian (x_0, x_1, x_t), the regression residual
        x_0 - E[x_0|x_t] has variance Var(x_0|x_t), independent of x.
        Using regression residuals avoids binning artifacts.
        """
        rng = np.random.RandomState(0)
        n = 500_000
        x0 = sigma_0 * rng.randn(n)
        x1 = mu_1 + sigma_1 * rng.randn(n)

        for t in [0.2, 0.5, 0.8]:
            x_t = (1 - t) * x0 + t * x1
            var_x0_exact, var_x1_exact, cov_01_exact = (
                _sample_coupling_conditional_cov(sigma_0, sigma_1, t)
            )

            var_t = _sample_coupling_marginal_variance(sigma_0, sigma_1, t)

            # E[x_0 | x_t] = E[x_0] + Cov(x_0,x_t)/Var(x_t) * (x_t - E[x_t])
            # Cov(x_0, x_t) = (1-t) * sigma_0^2
            cov_x0_xt = (1 - t) * sigma_0 ** 2
            beta_0 = cov_x0_xt / var_t
            resid_x0 = x0 - beta_0 * (x_t - t * mu_1)
            var_x0_mc = np.var(resid_x0)

            # E[x_1 | x_t] = mu_1 + Cov(x_1,x_t)/Var(x_t) * (x_t - t*mu_1)
            # Cov(x_1, x_t) = t * sigma_1^2
            cov_x1_xt = t * sigma_1 ** 2
            beta_1 = cov_x1_xt / var_t
            resid_x1 = (x1 - mu_1) - beta_1 * (x_t - t * mu_1)
            var_x1_mc = np.var(resid_x1)

            # Cov(x_0, x_1 | x_t) from residual covariance
            cov_01_mc = np.mean(resid_x0 * resid_x1) - (
                np.mean(resid_x0) * np.mean(resid_x1)
            )

            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([var_x0_mc]),
                numpy_bkd.asarray([var_x0_exact]),
                rtol=0.02,
            )
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([var_x1_mc]),
                numpy_bkd.asarray([var_x1_exact]),
                rtol=0.02,
            )
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([cov_01_mc]),
                numpy_bkd.asarray([cov_01_exact]),
                rtol=0.05,
            )

    @pytest.mark.parametrize(
        "sigma_0,sigma_1,mu_1",
        [(1.0, 1.5, 0.0), (1.0, 1.5, 2.0), (0.5, 3.0, -1.0)],
    )
    def test_conditional_velocity_variance_constant(
        self, numpy_bkd, sigma_0, sigma_1, mu_1,
    ) -> None:
        r"""Var(x_1 - x_0 | x_t) = sigma_0^2 * sigma_1^2 / sigma_t^2.

        Uses regression residuals: vel - E[vel|x_t] has the
        conditional variance, which is independent of x.
        """
        rng = np.random.RandomState(0)
        n = 500_000
        x0 = sigma_0 * rng.randn(n)
        x1 = mu_1 + sigma_1 * rng.randn(n)
        vel = x1 - x0

        for t in [0.2, 0.5, 0.8]:
            x_t = (1 - t) * x0 + t * x1
            var_t = _sample_coupling_marginal_variance(sigma_0, sigma_1, t)
            var_vel_exact = sigma_0 ** 2 * sigma_1 ** 2 / var_t

            # E[vel | x_t] is affine in x_t (jointly Gaussian).
            # Cov(vel, x_t) = Cov(x1-x0, (1-t)*x0 + t*x1)
            #   = t*sigma_1^2 - (1-t)*sigma_0^2
            cov_vel_xt = t * sigma_1 ** 2 - (1 - t) * sigma_0 ** 2
            beta = cov_vel_xt / var_t
            mean_vel = mu_1  # E[x1-x0] = mu_1
            resid = vel - mean_vel - beta * (x_t - t * mu_1)
            var_vel_mc = np.var(resid)

            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([var_vel_mc]),
                numpy_bkd.asarray([var_vel_exact]),
                rtol=0.02,
            )

    @pytest.mark.parametrize(
        "sigma_0,sigma_1",
        [(1.0, 1.0), (1.0, 1.5), (0.5, 3.0), (2.0, 0.7)],
    )
    def test_minimum_loss_formula(
        self, numpy_bkd, sigma_0, sigma_1,
    ) -> None:
        r"""Verify L* = pi * sigma_0 * sigma_1 / 2 by numerical quadrature.

        L* = integral_0^1 sigma_0^2 * sigma_1^2 / sigma_t^2 dt
        where sigma_t^2 = (1-t)^2 * sigma_0^2 + t^2 * sigma_1^2.
        """
        from scipy.integrate import quad

        s0, s1 = sigma_0, sigma_1

        def integrand(t):
            var_t = (1 - t) ** 2 * s0 ** 2 + t ** 2 * s1 ** 2
            return s0 ** 2 * s1 ** 2 / var_t

        loss_numerical, _ = quad(integrand, 0, 1)
        loss_formula = _sample_coupling_minimum_loss(s0, s1)

        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([loss_numerical]),
            numpy_bkd.asarray([loss_formula]),
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "sigma_0,sigma_1",
        [(1.0, 1.0), (1.0, 2.0), (0.5, 3.0)],
    )
    def test_minimum_loss_mc(
        self, numpy_bkd, sigma_0, sigma_1,
    ) -> None:
        r"""MC estimate of CFM loss at the true velocity matches L*.

        Draws (x_0, x_1) independently, computes x_t, evaluates the
        true conditional velocity u(x_t, t) = E[x_1 - x_0 | x_t],
        and checks that E[|v - u|^2] over the training distribution
        equals the irreducible variance floor L*.
        """
        rng = np.random.RandomState(0)
        n = 300_000
        mu_1 = 0.0  # mean doesn't affect conditional variance

        # MC estimate of L* = E_t E_{x0,x1} |vel - u(x_t, t)|^2
        # where vel = x1 - x0 and u is the marginal velocity
        n_t = 200
        t_vals = np.linspace(0.005, 0.995, n_t)
        loss_terms = []
        for t in t_vals:
            x0 = sigma_0 * rng.randn(n)
            x1 = mu_1 + sigma_1 * rng.randn(n)
            x_t = (1 - t) * x0 + t * x1
            vel = x1 - x0

            # Analytical marginal velocity (independent coupling, mu=0)
            var_t = (1 - t) ** 2 * sigma_0 ** 2 + t ** 2 * sigma_1 ** 2
            slope = (t * sigma_1 ** 2 - (1 - t) * sigma_0 ** 2) / var_t
            u_true = slope * x_t  # mu_1 = 0 simplifies

            loss_terms.append(np.mean((vel - u_true) ** 2))

        loss_mc = np.mean(loss_terms)
        loss_exact = _sample_coupling_minimum_loss(sigma_0, sigma_1)

        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([loss_mc]),
            numpy_bkd.asarray([loss_exact]),
            rtol=0.03,
        )

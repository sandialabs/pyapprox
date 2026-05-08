"""Tests for flow matching density utilities."""

import numpy as np

from pyapprox.generative.flowmatching.density import (
    compute_flow_density,
    compute_kl_divergence,
)


class _LinearVF:
    """Linear velocity field v(t, x) = a * x for testing.

    With v = a*x the backward ODE dx/dt = -a*x has solution x(0) = x(1)*exp(-a).
    The divergence is constant: div_x(v) = a.
    So log q(x1) = log p0(x1*exp(-a)) - a = log N(0, exp(2a)) evaluated at x1.
    """

    def __init__(self, a, bkd):
        self._a = a
        self._bkd = bkd

    def __call__(self, samples):
        # samples: (2, ns) -> [t; x]
        x = samples[1:2, :]
        return self._a * x

    def jacobian_batch(self, samples):
        bkd = self._bkd
        ns = samples.shape[1]
        # jac shape (ns, 1, 2): d(v)/d(t) = 0, d(v)/d(x) = a
        jac = bkd.zeros((ns, 1, 2))
        jac[:, 0, 1] = self._a
        return jac


class TestComputeFlowDensity:
    def test_linear_vf_euler(self, numpy_bkd):
        """Euler on linear VF should converge to analytical density."""
        bkd = numpy_bkd
        a = 0.5
        vf = _LinearVF(a, bkd)

        x_grid = bkd.reshape(bkd.linspace(-4, 4, 50), (1, -1))

        # Analytical: q(x1) = N(0, exp(2a)) at x1
        sigma_q = np.exp(a)
        p_exact = bkd.asarray(
            (1.0 / (sigma_q * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * (bkd.to_numpy(x_grid) / sigma_q) ** 2)
        )

        q = compute_flow_density(vf, x_grid, bkd, n_steps=1000, scheme="euler")
        bkd.assert_allclose(q, p_exact, rtol=1e-2)

    def test_linear_vf_heun(self, numpy_bkd):
        """Heun on linear VF should converge faster than Euler."""
        bkd = numpy_bkd
        a = 0.5
        vf = _LinearVF(a, bkd)

        x_grid = bkd.reshape(bkd.linspace(-4, 4, 50), (1, -1))

        sigma_q = np.exp(a)
        p_exact = bkd.asarray(
            (1.0 / (sigma_q * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * (bkd.to_numpy(x_grid) / sigma_q) ** 2)
        )

        # Heun with fewer steps should be more accurate than Euler
        q_heun = compute_flow_density(vf, x_grid, bkd, n_steps=100, scheme="heun")
        bkd.assert_allclose(q_heun, p_exact, rtol=1e-3)

    def test_invalid_scheme_raises(self, numpy_bkd):
        bkd = numpy_bkd
        vf = _LinearVF(0.5, bkd)
        x = bkd.reshape(bkd.linspace(-1, 1, 5), (1, -1))
        import pytest
        with pytest.raises(ValueError, match="Unknown ODE scheme"):
            compute_flow_density(vf, x, bkd, scheme="rk4")


class TestComputeKLDivergence:
    def test_perfect_model_zero_kl(self, numpy_bkd):
        """If the model density matches the target, KL should be near zero."""
        bkd = numpy_bkd
        # Use a linear VF where we know the exact density
        a = 0.3
        vf = _LinearVF(a, bkd)

        sigma_q = np.exp(a)

        def target_pdf(x):
            x_np = bkd.to_numpy(x)
            p = (1.0 / (sigma_q * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (x_np / sigma_q) ** 2
            )
            return bkd.asarray(p)

        # Quadrature points for the Gaussian target
        from pyapprox.surrogates.quadrature import gauss_quadrature_rule
        from pyapprox.probability import GaussianMarginal

        marginal = GaussianMarginal(0.0, sigma_q, bkd)
        pts, wts = gauss_quadrature_rule(marginal, 50, bkd)

        kl = compute_kl_divergence(
            vf, target_pdf, pts, wts, bkd, n_steps=500, scheme="heun"
        )
        assert abs(kl) < 0.05, f"Expected KL near 0, got {kl}"

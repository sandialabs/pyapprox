"""Tests for LeastSquaresFitter and OptimizerFitter."""

import copy

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
from pyapprox.generative.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)


def _make_vf(bkd, d, degree, m=0):
    """Create a BasisExpansion VF with input_dim = 1+d+m, nqoi = d."""
    marginals = [UniformMarginal(0.0, 1.0, bkd)]
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    bases_1d = create_bases_1d(marginals, bkd)
    nvars = 1 + d + m
    indices = compute_hyperbolic_indices(nvars, degree, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=d)


def _make_linear_quad_data(bkd, d, ns, m=0):
    """Create quad data for a simple linear transport x0 -> x1 = x0 + 1."""
    import numpy as np

    np.random.seed(42)
    t_vals = np.linspace(0.05, 0.95, ns)
    t = bkd.array([t_vals.tolist()])  # (1, ns)
    x0_np = np.random.randn(d, ns)
    x0 = bkd.array(x0_np.tolist())
    x1 = bkd.array((x0_np + 1.0).tolist())
    weights = bkd.array([1.0 / ns] * ns)
    c = bkd.array(np.random.randn(m, ns).tolist()) if m > 0 else None
    return FlowMatchingQuadData(t, x0, x1, weights, bkd, c)


class TestLeastSquaresFitter:
    def test_linear_vf_recovery(self, bkd) -> None:
        """For linear transport x1=x0+1, degree-1 VF should recover exactly."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)

        assert isinstance(result, FlowMatchingFitResult)
        assert result.training_loss() < 1e-10

    def test_deep_clone_isolation(self, bkd) -> None:
        """Original VF should not be modified by fitting."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        # Save original coefficients
        orig_coef = copy.deepcopy(vf.get_coefficients())

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)

        # Original should be unchanged
        bkd.assert_allclose(vf.get_coefficients(), orig_coef, rtol=1e-12)

        # Fitted should be different (nonzero coefficients)
        fitted_vf = result.surrogate()
        fitted_coef = fitted_vf.get_coefficients()  # type: ignore[union-attr]
        norm = float(bkd.to_numpy(bkd.sum(fitted_coef * fitted_coef)))
        assert norm > 1e-6

    def test_surrogate_callable(self, bkd) -> None:
        """Result surrogate should be callable."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)

        # Should be callable
        test_input = bkd.array([[0.5], [0.0]])  # (2, 1) for (t, x)
        output = result(test_input)
        assert output.shape[0] == d
        assert output.shape[1] == 1

    def test_multidim(self, bkd) -> None:
        """Test with d=2."""
        d, ns = 2, 30
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-10

    def test_with_conditioning(self, bkd) -> None:
        """Test fitting with conditioning variables."""
        d, ns, m = 1, 20, 1
        vf = _make_vf(bkd, d, degree=1, m=m)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns, m=m)

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)
        assert isinstance(result, FlowMatchingFitResult)
        # Loss should be small (target field doesn't depend on c,
        # but the VF has enough capacity)
        assert result.training_loss() < 1e-8


class TestOptimizerFitter:
    def test_linear_vf_recovery(self, bkd) -> None:
        """Optimizer should converge to near-zero loss for linear transport."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        fitter = OptimizerFitter(bkd)
        result = fitter.fit(vf, path, loss, qd)

        assert isinstance(result, FlowMatchingFitResult)
        assert result.training_loss() < 1e-6

    def test_deep_clone_isolation(self, bkd) -> None:
        """Original VF should not be modified."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        orig_coef = copy.deepcopy(vf.get_coefficients())

        fitter = OptimizerFitter(bkd)
        fitter.fit(vf, path, loss, qd)

        bkd.assert_allclose(vf.get_coefficients(), orig_coef, rtol=1e-12)

    def test_lstsq_optimizer_agreement(self, bkd) -> None:
        """Both fitters should produce similar loss for linear transport."""
        d, ns = 1, 20
        vf = _make_vf(bkd, d, degree=1)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        qd = _make_linear_quad_data(bkd, d, ns)

        lstsq_result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        opt_result = OptimizerFitter(bkd).fit(vf, path, loss, qd)

        # Both should achieve very low loss
        assert lstsq_result.training_loss() < 1e-10
        assert opt_result.training_loss() < 1e-6

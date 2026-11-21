"""Tests for jacobian methods (w.r.t. inputs) in FunctionTrain components.

Tests validate Jacobian computations using DerivativeChecker. Tests cover:
- ConstantExpansion (zero jacobian for constants)
- FunctionTrainCore (per-core input Jacobian)
- FunctionTrain (full input Jacobian via forward-backward sweep)
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain.additive import (
    ConstantExpansion,
    create_additive_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend


class TestConstantExpansionInputJacobian:
    """Tests for ConstantExpansion jacobian w.r.t. inputs."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_jacobian_batch_shape_nqoi_1(self, bkd) -> None:
        """Test jacobian_batch shape for nqoi=1."""
        const = ConstantExpansion(1.0, bkd, nqoi=1)

        nsamples = 5
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        assert jac.shape == (nsamples, 1, 1)

    def test_jacobian_batch_shape_nqoi_3(self, bkd) -> None:
        """Test jacobian_batch shape for nqoi=3."""
        const = ConstantExpansion(2.5, bkd, nqoi=3)

        nsamples = 7
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        assert jac.shape == (nsamples, 3, 1)

    def test_jacobian_batch_is_zero(self, bkd) -> None:
        """Test that jacobian_batch returns all zeros."""
        const = ConstantExpansion(5.0, bkd, nqoi=2)

        nsamples = 4
        samples = bkd.asarray(np.random.randn(1, nsamples))
        jac = const.jacobian_batch(samples)

        expected = bkd.zeros((nsamples, 2, 1))
        bkd.assert_allclose(jac, expected)

    def test_jacobian_shape(self, bkd) -> None:
        """Test single-sample jacobian shape."""
        const = ConstantExpansion(1.0, bkd, nqoi=2)

        sample = bkd.asarray(np.random.randn(1, 1))
        jac = const.jacobian(sample)

        # Shape: (nqoi, nvars=1)
        assert jac.shape == (2, 1)

    def test_jacobian_is_zero(self, bkd) -> None:
        """Test that single-sample jacobian returns zeros."""
        const = ConstantExpansion(3.0, bkd, nqoi=1)

        sample = bkd.asarray(np.random.randn(1, 1))
        jac = const.jacobian(sample)

        expected = bkd.zeros((1, 1))
        bkd.assert_allclose(jac, expected)


class TestFunctionTrainCoreInputJacobian:
    """Tests for FunctionTrainCore.jacobian_wrt_input."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_supports_input_jacobian(self, bkd) -> None:
        """Test supports_input_jacobian method."""
        max_level = 2
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        assert core.supports_input_jacobian()

    def test_supports_input_jacobian_with_constants(self, bkd) -> None:
        """Test supports_input_jacobian with ConstantExpansion."""
        max_level = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        # Middle core structure: [[1, 0], [f, 1]]
        core = FunctionTrainCore(
            [[const_1, const_0], [exp, const_1]],
            bkd,
        )

        assert core.supports_input_jacobian()

    def test_jacobian_wrt_input_shape(self, bkd) -> None:
        """Test jacobian_wrt_input output shape."""
        max_level = 2
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
        core = FunctionTrainCore([[exp]], bkd)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_input(samples)

        # Shape: (r_left, r_right, nsamples, nqoi)
        r_left, r_right = core.ranks()
        assert jac.shape == (r_left, r_right, nsamples, 1)

    def test_jacobian_wrt_input_shape_2x2(self, bkd) -> None:
        """Test jacobian_wrt_input shape for 2x2 core."""
        max_level = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi=1)
        const_0 = ConstantExpansion(0.0, bkd, nqoi=1)
        const_1 = ConstantExpansion(1.0, bkd, nqoi=1)

        core = FunctionTrainCore(
            [[const_1, const_0], [exp, const_1]],
            bkd,
        )

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = core.jacobian_wrt_input(samples)

        assert jac.shape == (2, 2, nsamples, 1)


class TestFunctionTrainInputJacobian:
    """Tests for FunctionTrain.jacobian and jacobian_batch."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, bkd, nvars=3, max_level=2, nqoi=1):
        """Create an additive FunctionTrain for testing."""
        univariate_bases = [
            self._create_univariate_expansion(bkd, max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch output shape."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars)
        assert jac.shape == (nsamples, 1, nvars)

    def test_jacobian_batch_shape_multi_qoi(self, bkd) -> None:
        """Test jacobian_batch shape with multiple QoIs."""
        nvars = 3
        nqoi = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars)
        assert jac.shape == (nsamples, nqoi, nvars)

    def test_jacobian_shape(self, bkd) -> None:
        """Test single-sample jacobian shape."""
        nvars = 3
        nqoi = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        jac = ft.jacobian(sample)

        # Shape: (nqoi, nvars)
        assert jac.shape == (nqoi, nvars)

    def test_jacobian_single_var_ft(self, bkd) -> None:
        """Test jacobian for single variable FT (nvars=1 edge case)."""
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi)

        # Create single-core FT (nvars=1)
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        jac = ft.jacobian_batch(samples)

        # Shape: (nsamples, nqoi, nvars=1)
        assert jac.shape == (nsamples, nqoi, 1)

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Test FunctionTrain input jacobian using DerivativeChecker."""
        nvars = 3
        max_level = 2
        nqoi = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Create wrapper: x -> f(x)
        def fun(x):
            return ft(x)

        def jacobian_func(x):
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        assert float(error_ratio) < 1e-6

    def test_jacobian_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test jacobian with multiple QoIs using DerivativeChecker."""
        nvars = 2
        max_level = 2
        nqoi = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=nqoi)

        # Set random parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Create wrapper
        def fun(x):
            return ft(x)

        def jacobian_func(x):
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        # Multi-QoI requires weights
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(sample, weights=weights, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        assert float(error_ratio) < 1e-6

    def test_jacobian_single_var_derivative_checker(self, bkd) -> None:
        """Test jacobian for nvars=1 using DerivativeChecker."""
        max_level = 2
        nqoi = 1
        exp = self._create_univariate_expansion(bkd, max_level, nqoi)

        # Set random coefficients
        nterms = exp.nterms()
        exp.set_coefficients(bkd.asarray(np.random.randn(nterms, nqoi)))

        # Create single-core FT
        core = FunctionTrainCore([[exp]], bkd)
        ft = FunctionTrain([core], bkd, nqoi)

        def fun(x):
            return ft(x)

        def jacobian_func(x):
            return ft.jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi,
            nvars=1,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-1, 1, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        error_ratio = checker.error_ratio(errors[0])
        assert float(error_ratio) < 1e-6


def _additive_ft(
    bkd: Backend[Array], nvars: int, max_level: int, nqoi: int
) -> FunctionTrain[Array]:
    """Additive FunctionTrain of univariate Legendre expansions."""
    univariate_bases = []
    for _ in range(nvars):
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        univariate_bases.append(BasisExpansion(basis, bkd, nqoi=nqoi))
    return create_additive_functiontrain(univariate_bases, bkd, nqoi)


class TestFunctionTrainInputJacobianAutograd:
    """The analytical input Jacobian matches the backend's autodiff
    Jacobian, which needs the computation graph to survive."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        np.random.seed(42)

    @pytest.mark.parametrize("nvars,nqoi", [(3, 1), (2, 2)])
    def test_jacobian_matches_autodiff(
        self, torch_bkd: AutodiffBackend[Array], nvars: int, nqoi: int
    ) -> None:
        bkd = torch_bkd
        nsamples = 3
        ft = _additive_ft(bkd, nvars=nvars, max_level=2, nqoi=nqoi)
        ft = ft.with_params(bkd.asarray(np.random.randn(ft.nparams())))
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        analytical_jac = ft.jacobian_batch(samples)  # (nsamples, nqoi, nvars)

        def flat_output(x: Array) -> Array:
            # One sample's (nqoi, 1) output as a vector of its (nvars,) inputs.
            return bkd.reshape(ft(bkd.reshape(x, (nvars, 1))), (nqoi,))

        for ii in range(nsamples):
            bkd.assert_allclose(
                analytical_jac[ii],
                bkd.jacobian(flat_output, samples[:, ii]),
                rtol=1e-10,
            )

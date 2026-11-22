"""
Tests for ExactGaussianProcess with a kernel that has no analytic input
jacobian (GeneralMaternKernel): prediction on every backend, and the
autograd-fallback derivatives on autodiff backends.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels import Matern52Kernel
from pyapprox.surrogates.kernels.general_matern import GeneralMaternKernel
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _make_gp(
    bkd: Backend[Array],
    nu: float = 2.5,
    nvars: int = 1,
    lenscale: Optional[List[float]] = None,
    nugget: float = 1e-6,
) -> ExactGaussianProcess[Array]:
    """Create an ExactGP with all hyperparameters inactive."""
    if lenscale is None:
        lenscale = [1.0] * nvars
    kernel = GeneralMaternKernel(
        nu=nu,
        lenscale=lenscale,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
    )
    gp = ExactGaussianProcess(kernel, nvars=nvars, bkd=bkd, nugget=nugget)
    # GeneralMaternKernel has no analytic param jacobian, so fix all params
    gp.hyp_list().set_all_inactive()
    return gp


def _sin_data(bkd: Backend[Array], ntrain: int) -> Tuple[Array, Array]:
    """Training data y = sin(x) on ntrain equispaced points in [-2, 2]."""
    X_train = bkd.reshape(bkd.linspace(-2.0, 2.0, ntrain), (1, -1))
    return X_train, bkd.sin(X_train)


def _fit(
    bkd: Backend[Array],
    gp: ExactGaussianProcess[Array],
    X_train: Array,
    y_train: Array,
) -> ExactGaussianProcess[Array]:
    return GPMaximumLikelihoodFitter(bkd).fit(gp, X_train, y_train).surrogate()


def _gp_jacobian(gp: ExactGaussianProcess[Array]) -> Callable[[Array], Array]:
    jacobian = gp.derivatives().jacobian
    assert jacobian is not None
    return jacobian


def _gp_jacobian_batch(
    gp: ExactGaussianProcess[Array],
) -> Callable[[Array], Array]:
    jacobian_batch = gp.derivatives().jacobian_batch
    assert jacobian_batch is not None
    return jacobian_batch


def _fitted_general_and_matern52(
    bkd: Backend[Array],
) -> Tuple[ExactGaussianProcess[Array], ExactGaussianProcess[Array]]:
    """GeneralMatern(nu=2.5) and Matern52 GPs fit to the same data with the
    same fixed hyperparameters."""
    general_kernel = GeneralMaternKernel(
        nu=2.5,
        lenscale=[1.0],
        lenscale_bounds=(0.1, 10.0),
        nvars=1,
        bkd=bkd,
    )
    ref_kernel = Matern52Kernel(
        lenscale=bkd.array([1.0]),
        lenscale_bounds=(0.1, 10.0),
        nvars=1,
        bkd=bkd,
    )
    general_gp = ExactGaussianProcess(general_kernel, nvars=1, bkd=bkd)
    ref_gp = ExactGaussianProcess(ref_kernel, nvars=1, bkd=bkd)
    general_gp.hyp_list().set_all_inactive()
    ref_gp.hyp_list().set_all_inactive()

    X_train, y_train = _sin_data(bkd, 15)
    return (
        _fit(bkd, general_gp, X_train, y_train),
        _fit(bkd, ref_gp, X_train, y_train),
    )


class TestExactGPGeneralMaternPrediction:
    """Prediction with fixed hyperparameters; needs no derivatives."""

    def test_fit_and_predict(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd), X_train, y_train)
        assert gp.is_fitted()

        mean = gp.predict(bkd.reshape(bkd.linspace(-2.0, 2.0, 5), (1, -1)))
        assert mean.shape == (1, 5)

    def test_predict_interpolates_training_data(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd, nugget=1e-10), X_train, y_train)

        bkd.assert_allclose(gp.predict(X_train), y_train, rtol=1e-4, atol=1e-4)

    def test_predict_std(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd, nugget=1e-10), X_train, y_train)

        std = gp.predict_std(bkd.reshape(bkd.linspace(-2.0, 2.0, 5), (1, -1)))
        assert std.shape == (1, 5)
        assert bkd.to_bool(bkd.all(std >= 0))

    def test_std_near_zero_at_training_points(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd, nugget=1e-10), X_train, y_train)

        assert bkd.to_bool(bkd.all(gp.predict_std(X_train) < 1e-4))

    def test_neg_log_marginal_likelihood(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd), X_train, y_train)

        nlml = gp.neg_log_marginal_likelihood()
        assert bkd.to_bool(bkd.all(bkd.isfinite(nlml)))

    @pytest.mark.parametrize("nu", [3.5, 4.5])
    def test_higher_half_integer_nu(self, bkd: Backend[Array], nu: float) -> None:
        X_train, y_train = _sin_data(bkd, 20)
        gp = _fit(bkd, _make_gp(bkd, nu=nu), X_train, y_train)

        X_test, y_true = _sin_data(bkd, 10)
        mean = gp.predict(X_test)
        assert mean.shape == (1, 10)
        assert bkd.to_float(bkd.mean(bkd.abs(mean - y_true))) < 0.5

    def test_call_interface(self, bkd: Backend[Array]) -> None:
        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, _make_gp(bkd), X_train, y_train)

        # __call__ returns (nqoi, n_test)
        result = gp(bkd.reshape(bkd.linspace(-2.0, 2.0, 5), (1, -1)))
        assert result.shape == (1, 5)

    def test_multidimensional_input(self, bkd: Backend[Array]) -> None:
        nvars = 3
        X_train = bkd.asarray(np.random.randn(nvars, 30))
        y_train = bkd.asarray(np.random.randn(1, 30))
        gp = _fit(bkd, _make_gp(bkd, nvars=nvars), X_train, y_train)

        mean = gp.predict(bkd.asarray(np.random.randn(nvars, 10)))
        assert mean.shape == (1, 10)

    def test_repr(self, bkd: Backend[Array]) -> None:
        gp = _make_gp(bkd)
        repr_str = repr(gp)
        assert "ExactGaussianProcess" in repr_str
        assert "not fitted" in repr_str

        X_train, y_train = _sin_data(bkd, 10)
        gp = _fit(bkd, gp, X_train, y_train)
        assert "fitted" in repr(gp)

    def test_matches_matern52_predictions(self, bkd: Backend[Array]) -> None:
        """GeneralMatern with nu=2.5 predicts as Matern52 does."""
        general_gp, ref_gp = _fitted_general_and_matern52(bkd)

        X_test = bkd.reshape(bkd.linspace(-2.0, 2.0, 10), (1, -1))
        bkd.assert_allclose(
            general_gp.predict(X_test), ref_gp.predict(X_test), rtol=1e-6, atol=1e-6
        )

    def test_no_input_derivatives_without_autodiff(self, numpy_bkd: NumpyBkd) -> None:
        """Without an analytic kernel jacobian or an autodiff backend, the
        fitted GP declares no input derivatives."""
        X_train, y_train = _sin_data(numpy_bkd, 10)
        gp = _fit(numpy_bkd, _make_gp(numpy_bkd), X_train, y_train)

        derivatives = gp.derivatives()
        assert derivatives.jacobian is None
        assert derivatives.hvp is None


class TestExactGPGeneralMaternAutogradDerivatives:
    """Autograd-fallback input derivatives and hyperparameter optimization."""

    def test_jacobian_shape(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        bkd = autodiff_bkd
        X_train = bkd.asarray(np.random.randn(2, 20))
        y_train = bkd.asarray(np.random.randn(1, 20))
        gp = _fit(bkd, _make_gp(bkd, nvars=2), X_train, y_train)

        jac = _gp_jacobian(gp)(bkd.asarray(np.random.randn(2, 1)))
        assert jac.shape == (1, 2)

        jac_batch = _gp_jacobian_batch(gp)(bkd.asarray(np.random.randn(2, 5)))
        assert jac_batch.shape == (5, 1, 2)

    def test_first_order_only(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        """The fallback declares a jacobian but no hvp: the cdist in the
        kernel evaluation does not support second-order autograd."""
        bkd = autodiff_bkd
        X_train = bkd.asarray(np.random.randn(2, 20))
        y_train = bkd.asarray(np.random.randn(1, 20))
        gp = _fit(bkd, _make_gp(bkd, nvars=2), X_train, y_train)

        assert gp.derivatives().jacobian is not None
        assert gp.derivatives().hvp is None

    def test_matches_matern52_jacobian(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Autograd jacobian of GeneralMatern(nu=2.5) matches the analytic
        Matern52 jacobian."""
        bkd = autodiff_bkd
        general_gp, ref_gp = _fitted_general_and_matern52(bkd)

        x_test = bkd.array([[0.5]])
        bkd.assert_allclose(
            _gp_jacobian(general_gp)(x_test),
            _gp_jacobian(ref_gp)(x_test),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("nu, nvars", [(2.5, 1), (2.5, 2), (3.5, 1)])
    def test_jacobian_derivative_checker(
        self, autodiff_bkd: AutodiffBackend[Array], nu: float, nvars: int
    ) -> None:
        bkd = autodiff_bkd
        if nvars == 1:
            X_train, y_train = _sin_data(bkd, 20)
        else:
            X_train = bkd.asarray(np.random.randn(nvars, 15))
            y_train = bkd.reshape(bkd.sin(bkd.sum(X_train, axis=0)), (1, -1))
        gp = _fit(bkd, _make_gp(bkd, nu=nu, nvars=nvars), X_train, y_train)

        checker = DerivativeChecker(gp)
        x0 = bkd.full((nvars, 1), 0.5)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        jac_error = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0
        )[0]

        assert bkd.to_bool(bkd.all(bkd.isfinite(jac_error)))
        min_error = bkd.to_float(bkd.min(jac_error))
        assert min_error < 1e-6, f"Minimum jacobian relative error {min_error}"
        error_ratio = bkd.to_float(checker.error_ratio(jac_error))
        assert error_ratio < 1e-6, f"Error ratio {error_ratio:.2e}"

    @pytest.mark.parametrize("initial_lenscale", [0.3, 0.5])
    def test_fit_optimizes_hyperparameters(
        self, autodiff_bkd: AutodiffBackend[Array], initial_lenscale: float
    ) -> None:
        """GeneralMaternKernel has no parameter jacobian, so the fitter
        optimizes the hyperparameters through autograd of the loss."""
        bkd = autodiff_bkd
        kernel = GeneralMaternKernel(
            nu=2.5,
            lenscale=[initial_lenscale],
            lenscale_bounds=(0.1, 10.0),
            nvars=1,
            bkd=bkd,
        )
        assert kernel.param_derivatives().jacobian is None
        gp = ExactGaussianProcess(kernel, nvars=1, bkd=bkd)
        initial_values = bkd.copy(kernel.hyp_list().get_values())

        X_train, y_train = _sin_data(bkd, 20)
        result = GPMaximumLikelihoodFitter(bkd).fit(gp, X_train, y_train)
        gp = result.surrogate()

        assert gp.is_fitted()
        nll = result.neg_log_marginal_likelihood()
        assert bkd.to_bool(bkd.all(bkd.isfinite(nll)))
        assert not bkd.allclose(initial_values, gp.hyp_list().get_values()), (
            "Hyperparameters should change during optimization"
        )

        X_test, y_true = _sin_data(bkd, 10)
        error = bkd.to_float(bkd.mean(bkd.abs(gp.predict(X_test) - y_true)))
        assert error < 0.5, f"Mean prediction error {error} too large"

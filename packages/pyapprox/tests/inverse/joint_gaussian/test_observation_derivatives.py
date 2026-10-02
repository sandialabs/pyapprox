"""Tests for the derivatives of LinearGaussianObservation in w and nu.

Each scalar f(w, nu) -- <G, Gamma_t|z> for a random G, log det A_w and
log det A_w|t -- is checked with DerivativeChecker at an interior point,
and against torch autograd at weights that include 0 and 1.
"""

from typing import Callable, Tuple

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.inverse.joint_gaussian import JointGaussian, LinearGaussianObservation
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

_Value = Callable[[LinearGaussianObservation[Array]], Array]
_Grads = Callable[[LinearGaussianObservation[Array]], Tuple[Array, Array]]


class TestObservationDerivatives:
    """Input in R^3, prediction target B (2x3), 4 observations, correlated noise."""

    _d = 4

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(13)
        root = rng.normal(size=(3, 3))
        prior_cov = root @ root.T + 0.2 * np.eye(3)
        amat = rng.normal(size=(4, 3))
        bmat = rng.normal(size=(2, 3))
        noise_root = rng.normal(size=(4, 4))
        noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(4)
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(amat),
            bkd.zeros((3, 1)),
            bkd.asarray(prior_cov),
            [bkd.asarray(bmat)],
            bkd,
        )
        self._joint = JointGaussian(
            blocks, DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        )
        self._cov_bar = bkd.asarray(rng.normal(size=(2, 2)))
        self._noise_var = np.diag(noise_cov)

    def _observe(
        self, bkd: Backend[Array], sample: Array
    ) -> LinearGaussianObservation[Array]:
        d = self._d
        return self._joint.observe(sample[:d], sample[d:], 0)

    def _cases(self) -> list[Tuple[_Value[Array], _Grads[Array]]]:
        cov_bar = self._cov_bar
        return [
            (
                lambda obs: obs.bkd().sum(cov_bar * obs.covariance()),
                lambda obs: obs.covariance_vjp(cov_bar),
            ),
            (lambda obs: obs.logdet_zz()[0], lambda obs: obs.logdet_zz_gradient()),
            (
                lambda obs: obs.logdet_zz_given_t()[0],
                lambda obs: obs.logdet_zz_given_t_gradient(),
            ),
        ]

    def test_derivative_checker(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        d = self._d
        # nu is a free input; 0.5 with w >= 0.5 lets steps reach 0.4 while
        # every entry stays positive (the direction is a unit vector).
        w = np.array([0.6, 0.5, 0.8, 0.7])
        nu = np.full(d, 0.5)
        sample = bkd.asarray(np.concatenate([w, nu])[:, None])
        for value, grads in self._cases():

            def fun(samples: Array, value: _Value[Array] = value) -> Array:
                columns = [
                    bkd.reshape(
                        value(self._observe(bkd, samples[:, ii : ii + 1])), (1,)
                    )
                    for ii in range(samples.shape[1])
                ]
                return bkd.reshape(bkd.hstack(columns), (1, -1))

            def jac(single: Array, grads: _Grads[Array] = grads) -> Array:
                dw, dnu = grads(self._observe(bkd, single))
                return bkd.reshape(bkd.vstack([dw, dnu]), (1, 2 * d))

            function = FunctionWithJacobianFromCallable(1, 2 * d, fun, jac, bkd)
            checker = DerivativeChecker(function)
            # First-order differences: the error falls with the step until
            # rounding, so the largest step must be large for the ratio of
            # smallest to largest error to reach 1e-6.
            fd_eps = bkd.flip(bkd.logspace(-12, -0.4, 13))
            errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
            assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_against_autograd(self, torch_bkd: Backend[Array]) -> None:
        """Including w = 0 and w = 1, where finite differences cannot reach."""
        import torch

        bkd = torch_bkd
        self._setup(bkd)
        w_np = np.array([0.0, 0.4, 1.0, 0.7])
        nu_np = (1.0 - w_np) * self._noise_var
        for value, grads in self._cases():
            w = torch.tensor(w_np[:, None], requires_grad=True)
            nu = torch.tensor(nu_np[:, None], requires_grad=True)
            obs = self._joint.observe(w, nu, 0)
            auto_w, auto_nu = torch.autograd.grad(value(obs), [w, nu])
            dw, dnu = grads(self._joint.observe(w.detach(), nu.detach(), 0))
            bkd.assert_allclose(dw, auto_w, rtol=1e-10, atol=1e-12)
            bkd.assert_allclose(dnu, auto_nu, rtol=1e-10, atol=1e-12)

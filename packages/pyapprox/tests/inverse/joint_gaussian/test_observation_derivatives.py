"""Tests for the derivatives of LinearGaussianObservation in w and nu.

Each scalar f(w, nu) -- <G, Gamma_t|z> for a random G, log det A_w and
log det A_w|t -- is checked with DerivativeChecker at an interior point
and at a point on the bounds (with an inward direction), and against
torch autograd, which also shows the computation graph is intact.
"""

from typing import Callable, Optional, Tuple

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

from tests._helpers.inward_direction import inward_direction

_Value = Callable[[LinearGaussianObservation[Array]], Array]
_Grads = Callable[[LinearGaussianObservation[Array]], Tuple[Array, Array]]


class TestObservationDerivatives:
    """Input in R^3, prediction target B (2x3), 4 observations, correlated noise."""

    _d = 4

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(13)
        # Well-conditioned covariances keep rounding small enough for the
        # first-order DerivativeChecker to reach error_ratio <= 1e-6.
        root = rng.normal(size=(3, 3))
        prior_cov = root @ root.T / 3 + np.eye(3)
        amat = rng.normal(size=(4, 3))
        bmat = rng.normal(size=(2, 3))
        noise_root = rng.normal(size=(4, 4))
        noise_cov = 0.1 * (noise_root @ noise_root.T / 4 + np.eye(4))
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

    def _error_ratios(
        self,
        bkd: Backend[Array],
        sample: np.ndarray,
        max_step: float,
        direction: Optional[Array] = None,
    ) -> list[float]:
        """DerivativeChecker error ratio of each case at ``sample = (w, nu)``."""
        d = self._d
        ratios = []
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
            fd_eps = bkd.flip(bkd.logspace(-12, float(np.log10(max_step)), 13))
            errors = checker.check_derivatives(
                bkd.asarray(sample), fd_eps=fd_eps, direction=direction, verbosity=0
            )
            ratios.append(bkd.to_float(checker.error_ratio(errors[0])))
        return ratios

    def test_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Interior: every entry >= 0.5, so steps up to 0.5 stay non-negative."""
        self._setup(bkd)
        w = np.array([0.6, 0.5, 0.8, 0.7])
        sample = np.concatenate([w, np.full(self._d, 0.5)])[:, None]
        assert max(self._error_ratios(bkd, sample, 0.5)) <= 1e-6

    def test_derivative_checker_at_bounds(self, bkd: Backend[Array]) -> None:
        """w_1 = 0 with nu_1 > 0, and w_3 = 1 with nu_3 = 0.

        ``observe`` needs w >= 0 and nu >= 0, so the direction points
        inward on the entries at zero; every step up to 0.5 stays feasible.
        """
        self._setup(bkd)
        w = np.array([0.0, 0.5, 1.0, 0.5])
        nu = np.array([0.5, 0.5, 0.0, 0.5])
        sample = np.concatenate([w, nu])[:, None]
        direction = inward_direction(sample, bkd, lower=np.zeros_like(sample))
        ratios = self._error_ratios(bkd, sample, 0.5, direction)
        assert max(ratios) <= 1e-6

    def test_against_autograd(self, torch_bkd: Backend[Array]) -> None:
        """At weights including 0 and 1; also shows the computation graph
        through the backend is intact."""
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

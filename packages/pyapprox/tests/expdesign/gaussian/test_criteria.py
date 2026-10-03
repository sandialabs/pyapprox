"""Tests for AOptimal, DOptimal and ExpectedInformationGain.

Each criterion is evaluated on JointGaussian.observe with the blended
relaxation, and checked against an independent reference. Gradients are
checked through the relaxation, f(w) = criterion(w, nu(w)).
"""

from typing import Tuple

import numpy as np
import pytest
from pyapprox.expdesign.analytical import (
    lognormal_goal_mg_blocks,
    relaxed_linear_target_eig,
    relaxed_lognormal_expected_variance,
)
from pyapprox.expdesign.gaussian import (
    AOptimal,
    BlendedObservation,
    DOptimal,
    ExpectedInformationGain,
)
from pyapprox.expdesign.protocols import GaussianDesignCriterionProtocol
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.joint import JointOutputs
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import CachedMoments, DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

_Criterion = GaussianDesignCriterionProtocol[Array]


class TestCriteria:
    """xi ~ N(mu, P) in R^4, data H xi (5 x 4), targets B xi (2 x 4) and xi."""

    _nx, _nobs = 4, 5

    def _setup(self, bkd: Backend[Array], correlated: bool = True) -> None:
        rng = np.random.default_rng(23)
        root = rng.normal(size=(self._nx, self._nx))
        self._prior_cov = root @ root.T + 0.2 * np.eye(self._nx)
        self._prior_mean = 0.1 * rng.normal(size=(self._nx, 1))
        self._hmat = rng.normal(size=(self._nobs, self._nx))
        self._bmat = 0.5 * rng.normal(size=(2, self._nx))
        if correlated:
            noise_root = rng.normal(size=(self._nobs, self._nobs))
            self._noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(
                self._nobs
            )
        else:
            self._noise_cov = np.diag(rng.uniform(0.05, 0.2, self._nobs))
        self._joint = self._make_joint(
            bkd,
            DenseBlocks.from_linear_model(
                bkd.asarray(self._hmat),
                bkd.asarray(self._prior_mean),
                bkd.asarray(self._prior_cov),
                [bkd.asarray(self._bmat), bkd.eye(self._nx)],
                bkd,
            ),
        )
        self._relax = BlendedObservation.from_noise(self._joint.noise())

    def _make_joint(
        self, bkd: Backend[Array], blocks: DenseBlocks[Array]
    ) -> JointGaussian[Array]:
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self._noise_cov), bkd)
        return JointGaussian(blocks, noise)

    def _composite(
        self,
        criterion: _Criterion[Array],
        w: Array,
        index: int,
        joint: "JointGaussian[Array] | None" = None,
    ) -> Tuple[Array, Array]:
        """criterion(w, nu(w)) and its gradient in w, by the chain rule."""
        joint = self._joint if joint is None else joint
        obs = joint.observe(w, self._relax.variances(w), index)
        dw, dnu = criterion.gradient(obs)
        grad = dw + dnu * self._relax.variances_jacobian_diagonal(w)
        return criterion.value(obs), grad

    def _w(self, bkd: Backend[Array]) -> Array:
        return bkd.asarray([[0.6], [0.0], [0.3], [1.0], [0.5]])

    def test_satisfy_protocol(self, bkd: Backend[Array]) -> None:
        for criterion in (AOptimal(), DOptimal(), ExpectedInformationGain()):
            assert isinstance(criterion, GaussianDesignCriterionProtocol)

    def test_a_matches_conjugate_posterior_trace(self, bkd: Backend[Array]) -> None:
        """Diagonal noise, s = sigma: precision weighting sigma^2 / w."""
        self._setup(bkd, correlated=False)
        w_np = np.array([0.6, 0.2, 0.3, 0.9, 0.5])
        post = DenseGaussianConjugatePosterior(
            bkd.asarray(self._hmat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(np.diag(np.diag(self._noise_cov) / w_np)),
            bkd,
        )
        post.compute(bkd.zeros((self._nobs, 1)))
        value, _ = self._composite(AOptimal(), bkd.asarray(w_np[:, None]), 1)
        bkd.assert_allclose(
            value, bkd.reshape(bkd.trace(post.posterior_covariance()), (1,)), rtol=1e-10
        )

    def test_a_with_target_map(self, bkd: Backend[Array]) -> None:
        """A row vector c gives c-optimality: c Gamma c^T."""
        self._setup(bkd)
        c = bkd.asarray([[0.3, -1.2]])
        obs = self._joint.observe(self._w(bkd), self._relax.variances(self._w(bkd)), 0)
        expected = bkd.dot(bkd.dot(c, obs.covariance()), c.T)[0]
        bkd.assert_allclose(AOptimal(c).value(obs), expected, rtol=1e-12)

    def test_eig_matches_relaxed_oracle(self, bkd: Backend[Array]) -> None:
        """Value is -EIG; correlated noise, weights including 0 and 1."""
        self._setup(bkd)
        value, _ = self._composite(ExpectedInformationGain(), self._w(bkd), 0)
        eig = relaxed_linear_target_eig(
            bkd.asarray(self._bmat),
            bkd.asarray(self._hmat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            self._w(bkd),
            bkd,
        )
        bkd.assert_allclose(value, -eig[0], rtol=1e-10)

    def test_d_and_eig_rank_designs_alike(self, bkd: Backend[Array]) -> None:
        """log det Gamma_t|z = log det Gamma_tt - 2 EIG, in value and gradient."""
        self._setup(bkd)
        d_value, d_grad = self._composite(DOptimal(), self._w(bkd), 0)
        e_value, e_grad = self._composite(ExpectedInformationGain(), self._w(bkd), 0)
        _, logdet_tt = bkd.slogdet(self._joint.blocks().target_covariance(0))
        bkd.assert_allclose(d_value, logdet_tt + 2.0 * e_value, rtol=1e-10)
        bkd.assert_allclose(d_grad, 2.0 * e_grad, rtol=1e-9, atol=1e-12)

    def test_d_with_target_map(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        c = bkd.asarray([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -0.5]])
        obs = self._joint.observe(self._w(bkd), self._relax.variances(self._w(bkd)), 1)
        _, expected = bkd.slogdet(bkd.dot(bkd.dot(c, obs.covariance()), c.T))
        bkd.assert_allclose(DOptimal(c).value(obs), bkd.reshape(expected, (1,)))

    def test_d_refuses_rank_deficient_map(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        c = bkd.asarray([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        obs = self._joint.observe(self._w(bkd), self._relax.variances(self._w(bkd)), 1)
        with pytest.raises(ValueError, match="positive definite"):
            DOptimal(c).value(obs)

    def test_log_det_criteria_refuse_too_few_samples(self, bkd: Backend[Array]) -> None:
        """A sampled 4D target from 5 = n_t + 1 samples; A is unaffected."""
        self._setup(bkd)
        rng = np.random.default_rng(24)
        xi = rng.multivariate_normal(self._prior_mean[:, 0], self._prior_cov, 5).T
        outputs = JointOutputs(
            targets=(bkd.asarray(xi),), observations=bkd.asarray(self._hmat @ xi)
        )
        blocks = CachedMoments(outputs, bkd.full((1, 5), 0.2), bkd).blocks()
        assert isinstance(blocks, DenseBlocks)
        joint = self._make_joint(bkd, blocks)
        w = self._w(bkd)
        self._composite(AOptimal(), w, 0, joint)
        for criterion in (DOptimal(), ExpectedInformationGain()):
            with pytest.raises(ValueError, match="N <= n_t \\+ 1"):
                self._composite(criterion, w, 0, joint)

    def _criteria(self) -> list[Tuple[_Criterion[Array], int]]:
        return [(AOptimal(), 1), (DOptimal(), 0), (ExpectedInformationGain(), 0)]

    def test_gradients_derivative_checker(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        # Interior weights; steps up to 0.4 along a unit direction stay in [0, 1].
        sample = bkd.asarray([[0.5], [0.45], [0.55], [0.5], [0.5]])
        for criterion, index in self._criteria():

            def fun(
                samples: Array,
                criterion: _Criterion[Array] = criterion,
                index: int = index,
            ) -> Array:
                values = [
                    self._composite(criterion, samples[:, ii : ii + 1], index)[0]
                    for ii in range(samples.shape[1])
                ]
                return bkd.reshape(bkd.hstack(values), (1, -1))

            def jac(
                single: Array,
                criterion: _Criterion[Array] = criterion,
                index: int = index,
            ) -> Array:
                return bkd.reshape(
                    self._composite(criterion, single, index)[1], (1, -1)
                )

            checker = DerivativeChecker(
                FunctionWithJacobianFromCallable(1, self._nobs, fun, jac, bkd)
            )
            errors = checker.check_derivatives(
                sample, fd_eps=bkd.flip(bkd.logspace(-12, -0.4, 13))
            )
            assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_gradients_against_autograd(self, torch_bkd: Backend[Array]) -> None:
        """At weights including 0 and 1, where finite differences cannot reach."""
        import torch

        bkd = torch_bkd
        self._setup(bkd)
        for criterion, index in self._criteria():
            w = torch.tensor([[0.0], [0.4], [1.0], [0.7], [0.2]], requires_grad=True)
            value, _ = self._composite(criterion, w, index)
            (auto,) = torch.autograd.grad(value[0], [w])
            _, grad = self._composite(criterion, w.detach(), index)
            bkd.assert_allclose(grad, auto, rtol=1e-9, atol=1e-12)


class TestAgainstExistingDOptimal:
    """DOptimalLinearModelObjective computes -1/2 log det(I + A^T W A sp^2 / s^2):
    minus the EIG about xi for g = A xi, prior sp^2 I, noise s^2 I, and
    precision weighting. With diagonal noise and s = sigma the blended
    relaxation is the same distribution, so ExpectedInformationGain must
    equal it in value and gradient, and DOptimal = 2 * it + n log sp^2."""

    def test_agree(self, bkd: Backend[Array]) -> None:
        from pyapprox.expdesign.objective import DOptimalLinearModelObjective

        rng = np.random.default_rng(26)
        nobs, nparams, prior_var, noise_var = 5, 3, 0.7, 0.2
        amat = bkd.asarray(rng.normal(size=(nobs, nparams)))
        existing = DOptimalLinearModelObjective(
            amat, bkd.asarray(noise_var), bkd.asarray(prior_var), bkd
        )
        blocks = DenseBlocks.from_linear_model(
            amat,
            bkd.zeros((nparams, 1)),
            prior_var * bkd.eye(nparams),
            [bkd.eye(nparams)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(noise_var * bkd.eye(nobs), bkd)
        joint = JointGaussian(blocks, noise)
        relax = BlendedObservation.from_noise(noise)
        w = bkd.asarray([[0.6], [0.0], [0.3], [1.0], [0.5]])
        obs = joint.observe(w, relax.variances(w), 0)
        eig = ExpectedInformationGain()
        dw, dnu = eig.gradient(obs)
        grad = dw + dnu * relax.variances_jacobian_diagonal(w)
        bkd.assert_allclose(eig.value(obs), existing(w)[0], rtol=1e-10)
        bkd.assert_allclose(grad.T, existing.jacobian(w), rtol=1e-9, atol=1e-12)
        bkd.assert_allclose(
            DOptimal().value(obs),
            2.0 * existing(w)[0] + nparams * np.log(prior_var),
            rtol=1e-10,
        )


class TestLognormalOracle:
    """The moment-Gaussian criteria on the exact blocks of (exp(F xi), H xi)."""

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(25)
        nx, nobs = 4, 5
        root = rng.normal(size=(nx, nx))
        self._prior_cov = 0.1 * (root @ root.T) + 0.05 * np.eye(nx)
        self._prior_mean = 0.1 * rng.normal(size=(nx, 1))
        self._hmat = rng.normal(size=(nobs, nx))
        self._fmat = 0.5 * rng.normal(size=(2, nx))
        self._noise_cov = np.diag(rng.uniform(0.05, 0.2, nobs))
        mg = lognormal_goal_mg_blocks(
            bkd.asarray(self._hmat),
            bkd.asarray(self._fmat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd,
        )
        self._qoi_cov = bkd.to_numpy(mg.qoi_cov)
        self._qoi_obs_cov = bkd.to_numpy(mg.qoi_obs_cov)
        self._obs_cov = bkd.to_numpy(mg.obs_cov)
        mean = bkd.vstack([mg.qoi_mean, mg.obs_mean])
        cov = bkd.vstack(
            [
                bkd.hstack([mg.qoi_cov, mg.qoi_obs_cov]),
                bkd.hstack([mg.qoi_obs_cov.T, mg.obs_cov]),
            ]
        )
        blocks = DenseBlocks(mean, cov, (2,), nobs, bkd)
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self._noise_cov), bkd)
        self._joint = JointGaussian(blocks, noise)
        self._relax = BlendedObservation.from_noise(noise)

    def _explicit(self, w: np.ndarray) -> Tuple[float, float]:
        """Moment-Gaussian A and EIG from the sqrt(w) construction."""
        d = np.diag(np.sqrt(w[:, 0]))
        nu = np.diag((1.0 - w[:, 0]) * np.diag(self._noise_cov))
        syy = self._obs_cov + self._noise_cov
        szz = d @ syy @ d + nu
        cqz = self._qoi_obs_cov @ d
        post = self._qoi_cov - cqz @ np.linalg.solve(szz, cqz.T)
        syy_q = syy - self._qoi_obs_cov.T @ np.linalg.solve(
            self._qoi_cov, self._qoi_obs_cov
        )
        szz_q = d @ syy_q @ d + nu
        eig = 0.5 * (np.linalg.slogdet(szz)[1] - np.linalg.slogdet(szz_q)[1])
        return float(np.trace(post)), float(eig)

    def _value(
        self, bkd: Backend[Array], criterion: _Criterion[Array], w: np.ndarray
    ) -> float:
        weights = bkd.asarray(w)
        obs = self._joint.observe(weights, self._relax.variances(weights), 0)
        return bkd.to_float(criterion.value(obs)[0])

    @pytest.mark.parametrize(
        "w",
        [[0.6, 0.0, 0.3, 1.0, 0.5], [1.0, 0.0, 1.0, 0.0, 1.0], [1.0] * 5],
    )
    def test_matches_explicit_and_bounds_truth(
        self, bkd: Backend[Array], w: list[float]
    ) -> None:
        """Goal-A and EIG equal the closed MG values, and MG goal-A bounds the
        true expected posterior variance of q = exp(F xi) from above."""
        self._setup(bkd)
        w_np = np.array(w)[:, None]
        a_value = self._value(bkd, AOptimal(), w_np)
        eig_value = self._value(bkd, ExpectedInformationGain(), w_np)
        a_ref, eig_ref = self._explicit(w_np)
        bkd.assert_allclose(bkd.asarray([a_value]), bkd.asarray([a_ref]), rtol=1e-10)
        bkd.assert_allclose(
            bkd.asarray([eig_value]), bkd.asarray([-eig_ref]), rtol=1e-10
        )
        true_var = relaxed_lognormal_expected_variance(
            bkd.asarray(self._fmat),
            bkd.asarray(self._hmat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            bkd.asarray(w_np),
            bkd,
        )
        assert a_value >= bkd.to_float(bkd.sum(true_var)) - 1e-12

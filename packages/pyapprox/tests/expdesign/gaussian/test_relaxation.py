"""Tests for BlendedObservation composed with JointGaussian.observe.

Stacked inputs xi = (m, a): a parameter m in R^2 and a nuisance a in R^2.
Data are H xi + e with H = [A, B_a]; the target is m, so the nuisance is
marginalized exactly.
"""

from itertools import product

import numpy as np
import pytest
from pyapprox.expdesign.analytical import relaxed_linear_target_covariance
from pyapprox.expdesign.gaussian import BlendedObservation
from pyapprox.expdesign.protocols import ObservationRelaxationProtocol
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.joint import SeparateFunctions
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import CachedMoments, DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend


class TestBlendedObservation:
    _nm, _na, _nobs = 2, 2, 4

    def _setup(self, bkd: Backend[Array], correlated: bool = True) -> None:
        rng = np.random.default_rng(18)
        nx = self._nm + self._na
        root = rng.normal(size=(nx, nx))
        self._prior_cov = root @ root.T + 0.2 * np.eye(nx)
        self._hmat = rng.normal(size=(self._nobs, nx))
        if correlated:
            noise_root = rng.normal(size=(self._nobs, self._nobs))
            self._noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(
                self._nobs
            )
        else:
            self._noise_cov = np.diag(rng.uniform(0.05, 0.2, self._nobs))
        self._param_mat = np.eye(nx)[: self._nm]
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(self._hmat),
            bkd.zeros((nx, 1)),
            bkd.asarray(self._prior_cov),
            [bkd.asarray(self._param_mat)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self._noise_cov), bkd)
        self._joint = JointGaussian(blocks, noise)
        self._relax = BlendedObservation.from_noise(noise)

    def _posterior_cov(self, bkd: Backend[Array], w: Array) -> Array:
        return self._joint.observe(w, self._relax.variances(w), 0).covariance()

    def test_satisfies_protocol(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        assert isinstance(self._relax, ObservationRelaxationProtocol)

    def test_matches_relaxed_oracle(self, bkd: Backend[Array]) -> None:
        """Interior weights with a zero, correlated noise."""
        self._setup(bkd)
        w = bkd.asarray([[0.6], [0.0], [0.25], [0.9]])
        oracle = relaxed_linear_target_covariance(
            bkd.asarray(self._param_mat),
            bkd.asarray(self._hmat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            w,
            bkd,
        )
        bkd.assert_allclose(self._posterior_cov(bkd, w), oracle, rtol=1e-10)

    def test_binary_weights_equal_selection(self, bkd: Backend[Array]) -> None:
        """At every nonempty binary design, with correlated noise."""
        self._setup(bkd)
        data = bkd.zeros((self._nobs, 1))
        for bits in product([0.0, 1.0], repeat=self._nobs):
            rows = [ii for ii, bit in enumerate(bits) if bit == 1.0]
            if not rows:
                continue
            w = bkd.asarray(np.array(bits)[:, None])
            _, selected = self._joint.select(rows).condition(data[rows], 0)
            bkd.assert_allclose(
                self._posterior_cov(bkd, w), selected, rtol=1e-10, atol=1e-12
            )

    def _sampled_joint(self, bkd: Backend[Array]) -> JointGaussian[Array]:
        """Blocks of (m, G(xi)) from 30 Monte Carlo samples of a nonlinear G,
        so they are neither exact nor Gaussian."""
        nx = self._nm + self._na
        chol = bkd.asarray(np.linalg.cholesky(self._prior_cov))
        hmat = bkd.asarray(self._hmat)

        def model(z: Array) -> Array:
            xi = bkd.dot(chol, z)
            return bkd.sin(bkd.dot(hmat, xi)) + 0.2 * xi[0:1] * xi[2:3]

        evaluator = SeparateFunctions(
            FunctionFromCallable(self._nobs, nx, model, bkd),
            [
                FunctionFromCallable(
                    self._nm, nx, lambda z: bkd.dot(chol, z)[: self._nm], bkd
                )
            ],
        )
        rng = np.random.default_rng(22)
        points = bkd.asarray(rng.standard_normal((nx, 30)))
        outputs = evaluator.evaluate(points)
        blocks = CachedMoments(outputs, bkd.full((1, 30), 1.0 / 30), bkd).blocks()
        assert isinstance(blocks, DenseBlocks)
        return JointGaussian(blocks, self._joint.noise())

    def test_binary_weights_equal_selection_sampled(self, bkd: Backend[Array]) -> None:
        """A zero weight removes a sensor exactly, even for sampled blocks.

        Only the noise-free outputs are sampled; the noise and nu enter
        analytically, so the removal is algebraic. Checked for the posterior
        covariance and for the EIG, whose log nu_i terms from unselected
        sensors must cancel.
        """
        self._setup(bkd)
        joint = self._sampled_joint(bkd)
        data = bkd.zeros((self._nobs, 1))
        for bits in product([0.0, 1.0], repeat=self._nobs):
            rows = [ii for ii, bit in enumerate(bits) if bit == 1.0]
            if not rows:
                continue
            w = bkd.asarray(np.array(bits)[:, None])
            obs = joint.observe(w, self._relax.variances(w), 0)
            subset = joint.select(rows)
            _, selected_cov = subset.condition(data[rows], 0)
            bkd.assert_allclose(obs.covariance(), selected_cov, rtol=1e-10, atol=1e-12)
            ones = bkd.ones((len(rows), 1))
            sub_obs = subset.observe(ones, bkd.zeros((len(rows), 1)), 0)
            eig = obs.logdet_zz() - obs.logdet_zz_given_t()
            sub_eig = sub_obs.logdet_zz() - sub_obs.logdet_zz_given_t()
            bkd.assert_allclose(eig, sub_eig, rtol=1e-10, atol=1e-12)

    def test_matches_precision_weighting(self, bkd: Backend[Array]) -> None:
        """Diagonal noise, s = sigma: the conjugate posterior with noise
        sigma^2 / w, nuisance marginalized."""
        self._setup(bkd, correlated=False)
        w_np = np.array([0.6, 0.15, 0.4, 0.9])
        post = DenseGaussianConjugatePosterior(
            bkd.asarray(self._hmat),
            bkd.zeros((self._nm + self._na, 1)),
            bkd.asarray(self._prior_cov),
            bkd.asarray(np.diag(np.diag(self._noise_cov) / w_np)),
            bkd,
        )
        post.compute(bkd.zeros((self._nobs, 1)))
        expected = post.posterior_covariance()[: self._nm, : self._nm]
        bkd.assert_allclose(
            self._posterior_cov(bkd, bkd.asarray(w_np[:, None])), expected, rtol=1e-10
        )

    def test_loewner_monotone(self, bkd: Backend[Array]) -> None:
        """Raising any weight never increases the posterior covariance."""
        self._setup(bkd)
        rng = np.random.default_rng(19)
        for _ in range(5):
            low = rng.uniform(0.0, 1.0, (self._nobs, 1))
            high = np.minimum(low + rng.uniform(0.0, 0.5, (self._nobs, 1)), 1.0)
            diff = self._posterior_cov(bkd, bkd.asarray(low)) - self._posterior_cov(
                bkd, bkd.asarray(high)
            )
            assert bkd.to_float(bkd.min(bkd.eigvalsh(diff))) > -1e-12

    def test_conditioning_bounded_as_weight_vanishes(self, bkd: Backend[Array]) -> None:
        """cond(A_w) approaches its finite value at w_1 = 0 rather than blowing up,
        unlike precision weighting, whose noise sigma^2 / w diverges."""
        self._setup(bkd)
        syy = self._joint.obs_covariance()

        def cond(w1: float) -> float:
            w = bkd.asarray([[0.5], [w1], [0.5], [0.5]])
            a_w = w * syy + bkd.diag(self._relax.variances(w)[:, 0])
            return float(np.linalg.cond(bkd.to_numpy(a_w)))

        at_zero = cond(0.0)
        for w1 in (1e-2, 1e-4, 1e-8):
            assert cond(w1) < 2.0 * at_zero

    def _composite(
        self, bkd: Backend[Array], cov_bar: Array, w: Array
    ) -> tuple[Array, Array]:
        """f(w) = <cov_bar, Gamma_t|z(w, nu(w))> and its gradient in w."""
        nu = self._relax.variances(w)
        obs = self._joint.observe(w, nu, 0)
        value = bkd.sum(cov_bar * obs.covariance())
        dw, dnu = obs.covariance_vjp(cov_bar)
        return value, dw + dnu * self._relax.variances_jacobian_diagonal(w)

    def test_composite_derivative_checker(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        cov_bar = bkd.asarray(np.random.default_rng(20).normal(size=(2, 2)))

        def fun(samples: Array) -> Array:
            values = [
                bkd.reshape(
                    self._composite(bkd, cov_bar, samples[:, ii : ii + 1])[0], (1,)
                )
                for ii in range(samples.shape[1])
            ]
            return bkd.reshape(bkd.hstack(values), (1, -1))

        def jac(sample: Array) -> Array:
            return bkd.reshape(self._composite(bkd, cov_bar, sample)[1], (1, -1))

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(1, self._nobs, fun, jac, bkd)
        )
        # w in [0.45, 0.55] with steps up to 0.4 stays inside [0, 1]
        # because the direction is a unit vector.
        sample = bkd.asarray([[0.5], [0.45], [0.55], [0.5]])
        errors = checker.check_derivatives(
            sample, fd_eps=bkd.flip(bkd.logspace(-12, -0.4, 13))
        )
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_composite_against_autograd(self, torch_bkd: Backend[Array]) -> None:
        """At weights including 0 and 1, where finite differences cannot reach."""
        import torch

        bkd = torch_bkd
        self._setup(bkd)
        cov_bar = bkd.asarray(np.random.default_rng(21).normal(size=(2, 2)))
        w = torch.tensor([[0.0], [0.4], [1.0], [0.7]], requires_grad=True)
        value, _ = self._composite(bkd, cov_bar, w)
        (auto,) = torch.autograd.grad(value, [w])
        _, grad = self._composite(bkd, cov_bar, w.detach())
        bkd.assert_allclose(grad, auto, rtol=1e-10, atol=1e-12)

    def test_variances_and_derivative(self, bkd: Backend[Array]) -> None:
        s2 = bkd.asarray([[0.1], [0.2], [0.3]])
        relax = BlendedObservation(s2, bkd)
        w = bkd.asarray([[0.0], [0.5], [1.0]])
        bkd.assert_allclose(relax.variances(w), bkd.asarray([[0.1], [0.1], [0.0]]))
        bkd.assert_allclose(relax.variances_jacobian_diagonal(w), -s2)

    def test_rejects_nonpositive_reference_variance(self, bkd: Backend[Array]) -> None:
        """s_i^2 = 0 would leave a zero-weight sensor with no noise."""
        with pytest.raises(ValueError, match="positive"):
            BlendedObservation(bkd.asarray([[0.1], [0.0]]), bkd)

    @pytest.mark.parametrize("w", [[[0.5], [1.5]], [[-0.1], [0.5]], [[0.5]]])
    def test_rejects_bad_weights(
        self, bkd: Backend[Array], w: list[list[float]]
    ) -> None:
        """Weights outside [0, 1], or of the wrong shape."""
        relax = BlendedObservation(bkd.asarray([[0.1], [0.2]]), bkd)
        with pytest.raises(ValueError):
            relax.variances(bkd.asarray(w))

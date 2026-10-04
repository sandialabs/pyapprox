"""Tests for DesignObjective: inference, relaxation and criterion combined."""

import numpy as np
import pytest
from pyapprox.expdesign.analytical import (
    lognormal_goal_mg_blocks,
    relaxed_lognormal_expected_variance,
)
from pyapprox.expdesign.design_space import BoxBudgetDesignSpace
from pyapprox.expdesign.gaussian import (
    AOptimal,
    BlendedObservation,
    DesignObjective,
    DOptimal,
    ExpectedInformationGain,
)
from pyapprox.expdesign.protocols import (
    GaussianDesignCriterionProtocol,
    OEDObjectiveProtocol,
)
from pyapprox.expdesign.solver import RelaxedOEDSolver
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

from tests._helpers.inward_direction import inward_direction


class TestDesignObjective:
    """xi ~ N(0, P) in R^4, data H xi (5 x 4), targets B xi (2 x 4) and xi."""

    _nx, _nobs = 4, 5

    def _setup(self, bkd: Backend[Array]) -> None:
        # Well-conditioned covariances (cond 4 and 3): rounding in the
        # criteria then stays small enough for the first-order finite
        # differences of DerivativeChecker to reach error_ratio <= 1e-6.
        rng = np.random.default_rng(27)
        root = rng.normal(size=(self._nx, self._nx))
        prior_cov = root @ root.T / self._nx + np.eye(self._nx)
        hmat = rng.normal(size=(self._nobs, self._nx))
        bmat = 0.5 * rng.normal(size=(2, self._nx))
        noise_root = rng.normal(size=(self._nobs, self._nobs))
        noise_cov = 0.1 * (noise_root @ noise_root.T / self._nobs + np.eye(self._nobs))
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(hmat),
            bkd.zeros((self._nx, 1)),
            bkd.asarray(prior_cov),
            [bkd.asarray(bmat), bkd.eye(self._nx)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        self._joint = JointGaussian(blocks, noise)
        self._relax = BlendedObservation.from_noise(noise)

    def _objective(
        self, criterion: GaussianDesignCriterionProtocol[Array], index: int
    ) -> DesignObjective[Array]:
        return DesignObjective(self._joint, self._relax, criterion, index)

    def _criteria(self) -> list[tuple[GaussianDesignCriterionProtocol[Array], int]]:
        return [(AOptimal(), 1), (DOptimal(), 0), (ExpectedInformationGain(), 0)]

    def test_satisfies_protocol_and_shapes(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        objective = self._objective(AOptimal(), 0)
        assert isinstance(objective, OEDObjectiveProtocol)
        assert objective.nvars() == self._nobs
        assert objective.nqoi() == 1
        w = bkd.full((self._nobs, 1), 0.5)
        assert objective(w).shape == (1, 1)
        jacobian = objective.derivatives().jacobian
        assert jacobian is not None
        assert jacobian(w).shape == (1, self._nobs)

    def test_value_is_criterion_of_observation(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        w = bkd.asarray([[0.6], [0.0], [0.3], [1.0], [0.5]])
        nu = self._relax.variances(w)
        for criterion, index in self._criteria():
            objective = self._objective(criterion, index)
            expected = criterion.value(self._joint.observe(w, nu, index))
            bkd.assert_allclose(objective(w)[0], expected, rtol=1e-12)

    def test_batch_evaluation(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        objective = self._objective(ExpectedInformationGain(), 0)
        batch = bkd.asarray(np.random.default_rng(28).uniform(0.0, 1.0, (5, 3)))
        values = objective(batch)
        assert values.shape == (1, 3)
        for ii in range(3):
            bkd.assert_allclose(
                values[:, ii : ii + 1], objective(batch[:, ii : ii + 1]), rtol=1e-12
            )

    def test_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Checked through the objective's own Derivatives bundle."""
        self._setup(bkd)
        # All weights 0.5: steps up to 0.5 along a unit direction stay in
        # [0, 1], and the largest step sets the ratio's denominator.
        sample = bkd.full((self._nobs, 1), 0.5)
        for criterion, index in self._criteria():
            checker = DerivativeChecker(self._objective(criterion, index))
            errors = checker.check_derivatives(
                sample, fd_eps=bkd.flip(bkd.logspace(-12, float(np.log10(0.5)), 13))
            )
            assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_derivative_checker_at_bounds(self, bkd: Backend[Array]) -> None:
        """w_1 = 0 and w_4 = 1, with a direction pointing into [0, 1]^d."""
        self._setup(bkd)
        sample_np = np.array([[0.5], [0.0], [0.5], [1.0], [0.5]])
        direction = inward_direction(
            sample_np,
            bkd,
            lower=np.zeros_like(sample_np),
            upper=np.ones_like(sample_np),
        )
        for criterion, index in self._criteria():
            checker = DerivativeChecker(self._objective(criterion, index))
            errors = checker.check_derivatives(
                bkd.asarray(sample_np),
                fd_eps=bkd.flip(bkd.logspace(-12, float(np.log10(0.5)), 13)),
                direction=direction,
            )
            assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_relaxed_solver_with_budget(self, bkd: Backend[Array]) -> None:
        """The objective plugs into RelaxedOEDSolver with a budget of 2."""
        self._setup(bkd)
        objective = self._objective(AOptimal(), 1)
        space = BoxBudgetDesignSpace(self._nobs, 2.0, bkd)
        solver = RelaxedOEDSolver(objective, design_space=space)
        weights, value = solver.solve()
        bkd.assert_allclose(bkd.sum(weights, axis=0), bkd.asarray([2.0]), rtol=1e-6)
        assert bkd.all_bool(weights >= -1e-8)
        assert bkd.all_bool(weights <= 1.0 + 1e-8)
        start = bkd.to_float(objective(space.initial())[0, 0])
        assert value <= start

    def test_rejects_bad_inputs(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        wrong_size = BlendedObservation(bkd.full((3, 1), 0.1), bkd)
        with pytest.raises(ValueError):
            DesignObjective(self._joint, wrong_size, AOptimal(), 0)
        with pytest.raises(ValueError):
            self._objective(AOptimal(), 2)
        with pytest.raises(TypeError):
            DesignObjective(self._joint, self._relax, object(), 0)
        with pytest.raises(ValueError):
            self._objective(AOptimal(), 0)(bkd.ones((3, 1)))


class TestDesignObjectiveLognormalOracle:
    """The moment-Gaussian A-criterion bounds the true expected variance.

    The QoI is lognormal, ``q = exp(F xi)``, observed through linear data
    ``y = H xi + e``. Two numbers are compared at the same design:

    - the truth, ``sum_i E_y[Var(q_i | y)]``, the expected posterior
      variance of the QoI, exact here because ``log q`` is linear-Gaussian
      (``relaxed_lognormal_expected_variance``);
    - the moment-Gaussian goal-A, ``tr Gamma_q|z``, from treating ``(q, y)``
      as jointly Gaussian with the exact moments of
      ``lognormal_goal_mg_blocks`` and conditioning linearly.

    ``Gamma_q|z`` is the error covariance of the best linear estimator of
    ``q`` from the data. The conditional mean ``E[q | y]`` is the best
    estimator of any form, with mean squared error ``E_y[Var(q | y)]``. A
    linear estimator cannot beat it, so goal-A >= truth: the criterion is
    conservative. This needs exact moments only, not a Gaussian target.
    """

    @pytest.mark.parametrize(
        "w", [[0.6, 0.0, 0.3, 1.0, 0.5], [1.0, 0.0, 1.0, 0.0, 1.0]]
    )
    def test_a_bound(self, bkd: Backend[Array], w: list[float]) -> None:
        rng = np.random.default_rng(29)
        nx, nobs = 4, 5
        root = rng.normal(size=(nx, nx))
        prior_cov = 0.1 * (root @ root.T) + 0.05 * np.eye(nx)
        prior_mean = 0.1 * rng.normal(size=(nx, 1))
        hmat = rng.normal(size=(nobs, nx))
        fmat = 0.5 * rng.normal(size=(2, nx))
        noise_cov = np.diag(rng.uniform(0.05, 0.2, nobs))
        mg = lognormal_goal_mg_blocks(
            bkd.asarray(hmat),
            bkd.asarray(fmat),
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd,
        )
        mean = bkd.vstack([mg.qoi_mean, mg.obs_mean])
        cov = bkd.vstack(
            [
                bkd.hstack([mg.qoi_cov, mg.qoi_obs_cov]),
                bkd.hstack([mg.qoi_obs_cov.T, mg.obs_cov]),
            ]
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        joint = JointGaussian(DenseBlocks(mean, cov, (2,), nobs, bkd), noise)
        objective = DesignObjective(
            joint, BlendedObservation.from_noise(noise), AOptimal(), 0
        )
        weights = bkd.asarray(np.array(w)[:, None])
        true_var = relaxed_lognormal_expected_variance(
            bkd.asarray(fmat),
            bkd.asarray(hmat),
            bkd.asarray(prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(noise_cov),
            weights,
            bkd,
        )
        mg_value = bkd.to_float(objective(weights)[0, 0])
        assert mg_value >= bkd.to_float(bkd.sum(true_var)) - 1e-12

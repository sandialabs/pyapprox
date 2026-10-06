"""Tests for GroupedDesign and ParameterizedObjective.

Six observations: three sensors at two times, observation ``s + 3 t`` for
sensor ``s`` and time ``t``. Grouping by sensor gives three design
variables, each switching one sensor on at both times.
"""

import pickle

import numpy as np
import pytest

from pyapprox.expdesign.design_space import (
    BoxBudgetDesignSpace,
    GroupedDesign,
    ParameterizedObjective,
)
from pyapprox.expdesign.gaussian import (
    AOptimal,
    BlendedObservation,
    DesignObjective,
    ExpectedInformationGain,
)
from pyapprox.expdesign.protocols import OEDObjectiveProtocol
from pyapprox.expdesign.solver import RelaxedOEDSolver
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.inward_direction import inward_direction

_BY_SENSOR = [[0, 3], [1, 4], [2, 5]]


class TestGroupedDesign:
    def test_matrix_by_sensor(self, bkd: Backend[Array]) -> None:
        design = GroupedDesign(_BY_SENSOR, 6, bkd)
        expected = np.zeros((6, 3))
        for jj, group in enumerate(_BY_SENSOR):
            expected[group, jj] = 1.0
        bkd.assert_allclose(design.matrix(), bkd.asarray(expected))
        v = bkd.asarray([[1.0], [0.0], [0.5]])
        bkd.assert_allclose(
            design(v), bkd.asarray([[1.0], [0.0], [0.5], [1.0], [0.0], [0.5]])
        )
        assert design.nvars() == 3
        assert design.nqoi() == 6

    def test_identity_and_uncovered_observations(self, bkd: Backend[Array]) -> None:
        identity = GroupedDesign([[ii] for ii in range(4)], 4, bkd)
        bkd.assert_allclose(identity.matrix(), bkd.eye(4))
        partial = GroupedDesign([[0, 2]], 4, bkd)
        bkd.assert_allclose(
            partial(bkd.ones((1, 1))), bkd.asarray([[1.0], [0.0], [1.0], [0.0]])
        )

    @pytest.mark.parametrize(
        "groups", [[], [[0], []], [[0, 6]], [[0, 1], [1, 2]], [[0, 0]]]
    )
    def test_rejects_bad_groups(
        self, bkd: Backend[Array], groups: list[list[int]]
    ) -> None:
        with pytest.raises(ValueError):
            GroupedDesign(groups, 6, bkd)


class TestParameterizedObjective:
    """A linear-Gaussian model with 6 observations, grouped by sensor."""

    def _objective(
        self, bkd: Backend[Array], criterion: object = None
    ) -> DesignObjective[Array]:
        rng = np.random.default_rng(30)
        root = rng.normal(size=(3, 3))
        prior_cov = root @ root.T / 3 + np.eye(3)
        noise_root = rng.normal(size=(6, 6))
        noise_cov = 0.1 * (noise_root @ noise_root.T / 6 + np.eye(6))
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(rng.normal(size=(6, 3))),
            bkd.zeros((3, 1)),
            bkd.asarray(prior_cov),
            [bkd.eye(3)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        joint = JointGaussian(blocks, noise)
        chosen = AOptimal() if criterion is None else criterion
        return DesignObjective(joint, BlendedObservation.from_noise(noise), chosen, 0)

    def test_satisfies_protocol_and_composes(self, bkd: Backend[Array]) -> None:
        objective = self._objective(bkd)
        design = GroupedDesign(_BY_SENSOR, 6, bkd)
        grouped = ParameterizedObjective(objective, design)
        assert isinstance(grouped, OEDObjectiveProtocol)
        assert grouped.nvars() == 3
        v = bkd.asarray([[0.7], [0.0], [1.0]])
        bkd.assert_allclose(grouped(v), objective(design(v)), rtol=1e-12)

    def test_identity_grouping_is_the_objective(self, bkd: Backend[Array]) -> None:
        objective = self._objective(bkd)
        identity = ParameterizedObjective(
            objective, GroupedDesign([[ii] for ii in range(6)], 6, bkd)
        )
        w = bkd.asarray([[0.7], [0.0], [1.0], [0.2], [0.5], [0.9]])
        bkd.assert_allclose(identity(w), objective(w), rtol=1e-12)
        jacobian = identity.derivatives().jacobian
        objective_jacobian = objective.derivatives().jacobian
        assert jacobian is not None and objective_jacobian is not None
        bkd.assert_allclose(jacobian(w), objective_jacobian(w), rtol=1e-12)

    @pytest.mark.parametrize("criterion", [AOptimal(), ExpectedInformationGain()])
    def test_jacobian_interior(self, bkd: Backend[Array], criterion: object) -> None:
        """Central differences at v = 0.5, with the V-shape check."""
        grouped = ParameterizedObjective(
            self._objective(bkd, criterion), GroupedDesign(_BY_SENSOR, 6, bkd)
        )
        checker = DerivativeChecker(grouped)
        steps = bkd.flip(bkd.logspace(-12, float(np.log10(0.5)), 23))
        errors = checker.check_derivatives(
            bkd.full((3, 1), 0.5), fd_eps=steps, central=True
        )[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6
        assert checker.check_v_shape(errors, steps, central=True).passed

    @pytest.mark.parametrize("criterion", [AOptimal(), ExpectedInformationGain()])
    def test_jacobian_at_bounds(self, bkd: Backend[Array], criterion: object) -> None:
        """Forward differences at v = (0, 0.5, 1) along an inward direction."""
        grouped = ParameterizedObjective(
            self._objective(bkd, criterion), GroupedDesign(_BY_SENSOR, 6, bkd)
        )
        sample = np.array([[0.0], [0.5], [1.0]])
        direction = inward_direction(
            sample, bkd, lower=np.zeros_like(sample), upper=np.ones_like(sample)
        )
        checker = DerivativeChecker(grouped)
        steps = bkd.flip(bkd.logspace(-12, float(np.log10(0.5)), 23))
        errors = checker.check_derivatives(
            bkd.asarray(sample), fd_eps=steps, direction=direction
        )[0]
        assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6
        assert checker.check_v_shape(errors, steps).passed

    def test_map_without_jacobian_gives_empty_bundle(self, bkd: Backend[Array]) -> None:
        matrix = GroupedDesign(_BY_SENSOR, 6, bkd).matrix()
        no_jacobian = FunctionFromCallable(6, 3, lambda v: bkd.dot(matrix, v), bkd)
        grouped = ParameterizedObjective(self._objective(bkd), no_jacobian)
        assert grouped.derivatives().jacobian is None

    def test_relaxed_solver_over_groups(self, bkd: Backend[Array]) -> None:
        """Choose one sensor's worth of weight across three sensor groups."""
        grouped = ParameterizedObjective(
            self._objective(bkd), GroupedDesign(_BY_SENSOR, 6, bkd)
        )
        space = BoxBudgetDesignSpace(3, 1.0, bkd)
        v, value = RelaxedOEDSolver(grouped, design_space=space).solve()
        bkd.assert_allclose(bkd.sum(v, axis=0), bkd.asarray([1.0]), rtol=1e-6)
        assert value <= bkd.to_float(grouped(space.initial())[0, 0])

    def test_picklable(self, bkd: Backend[Array]) -> None:
        """Bundles hold bound methods or small classes, never closures."""
        design = GroupedDesign(_BY_SENSOR, 6, bkd)
        objective = self._objective(bkd)
        grouped = ParameterizedObjective(objective, design)
        v = bkd.asarray([[0.7], [0.0], [1.0]])
        for original, point in ((design, v), (objective, design(v)), (grouped, v)):
            restored = pickle.loads(pickle.dumps(original))
            bkd.assert_allclose(restored(point), original(point), rtol=1e-12)
        restored_grouped = pickle.loads(pickle.dumps(grouped))
        jacobian = restored_grouped.derivatives().jacobian
        original_jacobian = grouped.derivatives().jacobian
        assert jacobian is not None and original_jacobian is not None
        bkd.assert_allclose(jacobian(v), original_jacobian(v), rtol=1e-12)

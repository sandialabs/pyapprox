"""Tests for the reference design criteria.

Every model is defined inline. Linear-Gaussian models give closed forms at
any weights, including correlated noise and nuisances; nonlinear models
are checked against the moment-Gaussian criteria, which the references
must bound. Inner rules are trapezoid rules with prior-density weights,
dense enough to resolve the likelihood; outer rules are Gauss-Hermite in
every input and noise variable.
"""

import math
from typing import Callable, Generic, List, Tuple

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import NDArray

from pyapprox.expdesign.analytical import (
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
)
from pyapprox.expdesign.design_space import BinaryDesignSubsetObjective
from pyapprox.expdesign.gaussian import (
    AOptimal,
    BlendedObservation,
    DesignObjective,
    ExpectedInformationGain,
)
from pyapprox.expdesign.protocols import (
    GaussianDesignCriterionProtocol,
    OEDObjectiveProtocol,
)
from pyapprox.expdesign.reference import (
    ReferenceAOptimal,
    ReferenceExpectedInformationGain,
)
from pyapprox.expdesign.solver import ExhaustiveSubsetSolver
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.joint import InputTarget, SeparateFunctions
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend

_Rule1D = Tuple[NDArray[np.float64], NDArray[np.float64]]
_NOISE_VAR = 0.05


class _ArrayRule(Generic[Array]):
    """A fixed rule from points (nvars, n) and weights (n,)."""

    def __init__(
        self,
        points: NDArray[np.float64],
        weights: NDArray[np.float64],
        bkd: Backend[Array],
    ) -> None:
        self._points = bkd.asarray(points)
        self._weights = bkd.asarray(weights)
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return int(self._points.shape[0])

    def __call__(self) -> Tuple[Array, Array]:
        return self._points, self._weights


def _trapezoid(npoints: int, std: float) -> _Rule1D:
    """Trapezoid nodes on [-8 std, 8 std] with N(0, std^2) density weights."""
    nodes = np.linspace(-8.0 * std, 8.0 * std, npoints)
    weights = np.exp(-(nodes**2) / (2.0 * std**2))
    return nodes, weights / weights.sum()


def _hermite(npoints: int, std: float = 1.0) -> _Rule1D:
    """Gauss-Hermite nodes for N(0, std^2)."""
    nodes, weights = hermegauss(npoints)
    return std * nodes, weights / weights.sum()


def _tensor(bkd: Backend[Array], *rules: _Rule1D) -> _ArrayRule[Array]:
    nodes = np.meshgrid(*[rule[0] for rule in rules], indexing="ij")
    weights = np.meshgrid(*[rule[1] for rule in rules], indexing="ij")
    points = np.vstack([grid.ravel() for grid in nodes])
    return _ArrayRule(points, np.prod([grid.ravel() for grid in weights], axis=0), bkd)


def _evaluator(
    bkd: Backend[Array],
    nvars: int,
    nobs: int,
    obs: Callable[[Array], Array],
    target_rows: List[int],
) -> SeparateFunctions[Array]:
    return SeparateFunctions(
        FunctionFromCallable(nobs, nvars, obs, bkd),
        [InputTarget(nvars, bkd, rows=target_rows)],
    )


def _noise(
    bkd: Backend[Array], cov: NDArray[np.float64]
) -> DenseCholeskyCovarianceOperator[Array]:
    return DenseCholeskyCovarianceOperator(bkd.asarray(cov), bkd)


def _scalar_model(
    bkd: Backend[Array], alpha: float, beta: float
) -> Tuple[SeparateFunctions[Array], _ArrayRule[Array], _ArrayRule[Array]]:
    """``y = beta m + alpha m^2 + e`` with ``m ~ N(0, 1)``."""
    evaluator = _evaluator(bkd, 1, 1, lambda x: beta * x + alpha * x**2, [0])
    nodes, weights = _trapezoid(1000, 1.0)
    inner = _ArrayRule(nodes[None, :], weights, bkd)
    return evaluator, inner, _tensor(bkd, _hermite(40), _hermite(40))


def _assert_value(
    bkd: Backend[Array],
    actual: float,
    expected: float,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    bkd.assert_allclose(
        bkd.asarray([actual]), bkd.asarray([expected]), rtol=rtol, atol=atol
    )


class TestLinearScalar:
    """``y = m + e``: posterior variance ``s/(s + w)``, EIG ``log(1 + w/s)/2``."""

    @pytest.mark.parametrize("w", [1.0, 0.4, 0.0])
    def test_closed_forms(self, bkd: Backend[Array], w: float) -> None:
        evaluator, inner, outer = _scalar_model(bkd, 0.0, 1.0)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        relaxation = BlendedObservation.from_noise(noise)
        a_opt = ReferenceAOptimal(evaluator, inner, outer, noise, relaxation, 0)
        eig = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, relaxation
        )
        assert isinstance(a_opt, OEDObjectiveProtocol)
        assert isinstance(eig, OEDObjectiveProtocol)
        weights = bkd.asarray([[w]])
        _assert_value(
            bkd,
            bkd.to_float(a_opt(weights)[0, 0]),
            _NOISE_VAR / (_NOISE_VAR + w),
            rtol=1e-10,
        )
        _assert_value(
            bkd,
            eig.expected_information_gain(weights),
            0.5 * math.log(1.0 + w / _NOISE_VAR),
            rtol=1e-10,
            atol=1e-12,
        )
        _assert_value(
            bkd,
            bkd.to_float(eig(weights)[0, 0]),
            -eig.expected_information_gain(weights),
        )

    def test_batching_does_not_change_values(self, bkd: Backend[Array]) -> None:
        evaluator, inner, outer = _scalar_model(bkd, 0.3, 1.0)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        relaxation = BlendedObservation.from_noise(noise)
        weights = bkd.asarray([[0.6]])
        values = []
        for max_entries in (2**22, 5000):
            a_opt = ReferenceAOptimal(
                evaluator, inner, outer, noise, relaxation, 0, max_entries=max_entries
            )
            eig = ReferenceExpectedInformationGain(
                evaluator, inner, outer, noise, relaxation, max_entries=max_entries
            )
            values.append(
                [
                    bkd.to_float(a_opt(weights)[0, 0]),
                    eig.expected_information_gain(weights),
                ]
            )
        bkd.assert_allclose(bkd.asarray(values[1]), bkd.asarray(values[0]), rtol=1e-12)


class TestLinearCorrelatedNoise:
    """One parameter, two correlated observations, against the relaxed
    linear-Gaussian closed forms, also for a prediction ``q = 2m``."""

    @pytest.mark.parametrize("w", [[1.0, 1.0], [0.7, 0.0], [0.3, 0.9]])
    def test_closed_forms(self, bkd: Backend[Array], w: List[float]) -> None:
        obs_mat = np.array([[1.0], [0.5]])
        noise_cov = 0.1 * np.array([[1.0, 0.6], [0.6, 1.0]])
        evaluator = _evaluator(bkd, 1, 2, lambda x: bkd.asarray(obs_mat) @ x, [0])
        nodes, weights = _trapezoid(1000, 1.0)
        inner = _ArrayRule(nodes[None, :], weights, bkd)
        outer = _tensor(bkd, _hermite(12), _hermite(12), _hermite(12))
        noise = _noise(bkd, noise_cov)
        relaxation = BlendedObservation.from_noise(noise)
        design = bkd.asarray(np.array(w)[:, None])
        prediction = bkd.asarray([[2.0]])
        a_opt = ReferenceAOptimal(
            evaluator, inner, outer, noise, relaxation, 0, target_map=prediction
        )
        eig = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, relaxation
        )
        args = (bkd.asarray(obs_mat), bkd.eye(1), bkd.asarray(noise_cov), design, bkd)
        expected_cov = relaxed_linear_target_covariance(prediction, *args)
        expected_eig = relaxed_linear_target_eig(bkd.eye(1), *args)
        _assert_value(
            bkd,
            bkd.to_float(a_opt(design)[0, 0]),
            bkd.to_float(bkd.trace(expected_cov)),
            rtol=1e-8,
        )
        _assert_value(
            bkd,
            eig.expected_information_gain(design),
            bkd.to_float(bkd.reshape(expected_eig, (-1,))[0]),
            rtol=1e-8,
        )


class TestNuisance:
    """``y = m + a + e`` with ``a ~ N(0, tau^2)``: the A-criterion of ``m``
    marginalizes ``a``; the EIG is about ``m`` with a nuisance rule and
    about ``(m, a)`` without."""

    def test_closed_forms(self, bkd: Backend[Array]) -> None:
        tau = math.sqrt(0.5)
        evaluator = _evaluator(bkd, 2, 1, lambda x: x[0:1] + x[1:2], [0])
        inner = _tensor(bkd, _trapezoid(100, 1.0), _trapezoid(100, tau))
        outer = _tensor(bkd, _hermite(10), _hermite(10, tau), _hermite(10))
        nodes, weights = _trapezoid(100, tau)
        nuisance_rule = _ArrayRule(nodes[None, :], weights, bkd)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        relaxation = BlendedObservation.from_noise(noise)
        design = bkd.asarray([[1.0]])
        a_opt = ReferenceAOptimal(evaluator, inner, outer, noise, relaxation, 0)
        eig_all = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, relaxation
        )
        eig_param = ReferenceExpectedInformationGain(
            evaluator,
            inner,
            outer,
            noise,
            relaxation,
            nuisance_rule=nuisance_rule,
            nuisance_indices=[1],
        )
        total = 1.0 + tau**2 + _NOISE_VAR
        _assert_value(
            bkd, bkd.to_float(a_opt(design)[0, 0]), 1.0 - 1.0 / total, rtol=1e-10
        )
        _assert_value(
            bkd,
            eig_all.expected_information_gain(design),
            0.5 * math.log(1.0 + (1.0 + tau**2) / _NOISE_VAR),
            rtol=1e-10,
        )
        _assert_value(
            bkd,
            eig_param.expected_information_gain(design),
            0.5 * math.log(1.0 + 1.0 / (tau**2 + _NOISE_VAR)),
            rtol=1e-10,
        )


def _moment_gaussian(
    bkd: Backend[Array],
    evaluator: SeparateFunctions[Array],
    inner: _ArrayRule[Array],
    noise: DenseCholeskyCovarianceOperator[Array],
    criterion: GaussianDesignCriterionProtocol[Array],
) -> DesignObjective[Array]:
    """The moment-Gaussian objective with the inner rule's moments."""
    points, weights = inner()
    outputs = evaluator.evaluate(points)
    stacked = bkd.vstack([outputs.targets[0], outputs.observations])
    mean = bkd.dot(stacked, weights)[:, None]
    centered = stacked - mean
    cov = bkd.dot(centered * weights[None, :], centered.T)
    blocks = DenseBlocks(
        mean, cov, (outputs.targets[0].shape[0],), evaluator.nobs(), bkd
    )
    return DesignObjective(
        JointGaussian(blocks, noise),
        BlendedObservation.from_noise(noise),
        criterion,
        0,
    )


class TestAgainstMomentGaussian:
    """The moment-Gaussian A bounds the reference from above, and its EIG
    bounds it from below for a Gaussian target."""

    @pytest.mark.parametrize("alpha, beta", [(0.3, 1.0), (1.0, 0.5), (1.0, 0.0)])
    @pytest.mark.parametrize("w", [1.0, 0.5])
    def test_bounds(
        self, bkd: Backend[Array], alpha: float, beta: float, w: float
    ) -> None:
        evaluator, inner, outer = _scalar_model(bkd, alpha, beta)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        relaxation = BlendedObservation.from_noise(noise)
        design = bkd.asarray([[w]])
        a_ref = bkd.to_float(
            ReferenceAOptimal(evaluator, inner, outer, noise, relaxation, 0)(design)[
                0, 0
            ]
        )
        eig_ref = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, relaxation
        ).expected_information_gain(design)
        a_mg = bkd.to_float(
            _moment_gaussian(bkd, evaluator, inner, noise, AOptimal())(design)[0, 0]
        )
        eig_mg = -bkd.to_float(
            _moment_gaussian(bkd, evaluator, inner, noise, ExpectedInformationGain())(
                design
            )[0, 0]
        )
        assert a_mg >= a_ref - 1e-8
        assert eig_mg <= eig_ref + 1e-8

    def test_even_model_is_invisible_to_two_moments(self, bkd: Backend[Array]) -> None:
        """``y = m^2 + e``: Cov(m, y) = 0, so the moment-Gaussian EIG is 0,
        while the data still carry information about ``|m|``."""
        evaluator, inner, outer = _scalar_model(bkd, 1.0, 0.0)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        design = bkd.asarray([[1.0]])
        eig_mg = -bkd.to_float(
            _moment_gaussian(bkd, evaluator, inner, noise, ExpectedInformationGain())(
                design
            )[0, 0]
        )
        eig_ref = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, BlendedObservation.from_noise(noise)
        ).expected_information_gain(design)
        _assert_value(bkd, eig_mg, 0.0, atol=1e-10)
        assert eig_ref > 1.0

    def test_two_sensor_designs_disagree(self, bkd: Backend[Array]) -> None:
        """Sensor 0 sees ``m^2``, sensor 1 sees ``0.3 m``. The reference
        chooses the even sensor; two moments see only the linear one."""
        evaluator = _evaluator(bkd, 1, 2, lambda x: bkd.vstack([x**2, 0.3 * x]), [0])
        nodes, weights = _trapezoid(1000, 1.0)
        inner = _ArrayRule(nodes[None, :], weights, bkd)
        outer = _tensor(bkd, _hermite(30), _hermite(12), _hermite(12))
        noise = _noise(bkd, _NOISE_VAR * np.eye(2))
        reference = ReferenceExpectedInformationGain(
            evaluator, inner, outer, noise, BlendedObservation.from_noise(noise)
        )
        moment_gaussian = _moment_gaussian(
            bkd, evaluator, inner, noise, ExpectedInformationGain()
        )
        ref_best = ExhaustiveSubsetSolver(BinaryDesignSubsetObjective(reference)).solve(
            1
        )
        mg_best = ExhaustiveSubsetSolver(
            BinaryDesignSubsetObjective(moment_gaussian)
        ).solve(1)
        assert ref_best.subset == (0,)
        assert mg_best.subset == (1,)


class TestValidation:
    def _parts(
        self, bkd: Backend[Array]
    ) -> Tuple[
        SeparateFunctions[Array],
        _ArrayRule[Array],
        _ArrayRule[Array],
        DenseCholeskyCovarianceOperator[Array],
        BlendedObservation[Array],
    ]:
        evaluator, inner, outer = _scalar_model(bkd, 0.0, 1.0)
        noise = _noise(bkd, np.array([[_NOISE_VAR]]))
        return evaluator, inner, outer, noise, BlendedObservation.from_noise(noise)

    def test_rejects_outer_rule_without_noise_dimensions(
        self, bkd: Backend[Array]
    ) -> None:
        evaluator, inner, _, noise, relaxation = self._parts(bkd)
        with pytest.raises(ValueError, match="outer_rule"):
            ReferenceAOptimal(evaluator, inner, inner, noise, relaxation, 0)

    def test_rejects_bad_target(self, bkd: Backend[Array]) -> None:
        evaluator, inner, outer, noise, relaxation = self._parts(bkd)
        with pytest.raises(ValueError, match="index"):
            ReferenceAOptimal(evaluator, inner, outer, noise, relaxation, 1)
        with pytest.raises(ValueError, match="target_map"):
            ReferenceAOptimal(
                evaluator, inner, outer, noise, relaxation, 0, target_map=bkd.eye(2)
            )

    def test_rejects_bad_nuisances(self, bkd: Backend[Array]) -> None:
        evaluator, inner, outer, noise, relaxation = self._parts(bkd)
        with pytest.raises(ValueError, match="without a nuisance_rule"):
            ReferenceExpectedInformationGain(
                evaluator, inner, outer, noise, relaxation, nuisance_indices=[0]
            )
        with pytest.raises(ValueError, match="nuisance_indices"):
            ReferenceExpectedInformationGain(
                evaluator, inner, outer, noise, relaxation, nuisance_rule=inner
            )

    def test_rejects_bad_design(self, bkd: Backend[Array]) -> None:
        evaluator, inner, outer, noise, relaxation = self._parts(bkd)
        a_opt = ReferenceAOptimal(evaluator, inner, outer, noise, relaxation, 0)
        with pytest.raises(ValueError, match="design_weights"):
            a_opt(bkd.ones((2, 1)))

    def test_rejects_non_protocol_inputs(self, bkd: Backend[Array]) -> None:
        evaluator, inner, outer, noise, relaxation = self._parts(bkd)
        with pytest.raises(TypeError, match="inner_rule"):
            ReferenceAOptimal(
                evaluator,
                object(),  # type: ignore[arg-type]
                outer,
                noise,
                relaxation,
                0,
            )

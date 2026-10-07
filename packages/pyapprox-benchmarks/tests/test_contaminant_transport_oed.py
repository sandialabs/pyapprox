"""Tests for ContaminantTransportOEDProblem.

NumPy only (Galerkin uses skfem which is NumPy-based). Most tests use a
coarse mesh and few steps; the default resolution is exercised by one
slow test.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from skfem import LinearForm, asm

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.expdesign.contaminant_transport import (
    CellsPlusUniformStreamFunction,
    ContaminantTransportOEDProblem,
    FixedVelocity,
    GaussianPlume,
    StreamFunctionFlow,
    UncertainVelocity,
)
from tests._helpers.markers import slow_test


class _ZeroStreamFunction:
    def __init__(self, bkd: NumpyBkd) -> None:
        self._bkd = bkd

    def gradient(self, points: Any) -> Any:
        return self._bkd.zeros((2, points.shape[1]))


def _coarse(
    bkd: NumpyBkd, velocity_model: Any, **kwargs: Any
) -> ContaminantTransportOEDProblem[Any]:
    """A 10x10-node mesh and 20 steps."""
    return ContaminantTransportOEDProblem(
        bkd, velocity_model, nelements_per_dim=9, deltat=0.05, **kwargs
    )


def _flow(bkd: NumpyBkd) -> StreamFunctionFlow[Any]:
    return StreamFunctionFlow(bkd, CellsPlusUniformStreamFunction(bkd))


def _still(
    bkd: NumpyBkd, center: tuple[float, float] = (0.5, 0.25)
) -> ContaminantTransportOEDProblem[Any]:
    """Coarse problem with zero velocity."""
    flow = StreamFunctionFlow(bkd, _ZeroStreamFunction(bkd))
    return _coarse(
        bkd,
        FixedVelocity(flow, bkd.zeros((flow.nmodes(),))),
        initial_condition=GaussianPlume(bkd, center=center),
    )


class TestContaminantTransportOEDProblem:
    def test_dimensions(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        flow = _flow(bkd)
        uncertain = _coarse(bkd, UncertainVelocity(flow))
        fixed = _coarse(bkd, FixedVelocity(flow, bkd.zeros((36,))))
        assert uncertain.nparams() == fixed.nparams() == 100
        assert uncertain.nvars() == 136
        assert fixed.nvars() == 100
        assert uncertain.prior().nvars() == 136
        assert uncertain.obs_map().nqoi() == 36 * 5
        assert uncertain.qoi_map().nqoi() == 10 * 5
        assert uncertain.observation_steps() == [4, 8, 12, 16, 20]
        problem = uncertain.problem()
        assert problem.nobs() == 180
        assert tuple(problem.design_conditions().shape) == (3, 180)
        bkd.assert_allclose(
            problem.noise_variances(), bkd.full((180,), 1e-4)
        )

    def test_design_conditions_and_groups(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        bench = _coarse(bkd, UncertainVelocity(_flow(bkd)))
        conditions = bench.problem().design_conditions()
        locations = bench.sensor_locations()
        times = bkd.asarray([0.2, 0.4, 0.6, 0.8, 1.0])
        by_sensor = bench.groups_by_sensor()
        by_time = bench.groups_by_time()
        assert len(by_sensor) == 36 and len(by_time) == 5
        for groups in (by_sensor, by_time):
            assert sorted(ii for group in groups for ii in group) == list(
                range(180)
            )
        for ss, group in enumerate(by_sensor):
            for ii in group:
                bkd.assert_allclose(conditions[:2, ii], locations[:, ss])
        for jj, group in enumerate(by_time):
            bkd.assert_allclose(
                conditions[2, group], bkd.full((36,), float(times[jj]))
            )

    def test_invalid_inputs_raise(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        flow = _flow(bkd)
        with pytest.raises(ValueError, match="coefficients must have shape"):
            FixedVelocity(flow, bkd.zeros((5,)))
        with pytest.raises(ValueError, match="not a multiple of deltat"):
            _coarse(bkd, UncertainVelocity(flow), observation_times=(0.21,))
        not_a_velocity_model: Any = flow
        with pytest.raises(TypeError, match="VelocityModelProtocol"):
            _coarse(bkd, not_a_velocity_model)

    def test_evaluate_both_matches_maps(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        bench = _coarse(bkd, UncertainVelocity(_flow(bkd)))
        samples = bench.prior().rvs(2)
        obs, qoi = bench.evaluate_both(samples)
        bkd.assert_allclose(obs, bench.obs_map()(samples))
        bkd.assert_allclose(qoi, bench.qoi_map()(samples))

    def test_oracle_matches_uncertain_at_truth(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        flow = _flow(bkd)
        uncertain = _coarse(bkd, UncertainVelocity(flow))
        zeta, eta = uncertain.truth(3)
        oracle = _coarse(bkd, FixedVelocity(flow, eta))
        obs, qoi = uncertain.evaluate_both(bkd.vstack([zeta, eta[:, None]]))
        obs_oracle, qoi_oracle = oracle.evaluate_both(zeta)
        bkd.assert_allclose(obs_oracle, obs, rtol=1e-12)
        bkd.assert_allclose(qoi_oracle, qoi, rtol=1e-12)

    def test_outflow_normal_velocity_is_one(self, numpy_bkd: NumpyBkd) -> None:
        # Every mode vanishes on the boundary, so the outflux QoI's fixed
        # normal velocity is exact for every eta.
        bkd = numpy_bkd
        bench = _coarse(bkd, UncertainVelocity(_flow(bkd)))
        velocity = bench.velocity_field(bench.prior().rvs(1))
        bkd.assert_allclose(
            velocity[0, bench.outflow_nodes()],
            bkd.ones((10,)),
            atol=1e-12,
        )

    def test_mass_conserved_without_velocity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        bench = _still(bkd)
        states, _ = bench.solve_trajectory(bench.prior().rvs(1))
        weights = asm(
            LinearForm(lambda v, w: v), bench.basis().skfem_basis()
        )
        mass = bkd.asarray(weights) @ states
        bkd.assert_allclose(mass, bkd.full(mass.shape, float(mass[0])))
        # Diffusion has moved mass; the check is not vacuous.
        assert float(bkd.max(bkd.abs(states[:, -1] - states[:, 0]))) > 1e-3

    def test_point_symmetry_without_velocity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        # With no velocity and a constant diffusivity the problem is
        # symmetric under the point reflection x -> 1 - x, and so is the
        # mesh, whose diagonals all point the same way.
        bkd = numpy_bkd
        bench = _still(bkd, center=(0.5, 0.5))
        states, _ = bench.solve_trajectory(bkd.zeros((bench.nvars(), 1)))
        nodes = bkd.to_numpy(bench.mesh().nodes())
        reflected = 1.0 - nodes
        partner = [
            int(np.argmin(np.sum((nodes - reflected[:, [ii]]) ** 2, axis=0)))
            for ii in range(nodes.shape[1])
        ]
        bkd.assert_allclose(states[partner, -1], states[:, -1], rtol=1e-10)

    def test_truth_is_reproducible(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        bench = _coarse(bkd, FixedVelocity(_flow(bkd), bkd.zeros((36,))))
        zeta, eta = bench.truth(7)
        assert tuple(zeta.shape) == (100, 1)
        assert tuple(eta.shape) == (36,)
        zeta2, eta2 = bench.truth(7)
        bkd.assert_allclose(zeta2, zeta)
        bkd.assert_allclose(eta2, eta)

    @slow_test
    def test_default_resolution(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        bench = ContaminantTransportOEDProblem(
            bkd, UncertainVelocity(_flow(bkd))
        )
        assert bench.nvars() == 1636
        assert bench.observation_steps() == [20, 40, 60, 80, 100]
        zeta, eta = bench.truth(0)
        obs, qoi = bench.evaluate_both(bkd.vstack([zeta, eta[:, None]]))
        assert tuple(obs.shape) == (180, 1)
        assert tuple(qoi.shape) == (200, 1)
        states, times = bench.solve_trajectory(
            bkd.vstack([zeta, eta[:, None]])
        )
        assert tuple(states.shape) == (1600, 101)
        bkd.assert_allclose(times[-1:], bkd.ones((1,)))

r"""A Galerkin steady state as an implicit function of the parameters.

``SteadyImplicitFunction`` over ``galerkin_steady_residual`` (stack
``Newton``, atol 1e-13, SuperLU, ``KeepAll``) on the four problems of
``test_galerkin_steady_residual``: the 1D diffusion problem, linear and
with :math:`u^2`, at p = (0.4, -0.3, 0.2); the elastic plate at
(E, nu) = (1, 0.3); the heat problem frozen at t* = 0.03.

- The state at the middle DOF, at the parameters, half of them, and them
  plus 0.1 (one call): values equal a fresh ``solve_galerkin_steady`` at
  each sample (sparse LU each iteration) to 1e-9 relative. The Jacobian
  and HVP at the parameters pass central finite differences (smallest
  relative error below 1e-6).
- The states at the middle and last-but-one DOFs (two entries, at most the
  number of parameters): the Jacobian passes central finite differences
  (below 1e-6).
- Downgrade: the :math:`u^2` reaction declared without its second
  derivative gives a residual bundle with no second-order fields (absent,
  not a stated zero), so the function has no HVP; its Jacobian still
  passes central finite differences (below 1e-6).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.setups import KeepAll, RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import (
    galerkin_steady_residual,
    solve_galerkin_steady,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import HEAT_PARAM, MovingEndHeat
from tests._helpers.galerkin_steady import (
    ADR_PARAM,
    PLATE_PARAM,
    elastic_plate,
    steady_adr,
)

_Np = NDArray[Any]
_HEAT_TIME = 0.03
_CASES = ["linear", "nonlinear", "elastic plate", "heat snapshot"]


def _problem(
    bkd: NumpyBkd, name: str
) -> Tuple[GalerkinSystem[_Np], ParameterizationProtocol[_Np], _Np, Optional[float]]:
    if name == "heat snapshot":
        heat = MovingEndHeat(bkd)
        return heat.system, heat.parameterization, HEAT_PARAM, _HEAT_TIME
    if name == "elastic plate":
        system, parameterization = elastic_plate(bkd)
        return system, parameterization, PLATE_PARAM, None
    system, parameterization, _ = steady_adr(bkd, name == "nonlinear")
    return system, parameterization, ADR_PARAM, None


class _Observed:
    """One problem's steady state at ``rows`` as a function of the
    parameters."""

    def __init__(self, bkd: NumpyBkd, name: str, rows: List[int]) -> None:
        self.name, self.rows = name, rows
        system, parameterization, self.param, time = _problem(bkd, name)
        n = system.nstates()
        discretization = skfem_discretization(system)
        space = discretization.space()
        assert isinstance(space, InnerProductSpaceProtocol)
        parameters = ParameterState(parameterization)
        functional: StateFunctional[_Np, _Np] = StateFunctional.from_terms(
            bkd, len(rows), state_terms=(ObservedValue(StateEntries(bkd, n, rows)),)
        )
        self.function: SteadyImplicitFunction[_Np, Any] = SteadyImplicitFunction(
            galerkin_steady_residual(discretization, parameters, time),
            space,
            parameters,
            KeepAll(ScipySparseLU()),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            functional,
            np.zeros((n, 1)),
            bkd,
        )

    def solved(self, bkd: NumpyBkd, sample: _Np) -> _Np:
        """The states at ``rows`` from a fresh solve at ``sample``, shape
        ``(nparams, 1)``, on a system of its own."""
        system, parameterization, _, time = _problem(bkd, self.name)
        parameters = ParameterState(parameterization)
        parameters.set_param(sample[:, 0])
        _, point = solve_galerkin_steady(
            skfem_discretization(system),
            RecomputeAlways(ScipySparseLU()),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            parameters,
            time,
        )
        assert point.converged()
        state: _Np = point.solution()
        return np.asarray(state[self.rows, :])


def _min_errors(bkd: NumpyBkd, function: _Observed) -> List[float]:
    errors = DerivativeChecker(function.function).check_derivatives(
        function.param, relative=True, central=True
    )
    return [bkd.to_float(bkd.min(e)) for e in errors]


def _close(actual: _Np, expected: _Np, rtol: float) -> None:
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


def _middle(bkd: NumpyBkd, name: str) -> int:
    system, _, _, _ = _problem(bkd, name)
    return system.nstates() // 2


@pytest.mark.parametrize("name", _CASES)
def test_scalar_values_equal_fresh_solves(numpy_bkd: NumpyBkd, name: str) -> None:
    bkd = numpy_bkd
    function = _Observed(bkd, name, [_middle(bkd, name)])
    p = function.param
    samples = np.hstack([p, 0.5 * p, p + 0.1])
    expected = np.hstack(
        [function.solved(bkd, samples[:, j : j + 1]) for j in range(3)]
    )
    _close(function.function(samples), expected, 1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_scalar_jacobian_and_hvp_pass_finite_differences(
    numpy_bkd: NumpyBkd, name: str
) -> None:
    bkd = numpy_bkd
    function = _Observed(bkd, name, [_middle(bkd, name)])
    assert function.function.derivatives().hvp is not None
    errors = _min_errors(bkd, function)
    assert len(errors) == 2
    assert max(errors) < 1e-6


@pytest.mark.parametrize("name", _CASES)
def test_vector_jacobian_passes_finite_differences(
    numpy_bkd: NumpyBkd, name: str
) -> None:
    bkd = numpy_bkd
    middle = _middle(bkd, name)
    function = _Observed(bkd, name, [middle, 2 * middle - 1])
    errors = _min_errors(bkd, function)
    assert len(errors) == 1
    assert errors[0] < 1e-6


def test_missing_state_curvature_downgrades_to_the_jacobian(
    numpy_bkd: NumpyBkd,
) -> None:
    bkd = numpy_bkd
    system, parameterization, _ = steady_adr(bkd, True, curvature=False)
    n = system.nstates()
    discretization = skfem_discretization(system)
    space = discretization.space()
    assert isinstance(space, InnerProductSpaceProtocol)
    parameters = ParameterState(parameterization)
    residual = galerkin_steady_residual(discretization, parameters)
    bundle = residual.derivatives()
    assert bundle.param is not None
    assert bundle.state_state is None and bundle.param_param is None
    functional: StateFunctional[_Np, _Np] = StateFunctional.from_terms(
        bkd, 1, state_terms=(ObservedValue(StateEntries(bkd, n, [n // 2])),)
    )
    function: SteadyImplicitFunction[_Np, Any] = SteadyImplicitFunction(
        residual,
        space,
        parameters,
        KeepAll(ScipySparseLU()),
        Newton(atol=1e-13, rtol=0.0, maxiters=30),
        functional,
        np.zeros((n, 1)),
        bkd,
    )
    derivs = function.derivatives()
    assert derivs.jacobian is not None and derivs.hvp is None
    errors = DerivativeChecker(function).check_derivatives(
        ADR_PARAM, relative=True, central=True
    )
    assert len(errors) == 1
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6

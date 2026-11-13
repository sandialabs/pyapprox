r"""A Galerkin steady state as an implicit function equals the steady model.

``SteadyImplicitFunction`` over ``galerkin_steady_residual`` (stack
``Newton``, atol 1e-13, SuperLU, ``KeepAll``) is compared with
``GalerkinSteadyForwardModel`` (``SteadyStateSolver``, tol 1e-13) on the
four problems of ``test_galerkin_steady_residual``: the 1D diffusion
problem, linear and with :math:`u^2`, at p = (0.4, -0.3, 0.2); the elastic
plate at (E, nu) = (1, 0.3); the heat problem frozen at t* = 0.03.

- The state at the middle DOF, at the parameters, half of them, and them
  plus 0.1: values to 1e-9 relative; the Jacobian at the parameters to
  1e-8 relative. The old model has no HVP with this functional, so the
  new HVP is checked by central finite differences (smallest relative
  error below 1e-6).
- The states at the middle and last-but-one DOFs (two entries, at most the
  number of parameters): the Jacobian to 1e-8 relative.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesFunctional,
)
from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady import GalerkinSteadyForwardModel
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
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


class _Pair:
    """The new function and the old model of one problem, each on its own
    system, observing the same states."""

    def __init__(self, bkd: NumpyBkd, name: str, rows: List[int]) -> None:
        system, parameterization, self.param, time = _problem(bkd, name)
        n = system.nstates()
        discretization = skfem_discretization(system)
        space = discretization.space()
        assert isinstance(space, InnerProductSpaceProtocol)
        parameters = ParameterState(parameterization)
        functional: StateFunctional[_Np, _Np] = StateFunctional.from_terms(
            bkd, len(rows), state_terms=(ObservedValue(StateEntries(bkd, n, rows)),)
        )
        self.new: SteadyImplicitFunction[_Np, Any] = SteadyImplicitFunction(
            galerkin_steady_residual(discretization, parameters, time),
            space,
            parameters,
            KeepAll(ScipySparseLU()),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            functional,
            np.zeros((n, 1)),
            bkd,
        )
        old_system, old_parameterization, _, _ = _problem(bkd, name)
        view = old_system.steady() if time is None else old_system.steady_snapshot(time)
        self.old = GalerkinSteadyForwardModel(
            view,
            old_parameterization,
            np.zeros(n),
            bkd,
            functional=SubsetOfStatesFunctional(
                n, self.param.shape[0], np.asarray(rows), bkd
            ),
            solver=SteadyStateSolver(view, tol=1e-13),
        )


def _close(actual: _Np, expected: _Np, rtol: float) -> None:
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


def _middle(bkd: NumpyBkd, name: str) -> int:
    system, _, _, _ = _problem(bkd, name)
    return system.nstates() // 2


@pytest.mark.parametrize("name", _CASES)
def test_scalar_values_and_jacobian(numpy_bkd: NumpyBkd, name: str) -> None:
    bkd = numpy_bkd
    pair = _Pair(bkd, name, [_middle(bkd, name)])
    p = pair.param
    samples = np.hstack([p, 0.5 * p, p + 0.1])
    _close(pair.new(samples), pair.old(samples), 1e-9)
    new, old = pair.new.derivatives(), pair.old.derivatives()
    assert new.jacobian is not None and old.jacobian is not None
    _close(new.jacobian(p), old.jacobian(p), 1e-8)


@pytest.mark.parametrize("name", _CASES)
def test_scalar_hvp_passes_finite_differences(numpy_bkd: NumpyBkd, name: str) -> None:
    """The old model has no HVP with this functional, so the oracle is
    finite differences of the Jacobian."""
    bkd = numpy_bkd
    pair = _Pair(bkd, name, [_middle(bkd, name)])
    assert pair.new.derivatives().hvp is not None
    errors = DerivativeChecker(pair.new).check_derivatives(
        pair.param, relative=True, central=True
    )
    assert len(errors) == 2
    assert max(bkd.to_float(bkd.min(e)) for e in errors) < 1e-6


@pytest.mark.parametrize("name", _CASES)
def test_vector_jacobian(numpy_bkd: NumpyBkd, name: str) -> None:
    bkd = numpy_bkd
    middle = _middle(bkd, name)
    pair = _Pair(bkd, name, [middle, 2 * middle - 1])
    new, old = pair.new.derivatives(), pair.old.derivatives()
    assert new.jacobian is not None and old.jacobian is not None
    _close(new.jacobian(pair.param), old.jacobian(pair.param), 1e-8)

r"""Apply-only parameterizations by the physics coefficients themselves.

``BendingStiffnessApplyOnlyParameterization`` (EI per element of an
Euler-Bernoulli cantilever: ``CubicHermiteBasis`` on 8 cells of [0, 1],
unit load, clamped left) and ``LameFieldApplyOnlyParameterization`` (Lame
fields per element of ``CompositeLinearElasticity`` on a 4 x 4 square,
clamped left, unit downward traction on the right).

The case the design rests on: a ``SteadySystem`` built once over
``galerkin_steady_residual`` with ``KeepAll``, cleared after each
``ParameterState.set_param`` (kept setups are valid for one parameter
value; whoever changes the parameters clears them), gives the new
solution when the coefficients change. Both problems are linear with the solution
proportional to 1/stiffness, so doubling the coefficients halves the
solution (to 1e-12, relative); the EI = 1 tip deflection is
:math:`qL^4/(8EI) = 0.125`. Also: the Lame fields applied uniformly give
the residual of the same material built by ``from_uniform`` (to 1e-12);
both refuse parameters of the wrong shape; both satisfy
``ParameterizationProtocol``; the Lame one pickles (the beam's skfem
Hermite element holds a lambda, so the beam does not). The first solve's
point refuses to be read after the parameters change.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import pickle
from typing import Any

import numpy as np
from skfem.models.elasticity import lame_parameters

from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import CubicHermiteBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC, NeumannBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
from pyapprox.pde.galerkin.physics.euler_bernoulli import EulerBernoulliBeamFEM
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.euler_bernoulli import (
    BendingStiffnessApplyOnlyParameterization,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    LameFieldApplyOnlyParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray

_NCELLS = 8
_E, _NU = 1.0, 0.3


class _UnitLoad:
    def __call__(self, x: NumpyArray) -> NumpyArray:
        return np.ones_like(x)


class _DownwardTraction:
    def __call__(self, coords: NumpyArray) -> NumpyArray:
        return np.vstack([np.zeros(coords.shape[1]), -np.ones(coords.shape[1])])


def _beam(bkd: NumpyBkd) -> EulerBernoulliBeamFEM[NumpyArray]:
    basis = CubicHermiteBasis(StructuredMesh1D(_NCELLS, (0.0, 1.0), bkd))
    return EulerBernoulliBeamFEM(basis, np.ones(_NCELLS), _UnitLoad(), bkd)


def _plate(bkd: NumpyBkd) -> CompositeLinearElasticity[NumpyArray]:
    mesh = StructuredMesh2D(nx=4, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    return CompositeLinearElasticity.from_uniform(
        basis=VectorLagrangeBasis(mesh, degree=1),
        youngs_modulus=_E,
        poisson_ratio=_NU,
        bkd=bkd,
    )


def _lame_params(physics: CompositeLinearElasticity[Any], scale: float) -> NumpyArray:
    lam, mu = lame_parameters(scale * _E, _NU)
    nelems = physics.basis().mesh().nelements()
    return np.concatenate([np.full(nelems, lam), np.full(nelems, mu)])


def _solve_twice(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    parameters: ParameterState[NumpyArray],
    first: NumpyArray,
    second: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """Solve at ``first``, then at ``second``, on one ``SteadySystem``
    with ``KeepAll``, clearing it on the parameter change as the setup
    contract requires of whoever changes the parameters."""
    discretization = skfem_discretization(system)
    space = discretization.space()
    assert isinstance(space, InnerProductSpaceProtocol)
    parameters.set_param(first)
    setups: KeepAll[NumpyArray, Any] = KeepAll(ScipySparseLU())
    steady: SteadySystem[NumpyArray, NumpyArray, Any] = SteadySystem(
        galerkin_steady_residual(discretization, parameters), space, setups
    )
    newton: Newton[NumpyArray, NumpyArray] = Newton(
        atol=1e-12, rtol=0.0, maxiters=5
    )
    one = steady.solve(newton, space.zeros())
    # A point refuses to be read once the parameters move on: read it first.
    assert one.converged()
    first_solution = np.array(one.solution()[:, 0])
    parameters.set_param(second)
    setups.clear()
    two = steady.solve(newton, space.zeros())
    assert two.converged()
    with pytest.raises(ValueError, match="parameters changed"):
        one.solution()
    return first_solution, np.array(two.solution()[:, 0])


def _close(actual: NumpyArray, expected: NumpyArray, rtol: float = 1e-12) -> None:
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


class TestBendingStiffness:
    def test_a_kept_solve_sees_the_new_stiffness(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        beam = _beam(bkd)
        system = compose_galerkin_system(
            beam, [DirichletBC(beam.basis(), "left", 0.0, bkd)]
        )
        parameters = ParameterState(
            BendingStiffnessApplyOnlyParameterization(beam, bkd)
        )
        one, two = _solve_twice(
            bkd, system, parameters, np.ones(_NCELLS), 2.0 * np.ones(_NCELLS)
        )
        # Free-end deflection is the second-to-last DOF (value, then slope).
        _close(one[-2:-1], np.asarray([0.125]))
        _close(two, 0.5 * one)

    def test_apply_sets_the_element_stiffnesses(self, numpy_bkd: NumpyBkd) -> None:
        beam = _beam(numpy_bkd)
        parameterization = BendingStiffnessApplyOnlyParameterization(beam, numpy_bkd)
        assert parameterization.nparams() == _NCELLS
        values = np.linspace(1.0, 2.0, _NCELLS)
        parameterization.apply(values)
        np.testing.assert_array_equal(beam.EI(), values)

    def test_refuses_the_wrong_shape(self, numpy_bkd: NumpyBkd) -> None:
        parameterization = BendingStiffnessApplyOnlyParameterization(
            _beam(numpy_bkd), numpy_bkd
        )
        with pytest.raises(ValueError, match="params_1d"):
            parameterization.apply(np.ones(_NCELLS + 1))

    def test_refuses_an_object_that_is_not_a_beam(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="EulerBernoulliBeamFEM"):
            BendingStiffnessApplyOnlyParameterization(
                object(),  # type: ignore[arg-type]
                numpy_bkd,
            )

    def test_is_a_parameterization(self, numpy_bkd: NumpyBkd) -> None:
        """Not pickled: the beam's skfem Hermite element holds a lambda."""
        beam = _beam(numpy_bkd)
        parameterization = BendingStiffnessApplyOnlyParameterization(beam, numpy_bkd)
        assert isinstance(parameterization, ParameterizationProtocol)
        assert parameterization.targets() == (beam,)
        assert parameterization.param_derivatives().param_jacobian is None


class TestLameField:
    def test_a_kept_solve_sees_the_new_fields(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        physics = _plate(bkd)
        basis = physics.basis()
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                NeumannBC(basis, "right", TimeIndependent(_DownwardTraction()), bkd),
            ],
        )
        nelems = basis.mesh().nelements()
        parameters = ParameterState(
            LameFieldApplyOnlyParameterization(physics, (nelems,), bkd)
        )
        one, two = _solve_twice(
            bkd,
            system,
            parameters,
            _lame_params(physics, 1.0),
            _lame_params(physics, 2.0),
        )
        assert float(np.max(np.abs(one))) > 1e-2
        _close(two, 0.5 * one)

    def test_uniform_fields_give_the_uniform_material(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        reference = _plate(bkd)
        physics = _plate(bkd)
        nelems = physics.basis().mesh().nelements()
        LameFieldApplyOnlyParameterization(physics, (nelems,), bkd).apply(
            _lame_params(physics, 1.0)
        )
        state = np.linspace(-0.1, 0.1, physics.nstates())
        _close(
            compose_galerkin_system(physics)
            .spatial_operator()
            .spatial_residual(state, 0.0),
            compose_galerkin_system(reference)
            .spatial_operator()
            .spatial_residual(state, 0.0),
        )

    def test_refuses_the_wrong_shape(self, numpy_bkd: NumpyBkd) -> None:
        physics = _plate(numpy_bkd)
        nelems = physics.basis().mesh().nelements()
        parameterization = LameFieldApplyOnlyParameterization(
            physics, (nelems,), numpy_bkd
        )
        assert parameterization.nparams() == 2 * nelems
        with pytest.raises(ValueError, match="params_1d"):
            parameterization.apply(np.ones(nelems))

    def test_refuses_an_empty_field_shape(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(ValueError, match="field_shape"):
            LameFieldApplyOnlyParameterization(_plate(numpy_bkd), (), numpy_bkd)

    def test_is_a_parameterization_and_pickles(self, numpy_bkd: NumpyBkd) -> None:
        physics = _plate(numpy_bkd)
        nelems = physics.basis().mesh().nelements()
        parameterization = LameFieldApplyOnlyParameterization(
            physics, (nelems, 3), numpy_bkd
        )
        assert isinstance(parameterization, ParameterizationProtocol)
        assert parameterization.nparams() == 6 * nelems
        assert parameterization.param_derivatives().param_jacobian is None
        clone = pickle.loads(pickle.dumps(parameterization))
        assert clone.nparams() == 6 * nelems

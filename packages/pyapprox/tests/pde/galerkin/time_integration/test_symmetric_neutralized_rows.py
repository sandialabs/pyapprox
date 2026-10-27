"""The symmetric essential-row treatment against the default one.

1D advection-diffusion, P1 on 8 cells, with a time-varying forcing and a
time-varying Dirichlet value g(t) = (1 + x) t^2 on the left, its rate
supplied. The symmetric treatment's ``f`` and mass are checked against
references sliced from the dense consistent mass, its stack mass keeps
the assembled mass's symmetry where the default's does not, and the two
treatments give the same stage slopes k = M^{-1} f: the coupling M_ID
g_dot only moves from the mass into ``f``. With a lumped mass the two
coincide. Only the symmetric treatment refuses a missing rate.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    SymmetricNeutralizedRows,
    skfem_ode_residual,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]
_TIMES = (0.2, 0.55, 0.9)


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = np.sin(np.pi * x[0]) * (1.0 + 3.0 * t)
    return value


def _ramp(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * t**2
    return value


def _ramp_dot(x: _Np, t: float) -> _Np:
    value: _Np = 2.0 * (1.0 + x[0]) * t
    return value


def _system(bkd: NumpyBkd, with_rate: bool = True) -> GalerkinSystem[_Np]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    rates = [TimeDependent(_ramp_dot)] if with_rate else []
    bc = DirichletBC(basis, "left", BoundarySignal(TimeDependent(_ramp), rates), bkd)
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=0.1, bkd=bkd, forcing=TimeDependent(_forcing)
    )
    return compose_galerkin_system(physics, [bc])


def _dense(matrix: Union[spmatrix, _Np]) -> _Np:
    if isinstance(matrix, spmatrix):
        dense: _Np = np.asarray(matrix.toarray())
        return dense
    return np.asarray(matrix)


def _split(system: GalerkinSystem[_Np]) -> tuple[_Np, _Np, _Np]:
    """The dense consistent mass and the essential and interior DOFs."""
    mass = _dense(system.mass_matrix())
    dofs = np.asarray(system.constraint_set().dofs())
    interior = np.setdiff1d(np.arange(mass.shape[0]), dofs)
    return mass, dofs, interior


def _state(system: GalerkinSystem[_Np]) -> _Np:
    nstates = system.spatial_operator().nstates()
    state: _Np = np.cos(np.linspace(0.0, 2.0, nstates))
    return state


class TestSymmetricNeutralizedRows:
    def test_residual_moves_the_coupling_into_f(self, numpy_bkd: NumpyBkd) -> None:
        """g_dot on the essential rows, F - M_ID g_dot on the others."""
        system = _system(numpy_bkd)
        adapter = skfem_ode_residual(system, row_treatment=SymmetricNeutralizedRows)
        mass, dofs, interior = _split(system)
        velocity = system.constraint_set().values_derivative(1)
        assert velocity is not None
        state = _state(system)
        for time in _TIMES:
            gdot = np.asarray(velocity(time))
            expected = np.asarray(
                system.spatial_operator().spatial_residual(state, time)
            ).copy()
            expected[interior] -= mass[np.ix_(interior, dofs)] @ gdot
            expected[dofs] = gdot
            numpy_bkd.assert_allclose(
                adapter(state, time), expected, rtol=1e-12, atol=1e-14
            )

    def test_mass_is_the_symmetric_elimination(self, numpy_bkd: NumpyBkd) -> None:
        """[M_II 0; 0 I], symmetric; the default keeps M_ID, so is not."""
        system = _system(numpy_bkd)
        mass, dofs, interior = _split(system)
        expected = np.eye(mass.shape[0])
        expected[np.ix_(interior, interior)] = mass[np.ix_(interior, interior)]
        symmetric = _dense(
            skfem_ode_residual(system, row_treatment=SymmetricNeutralizedRows)
            .mass_matrix()
            .as_matrix()
        )
        numpy_bkd.assert_allclose(symmetric, expected, rtol=1e-14)
        default = _dense(skfem_ode_residual(system).mass_matrix().as_matrix())
        assert np.abs(default - default.T).max() > 1e-3

    def test_stack_mass_keeps_the_assembled_symmetry(self, numpy_bkd: NumpyBkd) -> None:
        system = _system(numpy_bkd)
        symmetric = skfem_ode_residual(
            system, row_treatment=SymmetricNeutralizedRows
        ).implicit_system()
        assert symmetric.mass().properties().symmetric
        default = skfem_ode_residual(system).implicit_system()
        assert not default.mass().properties().symmetric

    def test_stage_slopes_agree_with_the_default(self, numpy_bkd: NumpyBkd) -> None:
        """k = M^{-1} f is the same under both treatments: the coupling
        only moves between the mass and f."""
        system = _system(numpy_bkd)
        symmetric = skfem_ode_residual(system, row_treatment=SymmetricNeutralizedRows)
        default = skfem_ode_residual(system)
        state = _state(system)
        for time in _TIMES:
            numpy_bkd.assert_allclose(
                symmetric.mass_matrix().solve(symmetric(state, time)),
                default.mass_matrix().solve(default(state, time)),
                rtol=1e-11,
                atol=1e-13,
            )

    def test_lumped_mass_is_the_same_under_both(self, numpy_bkd: NumpyBkd) -> None:
        """M_ID = 0 for a lumped mass, so the treatments coincide."""
        system = _system(numpy_bkd)
        symmetric = skfem_ode_residual(
            system, lumped_mass=True, row_treatment=SymmetricNeutralizedRows
        )
        default = skfem_ode_residual(system, lumped_mass=True)
        state = _state(system)
        numpy_bkd.assert_allclose(
            _dense(symmetric.mass_matrix().as_matrix()),
            _dense(default.mass_matrix().as_matrix()),
        )
        numpy_bkd.assert_allclose(symmetric(state, 0.4), default(state, 0.4))


class TestMissingRate:
    def test_symmetric_treatment_refuses_it(self, numpy_bkd: NumpyBkd) -> None:
        """The rate couples into every interior row, so it is required, and
        the refusal names the boundary condition lacking it."""
        system = _system(numpy_bkd, with_rate=False)
        with pytest.raises(TypeError, match="DirichletBC\\(boundary='left'"):
            skfem_ode_residual(system, row_treatment=SymmetricNeutralizedRows)

    def test_default_and_lumped_accept_it(self, numpy_bkd: NumpyBkd) -> None:
        system = _system(numpy_bkd, with_rate=False)
        skfem_ode_residual(system)
        skfem_ode_residual(
            system, lumped_mass=True, row_treatment=SymmetricNeutralizedRows
        )

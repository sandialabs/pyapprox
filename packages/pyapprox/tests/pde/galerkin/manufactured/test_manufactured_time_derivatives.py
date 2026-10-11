"""The manufactured Dirichlet builders' time derivatives are derivatives.

Each builder pairs boundary values with an analytic d/dT (from sympy, or
supplied by the caller). The framework trusts that pairing, so a
mismatched expression would surface only as lost convergence in a
stage-based stepper. These tests check every time-dependent essential
BC each builder produces against finite differences of its values.

Solutions are non-polynomial in T and nonzero on the boundary: a
solution vanishing there has zero values and derivatives, and would
pass vacuously.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
    create_hyperelasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.time_derivative_checks import (
    assert_time_derivatives_match,
)

_BOUNDARIES_1D = ["left", "right"]
_BOUNDARIES_2D = ["left", "right", "bottom", "top"]


def _dirichlet_all(
    ms: GalerkinManufacturedSolution[NDArray[Any]], names: List[str]
) -> List[DirichletBC[NDArray[Any]]]:
    return [ms.dirichlet(name) for name in names]


def _assert_bcs(bcs: List[DirichletBC[NDArray[Any]]], bkd: NumpyBkd) -> None:
    assert bcs, "the builder produced no essential BCs to check"
    for bc in bcs:
        assert not bc.is_time_invariant()
        assert_time_derivatives_match(
            bc.constrained_values,
            bc.constrained_values_derivative,
            1,
            bc.constrained_dofs().shape[0],
            bkd,
        )


class TestManufacturedAdapterTimeDerivatives:
    def test_adr_1d(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="(1+x)*cos(2*T)+x**2*exp(T)",
            diff_str="4+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        ms = GalerkinManufacturedSolution(
            LagrangeBasis(mesh, degree=2), functions, bkd
        )
        _assert_bcs(_dirichlet_all(ms, _BOUNDARIES_1D), bkd)

    def test_adr_2d(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_str="(1+x*y)*sin(2*T+1)+y*exp(-T)",
            diff_str="4+1e-16*x",
            react_str="0*u",
            vel_strs=["0+1e-16*x", "0+1e-16*y"],
            bkd=bkd,
            time_dependent=True,
        )
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=bkd)
        ms = GalerkinManufacturedSolution(
            LagrangeBasis(mesh, degree=1), functions, bkd
        )
        _assert_bcs(_dirichlet_all(ms, _BOUNDARIES_2D), bkd)

    def test_hyperelasticity_1d(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        functions, _ = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=["0.1*(1+x)*sin(2*T+1)"],
            stress_model=NeoHookeanStress(1.0, 1.0),
            bkd=bkd,
        )
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        ms = GalerkinManufacturedSolution(
            VectorLagrangeBasis(mesh, degree=2), functions, bkd
        )
        _assert_bcs(_dirichlet_all(ms, _BOUNDARIES_1D), bkd)

    def test_hyperelasticity_2d(self, numpy_bkd: NumpyBkd) -> None:
        """Vector values: each interleaved DOF takes its own component."""
        bkd = numpy_bkd
        functions, _ = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=[
                "0.1*(1+x*y)*sin(2*T+1)",
                "0.05*(1+x+y)*cos(3*T)",
            ],
            stress_model=NeoHookeanStress(1.0, 1.0),
            bkd=bkd,
        )
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=bkd)
        ms = GalerkinManufacturedSolution(
            VectorLagrangeBasis(mesh, degree=1), functions, bkd
        )
        _assert_bcs(_dirichlet_all(ms, _BOUNDARIES_2D), bkd)


def _solution(x: NDArray[Any], t: float) -> NDArray[Any]:
    return (1.0 + x[0]) * np.sin(2.0 * t + 1.0)


def _solution_dot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 2.0 * (1.0 + x[0]) * np.cos(2.0 * t + 1.0)


def _solution_dot_missing_chain_rule(
    x: NDArray[Any], t: float
) -> NDArray[Any]:
    return (1.0 + x[0]) * np.cos(2.0 * t + 1.0)


class TestSuppliedTimeDerivatives:
    """Hand-written functions: the supplied d/dT is trusted, so the check
    must catch a wrong one."""

    def _bcs(
        self, bkd: NumpyBkd, solution_dot: Any
    ) -> List[DirichletBC[NDArray[Any]]]:
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        ms = GalerkinManufacturedSolution(
            LagrangeBasis(mesh, degree=1),
            {
                "solution": TimeDependent(_solution),
                "solution_time_derivative": TimeDependent(solution_dot),
            },
            bkd,
        )
        return _dirichlet_all(ms, _BOUNDARIES_1D)

    def test_supplied_derivative_passes(self, numpy_bkd: NumpyBkd) -> None:
        _assert_bcs(self._bcs(numpy_bkd, _solution_dot), numpy_bkd)

    def test_wrong_supplied_derivative_is_caught(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        with pytest.raises(AssertionError, match="order 1"):
            _assert_bcs(
                self._bcs(numpy_bkd, _solution_dot_missing_chain_rule),
                numpy_bkd,
            )

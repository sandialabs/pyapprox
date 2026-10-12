"""Tests for GalerkinManufacturedSolution.

Natural boundary data comes from the physics' flux law at the exact
solution, with the mesh's facet normals. The convergence tests check that
the data is what the physics enforces, on a square and on a disk, whose
facet normals are not the coordinate directions; a wrong-normal control
shows the disk test would catch a wrong normal.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
from skfem import MeshTri

from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
    create_helmholtz_manufactured_test,
)
from pyapprox.pde.galerkin.manufactured.adapter import (
    create_elasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    UnstructuredMesh2D,
)
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.protocols.basis import (
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_Arr = NDArray[np.floating[Any]]

_ADR_SOL = "sin(1.3*x)*cos(0.7*y) + 0.5*x*y"
_ADR_DIFF = 0.7
_ADR_VEL = ["1.0+1e-16*x", "0.5+1e-16*x"]
_ADR_REACT = 1.0


def _adr_functions(
    bkd: Backend[Array], advection_form: AdvectionFormProtocol
) -> Dict[str, Callable[..., Any]]:
    functions, _ = create_adr_manufactured_test(
        bounds=[-1.0, 1.0, -1.0, 1.0],
        sol_str=_ADR_SOL,
        diff_str=f"{_ADR_DIFF}+1e-16*x",
        react_str=f"{_ADR_REACT}*u",
        vel_strs=_ADR_VEL,
        bkd=bkd,
        advection_form=advection_form,
    )
    return functions


def _adr_physics(
    ms: GalerkinManufacturedSolution[Array],
    advection_form: AdvectionFormProtocol,
) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        basis=ms.basis(),
        diffusivity=_ADR_DIFF,
        bkd=ms.bkd(),
        velocity=ms.velocity(),
        reaction=_ADR_REACT,
        forcing=ms.forcing(),
        advection_form=advection_form,
    )


def _exact_at_dofs(ms: GalerkinManufacturedSolution[Array]) -> _Arr:
    """The exact solution at the DOFs, each DOF taking its own component."""
    bkd = ms.bkd()
    basis = ms.basis()
    coords = np.asarray(bkd.to_numpy(basis.dof_coordinates()))
    values = np.asarray(ms.solution()(coords, 0.0))
    if not isinstance(basis, ComponentDofsBasisProtocol):
        return values
    components = np.asarray(
        bkd.to_numpy(basis.dof_components(bkd.arange(basis.ndofs())))
    )
    return values[np.arange(basis.ndofs()), components]


def _mass_norm_error(
    system: GalerkinSystem[Array],
    ms: GalerkinManufacturedSolution[Array],
) -> float:
    """L2 norm of the error against the exact solution's interpolant."""
    bkd = system.bkd()
    result = SteadyStateSolver(system.steady(), tol=1e-11).solve()
    assert result.converged
    err = np.asarray(bkd.to_numpy(result.solution)) - _exact_at_dofs(ms)
    mass = system.mass_matrix()
    mass_np = mass.toarray() if issparse(mass) else bkd.to_numpy(mass)
    return float(np.sqrt(err @ mass_np @ err))


def _rates(errors: List[float]) -> _Arr:
    errs = np.asarray(errors)
    return np.log2(errs[:-1] / errs[1:])


def _disk_mesh(
    nrefs: int, tmp_path: Path, bkd: Backend[Array]
) -> UnstructuredMesh2D[Array]:
    """A polygonal unit disk with one boundary, ``"circle"``."""
    mesh = MeshTri.init_circle(nrefs)
    path = tmp_path / f"disk_{nrefs}.json"
    path.write_text(
        json.dumps(
            {
                "p": mesh.p.T.tolist(),
                "t": mesh.t.T.tolist(),
                "boundaries": {"circle": mesh.boundary_facets().tolist()},
            }
        )
    )
    return UnstructuredMesh2D(str(path), bkd)


class _ReversedNormal:
    """Wrong-model control: the law evaluated with the inward normal."""

    def __init__(self, law: NormalFluxLawProtocol) -> None:
        self._law = law

    def normal_flux(
        self, coords: _Arr, u: _Arr, grad_u: _Arr, normal: _Arr, time: float
    ) -> _Arr:
        return self._law.normal_flux(coords, u, grad_u, -normal, time)

    def is_time_dependent(self) -> bool:
        return self._law.is_time_dependent()


class TestCreateHelpers:
    def test_create_adr_1d(self, numpy_bkd: NumpyBkd) -> None:
        functions, nvars = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=numpy_bkd,
        )
        assert nvars == 1
        assert {"solution", "gradient", "forcing"} <= set(functions)

    def test_create_helmholtz_2d(self, numpy_bkd: NumpyBkd) -> None:
        functions, nvars = create_helmholtz_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_str="x**2*y**2",
            sqwavenum_str="1+1e-16*x",
            bkd=numpy_bkd,
        )
        assert nvars == 2
        assert {"solution", "forcing", "sqwavenum"} <= set(functions)


class TestValuesAndSuppliers:
    def test_dirichlet_values_are_exact(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3, ny=3, bounds=[[-1.0, 1.0], [-1.0, 1.0]], bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)
        ms = GalerkinManufacturedSolution(
            basis, _adr_functions(bkd, NonConservativeAdvection()), bkd
        )
        bc = ms.dirichlet("top")
        coords = bkd.to_numpy(basis.dof_coordinates())[
            :, bkd.to_numpy(bc.boundary_dofs())
        ]
        bkd.assert_allclose(
            bc.boundary_values(0.0),
            bkd.asarray(ms.solution()(coords, 0.0)),
            rtol=1e-14,
        )
        assert bc.is_time_invariant()

    def test_vector_dirichlet_selects_components(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        functions, _ = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=["x*y + 0.1", "x - 2*y"],
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
        )
        mesh = StructuredMesh2D(
            nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        ms = GalerkinManufacturedSolution(basis, functions, bkd)
        bc = ms.dirichlet("left")
        dofs = bc.boundary_dofs()
        bkd.assert_allclose(
            bc.boundary_values(0.0),
            bkd.asarray(_exact_at_dofs(ms)[bkd.to_numpy(dofs)]),
            rtol=1e-14,
        )

    def test_transient_dirichlet_carries_exact_derivative(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="x**2*(1+T**2)",
            diff_str="1.0",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            time_dependent=True,
        )
        basis = LagrangeBasis(
            StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), degree=1
        )
        ms = GalerkinManufacturedSolution(basis, functions, bkd)
        bc = ms.dirichlet("right")
        assert not bc.is_time_invariant()
        bkd.assert_allclose(
            bc.boundary_values(2.0), bkd.asarray([5.0]), rtol=1e-14
        )
        derivative = bc.constrained_values_derivative(1)
        assert derivative is not None
        bkd.assert_allclose(derivative(2.0), bkd.asarray([4.0]), rtol=1e-14)
        assert ms.forcing().is_time_dependent()

    def test_forcing_layout_follows_basis(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=2, ny=2, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        coords = np.random.rand(2, 5)
        scalar = GalerkinManufacturedSolution(
            LagrangeBasis(mesh, degree=1),
            _adr_functions(bkd, NonConservativeAdvection()),
            bkd,
        )
        assert scalar.forcing()(coords, 0.0).shape == (5,)
        assert not scalar.forcing().is_time_dependent()
        functions, _ = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=["x*y", "x**2"],
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
        )
        vector = GalerkinManufacturedSolution(
            VectorLagrangeBasis(mesh, degree=1), functions, bkd
        )
        bkd.assert_allclose(
            bkd.asarray(vector.forcing()(coords, 0.0)),
            bkd.asarray(np.asarray(functions["forcing"](coords)).T),
            rtol=1e-14,
        )

    def test_velocity_matches_coordinate_shape(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=2, ny=2, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        ms = GalerkinManufacturedSolution(
            LagrangeBasis(mesh, degree=1),
            _adr_functions(bkd, NonConservativeAdvection()),
            bkd,
        )
        coords = np.random.rand(2, 3, 4)
        values = ms.velocity()(coords, 0.0)
        assert values.shape == (2, 3, 4)
        bkd.assert_allclose(
            bkd.asarray(values[0]), bkd.asarray(np.ones((3, 4))), rtol=1e-12
        )

    def test_natural_data_needs_gradient(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        functions = dict(_adr_functions(bkd, NonConservativeAdvection()))
        ms = GalerkinManufacturedSolution(basis, functions, bkd)
        law = _adr_physics(ms, NonConservativeAdvection()).flux_law()
        del functions["gradient"]
        no_gradient = GalerkinManufacturedSolution(basis, functions, bkd)
        with pytest.raises(ValueError, match="gradient"):
            no_gradient.neumann("right", law)

    def test_rejects_non_law(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = LagrangeBasis(
            StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), degree=1
        )
        ms = GalerkinManufacturedSolution(
            basis, _adr_functions(bkd, NonConservativeAdvection()), bkd
        )
        with pytest.raises(TypeError, match="NormalFluxLawProtocol"):
            ms.robin("right", 1.0, object())  # type: ignore[arg-type]


def _square_adr_error(
    nx: int,
    degree: int,
    advection_form: AdvectionFormProtocol,
    bkd: Backend[Array],
) -> float:
    mesh = StructuredMesh2D(
        nx=nx, ny=nx, bounds=[[-1.0, 1.0], [-1.0, 1.0]], bkd=bkd
    )
    basis: GalerkinBasisProtocol[Array] = LagrangeBasis(mesh, degree=degree)
    ms = GalerkinManufacturedSolution(
        basis, _adr_functions(bkd, advection_form), bkd
    )
    physics = _adr_physics(ms, advection_form)
    law = physics.flux_law()
    bcs: List[BoundaryConditionRole[Array]] = [
        ms.dirichlet("left"),
        ms.neumann("right", law),
        ms.robin("bottom", 2.0, law),
        ms.neumann("top", law),
    ]
    return _mass_norm_error(compose_galerkin_system(physics, bcs), ms)


def _disk_adr_error(
    nrefs: int,
    degree: int,
    tmp_path: Path,
    bkd: Backend[Array],
    robin: bool,
    reverse_normal: bool = False,
) -> float:
    basis: GalerkinBasisProtocol[Array] = LagrangeBasis(
        _disk_mesh(nrefs, tmp_path, bkd), degree=degree
    )
    form = NonConservativeAdvection()
    ms = GalerkinManufacturedSolution(basis, _adr_functions(bkd, form), bkd)
    physics = _adr_physics(ms, form)
    law: NormalFluxLawProtocol = physics.flux_law()
    if reverse_normal:
        law = _ReversedNormal(law)
    bc: BoundaryConditionRole[Array] = (
        ms.robin("circle", 1.5, law) if robin else ms.neumann("circle", law)
    )
    return _mass_norm_error(compose_galerkin_system(physics, [bc]), ms)


class TestNaturalDataConverges:
    """The law-built data converges at the element order, so it is the
    data the physics enforces."""

    @pytest.mark.parametrize(
        "advection_form",
        [NonConservativeAdvection(), ConservativeAdvection()],
        ids=["nonconservative", "conservative"],
    )
    @pytest.mark.parametrize("degree", [1, 2])
    def test_square_mixed_conditions(
        self,
        numpy_bkd: NumpyBkd,
        degree: int,
        advection_form: AdvectionFormProtocol,
    ) -> None:
        errors = [
            _square_adr_error(nx, degree, advection_form, numpy_bkd)
            for nx in (4, 8, 16)
        ]
        rates = _rates(errors)
        assert np.all(rates > degree + 0.8), (errors, rates)

    @pytest.mark.parametrize("robin", [False, True])
    @pytest.mark.parametrize("degree", [1, 2])
    def test_disk_uses_facet_normals(
        self, numpy_bkd: NumpyBkd, tmp_path: Path, degree: int, robin: bool
    ) -> None:
        """The whole boundary is natural, with normals in every direction."""
        errors = [
            _disk_adr_error(nrefs, degree, tmp_path, numpy_bkd, robin)
            for nrefs in (2, 3, 4)
        ]
        rates = _rates(errors)
        assert np.all(rates > degree + 0.8), (errors, rates)

    def test_disk_with_reversed_normal_does_not_converge(
        self, numpy_bkd: NumpyBkd, tmp_path: Path
    ) -> None:
        """Wrong-model control: inward normals give data for a different
        problem, so the error stalls at O(1)."""
        errors = [
            _disk_adr_error(
                nrefs, 1, tmp_path, numpy_bkd, False, reverse_normal=True
            )
            for nrefs in (2, 3, 4)
        ]
        assert np.all(np.abs(_rates(errors)) < 0.3), errors
        assert errors[-1] > 1e-2

    def test_vector_traction(self, numpy_bkd: NumpyBkd) -> None:
        """Linear elasticity with traction on three sides, P1."""
        bkd = numpy_bkd
        lam, mu = 1.5, 0.8
        functions, _ = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=["0.1*sin(x + 0.5*y) + 0.05", "0.05*cos(0.3*x + y)"],
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=bkd,
        )
        errors = []
        for nx in (4, 8, 16):
            mesh = StructuredMesh2D(
                nx=nx, ny=nx, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
            )
            basis = VectorLagrangeBasis(mesh, degree=1)
            ms = GalerkinManufacturedSolution(basis, functions, bkd)
            physics = CompositeLinearElasticity.from_uniform(
                basis, 1.0, 0.3, bkd, body_force=ms.forcing()
            )
            nelems = mesh.nelements()
            physics.set_lame_parameters(
                np.full(nelems, lam), np.full(nelems, mu)
            )
            law = physics.flux_law()
            bcs: List[BoundaryConditionRole[Array]] = [
                ms.dirichlet("left"),
                ms.neumann("right", law),
                ms.neumann("bottom", law),
                ms.robin("top", 2.0, law),
            ]
            errors.append(
                _mass_norm_error(compose_galerkin_system(physics, bcs), ms)
            )
        rates = _rates(errors)
        assert np.all(rates > 1.8), (errors, rates)

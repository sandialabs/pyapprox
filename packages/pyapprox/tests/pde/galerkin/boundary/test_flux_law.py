"""Tests for the pointwise normal-flux laws and each physics' ``flux_law``.

Each physics' law is compared, pointwise, with the flux the manufactured
layer derives symbolically (sympy) from the same strong form. The two are
independent: the law is NumPy arithmetic written once per constitutive
relation, the manufactured flux is differentiated from the solution string.
Points and unit normals are arbitrary because the laws are pointwise; a
facet normal is one particular choice.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    ConstantDiffusion,
    NodalFieldDiffusion,
)
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import (
    AdvectiveFlux,
    DiffusiveFlux,
    FluxLawProviderProtocol,
    LinearElasticTraction,
    NeoHookeanTraction,
    NormalFluxLawProtocol,
    PK1Traction,
    SumFlux,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    BurgersPhysics,
    CompositeHyperelasticityPhysics,
    CompositeLinearElasticity,
    Helmholtz,
    HyperelasticityPhysics,
)
from pyapprox.pde.galerkin.physics.quasilinear_diffusion import (
    QuasilinearDiffusion,
)
from pyapprox.pde.manufactured import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
)
from pyapprox.pde.manufactured.burgers import ManufacturedBurgers1D
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.manufactured.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Pts = NDArray[np.floating[Any]]
_Fn = Callable[..., Any]


def _points_and_normals(ndim: int, npts: int = 7) -> Tuple[_Pts, _Pts]:
    """Points inside the unit box and arbitrary unit normals."""
    rng = np.random.default_rng(3)
    coords = 0.05 + 0.9 * rng.random((ndim, npts))
    normal = rng.normal(size=(ndim, npts))
    return coords, normal / np.linalg.norm(normal, axis=0)


def _scalar_state(functions: dict, coords: _Pts) -> Tuple[_Pts, _Pts]:
    """Exact ``u`` ``(1, npts)`` and ``grad_u`` ``(1, ndim, npts)``."""
    u = np.asarray(functions["solution"](coords)).reshape(1, -1)
    grad_u = np.asarray(functions["gradient"](coords)).T[np.newaxis]
    return u, grad_u


def _vector_state(functions: dict, coords: _Pts) -> Tuple[_Pts, _Pts]:
    """Exact ``u`` ``(ncomp, npts)`` and ``grad_u`` ``(ncomp, ndim, npts)``."""
    u = np.asarray(functions["solution"](coords)).T
    grad_u = np.transpose(np.asarray(functions["gradient"](coords)), (0, 2, 1))
    return u, grad_u


def _scalar_flux_dot_n(flux: _Fn, coords: _Pts, normal: _Pts) -> _Pts:
    """A manufactured scalar flux vector, ``(npts, ndim)``, dotted with n."""
    ret: _Pts = np.sum(np.asarray(flux(coords)).T * normal, axis=0)[np.newaxis]
    return ret


def _tensor_dot_n(tensor: _Fn, coords: _Pts, normal: _Pts) -> _Pts:
    """A manufactured stress, ``(ncomp, npts, ndim)``, applied to n."""
    ret: _Pts = np.einsum("ipj,jp->ip", np.asarray(tensor(coords)), normal)
    return ret


def _check_law(
    physics: FluxLawProviderProtocol,
    coords: _Pts,
    u: _Pts,
    grad_u: _Pts,
    normal: _Pts,
    expected: _Pts,
) -> None:
    bkd = NumpyBkd()
    assert isinstance(physics, FluxLawProviderProtocol)
    law = physics.flux_law()
    assert isinstance(law, NormalFluxLawProtocol)
    actual = law.normal_flux(coords, u, grad_u, normal, 0.0)
    assert actual.shape == expected.shape
    bkd.assert_allclose(
        bkd.asarray(actual), bkd.asarray(expected), rtol=1e-12, atol=1e-13
    )


def _scalar_basis_2d() -> LagrangeBasis[NDArray[Any]]:
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


def _vector_basis_2d(nx: int = 3) -> VectorLagrangeBasis[NDArray[Any]]:
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(
        nx=nx, ny=nx, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )
    return VectorLagrangeBasis(mesh, degree=1)


def _lame(E: float, nu: float) -> Tuple[float, float]:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


class TestScalarPhysicsLaws:
    """Diffusion-type physics against the manufactured diffusive flux."""

    @pytest.mark.parametrize("conservative", [False, True])
    def test_advection_diffusion(self, conservative: bool) -> None:
        """Non-conservative: D grad(u).n. Conservative also has -(v.n)u,
        so it matches the manufactured total flux -(-D grad u + v u).n."""
        bkd = NumpyBkd()
        man = ManufacturedAdvectionDiffusionReaction(
            sol_str="sin(x)*cos(2*y)+x*y",
            nvars=2,
            diff_str="1+x*y",
            react_str="0",
            vel_strs=["1+y", "x"],
            bkd=bkd,
            oned=True,
            conservative=conservative,
        )

        def diffusivity(x: _Pts) -> _Pts:
            return np.asarray(1.0 + x[0] * x[1])

        def velocity(x: _Pts) -> _Pts:
            return np.stack([1.0 + x[1], x[0]])

        physics = AdvectionDiffusionReaction(
            basis=_scalar_basis_2d(),
            diffusivity=diffusivity,
            bkd=bkd,
            velocity=velocity,
            conservative=conservative,
        )
        coords, normal = _points_and_normals(2)
        u, grad_u = _scalar_state(man.functions, coords)
        flux_key = "flux" if conservative else "diffusive_flux"
        expected = -_scalar_flux_dot_n(man.functions[flux_key], coords, normal)
        _check_law(physics, coords, u, grad_u, normal, expected)

    def test_conservative_without_velocity_is_diffusive(self) -> None:
        """No velocity, no advective boundary term."""
        bkd = NumpyBkd()
        physics = AdvectionDiffusionReaction(
            basis=_scalar_basis_2d(), diffusivity=2.0, bkd=bkd, conservative=True
        )
        assert isinstance(physics.flux_law(), DiffusiveFlux)

    def test_helmholtz(self) -> None:
        bkd = NumpyBkd()
        man = ManufacturedHelmholtz(
            sol_str="sin(x)*y**2", nvars=2, sqwavenum_str="2", bkd=bkd, oned=True
        )
        physics = Helmholtz(basis=_scalar_basis_2d(), wavenumber=2.0, bkd=bkd)
        coords, normal = _points_and_normals(2)
        u, grad_u = _scalar_state(man.functions, coords)
        expected = np.sum(grad_u[0] * normal, axis=0)[np.newaxis]
        _check_law(physics, coords, u, grad_u, normal, expected)

    def test_burgers(self) -> None:
        """The advective term is not integrated by parts: nu u_x n."""
        bkd = NumpyBkd()
        man = ManufacturedBurgers1D(
            sol_str="sin(x)+x**2", visc_str="0.3", bkd=bkd, oned=True
        )
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        physics = BurgersPhysics(LagrangeBasis(mesh, degree=1), 0.3, bkd)
        coords, normal = _points_and_normals(1)
        u, grad_u = _scalar_state(man.functions, coords)
        expected = -_scalar_flux_dot_n(
            man.functions["diffusive_flux"], coords, normal
        )
        _check_law(physics, coords, u, grad_u, normal, expected)

    def test_quasilinear(self) -> None:
        """a(x) kappa(u) grad(u).n; a(x) = 1 + x is linear, so the P1
        nodal field reproduces it exactly."""
        bkd = NumpyBkd()
        basis = _scalar_basis_2d()
        dofs = 1.0 + bkd.to_numpy(basis.dof_coordinates())[0]

        def kappa(u: _Pts) -> _Pts:
            return 1.0 + u**2

        def kappa_deriv(u: _Pts) -> _Pts:
            return 2.0 * u

        def kappa_second_deriv(u: _Pts) -> _Pts:
            return np.full_like(u, 2.0)

        physics = QuasilinearDiffusion(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis, dofs=bkd.asarray(dofs)),
            bkd=bkd,
            kappa=kappa,
            kappa_deriv=kappa_deriv,
            kappa_second_deriv=kappa_second_deriv,
        )
        man = ManufacturedHelmholtz(
            sol_str="sin(x)*y", nvars=2, sqwavenum_str="0", bkd=bkd, oned=True
        )
        coords, normal = _points_and_normals(2)
        u, grad_u = _scalar_state(man.functions, coords)
        expected = (
            (1.0 + coords[0])
            * kappa(u[0])
            * np.sum(grad_u[0] * normal, axis=0)
        )[np.newaxis]
        _check_law(physics, coords, u, grad_u, normal, expected)


class TestElasticPhysicsLaws:
    """Elastic physics against the manufactured stress applied to n."""

    def test_linear_elasticity(self) -> None:
        bkd = NumpyBkd()
        E, nu = 2.0, 0.3
        lam, mu = _lame(E, nu)
        man = ManufacturedLinearElasticityEquations(
            sol_strs=["sin(x)*y", "x**2+cos(y)"],
            nvars=2,
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=bkd,
            oned=True,
        )
        physics = CompositeLinearElasticity.from_uniform(
            _vector_basis_2d(), youngs_modulus=E, poisson_ratio=nu, bkd=bkd
        )
        coords, normal = _points_and_normals(2)
        u, grad_u = _vector_state(man.functions, coords)
        expected = _tensor_dot_n(man.functions["flux"], coords, normal)
        _check_law(physics, coords, u, grad_u, normal, expected)

    def test_hyperelasticity(self) -> None:
        bkd = NumpyBkd()
        stress = NeoHookeanStress(lamda=1.2, mu=0.7)
        man = ManufacturedHyperelasticityEquations(
            sol_strs=["0.1*sin(x)*y", "0.1*x**2"],
            nvars=2,
            stress_model=stress,
            bkd=bkd,
            oned=True,
        )
        physics = HyperelasticityPhysics(_vector_basis_2d(), stress, bkd)
        coords, normal = _points_and_normals(2)
        u, grad_u = _vector_state(man.functions, coords)
        expected = _tensor_dot_n(man.functions["flux"], coords, normal)
        _check_law(physics, coords, u, grad_u, normal, expected)


class TestCompositeMaterialLaws:
    """Each point takes the material of the element containing it."""

    _MATERIALS = {"left": (1.0, 0.3), "right": (5.0, 0.2)}

    def _split(self, nx: int) -> Tuple[Any, dict]:
        """A 4x4 mesh split at x = 1/2 into two materials."""
        basis = _vector_basis_2d(nx)
        mesh = basis.skfem_basis().mesh
        centers_x = mesh.p[0, mesh.t].mean(axis=0)
        elements = {
            "left": np.flatnonzero(centers_x < 0.5),
            "right": np.flatnonzero(centers_x >= 0.5),
        }
        return basis, elements

    def _expected(
        self,
        coords: _Pts,
        normal: _Pts,
        traction: Callable[[str], _Fn],
    ) -> _Pts:
        """Each point's traction from its own material's manufactured flux."""
        left = coords[0] < 0.5
        expected = np.zeros((2, coords.shape[1]))
        for name, mask in (("left", left), ("right", ~left)):
            expected[:, mask] = _tensor_dot_n(
                traction(name), coords, normal
            )[:, mask]
        return expected

    def _coords(self) -> Tuple[_Pts, _Pts]:
        # Keep points off the material interface x = 1/2.
        coords, normal = _points_and_normals(2, npts=11)
        coords[0] = np.where(
            np.abs(coords[0] - 0.5) < 0.05, coords[0] + 0.1, coords[0]
        )
        return coords, normal

    def test_composite_linear_elasticity(self) -> None:
        bkd = NumpyBkd()
        basis, elements = self._split(4)
        physics = CompositeLinearElasticity(
            basis, dict(self._MATERIALS), elements, bkd
        )
        sol_strs = ["sin(x)*y", "x**2+cos(y)"]

        def traction(name: str) -> _Fn:
            lam, mu = _lame(*self._MATERIALS[name])
            man = ManufacturedLinearElasticityEquations(
                sol_strs=sol_strs, nvars=2, lambda_str=str(lam),
                mu_str=str(mu), bkd=bkd, oned=True,
            )
            return man.functions["flux"]

        coords, normal = self._coords()
        man = ManufacturedLinearElasticityEquations(
            sol_strs=sol_strs, nvars=2, lambda_str="1", mu_str="1",
            bkd=bkd, oned=True,
        )
        u, grad_u = _vector_state(man.functions, coords)
        _check_law(
            physics, coords, u, grad_u, normal,
            self._expected(coords, normal, traction),
        )

    def test_composite_hyperelasticity(self) -> None:
        bkd = NumpyBkd()
        basis, elements = self._split(4)
        physics = CompositeHyperelasticityPhysics(
            basis, dict(self._MATERIALS), elements, bkd
        )
        sol_strs = ["0.1*sin(x)*y", "0.1*x**2"]

        def manufactured(name: str) -> ManufacturedHyperelasticityEquations:
            lam, mu = _lame(*self._MATERIALS[name])
            return ManufacturedHyperelasticityEquations(
                sol_strs=sol_strs, nvars=2,
                stress_model=NeoHookeanStress(lamda=lam, mu=mu),
                bkd=bkd, oned=True,
            )

        coords, normal = self._coords()
        u, grad_u = _vector_state(manufactured("left").functions, coords)
        _check_law(
            physics, coords, u, grad_u, normal,
            self._expected(
                coords, normal,
                lambda name: manufactured(name).functions["flux"],
            ),
        )

    def test_law_sees_updated_materials(self) -> None:
        """The law reads the current Lame values, not those at construction."""
        bkd = NumpyBkd()
        basis, elements = self._split(4)
        physics = CompositeLinearElasticity(
            basis, dict(self._MATERIALS), elements, bkd
        )
        law = physics.flux_law()
        physics.set_lame_material_values(np.array([3.0, 2.0, 3.0, 2.0]))
        coords, normal = self._coords()
        man = ManufacturedLinearElasticityEquations(
            sol_strs=["sin(x)*y", "x**2+cos(y)"], nvars=2,
            lambda_str="3", mu_str="2", bkd=bkd, oned=True,
        )
        u, grad_u = _vector_state(man.functions, coords)
        bkd.assert_allclose(
            bkd.asarray(law.normal_flux(coords, u, grad_u, normal, 0.0)),
            bkd.asarray(_tensor_dot_n(man.functions["flux"], coords, normal)),
            rtol=1e-12,
        )

    def test_per_quadrature_point_values_raise(self) -> None:
        bkd = NumpyBkd()
        basis, elements = self._split(4)
        physics = CompositeLinearElasticity(
            basis, dict(self._MATERIALS), elements, bkd
        )
        nelems = basis.skfem_basis().mesh.nelements
        nquad = basis.skfem_basis().X.shape[1]
        physics.set_lame_parameters(
            np.ones((nelems, nquad)), np.ones((nelems, nquad))
        )
        coords, normal = self._coords()
        u = np.zeros((2, coords.shape[1]))
        grad_u = np.zeros((2, 2, coords.shape[1]))
        with pytest.raises(ValueError, match="per-quadrature-point"):
            physics.flux_law().normal_flux(coords, u, grad_u, normal, 0.0)


class TestLawComposition:
    """The composite laws and their validation."""

    def test_sum_is_sum_of_terms(self) -> None:
        bkd = NumpyBkd()
        coords, normal = _points_and_normals(2)
        u = np.sin(coords[:1])
        grad_u = np.cos(coords)[np.newaxis]
        diffusive = DiffusiveFlux(ConstantDiffusion(2.0))
        stress_free = LinearElasticTraction(0.0, 0.0)
        assert np.all(
            stress_free.normal_flux(coords, u, grad_u, normal, 0.0) == 0.0
        )
        total = SumFlux([diffusive, diffusive])
        bkd.assert_allclose(
            bkd.asarray(total.normal_flux(coords, u, grad_u, normal, 0.0)),
            bkd.asarray(
                2.0 * diffusive.normal_flux(coords, u, grad_u, normal, 0.0)
            ),
            rtol=1e-14,
        )

    def test_neo_hookean_fields_match_uniform_model(self) -> None:
        """Constant fields reproduce the uniform stress model."""
        bkd = NumpyBkd()
        coords, normal = _points_and_normals(2)
        grad_u = 0.1 * np.sin(np.arange(4.0)).reshape(2, 2, 1) * coords[0]
        u = np.zeros((2, coords.shape[1]))
        bkd.assert_allclose(
            bkd.asarray(
                NeoHookeanTraction(1.2, 0.7).normal_flux(
                    coords, u, grad_u, normal, 0.0
                )
            ),
            bkd.asarray(
                PK1Traction(NeoHookeanStress(1.2, 0.7)).normal_flux(
                    coords, u, grad_u, normal, 0.0
                )
            ),
            rtol=1e-14,
        )

    def test_rejects_wrong_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SumFlux([])
        with pytest.raises(TypeError, match="NormalFluxLawProtocol"):
            SumFlux([object()])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="DiffusionFunctionProtocol"):
            DiffusiveFlux(2.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VelocityFunctionProtocol"):
            AdvectiveFlux(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StressModelProtocol"):
            PK1Traction(object())  # type: ignore[arg-type]

    def test_time_dependence_is_declared(self) -> None:
        assert not DiffusiveFlux(ConstantDiffusion(1.0)).is_time_dependent()
        assert not LinearElasticTraction(1.0, 1.0).is_time_dependent()

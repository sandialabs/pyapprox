"""Natural-BC composition: F = F_Omega + F_Gamma, terms added exactly once.

Every Galerkin physics supplies only its interior; the natural-BC terms are
added by the composition. ``check_natural_bc_composition`` verifies this for
each physics, and the two negative cases show it catches a physics that adds
a term itself or adds one twice.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.constitutive.coefficient_functions import NodalFieldDiffusion
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import NeumannBC, RobinBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    BurgersPhysics,
    CompositeHyperelasticityPhysics,
    CompositeLinearElasticity,
    Helmholtz,
    HyperelasticityPhysics,
    QuasilinearDiffusion,
)
from pyapprox.util.backends.numpy import NumpyBkd
from scipy.sparse import csr_matrix, issparse

from tests._helpers.natural_bc_conformance import check_natural_bc_composition

_Arr = NDArray[np.floating[Any]]
_BKD = NumpyBkd()


class _ScalarData:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.asarray(1.0 + coords[0] * coords[1])


class _VectorData:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.stack([1.0 + coords[1], 0.5 - coords[0] * coords[1]])


def _kappa(u: _Arr) -> _Arr:
    return np.asarray(1.0 + u**2)


def _kappa_deriv(u: _Arr) -> _Arr:
    return np.asarray(2.0 * u)


def _kappa_second_deriv(u: _Arr) -> _Arr:
    return np.full_like(u, 2.0)


def _mesh() -> StructuredMesh2D[_Arr]:
    return StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=_BKD)


def _scalar_basis() -> LagrangeBasis[_Arr]:
    return LagrangeBasis(_mesh(), degree=2)


def _vector_basis() -> VectorLagrangeBasis[_Arr]:
    return VectorLagrangeBasis(_mesh(), degree=2)


def _all_elements(basis: Any) -> Dict[str, NDArray[Any]]:
    return {"all": np.arange(basis.skfem_basis().mesh.nelements)}


# Each entry: (basis factory, physics factory taking (basis, bcs), vector?)
_PHYSICS: Dict[str, Any] = {
    "adr": (
        _scalar_basis,
        lambda b, bcs: AdvectionDiffusionReaction(
            b, 1.3, _BKD, reaction=0.4, boundary_conditions=bcs
        ),
        False,
    ),
    "helmholtz": (
        _scalar_basis,
        lambda b, bcs: Helmholtz(b, 2.0, _BKD, boundary_conditions=bcs),
        False,
    ),
    "burgers": (
        _scalar_basis,
        lambda b, bcs: BurgersPhysics(b, 0.1, _BKD, boundary_conditions=bcs),
        False,
    ),
    "quasilinear": (
        _scalar_basis,
        lambda b, bcs: QuasilinearDiffusion(
            b,
            NodalFieldDiffusion(b, dofs=1.0 + 0.1 * np.arange(b.ndofs()) / b.ndofs()),
            _BKD,
            kappa=_kappa,
            kappa_deriv=_kappa_deriv,
            kappa_second_deriv=_kappa_second_deriv,
            boundary_conditions=bcs,
        ),
        False,
    ),
    "linear_elasticity": (
        _vector_basis,
        lambda b, bcs: CompositeLinearElasticity.from_uniform(
            b, 1.0, 0.3, _BKD, boundary_conditions=bcs
        ),
        True,
    ),
    "hyperelasticity": (
        _vector_basis,
        lambda b, bcs: HyperelasticityPhysics(
            b, NeoHookeanStress(1.0, 1.0), _BKD, boundary_conditions=bcs
        ),
        True,
    ),
    "composite_hyperelasticity": (
        _vector_basis,
        lambda b, bcs: CompositeHyperelasticityPhysics(
            b, {"all": (1.0, 0.3)}, _all_elements(b), _BKD,
            boundary_conditions=bcs,
        ),
        True,
    ),
}


def _natural_bcs(basis: Any, vector: bool) -> List[Any]:
    data = _VectorData() if vector else _ScalarData()
    return [
        NeumannBC(basis, "right", data, _BKD),
        RobinBC(basis, "top", 1.7, data, _BKD),
    ]


class _SmoothScalarState:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.asarray(0.3 * np.sin(2.0 * coords[0]) * (1.0 + coords[1]))


class _SmoothVectorState:
    """A small smooth displacement: det(I + grad u) stays positive."""

    def __call__(self, coords: _Arr) -> _Arr:
        return np.stack(
            [
                0.05 * np.sin(2.0 * coords[0]) * coords[1],
                0.03 * coords[0] * coords[1],
            ]
        )


def _state(basis: Any, vector: bool) -> _Arr:
    """An admissible state: smooth, so the nonlinear physics are defined
    (random nodal noise inverts hyperelastic elements)."""
    func = _SmoothVectorState() if vector else _SmoothScalarState()
    state: _Arr = basis.interpolate(func)
    return state


@pytest.mark.parametrize("name", list(_PHYSICS))
def test_physics_composes_natural_bcs_once(name: str) -> None:
    make_basis, make_physics, vector = _PHYSICS[name]
    basis = make_basis()
    bcs = _natural_bcs(basis, vector)
    check_natural_bc_composition(
        lambda bc_list: make_physics(basis, bc_list),
        bcs,
        _state(basis, vector),
        time=0.2,
        rtol=1e-10,
        atol=1e-10,
    )


class _HelmholtzAddingRobinItself(Helmholtz[_Arr]):
    """Wrong: adds the Robin load inside its own interior."""

    def interior_residual(self, state: _Arr, time: float) -> _Arr:
        residual = super().interior_residual(state, time)
        for bc in self.weak_form_bcs():
            residual = bc.apply_to_load(residual, time)
        return residual


class _HelmholtzDoublingTerms(Helmholtz[_Arr]):
    """Wrong: adds the natural-BC terms a second time."""

    def spatial_residual(self, state: _Arr, time: float) -> _Arr:
        residual = super().spatial_residual(state, time)
        return self.natural_bc_operator().add_to_residual(residual, state, time)


@pytest.mark.parametrize(
    "cls", [_HelmholtzAddingRobinItself, _HelmholtzDoublingTerms]
)
def test_check_catches_wrong_composition(cls: Any) -> None:
    basis = _scalar_basis()
    with pytest.raises(AssertionError):
        check_natural_bc_composition(
            lambda bc_list: cls(basis, 2.0, _BKD, boundary_conditions=bc_list),
            _natural_bcs(basis, False),
            _state(basis, False),
        )


@pytest.mark.parametrize("essential", [False, True])
@pytest.mark.parametrize("natural", [False, True])
def test_bc_mix(essential: bool, natural: bool) -> None:
    """Any mix of BCs, including none: spatial - interior is exactly the
    natural terms present (essential BCs never enter F)."""
    from pyapprox.pde.galerkin.boundary.implementations import DirichletBC

    basis = _scalar_basis()
    natural_bcs = _natural_bcs(basis, False) if natural else []
    essential_bcs = [DirichletBC(basis, "left", 0.5, _BKD)] if essential else []
    physics = Helmholtz(
        basis, 2.0, _BKD, boundary_conditions=essential_bcs + natural_bcs
    )
    assert physics.natural_bc_operator().is_empty() == (not natural)
    state = _state(basis, False)
    expected = _BKD.zeros((basis.ndofs(),))
    for bc in natural_bcs:
        expected = bc.apply_to_residual(expected, state, 0.0)
    _BKD.assert_allclose(
        physics.spatial_residual(state, 0.0)
        - physics.interior_residual(state, 0.0),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_check_catches_factory_ignoring_bcs() -> None:
    """A factory that drops the BC list must not pass as conforming."""
    basis = _scalar_basis()
    with pytest.raises(AssertionError, match="exactly the given"):
        check_natural_bc_composition(
            lambda bc_list: Helmholtz(basis, 2.0, _BKD),
            _natural_bcs(basis, False),
            _state(basis, False),
        )


class TestNaturalBCOperator:
    """The operator on its own, with a stub interior."""

    def test_empty_operator_is_identity(self) -> None:
        operator: NaturalBCOperator[_Arr] = NaturalBCOperator([])
        assert operator.is_empty()
        residual = _BKD.asarray(np.arange(4.0))
        jacobian = csr_matrix(np.eye(4))
        assert operator.add_to_residual(residual, residual, 0.0) is residual
        assert operator.add_to_jacobian(jacobian, residual, 0.0) is jacobian

    def test_sums_the_terms(self) -> None:
        basis = _scalar_basis()
        bcs = _natural_bcs(basis, False)
        operator: NaturalBCOperator[_Arr] = NaturalBCOperator(bcs)
        n = basis.ndofs()
        state = _state(basis, False)
        interior = _BKD.asarray(np.linspace(-1.0, 1.0, n))
        expected = interior
        for bc in bcs:
            expected = bc.apply_to_residual(expected, state, 0.0)
        _BKD.assert_allclose(
            operator.add_to_residual(interior, state, 0.0), expected, rtol=1e-14
        )
        jac = operator.add_to_jacobian(csr_matrix((n, n)), state, 0.0)
        assert issparse(jac)
        expected_jac = np.zeros((n, n))
        for bc in bcs:
            expected_jac = bc.apply_to_jacobian(expected_jac, state, 0.0)
        _BKD.assert_allclose(
            _BKD.asarray(jac.toarray()), _BKD.asarray(expected_jac), rtol=1e-14
        )

    def test_rejects_non_term(self) -> None:
        with pytest.raises(TypeError, match="WeakFormBCProtocol"):
            NaturalBCOperator([object()])  # type: ignore[list-item]

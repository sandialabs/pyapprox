"""Natural-BC composition: F = F_Omega + F_Gamma, terms added exactly once.

Every Galerkin physics supplies only its interior; the natural-BC terms are
added by ``compose_galerkin_system``. The per-physics test checks that the
terms combine with each physics' own state layout and Jacobian format
(sparse or dense, scalar or interleaved vector DOFs). The terms' own
``apply_to_residual``/``apply_to_jacobian`` are the reference; their
correctness is tested independently.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse

from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.constitutive.coefficient_functions import NodalFieldDiffusion
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
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
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.util.backends.numpy import NumpyBkd

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


# Each entry: (basis factory, BC-free physics factory taking basis, vector?)
_PHYSICS: Dict[str, Any] = {
    "adr": (
        _scalar_basis,
        lambda b: AdvectionDiffusionReaction(b, 1.3, _BKD, reaction=0.4),
        False,
    ),
    "helmholtz": (
        _scalar_basis,
        lambda b: Helmholtz(b, 2.0, _BKD),
        False,
    ),
    "burgers": (
        _scalar_basis,
        lambda b: BurgersPhysics(b, 0.1, _BKD),
        False,
    ),
    "quasilinear": (
        _scalar_basis,
        lambda b: QuasilinearDiffusion(
            b,
            NodalFieldDiffusion(b, dofs=1.0 + 0.1 * np.arange(b.ndofs()) / b.ndofs()),
            _BKD,
            kappa=_kappa,
            kappa_deriv=_kappa_deriv,
            kappa_second_deriv=_kappa_second_deriv,
        ),
        False,
    ),
    "linear_elasticity": (
        _vector_basis,
        lambda b: CompositeLinearElasticity.from_uniform(b, 1.0, 0.3, _BKD),
        True,
    ),
    "hyperelasticity": (
        _vector_basis,
        lambda b: HyperelasticityPhysics(b, NeoHookeanStress(1.0, 1.0), _BKD),
        True,
    ),
    "composite_hyperelasticity": (
        _vector_basis,
        lambda b: CompositeHyperelasticityPhysics(
            b, {"all": (1.0, 0.3)}, _all_elements(b), _BKD
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


def _dense(matrix: Any) -> _Arr:
    """A Jacobian as a dense array (physics Jacobians may be sparse)."""
    if issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


@pytest.mark.parametrize("name", list(_PHYSICS))
def test_composition_adds_natural_terms_once(name: str) -> None:
    """F - F_Omega is exactly the sum of the natural terms, for the
    residual and the Jacobian, in each physics' own state layout and
    Jacobian format; the essential BC composed alongside stays out of F."""
    make_basis, make_physics, vector = _PHYSICS[name]
    basis = make_basis()
    physics = make_physics(basis)
    natural_bcs = _natural_bcs(basis, vector)
    essential_bc = DirichletBC(basis, "left", 0.0, _BKD)
    operator = compose_galerkin_system(
        physics, [*natural_bcs, essential_bc]
    ).spatial_operator()
    state = _state(basis, vector)
    time = 0.2
    nstates = basis.ndofs()
    expected_residual = _BKD.zeros((nstates,))
    expected_jacobian: Any = np.zeros((nstates, nstates))
    for bc in natural_bcs:
        expected_residual = bc.apply_to_residual(expected_residual, state, time)
        expected_jacobian = bc.apply_to_jacobian(expected_jacobian, state, time)
    _BKD.assert_allclose(
        operator.spatial_residual(state, time)
        - physics.interior_residual(state, time),
        expected_residual,
        rtol=1e-10,
        atol=1e-10,
    )
    _BKD.assert_allclose(
        _BKD.asarray(
            _dense(operator.spatial_jacobian(state, time))
            - _dense(physics.interior_jacobian(state, time))
        ),
        _BKD.asarray(_dense(expected_jacobian)),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("essential", [False, True])
@pytest.mark.parametrize("natural", [False, True])
def test_bc_mix(essential: bool, natural: bool) -> None:
    """Any mix of BCs, including none: spatial - interior is exactly the
    natural terms present (essential BCs never enter F)."""
    basis = _scalar_basis()
    natural_bcs = _natural_bcs(basis, False) if natural else []
    essential_bcs = [DirichletBC(basis, "left", 0.5, _BKD)] if essential else []
    physics = Helmholtz(basis, 2.0, _BKD)
    system = compose_galerkin_system(physics, essential_bcs + natural_bcs)
    operator = system.spatial_operator()
    assert isinstance(operator, ComposedSpatialOperator)
    assert operator.natural_bcs().is_empty() == (not natural)
    state = _state(basis, False)
    expected = _BKD.zeros((basis.ndofs(),))
    for bc in natural_bcs:
        expected = bc.apply_to_residual(expected, state, 0.0)
    _BKD.assert_allclose(
        operator.spatial_residual(state, 0.0)
        - physics.interior_residual(state, 0.0),
        expected,
        rtol=1e-12,
        atol=1e-12,
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

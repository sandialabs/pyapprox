"""Sign and shape convention of Galerkin weak-form (natural) BC terms.

A natural BC is a term ``c(u, t)`` in the spatial operator,
``F = F_interior + c``, with the physics' sign convention ``F = b - K u``:

    Neumann:  c = int h . phi                    (dc/du = 0)
    Robin:    c = int (g - a u) . phi            (dc/du = -K_Gamma)

so ``apply_to_residual`` adds ``c`` and ``apply_to_jacobian`` adds
``dc/du``.

The reference is independent of the BC code: on a uniform mesh with
linear elements and constant data, the boundary integrals along an edge
are closed-form. With node spacing ``h`` along the edge, the load
``int g . phi_i`` is ``g h`` at interior edge nodes and ``g h / 2`` at the
two end nodes, and ``K_Gamma = a M_edge`` with ``M_edge`` the 1D linear
mass matrix, ``(h/6) [[2, 1], [1, 2]]`` per segment, per component. The
expected term is assembled from DOF coordinates alone. (Comparing against a
physics would be circular: the physics adds these terms with the same
methods.)
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import NeumannBC, RobinBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.physics.helmholtz import Helmholtz
from pyapprox.util.backends.numpy import NumpyBkd
from scipy.sparse import issparse

_Arr = NDArray[np.floating[Any]]


def _dense(mat: Any) -> _Arr:
    return np.asarray(mat.toarray() if issparse(mat) else mat)


class _ScalarData:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.asarray(1.0 + coords[0] * coords[1])


class _VectorData:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.stack([1.0 + coords[1], 0.5 - coords[0] * coords[1]])


def _scalar_case(bcs: List[Any]) -> Any:
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    return Helmholtz(basis, 2.0, bkd, boundary_conditions=bcs), basis


def _vector_case(bcs: List[Any]) -> Any:
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=2)
    physics = CompositeLinearElasticity.from_uniform(
        basis, 1.0, 0.3, bkd, boundary_conditions=bcs
    )
    return physics, basis


def _make_bc(kind: str, basis: Any, vector: bool) -> Any:
    bkd = NumpyBkd()
    data = TimeIndependent(_VectorData() if vector else _ScalarData())
    if kind == "robin":
        return RobinBC(basis, "right", 1.7, data, bkd)
    return NeumannBC(basis, "right", data, bkd)


_CASES: List[Tuple[str, Callable[[List[Any]], Any], bool]] = [
    ("scalar", _scalar_case, False),
    ("vector", _vector_case, True),
]


@pytest.mark.parametrize("kind", ["robin", "neumann"])
@pytest.mark.parametrize("name,make_physics,vector", _CASES)
class TestWeakFormTermConvention:
    def _setup(self, make_physics: Any, vector: bool, kind: str) -> Any:
        bare, basis = make_physics([])
        bc = _make_bc(kind, basis, vector)
        with_bc, _ = make_physics([bc])
        rng = np.random.default_rng(3)
        state = bare.bkd().asarray(rng.normal(0.0, 1.0, bare.nstates()))
        return bare, with_bc, bc, state

    def test_jacobian_matches_finite_differences(
        self, name: str, make_physics: Any, vector: bool, kind: str
    ) -> None:
        bare, _, bc, state = self._setup(make_physics, vector, kind)
        bkd = bare.bkd()
        n = bare.nstates()

        def residual(samples: _Arr) -> _Arr:
            return bkd.stack(
                [
                    bc.apply_to_residual(bkd.zeros((n,)), samples[:, ii], 0.0)
                    for ii in range(samples.shape[1])
                ],
                axis=1,
            )

        def jacobian(sample: _Arr) -> _Arr:
            return bkd.asarray(
                _dense(bc.apply_to_jacobian(bkd.zeros((n, n)), sample[:, 0], 0.0))
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=n, nvars=n, fun=residual, jacobian=jacobian, bkd=bkd
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        # Linear in u: Neumann's Jacobian is exactly zero, so only the
        # Robin case has a sweep to check.
        if kind == "robin":
            assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-6


_ROBIN_A = 1.7
_G_SCALAR = 0.8
_G_VECTOR = np.array([0.8, -0.3])


class _ConstantVectorData:
    def __call__(self, coords: _Arr) -> _Arr:
        return np.outer(_G_VECTOR, np.ones(coords.shape[1]))


def _independent_term(
    basis: Any, kind: str, vector: bool, state: _Arr
) -> Tuple[_Arr, _Arr]:
    """Expected (c, dc/du) on the edge x = 1, from DOF coordinates alone.

    Linear elements, constant data: each edge segment of length h adds
    g h / 2 to the load at its two nodes and a (h/6)[[2, 1], [1, 2]] to
    K_Gamma, separately for each component.
    """
    coords = np.asarray(basis.dof_coordinates())
    n = coords.shape[1]
    ncomp = 2 if vector else 1
    load = np.zeros(n)
    k_gamma = np.zeros((n, n))
    for comp in range(ncomp):
        dofs = [
            i for i in range(n)
            if np.isclose(coords[0, i], 1.0) and i % ncomp == comp
        ]
        dofs.sort(key=lambda i: coords[1, i])
        g = _G_VECTOR[comp] if vector else _G_SCALAR
        for a, b in zip(dofs[:-1], dofs[1:]):
            h = coords[1, b] - coords[1, a]
            load[[a, b]] += g * h / 2.0
            if kind == "robin":
                k_gamma[np.ix_([a, b], [a, b])] += (
                    _ROBIN_A * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
                )
    return load - k_gamma @ state, -k_gamma


@pytest.mark.parametrize("kind", ["robin", "neumann"])
@pytest.mark.parametrize("vector", [False, True])
def test_term_matches_independent_assembly(kind: str, vector: bool) -> None:
    """The term and its Jacobian equal the closed-form boundary integrals."""
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=1) if vector else LagrangeBasis(
        mesh, degree=1
    )
    data: Any = _ConstantVectorData() if vector else _G_SCALAR
    bc = (
        RobinBC(basis, "right", _ROBIN_A, data, bkd)
        if kind == "robin"
        else NeumannBC(basis, "right", data, bkd)
    )
    n = basis.ndofs()
    state = bkd.asarray(np.random.default_rng(3).normal(0.0, 1.0, n))
    expected_c, expected_jac = _independent_term(
        basis, kind, vector, bkd.to_numpy(state)
    )
    bkd.assert_allclose(
        bc.apply_to_residual(bkd.zeros((n,)), state, 0.0),
        bkd.asarray(expected_c),
        rtol=1e-12,
        atol=1e-14,
    )
    bkd.assert_allclose(
        bkd.asarray(_dense(bc.apply_to_jacobian(bkd.zeros((n, n)), state, 0.0))),
        bkd.asarray(expected_jac),
        rtol=1e-12,
        atol=1e-14,
    )


class _TimeVaryingData:
    def __call__(self, coords: _Arr, time: float) -> _Arr:
        return np.asarray((1.0 + time) * (1.0 + coords[0] * coords[1]))


def _ambiguous_data(coords: _Arr, time: float = 0.0) -> _Arr:
    return np.asarray(1.0 + coords[0] + 0.0 * time)


@pytest.mark.parametrize("cls", [RobinBC, NeumannBC])
class TestWeakFormDataTimeDeclaration:
    """Time dependence of natural-BC data is declared, never inferred."""

    _NDOFS = 16  # degree-1 Lagrange on a 3 x 3 quad mesh: 4 x 4 nodes

    def _bc(self, cls: Any, data: Any) -> Any:
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        assert basis.ndofs() == self._NDOFS
        if cls is RobinBC:
            return RobinBC(basis, "right", 1.7, data, bkd)
        return NeumannBC(basis, "right", data, bkd)

    def _load(self, bc: Any, time: float) -> _Arr:
        return bc.apply_to_load(bc.bkd().zeros((self._NDOFS,)), time)

    def test_bare_coordinate_callable_is_time_independent(self, cls: Any) -> None:
        bare = self._bc(cls, _ScalarData())
        declared = self._bc(cls, TimeIndependent(_ScalarData()))
        assert not bare.is_time_dependent()
        bare.bkd().assert_allclose(
            self._load(bare, 0.3), self._load(declared, 0.3), rtol=1e-14
        )

    def test_declared_time_dependent_data(self, cls: Any) -> None:
        bc = self._bc(cls, TimeDependent(_TimeVaryingData()))
        assert bc.is_time_dependent()
        bc.bkd().assert_allclose(
            self._load(bc, 1.0), 2.0 * self._load(bc, 0.0), rtol=1e-12
        )

    def test_ambiguous_callable_is_rejected(self, cls: Any) -> None:
        with pytest.raises(TypeError, match="ambiguous"):
            self._bc(cls, _ambiguous_data)

    def test_constant_is_time_independent(self, cls: Any) -> None:
        assert not self._bc(cls, 2.5).is_time_dependent()


class TestNeumannSetFluxFunc:
    """Replacing Neumann data keeps the declaration rule."""

    def _bc(self) -> Any:
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        return NeumannBC(LagrangeBasis(mesh, degree=1), "right", 1.0, bkd)

    def test_new_data_reaches_the_load(self) -> None:
        bc = self._bc()
        zero = bc.bkd().zeros((16,))
        before = bc.apply_to_load(zero, 0.0)
        bc.set_flux_func(3.0)
        bc.bkd().assert_allclose(bc.apply_to_load(zero, 0.0), 3.0 * before)
        bc.set_flux_func(TimeDependent(_TimeVaryingData()))
        assert bc.is_time_dependent()

    def test_ambiguous_data_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ambiguous"):
            self._bc().set_flux_func(_ambiguous_data)

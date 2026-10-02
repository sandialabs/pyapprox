"""Sign and shape convention of Galerkin weak-form (natural) BC terms.

A natural BC is a term ``c(u, t)`` in the spatial operator,
``F = F_interior + c``, with the physics' sign convention ``F = b - K u``:

    Neumann:  c = int h . phi                    (dc/du = 0)
    Robin:    c = int (g - a u) . phi            (dc/du = -K_Gamma)

so ``apply_to_residual`` adds ``c`` and ``apply_to_jacobian`` adds
``dc/du``. The reference is the physics itself: each physics adds these
terms through its load and stiffness, so a BC's residual contribution must
equal the physics' spatial residual with the BC minus without it. This
holds for scalar and vector bases alike.
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
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
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

    def test_residual_is_the_physics_term(
        self, name: str, make_physics: Any, vector: bool, kind: str
    ) -> None:
        bare, with_bc, bc, state = self._setup(make_physics, vector, kind)
        bkd = bare.bkd()
        term = bc.apply_to_residual(bkd.zeros((bare.nstates(),)), state, 0.0)
        expected = with_bc.spatial_residual(state, 0.0) - bare.spatial_residual(
            state, 0.0
        )
        bkd.assert_allclose(term, expected, rtol=1e-12, atol=1e-12)

    def test_jacobian_is_the_physics_term(
        self, name: str, make_physics: Any, vector: bool, kind: str
    ) -> None:
        bare, with_bc, bc, state = self._setup(make_physics, vector, kind)
        bkd = bare.bkd()
        n = bare.nstates()
        term = _dense(bc.apply_to_jacobian(bkd.zeros((n, n)), state, 0.0))
        expected = _dense(with_bc.spatial_jacobian(state, 0.0)) - _dense(
            bare.spatial_jacobian(state, 0.0)
        )
        bkd.assert_allclose(
            bkd.asarray(term), bkd.asarray(expected), rtol=1e-12, atol=1e-12
        )

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

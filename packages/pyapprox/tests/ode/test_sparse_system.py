"""sparse_system and SparseJacobianOperator on a Galerkin model.

Bound to NumPy and SciPy, so tested on NumPy only: the shifted Jacobian
of the Galerkin advection-diffusion adapter against today's Newton matrix
:math:`M - cJ` (with :math:`\\sigma = 1/c`), and the default sparse LU
against a dense solve.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.ode.implicit_stage_solve import newton_matrix
from pyapprox.ode.protocols import ImplicitSystemProtocol, JacobianOperatorProtocol
from pyapprox.ode.semi_explicit_system import sparse_system
from pyapprox.ode.sparse_jacobian_operator import (
    SparseJacobianOperator,
    SparseJacobianSnapshot,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat

_Np = NDArray[Any]
_POINT = LinearizationPoint(0, ("step", 1))


def _adapter(bkd: NumpyBkd) -> GalerkinPhysicsToODEResidualAdapter[_Np]:
    mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
    physics = AdvectionDiffusionReaction(
        basis=LagrangeBasis(mesh, degree=2), diffusivity=0.5, bkd=bkd, reaction=2.0
    )
    return GalerkinPhysicsToODEResidualAdapter(compose_galerkin_system(physics))


class TestSparseSystem:
    def test_shifted_jacobian_is_todays_newton_matrix(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        adapter = _adapter(numpy_bkd)
        system: ImplicitSystemProtocol[_Np, _Np, CSRFormat] = sparse_system(adapter)
        n = adapter.mass_matrix().as_matrix().shape[0]
        state = np.linspace(0.0, 1.0, n)
        coefficient = 0.05
        op = system.shifted_jacobian(state, np.zeros(n), 0.0, 1.0 / coefficient, _POINT)
        assert op.linearization() == _POINT
        newton = csr_matrix(newton_matrix(adapter, state, 0.0, coefficient)).toarray()
        x = np.random.default_rng(0).normal(size=(n, 2))
        numpy_bkd.assert_allclose(coefficient * op.apply(x), newton @ x, rtol=1e-12)
        numpy_bkd.assert_allclose(
            coefficient * op.apply_transpose(x), newton.T @ x, rtol=1e-12
        )
        assert isinstance(op.format(), CSRFormat)

    def test_default_solver_solves(self, numpy_bkd: NumpyBkd) -> None:
        adapter = _adapter(numpy_bkd)
        system = sparse_system(adapter)
        n = adapter.mass_matrix().as_matrix().shape[0]
        state, shift = np.linspace(0.0, 1.0, n), 20.0
        op = system.shifted_jacobian(state, np.zeros(n), 0.0, shift, _POINT)
        dense = op.format().matrix().toarray()
        b = np.random.default_rng(1).normal(size=(n, 1))
        x, info = system.solver().setup(op).solve(b)
        assert info.converged
        numpy_bkd.assert_allclose(x, np.linalg.solve(dense, b), rtol=1e-10)


class TestSparseJacobianOperator:
    def test_snapshot_carries_its_point_and_the_jacobian(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        adapter = _adapter(numpy_bkd)
        n = adapter.mass_matrix().as_matrix().shape[0]
        producer: JacobianOperatorProtocol[_Np, _Np, CSRFormat] = (
            SparseJacobianOperator(adapter, SpaceDescriptor("state", n))
        )
        state = np.linspace(0.0, 1.0, n)
        op = producer.jacobian_operator(state, 0.0, _POINT)
        assert isinstance(op, SparseJacobianSnapshot)
        assert op.linearization() == _POINT
        numpy_bkd.assert_allclose(
            op.format().matrix().toarray(),
            csr_matrix(adapter.jacobian(state, 0.0)).toarray(),
            rtol=1e-14,
        )

    def test_refuses_a_model_without_a_jacobian(self) -> None:
        with pytest.raises(TypeError, match="JacobianMatrixProtocol"):
            SparseJacobianOperator(object(), SpaceDescriptor("state", 2))  # type: ignore[arg-type]

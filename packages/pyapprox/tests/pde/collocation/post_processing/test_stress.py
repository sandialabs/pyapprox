"""Tests for 2D stress post-processing.

The fixture is the Lame pressurized quarter-annulus (Ri 1, Ro 2,
pressure 1, E 1, nu 0.3) solved through ``CollocationSystem`` with
traction closure rows and symmetry Dirichlet rows, by Newton (atol
1e-10).
"""

import math
from dataclasses import dataclass
from typing import Generic, List, Tuple, Union

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    TractionNormalOperator,
)
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.post_processing.stress import (
    StressPostProcessor2D,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton

# ======================================================================
# Helpers
# ======================================================================

_R_INNER, _R_OUTER, _PRESSURE = 1.0, 2.0, 1.0


@dataclass(frozen=True)
class _LameSetup(Generic[Array]):
    """The solved Lame problem and its post-processor."""

    state: Array
    proc: StressPostProcessor2D[Array]
    # (mu, lamda) at the nodes, the material the state was solved with
    material: Tuple[Array, Array]
    mesh: TransformedMesh2D[Array]
    npts: int
    dx: Array
    dy: Array


def _traction(
    bkd: Backend[Array],
    mesh: TransformedMesh2D[Array],
    physics: LinearElasticityPhysics[Array],
    side: BoundarySide,
    component: int,
    data: Union[float, Array],
) -> RobinRows[Array]:
    """Component ``component`` of the traction on ``side`` equals ``data``."""
    indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
    operator = TractionNormalOperator(bkd, indices, normals, physics, component)
    value = data if isinstance(data, float) else NodalField(data)
    return RobinRows(
        bkd,
        mesh.npts(),
        2,
        [RobinPart(indices, operator, 0.0, 1.0, value, component)],
    )


def _setup_lame_problem(
    bkd: Backend[Array], npts_r: int = 12, npts_theta: int = 12
) -> _LameSetup[Array]:
    """Set up and solve the Lame analytical problem on a quarter-annulus."""
    poisson_ratio = 0.3
    youngs = 1.0
    mu = youngs / (2.0 * (1.0 + poisson_ratio))
    lamda = (
        youngs
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    transform = PolarTransform(
        (_R_INNER, _R_OUTER),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()

    physics = LinearElasticityPhysics(
        basis,
        bkd,
        lamda=lamda,
        mu=mu,
    )

    # Cylinder conditions, in the order whose later closure wins at a
    # shared corner; Dirichlet rows beat every closure.
    inner_normals = mesh.boundary_normals(BoundarySide.LEFT)
    closures: List[ClosureRowsProtocol[Array]] = [
        _traction(bkd, mesh, physics, BoundarySide.RIGHT, 0, 0.0),
        _traction(bkd, mesh, physics, BoundarySide.RIGHT, 1, 0.0),
        _traction(bkd, mesh, physics, BoundarySide.BOTTOM, 0, 0.0),
        _traction(bkd, mesh, physics, BoundarySide.TOP, 1, 0.0),
    ]
    closures.extend(
        _traction(
            bkd,
            mesh,
            physics,
            BoundarySide.LEFT,
            comp,
            -_PRESSURE * inner_normals[:, comp],
        )
        for comp in (0, 1)
    )
    essential = DirichletRows(
        bkd,
        npts,
        2,
        [
            DirichletPart(mesh.boundary_indices(BoundarySide.BOTTOM), 0.0, 1),
            DirichletPart(mesh.boundary_indices(BoundarySide.TOP), 0.0, 0),
        ],
    )
    system = CollocationSystem(physics, essential, closures)

    # Solve
    steady = SteadySystem(
        collocation_steady_residual(system),
        backend_state_space(bkd, 2 * npts),
        RecomputeAlways(BackendDenseLU(bkd)),
    )
    newton: Newton[Array, Array] = Newton(atol=1e-10, rtol=0.0, maxiters=50)
    point = steady.solve(newton, bkd.zeros((2 * npts, 1)))
    assert point.converged()

    # Create post-processor
    dx, dy = basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)
    curv_basis = transform.unit_curvilinear_basis(
        mesh.reference_points(),
    )
    proc = StressPostProcessor2D(
        dx,
        dy,
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    return _LameSetup(
        point.state()[:, 0],
        proc,
        (bkd.full((npts,), mu), bkd.full((npts,), lamda)),
        mesh,
        npts,
        dx,
        dy,
    )


def _lame_stresses(
    bkd: Backend[Array], pts: Array, r_inner: float, r_outer: float, pressure: float
) -> Tuple[Array, Array]:
    """Analytical Lame stresses in polar coordinates.

    Returns sigma_rr, sigma_tt, each shape (npts,).
    """
    x = pts[0, :]
    y = pts[1, :]
    r = bkd.sqrt(x**2 + y**2)
    Ri, Ro = r_inner, r_outer
    A = pressure * Ri**2 / (Ro**2 - Ri**2)
    B = -pressure * Ri**2 * Ro**2 / (Ro**2 - Ri**2)
    sigma_rr = A + B / r**2
    sigma_tt = A - B / r**2
    return sigma_rr, sigma_tt


# ======================================================================
# Tests
# ======================================================================


class TestStressPostProcessor2D:
    def test_hoop_stress_lame(self, bkd: Backend[Array]) -> None:
        """Hoop stress matches analytical Lame solution."""
        setup = _setup_lame_problem(bkd, npts_r=14, npts_theta=14)
        sigma_tt = setup.proc.hoop_stress(setup.state, *setup.material)
        pts = setup.mesh.points()
        _, sigma_tt_exact = _lame_stresses(
            bkd,
            pts,
            _R_INNER,
            _R_OUTER,
            _PRESSURE,
        )
        bkd.assert_allclose(sigma_tt, sigma_tt_exact, rtol=1e-6)

    @pytest.mark.slow_on("TorchBkd")
    def test_radial_stress_lame(self, bkd: Backend[Array]) -> None:
        """Radial stress matches analytical Lame solution."""
        setup = _setup_lame_problem(bkd, npts_r=16, npts_theta=16)
        sigma_rr = setup.proc.radial_stress(setup.state, *setup.material)
        pts = setup.mesh.points()
        sigma_rr_exact, _ = _lame_stresses(
            bkd,
            pts,
            _R_INNER,
            _R_OUTER,
            _PRESSURE,
        )
        bkd.assert_allclose(sigma_rr, sigma_rr_exact, atol=1e-6)

    def test_strain_energy_density_positive(self, bkd: Backend[Array]) -> None:
        """Strain energy density is positive for the Lame solution."""
        setup = _setup_lame_problem(bkd)
        psi = setup.proc.strain_energy_density(setup.state, *setup.material)
        assert float(bkd.min(psi)) > 0.0

    def test_hoop_stress_jacobian_fd(self, bkd: Backend[Array]) -> None:
        """Hoop stress Jacobian matches finite differences."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup.proc
        state = setup.state
        npts = setup.npts

        mat = setup.material
        jac = proc.hoop_stress_state_jacobian(state, *mat)  # (npts, 2*npts)
        assert jac.shape == (npts, 2 * npts)

        # Finite difference check at a few DOFs
        eps = 1e-7
        for dof in [0, npts // 2, npts, 3 * npts // 2]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.hoop_stress(state_p, *mat) - proc.hoop_stress(state_m, *mat)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_jacobian_fd(self, bkd: Backend[Array]) -> None:
        """Strain energy density Jacobian matches finite differences."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup.proc
        state = setup.state
        npts = setup.npts

        mat = setup.material
        jac = proc.strain_energy_density_state_jacobian(state, *mat)
        assert jac.shape == (npts, 2 * npts)

        eps = 1e-7
        for dof in [0, npts // 2, npts, 3 * npts // 2]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.strain_energy_density(state_p, *mat)
                - proc.strain_energy_density(state_m, *mat)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-4, atol=1e-8)

    def test_cartesian_stress_shapes(self, bkd: Backend[Array]) -> None:
        """Cartesian stresses have correct shapes."""
        setup = _setup_lame_problem(bkd)
        sxx, sxy, syy = setup.proc.cartesian_stress(
            setup.state, *setup.material
        )
        npts = setup.npts
        assert sxx.shape == (npts,)
        assert sxy.shape == (npts,)
        assert syy.shape == (npts,)

    def test_cartesian_stress_jacobian_shapes(self, bkd: Backend[Array]) -> None:
        """Cartesian stress Jacobians have correct shapes."""
        setup = _setup_lame_problem(bkd)
        npts = setup.npts
        dsxx, dsxy, dsyy = setup.proc.cartesian_stress_state_jacobian(
            *setup.material
        )
        assert dsxx.shape == (npts, 2 * npts)
        assert dsxy.shape == (npts, 2 * npts)
        assert dsyy.shape == (npts, 2 * npts)

    def test_no_curvilinear_basis_raises(self, bkd: Backend[Array]) -> None:
        """hoop_stress raises when curvilinear_basis not provided."""
        setup = _setup_lame_problem(bkd)
        proc_no_curv = StressPostProcessor2D(
            setup.dx,
            setup.dy,
            bkd=bkd,
        )
        with pytest.raises(ValueError):
            proc_no_curv.hoop_stress(setup.state, *setup.material)

    def test_lame_jacobians_fd(self, bkd: Backend[Array]) -> None:
        """The pointwise Lame derivatives of the hoop stress and the strain
        energy density equal central differences in uniform mu and
        lamda shifts (each quantity is pointwise in the material)."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup.proc
        state = setup.state
        mu, lamda = setup.material
        eps = 1e-6
        for name in ("hoop_stress", "strain_energy_density"):
            value = getattr(proc, name)
            dmu, dlamda = getattr(proc, name + "_lame_jacobian")(state, mu, lamda)
            fd_mu = (value(state, mu + eps, lamda) - value(state, mu - eps, lamda)) / (
                2.0 * eps
            )
            fd_lamda = (
                value(state, mu, lamda + eps) - value(state, mu, lamda - eps)
            ) / (2.0 * eps)
            bkd.assert_allclose(dmu, fd_mu, rtol=1e-6, atol=1e-10)
            bkd.assert_allclose(dlamda, fd_lamda, rtol=1e-6, atol=1e-10)

    def test_stress_scales_with_material(self, bkd: Backend[Array]) -> None:
        """Linear stress is homogeneous of degree one in (mu, lamda) at
        fixed strain, and the strain energy density likewise."""
        setup = _setup_lame_problem(bkd, npts_r=8, npts_theta=8)
        proc = setup.proc
        state = setup.state
        mu, lamda = setup.material
        for name in ("hoop_stress", "strain_energy_density"):
            value = getattr(proc, name)
            bkd.assert_allclose(
                value(state, 3.0 * mu, 3.0 * lamda),
                3.0 * value(state, mu, lamda),
                rtol=1e-12,
            )

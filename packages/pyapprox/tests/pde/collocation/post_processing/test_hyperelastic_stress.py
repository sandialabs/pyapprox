"""Tests for 2D hyperelastic stress post-processing."""

import math
from dataclasses import dataclass
from typing import Generic, List, Tuple

import numpy as np
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
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
)
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.constitutive import (
    NeoHookeanStress,
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


@dataclass(frozen=True)
class _HyperelasticSetup(Generic[Array]):
    """The post-processors on a quarter-annulus and their material."""

    proc: HyperelasticStressPostProcessor2D[Array]
    linear_proc: StressPostProcessor2D[Array]
    stress_model: NeoHookeanStress[Array]
    # (mu, lamda) at the nodes
    material: Tuple[Array, Array]
    mesh: TransformedMesh2D[Array]
    basis: ChebyshevBasis2D[Array]
    npts: int
    Dx: Array
    Dy: Array
    lamda: float
    mu: float


def _setup_hyperelastic_processor(
    bkd: Backend[Array], npts_r: int = 8, npts_theta: int = 8
) -> _HyperelasticSetup[Array]:
    """Create HyperelasticStressPostProcessor2D on a quarter-annulus."""
    r_inner, r_outer = 1.0, 2.0
    lamda, mu = 0.5769, 0.3846  # E=1, nu=0.3

    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()

    Dx = basis.derivative_matrix(1, 0)
    Dy = basis.derivative_matrix(1, 1)

    stress_model: NeoHookeanStress[Array] = NeoHookeanStress(lamda=lamda, mu=mu)
    curv_basis = transform.unit_curvilinear_basis(
        mesh.reference_points(),
    )

    proc = HyperelasticStressPostProcessor2D(
        Dx,
        Dy,
        NeoHookeanLaw(bkd),
        bkd,
        curvilinear_basis=curv_basis,
    )

    # Also create linear processor for comparison
    linear_proc = StressPostProcessor2D(
        Dx,
        Dy,
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    return _HyperelasticSetup(
        proc,
        linear_proc,
        stress_model,
        (bkd.full((npts,), mu), bkd.full((npts,), lamda)),
        mesh,
        basis,
        npts,
        Dx,
        Dy,
        lamda,
        mu,
    )


def _piola_traction(
    bkd: Backend[Array],
    mesh: TransformedMesh2D[Array],
    physics: HyperelasticityPhysics[Array],
    side: BoundarySide,
    component: int,
    data: NodalField[Array],
) -> RobinRows[Array]:
    """Component ``component`` of the PK1 traction on ``side`` equals
    ``data``."""
    indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
    operator = HyperelasticTractionNormalOperator(
        bkd, indices, normals, physics, component
    )
    return RobinRows(
        bkd,
        StateLayout.uniform(mesh.npts(), 2),
        [RobinPart(indices, operator, 0.0, 1.0, data, component)],
    )


# ======================================================================
# Tests
# ======================================================================


class TestHyperelasticStressPostProcessor2D:
    def test_cauchy_from_pk1(self, bkd: Backend[Array]) -> None:
        """Cauchy stress sigma = (1/J)*P*F^T matches direct computation."""
        setup = _setup_hyperelastic_processor(bkd)
        mat = setup.material
        proc = setup.proc
        npts = setup.npts

        # Small random displacement
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.01)

        sxx, sxy, syy = proc.cartesian_stress(state, *mat)

        # Direct computation
        (F11, F12), (F21, F22) = proc._compute_F(state)
        P11, P12, P21, P22 = setup.stress_model.compute_stress_2d(
            F11,
            F12,
            F21,
            F22,
            bkd,
        )
        J = F11 * F22 - F12 * F21
        inv_J = 1.0 / J
        sxx_direct = inv_J * (P11 * F11 + P12 * F12)
        sxy_direct = inv_J * (P11 * F21 + P12 * F22)
        syy_direct = inv_J * (P21 * F21 + P22 * F22)

        bkd.assert_allclose(sxx, sxx_direct, rtol=1e-12)
        bkd.assert_allclose(sxy, sxy_direct, rtol=1e-12)
        bkd.assert_allclose(syy, syy_direct, rtol=1e-12)

    def test_strain_energy_density_direct(self, bkd: Backend[Array]) -> None:
        """Strain energy matches Neo-Hookean formula directly."""
        setup = _setup_hyperelastic_processor(bkd)
        mat = setup.material
        proc = setup.proc
        npts = setup.npts
        mu = setup.mu
        lamda = setup.lamda

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.01)

        W = proc.strain_energy_density(state, *mat)

        # Direct: W = mu/2*(I1-3) - mu*ln(J) + lam/2*(ln(J))^2
        (F11, F12), (F21, F22) = proc._compute_F(state)
        J = F11 * F22 - F12 * F21
        ln_J = bkd.log(J)
        # Plane strain: F33=1, so I1 includes +1.0
        I1 = F11**2 + F12**2 + F21**2 + F22**2 + 1.0
        W_direct = 0.5 * mu * (I1 - 3.0) - mu * ln_J + 0.5 * lamda * ln_J**2

        bkd.assert_allclose(W, W_direct, rtol=1e-12)

    def test_hoop_stress_small_disp(self, bkd: Backend[Array]) -> None:
        """Small displacement: hyperelastic hoop stress ≈ linear hoop stress."""
        setup = _setup_hyperelastic_processor(bkd)
        mat = setup.material
        npts = setup.npts

        # Very small displacement to stay in linear regime
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 1e-5)

        hoop_hyper = setup.proc.hoop_stress(state, *mat)
        hoop_linear = setup.linear_proc.hoop_stress(state, *mat)

        bkd.assert_allclose(hoop_hyper, hoop_linear, atol=1e-6)

    def test_cauchy_stress_jacobian_fd(self, bkd: Backend[Array]) -> None:
        """Cauchy stress Jacobian matches finite differences."""
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        mat = setup.material
        proc = setup.proc
        npts = setup.npts

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        dsxx, dsxy, dsyy = proc.cartesian_stress_state_jacobian(state, *mat)
        assert dsxx.shape == (npts, 2 * npts)

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            sxx_p, sxy_p, syy_p = proc.cartesian_stress(state_p, *mat)
            sxx_m, sxy_m, syy_m = proc.cartesian_stress(state_m, *mat)
            bkd.assert_allclose(
                dsxx[:, dof],
                (sxx_p - sxx_m) / (2.0 * eps),
                rtol=1e-5,
                atol=1e-8,
            )
            bkd.assert_allclose(
                dsxy[:, dof],
                (sxy_p - sxy_m) / (2.0 * eps),
                rtol=1e-5,
                atol=1e-8,
            )
            bkd.assert_allclose(
                dsyy[:, dof],
                (syy_p - syy_m) / (2.0 * eps),
                rtol=1e-5,
                atol=1e-8,
            )

    def test_hoop_stress_jacobian_fd(self, bkd: Backend[Array]) -> None:
        """Hoop stress Jacobian matches finite differences."""
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        mat = setup.material
        proc = setup.proc
        npts = setup.npts

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        jac = proc.hoop_stress_state_jacobian(state, *mat)
        assert jac.shape == (npts, 2 * npts)

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (proc.hoop_stress(state_p, *mat) - proc.hoop_stress(state_m, *mat)) / (
                2.0 * eps
            )
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_jacobian_fd(self, bkd: Backend[Array]) -> None:
        """Strain energy density Jacobian matches finite differences."""
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        mat = setup.material
        proc = setup.proc
        npts = setup.npts

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        jac = proc.strain_energy_density_state_jacobian(state, *mat)
        assert jac.shape == (npts, 2 * npts)

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
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
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_positive(self, bkd: Backend[Array]) -> None:
        """Strain energy density is positive at an equilibrium state."""
        setup = _setup_hyperelastic_processor(bkd)
        mat = setup.material
        npts = setup.npts
        mesh = setup.mesh
        basis = setup.basis

        # Solve a pressurized quarter-annulus BVP
        physics = HyperelasticityPhysics(
            basis,
            bkd,
            NeoHookeanLaw(bkd),
            mu=setup.mu,
            lamda=setup.lamda,
        )

        closures: List[ClosureRowsProtocol[Array]] = []
        # Outer wall: traction-free (both components)
        nouter = mesh.boundary_indices(BoundarySide.RIGHT).shape[0]
        for comp in (0, 1):
            closures.append(
                _piola_traction(
                    bkd,
                    mesh,
                    physics,
                    BoundarySide.RIGHT,
                    comp,
                    NodalField(bkd.zeros((nouter,))),
                )
            )
        # Inner wall: pressure t = -p*n (both components)
        pressure = 0.01
        inner_normals = mesh.boundary_normals(BoundarySide.LEFT)
        for comp in (0, 1):
            pn = -pressure * inner_normals[:, comp]
            closures.append(
                _piola_traction(
                    bkd, mesh, physics, BoundarySide.LEFT, comp, NodalField(pn)
                )
            )
        # Bottom: v=0 (symmetry); top: u=0 (symmetry)
        essential = DirichletRows(
            bkd,
            StateLayout.uniform(npts, 2),
            [
                DirichletPart(mesh.boundary_indices(BoundarySide.BOTTOM), 0.0, 1),
                DirichletPart(mesh.boundary_indices(BoundarySide.TOP), 0.0, 0),
            ],
        )
        steady = SteadySystem(
            collocation_steady_residual(
                CollocationSystem(physics, essential, closures)
            ),
            backend_state_space(bkd, 2 * npts),
            RecomputeAlways(BackendDenseLU(bkd)),
        )
        newton: Newton[Array, Array] = Newton(atol=1e-10, rtol=0.0, maxiters=50)
        point = steady.solve(newton, bkd.zeros((2 * npts, 1)))
        assert point.converged()
        state = point.state()[:, 0]

        W = setup.proc.strain_energy_density(state, *mat)
        assert float(bkd.min(W)) > 0.0

    def test_no_curvilinear_basis_raises(self, bkd: Backend[Array]) -> None:
        """hoop_stress raises when curvilinear_basis not provided."""
        setup = _setup_hyperelastic_processor(bkd)
        mat = setup.material
        proc_no_curv = HyperelasticStressPostProcessor2D(
            setup.Dx,
            setup.Dy,
            NeoHookeanLaw(bkd),
            bkd,
        )
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * setup.npts) * 0.005)
        with pytest.raises(ValueError):
            proc_no_curv.hoop_stress(state, *mat)

    def test_non_law_refused(self, numpy_bkd: Backend[Array]) -> None:
        setup = _setup_hyperelastic_processor(numpy_bkd)
        with pytest.raises(TypeError, match="PostProcessingLawProtocol"):
            HyperelasticStressPostProcessor2D(
                setup.Dx,
                setup.Dy,
                setup.stress_model,  # type: ignore[arg-type]
                numpy_bkd,
            )

    def test_lame_jacobians_fd(self, bkd: Backend[Array]) -> None:
        """The pointwise Lame derivatives of the hoop stress and the strain
        energy density equal central differences in uniform mu and
        lamda shifts (each quantity is pointwise in the material)."""
        setup = _setup_hyperelastic_processor(bkd)
        proc = setup.proc
        mu, lamda = setup.material
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * setup.npts) * 0.01)
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

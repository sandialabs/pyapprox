"""Tests for 2D hyperelastic stress post-processing."""

import math
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics.stress_models import (
    NeoHookeanStress,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    hyperelastic_traction_neumann_bc,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)


# ======================================================================
# Helpers
# ======================================================================

def _setup_hyperelastic_processor(bkd, npts_r=8, npts_theta=8):
    """Create HyperelasticStressPostProcessor2D on a quarter-annulus."""
    r_inner, r_outer = 1.0, 2.0
    lamda, mu = 0.5769, 0.3846  # E=1, nu=0.3

    transform = PolarTransform(
        (r_inner, r_outer), (0.0, math.pi / 2.0), bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()

    Dx = basis.derivative_matrix(1, 0)
    Dy = basis.derivative_matrix(1, 1)

    stress_model = NeoHookeanStress(lamda=lamda, mu=mu)
    curv_basis = transform.unit_curvilinear_basis(
        mesh.reference_points(),
    )

    proc = HyperelasticStressPostProcessor2D(
        Dx, Dy, stress_model, bkd, curvilinear_basis=curv_basis,
    )

    # Also create linear processor for comparison
    linear_proc = StressPostProcessor2D(
        Dx, Dy,
        get_lamda=lambda: bkd.asarray([lamda] * npts),
        get_mu=lambda: bkd.asarray([mu] * npts),
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    return {
        "proc": proc, "linear_proc": linear_proc,
        "stress_model": stress_model,
        "mesh": mesh, "basis": basis, "transform": transform,
        "npts": npts, "Dx": Dx, "Dy": Dy,
        "lamda": lamda, "mu": mu,
    }


# ======================================================================
# Tests
# ======================================================================

class TestHyperelasticStressPostProcessor2D(
    Generic[Array], unittest.TestCase,
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_cauchy_from_pk1(self):
        """Cauchy stress sigma = (1/J)*P*F^T matches direct computation."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd)
        proc = setup["proc"]
        npts = setup["npts"]

        # Small random displacement
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.01)

        sxx, sxy, syy = proc.cartesian_stress(state)

        # Direct computation
        F11, F12, F21, F22 = proc._compute_F(state)
        P11, P12, P21, P22 = setup["stress_model"].compute_stress_2d(
            F11, F12, F21, F22, bkd,
        )
        J = F11 * F22 - F12 * F21
        inv_J = 1.0 / J
        sxx_direct = inv_J * (P11 * F11 + P12 * F12)
        sxy_direct = inv_J * (P11 * F21 + P12 * F22)
        syy_direct = inv_J * (P21 * F21 + P22 * F22)

        bkd.assert_allclose(sxx, sxx_direct, rtol=1e-12)
        bkd.assert_allclose(sxy, sxy_direct, rtol=1e-12)
        bkd.assert_allclose(syy, syy_direct, rtol=1e-12)

    def test_strain_energy_density_direct(self):
        """Strain energy matches Neo-Hookean formula directly."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd)
        proc = setup["proc"]
        npts = setup["npts"]
        mu = setup["mu"]
        lamda = setup["lamda"]

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.01)

        W = proc.strain_energy_density(state)

        # Direct: W = mu/2*(I1-3) - mu*ln(J) + lam/2*(ln(J))^2
        F11, F12, F21, F22 = proc._compute_F(state)
        J = F11 * F22 - F12 * F21
        ln_J = bkd.log(J)
        # Plane strain: F33=1, so I1 includes +1.0
        I1 = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2 + 1.0
        W_direct = (
            0.5 * mu * (I1 - 3.0)
            - mu * ln_J
            + 0.5 * lamda * ln_J ** 2
        )

        bkd.assert_allclose(W, W_direct, rtol=1e-12)

    def test_hoop_stress_small_disp(self):
        """Small displacement: hyperelastic hoop stress ≈ linear hoop stress."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd)
        npts = setup["npts"]

        # Very small displacement to stay in linear regime
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 1e-5)

        hoop_hyper = setup["proc"].hoop_stress(state)
        hoop_linear = setup["linear_proc"].hoop_stress(state)

        bkd.assert_allclose(hoop_hyper, hoop_linear, atol=1e-6)

    def test_cauchy_stress_jacobian_fd(self):
        """Cauchy stress Jacobian matches finite differences."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        proc = setup["proc"]
        npts = setup["npts"]

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        dsxx, dsxy, dsyy = proc.cartesian_stress_state_jacobian(state)
        self.assertEqual(dsxx.shape, (npts, 2 * npts))

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            sxx_p, sxy_p, syy_p = proc.cartesian_stress(state_p)
            sxx_m, sxy_m, syy_m = proc.cartesian_stress(state_m)
            bkd.assert_allclose(
                dsxx[:, dof], (sxx_p - sxx_m) / (2.0 * eps),
                rtol=1e-5, atol=1e-8,
            )
            bkd.assert_allclose(
                dsxy[:, dof], (sxy_p - sxy_m) / (2.0 * eps),
                rtol=1e-5, atol=1e-8,
            )
            bkd.assert_allclose(
                dsyy[:, dof], (syy_p - syy_m) / (2.0 * eps),
                rtol=1e-5, atol=1e-8,
            )

    def test_hoop_stress_jacobian_fd(self):
        """Hoop stress Jacobian matches finite differences."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        proc = setup["proc"]
        npts = setup["npts"]

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        jac = proc.hoop_stress_state_jacobian(state)
        self.assertEqual(jac.shape, (npts, 2 * npts))

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.hoop_stress(state_p) - proc.hoop_stress(state_m)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_jacobian_fd(self):
        """Strain energy density Jacobian matches finite differences."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd, npts_r=6, npts_theta=6)
        proc = setup["proc"]
        npts = setup["npts"]

        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * npts) * 0.005)

        jac = proc.strain_energy_density_state_jacobian(state)
        self.assertEqual(jac.shape, (npts, 2 * npts))

        eps = 1e-7
        for dof in [0, npts // 2, npts, npts + npts // 3]:
            if dof >= 2 * npts:
                continue
            state_p = bkd.copy(state)
            state_p[dof] = state_p[dof] + eps
            state_m = bkd.copy(state)
            state_m[dof] = state_m[dof] - eps
            fd = (
                proc.strain_energy_density(state_p)
                - proc.strain_energy_density(state_m)
            ) / (2.0 * eps)
            bkd.assert_allclose(jac[:, dof], fd, rtol=1e-5, atol=1e-8)

    def test_strain_energy_density_positive(self):
        """Strain energy density is positive at an equilibrium state."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd)
        npts = setup["npts"]
        mesh = setup["mesh"]
        basis = setup["basis"]

        # Solve a pressurized quarter-annulus BVP
        physics = HyperelasticityPhysics(
            basis, bkd, setup["stress_model"],
        )
        D_matrices = [setup["Dx"], setup["Dy"]]

        bcs = []
        # Outer wall: traction-free (both components)
        for comp in (0, 1):
            outer_idx = mesh.boundary_indices(1)
            outer_normals = mesh.boundary_normals(1)
            bcs.append(hyperelastic_traction_neumann_bc(
                bkd, outer_idx, outer_normals, D_matrices,
                setup["stress_model"], npts, comp, values=0.0,
            ))
        # Inner wall: pressure t = -p*n (both components)
        pressure = 0.01
        for comp in (0, 1):
            inner_idx = mesh.boundary_indices(0)
            inner_normals = mesh.boundary_normals(0)
            pn = -pressure * inner_normals[:, comp]
            bcs.append(hyperelastic_traction_neumann_bc(
                bkd, inner_idx, inner_normals, D_matrices,
                setup["stress_model"], npts, comp, values=pn,
            ))
        # Bottom (bnd 2): v=0 (symmetry)
        bot_idx = mesh.boundary_indices(2)
        bcs.append(zero_dirichlet_bc(bkd, bot_idx + npts))
        # Top (bnd 3): u=0 (symmetry)
        top_idx = mesh.boundary_indices(3)
        bcs.append(zero_dirichlet_bc(bkd, top_idx))

        physics.set_boundary_conditions(bcs)

        model = CollocationModel(physics, bkd)
        init_state = bkd.zeros((2 * npts,))
        state = model.solve_steady(init_state, tol=1e-10, maxiter=50)

        W = setup["proc"].strain_energy_density(state)
        self.assertGreater(float(bkd.min(W)), 0.0)

    def test_no_curvilinear_basis_raises(self):
        """hoop_stress raises when curvilinear_basis not provided."""
        bkd = self._bkd
        setup = _setup_hyperelastic_processor(bkd)
        proc_no_curv = HyperelasticStressPostProcessor2D(
            setup["Dx"], setup["Dy"],
            setup["stress_model"], bkd,
        )
        np.random.seed(42)
        state = bkd.array(np.random.randn(2 * setup["npts"]) * 0.005)
        with self.assertRaises(ValueError):
            proc_no_curv.hoop_stress(state)


class TestHyperelasticStressPostProcessor2DNumpy(
    TestHyperelasticStressPostProcessor2D[NDArray[Any]],
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHyperelasticStressPostProcessor2DTorch(
    TestHyperelasticStressPostProcessor2D[torch.Tensor],
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

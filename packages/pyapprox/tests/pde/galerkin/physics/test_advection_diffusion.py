"""Tests for LinearAdvectionDiffusionReaction physics.

This module contains:
1. Unit tests for matrix properties (TestLinearADRBase)
2. Parametrized convergence tests inspired by legacy test_finite_elements.py
   - 4 1D convergence rate tests with P2 elements
   - 4 2D convergence rate tests with P2 elements
   Uses manufactured solutions with natural (zero Neumann) boundary conditions.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.util.backends.numpy import NumpyBkd


class TestLinearADRBase:
    """Base test class for LinearAdvectionDiffusionReaction."""

    def _to_dense(self, bkd, matrix):
        """Convert sparse or backend matrix to dense numpy array."""
        if issparse(matrix):
            return matrix.toarray()
        return bkd.to_numpy(matrix)

    def test_1d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=bkd
        )

        M = physics.mass_matrix()
        M_np = self._to_dense(bkd, M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_mass_matrix_positive_definite(self, numpy_bkd) -> None:
        """Test mass matrix is positive definite."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=bkd
        )

        M = physics.mass_matrix()
        M_np = self._to_dense(bkd, M)

        eigenvalues = np.linalg.eigvalsh(M_np)
        assert np.all(eigenvalues > 0)

    def test_1d_stiffness_assembly(self, numpy_bkd: NumpyBkd) -> None:
        """Test stiffness matrix assembly in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=bkd
        )
        system = compose_galerkin_system(physics)

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = system.spatial_operator().spatial_jacobian(u0, 0.0)
        jac_np = self._to_dense(bkd, jac)

        # For pure diffusion (no reaction), -jacobian should be the
        # stiffness matrix, which should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)


    def test_3d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=bkd
        )

        M = physics.mass_matrix()
        M_np = self._to_dense(bkd, M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_stiffness_symmetric(self, numpy_bkd: NumpyBkd) -> None:
        """Test stiffness matrix is symmetric in 3D (pure diffusion)."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=bkd
        )
        system = compose_galerkin_system(physics)

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = system.spatial_operator().spatial_jacobian(u0, 0.0)
        jac_np = self._to_dense(bkd, jac)

        # For pure diffusion, -jacobian = stiffness matrix should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)


# =============================================================================
# Parametrized ADR Convergence Rate Tests with P2 Elements
# =============================================================================
#
# These tests verify convergence rates for the ADR solver with P2 elements.
# They are inspired by pyapprox/pde/galerkin/tests/test_finite_elements.py.
#
# Key design notes:
# - Use manufactured solutions with cos(k*pi*x) form that have zero Neumann
#   BCs (natural boundary conditions) to avoid explicit BC enforcement
# - Include reaction term (r > 0) to make the problem well-posed
# - Verify O(h^3) convergence rate for P2 elements with smooth solutions
#
# The 1D cases cover:
#   - Different wavenumbers: k=1, k=2
#   - With and without advection
# Total: 4 cases
#
# The 2D cases cover:
#   - Different wavenumbers
#   - With and without advection
# Total: 4 cases
# =============================================================================

# Test case format: (name, wavenumber_x, wavenumber_y, diffusivity, advection, reaction)
# For 1D: wavenumber_y is ignored (use 0)
# wavenumber: k in cos(k*pi*x) solution form


# =============================================================================
# Parametrized ADR Exact Reproduction Tests with P2 Elements and BCs
# =============================================================================
#
# These tests verify exact reproduction for linear/quadratic solutions with
# P2 elements and various boundary conditions (Dirichlet, Neumann, Robin).
# They are inspired by pyapprox/pde/galerkin/tests/test_finite_elements.py
# test_advection_diffusion_reaction (lines 617-765).
#
# With P2 (quadratic) elements:
# - Linear solutions (e.g., "x", "x+2*y") should be reproduced exactly
# - Quadratic solutions (e.g., "x**2", "x**2*y**2") should be reproduced exactly
# - Expected relative error: < 1e-8
#
# 1D cases (24 total):
#   - bounds: [0, 1] or [0, 1.1]
#   - bndry_types: ["D", "N"], ["R", "D"], ["R", "R"]
#   - velocity: none or advective
#   - reaction: 0 or linear
#
# 2D cases (9 total):
#   - Various BC combinations
#   - Linear and quadratic solutions
# =============================================================================

# Import manufactured solution infrastructure


class TestDiffusivityPositivity:
    """The operator's own requirement, enforced where it is consumed.

    :math:`-\\nabla\\cdot(D\\nabla u)` is elliptic only while
    :math:`D > 0`. Where D dips negative the local operator changes
    character and the linear solve returns a plausible field with NO
    error --- measured before this check existed: a single negative DOF
    changed the peak from 0.7535 to 0.7030 and raised nothing.

    Enforced by the physics rather than by a parameterization, because
    the requirement holds however the field arrived: built directly,
    mutated through ``set_dofs``, or written by an optimizer.
    """

    @staticmethod
    def _physics(numpy_bkd, dofs):
        from pyapprox.pde.constitutive.coefficient_functions import (
            NodalFieldDiffusion,
        )
        from pyapprox.pde.galerkin.basis import LagrangeBasis
        from pyapprox.pde.galerkin.mesh import StructuredMesh2D
        from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction

        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=numpy_bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        return AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis, dofs=dofs(basis)),
            bkd=numpy_bkd,
            forcing=TimeIndependent(lambda x: np.ones(x.shape[1])),
        )

    def test_positive_diffusivity_assembles(self, numpy_bkd) -> None:
        physics = self._physics(
            numpy_bkd, lambda b: np.full(b.ndofs(), 0.1)
        )
        state = numpy_bkd.zeros((physics.nstates(),))
        physics._assemble_stiffness(state, 0.0)

    def test_negative_diffusivity_raises(self, numpy_bkd) -> None:
        """One negative DOF is enough: it makes the operator
        non-elliptic there, and previously solved silently."""

        def _dofs(basis):
            values = np.full(basis.ndofs(), 0.1)
            values[5] = -0.5
            return values

        physics = self._physics(numpy_bkd, _dofs)
        state = numpy_bkd.zeros((physics.nstates(),))
        with pytest.raises(ValueError, match="must be positive"):
            physics._assemble_stiffness(state, 0.0)

    def test_a_field_mutated_after_construction_is_caught(
        self, numpy_bkd
    ) -> None:
        """The case a parameterization-side check could never cover on
        its own: the field goes bad through set_dofs, with no
        parameterization involved at all."""
        physics = self._physics(
            numpy_bkd, lambda b: np.full(b.ndofs(), 0.1)
        )
        state = numpy_bkd.zeros((physics.nstates(),))
        physics._assemble_stiffness(state, 0.0)

        diffusion = physics.diffusion_function()
        bad = np.full(diffusion.ndofs(), 0.1)
        bad[3] = -1.0
        diffusion.set_dofs(bad)
        with pytest.raises(ValueError, match="must be positive"):
            physics._assemble_stiffness(state, 0.0)

    def test_checks_quadrature_values_not_dofs(self, numpy_bkd: NumpyBkd) -> None:
        """The check is on what the operator integrates.

        A single zero DOF does NOT make the interpolant zero at any
        quadrature point --- the surrounding positive nodes carry it ---
        so this assembles. That is the correct behavior and a real
        difference from checking DOFs: what matters is the field the
        integrand sees, not the coefficients that generate it. Making
        the whole field zero does raise.
        """
        physics = self._physics(
            numpy_bkd,
            lambda b: np.where(
                np.arange(b.ndofs()) == 2, 0.0, 0.1
            ),
        )
        state = numpy_bkd.zeros((physics.nstates(),))
        physics._assemble_stiffness(state, 0.0)

        physics = self._physics(numpy_bkd, lambda b: np.zeros(b.ndofs()))
        state = numpy_bkd.zeros((physics.nstates(),))
        with pytest.raises(ValueError, match="must be positive"):
            physics._assemble_stiffness(state, 0.0)


class TestLinearReactionSelectedByCapability:
    """The stiffness picks a linear reaction on what it CAN do.

    Gating on one concrete class silently dropped every other
    spatially varying linear reaction from the stiffness: no error,
    just a missing term in the operator and in every adjoint derived
    from it.
    """

    @staticmethod
    def _physics(numpy_bkd, reaction):
        from pyapprox.pde.galerkin.basis import LagrangeBasis
        from pyapprox.pde.galerkin.mesh import StructuredMesh2D
        from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction

        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=numpy_bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        return AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=numpy_bkd,
            reaction=reaction,
            forcing=TimeIndependent(lambda x: np.ones(x.shape[1])),
        ), basis

    def test_a_custom_linear_reaction_reaches_the_stiffness(
        self, numpy_bkd
    ) -> None:
        """Satisfying the protocol is enough; being a particular class
        is not required."""

        class _CustomLinearReaction:
            def values(self, coords, time=0.0):
                return np.full(np.asarray(coords).shape[1:], 2.0)

            def value(self, coords, state, time=0.0):
                return self.values(coords, time) * np.asarray(state)

            def derivative(self, coords, state, time=0.0):
                return np.broadcast_to(
                    self.values(coords, time), np.asarray(state).shape
                ).copy()

            def second_derivative_function(self):
                return None

            def is_linear(self):
                return True

            def version(self):
                return 1

        without, basis = self._physics(numpy_bkd, None)
        with_reaction, _ = self._physics(
            numpy_bkd, _CustomLinearReaction()
        )
        state = numpy_bkd.zeros((basis.ndofs(),))
        bare = np.asarray(
            numpy_bkd.to_numpy(
                without._assemble_stiffness(state, 0.0)
            ).todense()
        )
        reacted = np.asarray(
            numpy_bkd.to_numpy(
                with_reaction._assemble_stiffness(state, 0.0)
            ).todense()
        )
        assert np.abs(reacted - bare).max() > 1e-10

    def test_a_modulated_reaction_reassembles_per_time(
        self, numpy_bkd
    ) -> None:
        """The end-to-end property time-modulated reactions exist for: a
        control that varies in time must change the operator it enters."""
        from pyapprox.pde.constitutive.coefficient_functions import (
            TimeModulatedNodalFieldLinearReaction,
        )
        from pyapprox.pde.field_maps.modulation import (
            PiecewiseLinearModulation,
        )
        from pyapprox.pde.galerkin.basis import LagrangeBasis

        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=numpy_bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        coords = numpy_bkd.to_numpy(basis.dof_coordinates())
        modes = np.stack(
            [
                np.exp(-20.0 * (coords[0] - center) ** 2)
                for center in (0.25, 0.5, 0.75)
            ],
            axis=1,
        )
        reaction = TimeModulatedNodalFieldLinearReaction(
            basis,
            modes,
            PiecewiseLinearModulation(numpy_bkd, [0.0, 0.5, 1.0]),
            np.array([2.0, 1.0, 3.0]),
        )
        physics, _ = self._physics(numpy_bkd, reaction)
        assert physics._stiffness_is_time_dependent()

        state = numpy_bkd.zeros((basis.ndofs(),))
        early = np.asarray(
            numpy_bkd.to_numpy(
                physics._assemble_stiffness(state, 0.0)
            ).todense()
        )
        late = np.asarray(
            numpy_bkd.to_numpy(
                physics._assemble_stiffness(state, 1.0)
            ).todense()
        )
        assert np.abs(late - early).max() > 1e-10

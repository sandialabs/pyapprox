"""Tests for AdvectionDiffusionReaction.stiffness_forms.

The exposed bilinear forms must (i) sum to the same stiffness the
physics assembles internally, (ii) support element-restricted assembly
that partitions the global matrix exactly, and (iii) leave the
constant-coefficient caching behaviour unchanged.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from skfem import BilinearForm, asm
from skfem.helpers import dot, grad


def _make_physics(numpy_bkd, diffusivity, velocity=None, reaction=None):
    mesh = StructuredMesh2D(
        nx=6,
        ny=6,
        bounds=[[-1.0, 1.0], [-1.0, 1.0]],
        bkd=numpy_bkd,
        element_type="quad",
    )
    basis = LagrangeBasis(mesh, degree=1)
    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
        bkd=numpy_bkd,
        velocity=velocity,
        reaction=reaction,
    )


class TestFormsMatchAssembly:
    def test_forms_sum_to_spatial_jacobian(self, numpy_bkd):
        physics = _make_physics(
            numpy_bkd,
            diffusivity=lambda x: 1.0 + 0.5 * x[0] ** 2,
            velocity=numpy_bkd.array([1.0, -0.5]),
            reaction=0.7,
        )
        skfem_basis = physics.basis().skfem_basis()
        total = sum(
            asm(form, skfem_basis) for form in physics.stiffness_forms()
        )
        zeros = numpy_bkd.zeros((physics.nstates(),))
        stiffness = -physics.spatial_jacobian(zeros, 0.0)
        assert np.abs((total - stiffness).toarray()).max() < 1e-14

    def test_diffusion_form_matches_hand_assembly(self, numpy_bkd):
        """Independent reference: the weak form written out in the test."""
        physics = _make_physics(numpy_bkd, diffusivity=0.85)
        skfem_basis = physics.basis().skfem_basis()

        @BilinearForm
        def reference(u, v, w):
            return 0.85 * dot(grad(u), grad(v))

        expected = asm(reference, skfem_basis)
        (form,) = physics.stiffness_forms()
        assert np.abs((asm(form, skfem_basis) - expected).toarray()).max() == 0.0

    def test_advection_form_present_only_with_velocity(self, numpy_bkd):
        assert len(_make_physics(numpy_bkd, 1.0).stiffness_forms()) == 1
        assert (
            len(
                _make_physics(
                    numpy_bkd, 1.0, velocity=numpy_bkd.array([1.0, 1.0])
                ).stiffness_forms()
            )
            == 2
        )


class TestRestrictedAssembly:
    def test_element_subsets_partition_the_global_matrix(self, numpy_bkd):
        """asm on a subset plus asm on its complement equals the global."""
        physics = _make_physics(
            numpy_bkd,
            diffusivity=lambda x: 1.0 + 0.25 * x[1] ** 2,
            velocity=numpy_bkd.array([0.5, 1.0]),
        )
        skfem_basis = physics.basis().skfem_basis()
        nelems = skfem_basis.nelems
        subset = np.arange(nelems // 2)
        complement = np.arange(nelems // 2, nelems)
        for form in physics.stiffness_forms():
            full = asm(form, skfem_basis)
            parts = asm(form, skfem_basis.with_elements(subset)) + asm(
                form, skfem_basis.with_elements(complement)
            )
            assert np.abs((full - parts).toarray()).max() < 1e-14


class TestCachingUnchanged:
    def test_constant_coefficients_still_cached(self, numpy_bkd):
        physics = _make_physics(
            numpy_bkd, diffusivity=2.0, velocity=numpy_bkd.array([1.0, 0.0])
        )
        zeros = numpy_bkd.zeros((physics.nstates(),))
        physics.spatial_jacobian(zeros, 0.0)
        assert physics._stiffness_cached is not None
        first = physics._stiffness_cached
        physics.spatial_jacobian(zeros, 0.0)
        assert physics._stiffness_cached is first

    def test_callable_coefficients_not_cached(self, numpy_bkd):
        physics = _make_physics(
            numpy_bkd, diffusivity=lambda x: 1.0 + 0.0 * x[0]
        )
        zeros = numpy_bkd.zeros((physics.nstates(),))
        physics.spatial_jacobian(zeros, 0.0)
        assert physics._stiffness_cached is None

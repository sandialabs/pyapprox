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
from skfem import BilinearForm, asm
from skfem.helpers import dot, grad

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction


def _make_physics(
    numpy_bkd, diffusivity, velocity=None, reaction=None, forcing=None
):
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
        forcing=forcing,
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


def _quadratic_reaction(x, u):
    return u**2


def _quadratic_reaction_deriv(x, u):
    return 2.0 * u


def _unit_forcing(x):
    return np.ones(x.shape[1])


class TestLoadAndReactionForms:
    def _nonlinear_physics(self, numpy_bkd):
        return _make_physics(
            numpy_bkd,
            diffusivity=1.0,
            velocity=numpy_bkd.array([0.5, 1.0]),
            reaction=(_quadratic_reaction, _quadratic_reaction_deriv),
            forcing=_unit_forcing,
        )

    def test_forms_reassemble_the_spatial_residual(self, numpy_bkd):
        """asm of the exposed load forms + stiffness equals the physics'
        spatial residual at a nonzero state."""
        physics = self._nonlinear_physics(numpy_bkd)
        skfem_basis = physics.basis().skfem_basis()
        rng = np.random.RandomState(0)
        state = numpy_bkd.array(rng.uniform(0.1, 1.0, physics.nstates()))
        stiffness = sum(
            asm(form, skfem_basis) for form in physics.stiffness_forms()
        )
        load = asm(physics.forcing_form(0.0), skfem_basis) + asm(
            physics.reaction_form(),
            skfem_basis,
            u_prev=skfem_basis.interpolate(numpy_bkd.to_numpy(state)),
        )
        reference = physics.spatial_residual(state, 0.0)
        assert np.abs(load - stiffness @ state - reference).max() < 1e-13

    def test_reaction_jacobian_form_matches_spatial_jacobian(
        self, numpy_bkd
    ):
        physics = self._nonlinear_physics(numpy_bkd)
        skfem_basis = physics.basis().skfem_basis()
        rng = np.random.RandomState(1)
        state = numpy_bkd.array(rng.uniform(0.1, 1.0, physics.nstates()))
        stiffness = sum(
            asm(form, skfem_basis) for form in physics.stiffness_forms()
        )
        reaction_jacobian = asm(
            physics.reaction_jacobian_form(),
            skfem_basis,
            u_prev=skfem_basis.interpolate(numpy_bkd.to_numpy(state)),
        )
        reference = physics.spatial_jacobian(state, 0.0)
        assert (
            np.abs(
                (-stiffness + reaction_jacobian - reference).toarray()
            ).max()
            < 1e-13
        )

    def test_spatial_jacobian_is_fd_consistent(self, numpy_bkd):
        """Ground truth for the reaction-Jacobian sign: dF/du of
        F = load - K*u gains +(w, R'(u)*du), so finite differences of
        spatial_residual must match spatial_jacobian."""
        physics = self._nonlinear_physics(numpy_bkd)
        rng = np.random.RandomState(3)
        state = numpy_bkd.array(rng.uniform(0.1, 1.0, physics.nstates()))
        jacobian = physics.spatial_jacobian(state, 0.0).toarray()
        step = 1e-7
        residual = physics.spatial_residual(state, 0.0)
        fd = np.zeros_like(jacobian)
        for j in range(physics.nstates()):
            perturbed = numpy_bkd.copy(state)
            perturbed[j] += step
            fd[:, j] = (
                physics.spatial_residual(perturbed, 0.0) - residual
            ) / step
        assert np.abs(jacobian - fd).max() < 1e-6

    def test_state_forms_partition_over_element_subsets(self, numpy_bkd):
        """Restricted assembly of the state-dependent forms partitions
        the global result (interpolation done per restricted basis)."""
        physics = self._nonlinear_physics(numpy_bkd)
        skfem_basis = physics.basis().skfem_basis()
        rng = np.random.RandomState(2)
        state_np = rng.uniform(0.1, 1.0, physics.nstates())
        nelems = skfem_basis.nelems
        halves = (
            np.arange(nelems // 2),
            np.arange(nelems // 2, nelems),
        )
        form = physics.reaction_form()
        full = asm(
            form, skfem_basis, u_prev=skfem_basis.interpolate(state_np)
        )
        parts = sum(
            asm(
                form,
                skfem_basis.with_elements(subset),
                u_prev=skfem_basis.with_elements(subset).interpolate(
                    state_np
                ),
            )
            for subset in halves
        )
        assert np.abs(full - parts).max() < 1e-14

    def test_exposed_forms_are_picklable(self, numpy_bkd):
        """Forms returned to consumers must survive pickling (kernels
        are module-level classes, not closures)."""
        import pickle

        physics = self._nonlinear_physics(numpy_bkd)
        forms = physics.stiffness_forms() + [
            physics.forcing_form(0.0),
            physics.reaction_form(),
            physics.reaction_jacobian_form(),
        ]
        for form in forms:
            assert form is not None
            pickle.loads(pickle.dumps(form))


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

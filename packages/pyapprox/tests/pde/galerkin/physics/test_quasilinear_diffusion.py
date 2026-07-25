"""FD validation of the quasilinear-diffusion typed assemblies.

The field-carrying term :math:`a(x) \\kappa(u) \\nabla u` is nonlinear
in the state, so unlike the ADR coefficients the mixed
second-derivative contractions are genuine assemblies; each is
DerivativeChecker-validated, and the two mixed contractions are tied by
the shared-tensor identity.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Tuple

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import QuasilinearDiffusion
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray


def _kappa(u):
    return 1.0 + u**2


def _kappa_deriv(u):
    return 2.0 * u


def _kappa_second_deriv(u):
    return np.full_like(u, 2.0)


def _forcing(x):
    return np.ones(x.shape[1])


def _build_physics(
    bkd: NumpyBkd, ndim: int
) -> Tuple[QuasilinearDiffusion[NumpyArray], LagrangeBasis[NumpyArray]]:
    if ndim == 1:
        mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    else:
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
        )
    basis = LagrangeBasis(mesh, degree=1)
    rng = np.random.default_rng(3)
    physics = QuasilinearDiffusion(
        basis=basis,
        diffusivity=NodalFieldDiffusion(
            basis, dofs=1.0 + 0.3 * rng.random(basis.ndofs())
        ),
        bkd=bkd,
        kappa=_kappa,
        kappa_deriv=_kappa_deriv,
        kappa_second_deriv=_kappa_second_deriv,
        forcing=_forcing,
        boundary_conditions=[DirichletBC(basis, "left", 0.0, bkd)],
    )
    return physics, basis


class TestQuasilinearDiffusion:
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_spatial_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """Newton jacobian (kappa' and kappa terms) vs FD of the
        residual in the state."""
        bkd = numpy_bkd
        physics, _ = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(5)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))

        def residual_of_state(samples: NumpyArray) -> NumpyArray:
            results = [
                bkd.to_numpy(
                    physics.spatial_residual(samples[:, ii], 0.0)
                ).copy()
                for ii in range(samples.shape[1])
            ]
            return bkd.asarray(np.stack(results, axis=1))

        def jac_of_state(sample: NumpyArray) -> NumpyArray:
            return bkd.asarray(
                np.asarray(
                    physics.spatial_jacobian(sample[:, 0], 0.0).todense()
                )
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=residual_of_state,
            jacobian=jac_of_state,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        # One-sided FD of the nonlinear residual bottoms near 1e-8
        # (verified V-shaped eps sweep) with roundoff blowup on the
        # small-eps side nudging the ratio past 1e-6; a genuine
        # jacobian bug plateaus at O(1).
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-7
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 5e-6

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_diffusivity_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """S(u) (with the kappa(u) weight) vs FD of the residual in the
        diffusivity DOFs."""
        bkd = numpy_bkd
        physics, _ = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        field = physics.diffusion_function()
        base_dofs = np.array(field.dofs(), copy=True)
        rng = np.random.default_rng(7)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))

        def residual_of_dofs(samples: NumpyArray) -> NumpyArray:
            results = []
            for ii in range(samples.shape[1]):
                field.set_dofs(bkd.to_numpy(samples[:, ii]))
                results.append(
                    bkd.to_numpy(physics.spatial_residual(state, 0.0)).copy()
                )
            field.set_dofs(base_dofs)
            return bkd.asarray(np.stack(results, axis=1))

        analytic = np.asarray(
            physics.residual_diffusivity_jacobian(state).todense()
        )

        def jac_of_dofs(sample: NumpyArray) -> NumpyArray:
            return bkd.asarray(analytic)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=base_dofs.shape[0],
            fun=residual_of_dofs,
            jacobian=jac_of_dofs,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            bkd.asarray(base_dofs)[:, None], relative=True
        )[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_state_field_hvp_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """The state-shaped mixed contraction is the exact gradient of
        u -> lambda^T (S(u) delta)."""
        bkd = numpy_bkd
        physics, _ = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(11)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        delta = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        adj_np = bkd.to_numpy(adj)
        delta_np = bkd.to_numpy(delta)

        def scalar_of_state(samples: NumpyArray) -> NumpyArray:
            results = [
                adj_np
                @ (
                    physics.residual_diffusivity_jacobian(samples[:, ii])
                    @ delta_np
                )
                for ii in range(samples.shape[1])
            ]
            return bkd.asarray(np.array(results)[None, :])

        def grad_of_state(sample: NumpyArray) -> NumpyArray:
            out = physics.residual_diffusivity_state_field_hvp(
                sample[:, 0], adj, delta
            )
            return bkd.asarray(bkd.to_numpy(out)[None, :])

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nstates,
            fun=scalar_of_state,
            jacobian=grad_of_state,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        # Noise-limited (V-shaped eps sweep; see spatial-jacobian test);
        # the scalar contraction's FD floor sits near 1e-7.
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-6
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 5e-6

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_mixed_contraction_shared_tensor_identity(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """Both mixed contractions read the same tensor:
        <field_state(w), delta> == <state_field(delta), w>."""
        bkd = numpy_bkd
        physics, _ = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(13)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        delta = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        lhs = bkd.sum(
            physics.residual_diffusivity_field_state_hvp(state, adj, wvec)
            * delta
        )
        rhs = bkd.sum(
            physics.residual_diffusivity_state_field_hvp(state, adj, delta)
            * wvec
        )
        bkd.assert_allclose(lhs, rhs, rtol=1e-12)

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_state_state_hvp_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """state_state_hvp (kappa'' pathway) is the exact gradient of
        u -> lambda^T (dR/du(u) w)."""
        bkd = numpy_bkd
        physics, _ = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(17)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        adj_np = bkd.to_numpy(adj)
        wvec_np = bkd.to_numpy(wvec)

        def scalar_of_state(samples: NumpyArray) -> NumpyArray:
            results = [
                adj_np
                @ (
                    physics.spatial_jacobian(samples[:, ii], 0.0)
                    @ wvec_np
                )
                for ii in range(samples.shape[1])
            ]
            return bkd.asarray(np.array(results)[None, :])

        def grad_of_state(sample: NumpyArray) -> NumpyArray:
            out = physics.state_state_hvp(sample[:, 0], adj, wvec, 0.0)
            return bkd.asarray(bkd.to_numpy(out)[None, :])

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nstates,
            fun=scalar_of_state,
            jacobian=grad_of_state,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        # Noise-limited (V-shaped eps sweep; see spatial-jacobian test);
        # the scalar contraction's FD floor sits near 1e-7.
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-6
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 5e-6

    def test_construction_validation(self, numpy_bkd: NumpyBkd) -> None:
        """Non-nodal diffusivity and DOF-count mismatch raise."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        with pytest.raises(TypeError, match="NodalFieldDiffusion"):
            QuasilinearDiffusion(
                basis=basis,
                diffusivity=1.0,  # type: ignore[arg-type]
                bkd=bkd,
                kappa=_kappa,
                kappa_deriv=_kappa_deriv,
                kappa_second_deriv=_kappa_second_deriv,
            )
        other_basis = LagrangeBasis(
            StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), degree=1
        )
        with pytest.raises(ValueError, match="DOFs"):
            QuasilinearDiffusion(
                basis=basis,
                diffusivity=NodalFieldDiffusion(other_basis),
                bkd=bkd,
                kappa=_kappa,
                kappa_deriv=_kappa_deriv,
                kappa_second_deriv=_kappa_second_deriv,
            )

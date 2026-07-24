"""FD validation of the galerkin ADR typed field-derivative assemblies.

Each ``residual_<field>_jacobian`` is DerivativeChecker-validated
against ``spatial_residual`` as a function of the field DOFs, and each
``residual_<field>_state_jacobian`` against the field Jacobian action
as a function of the state — the per-field unit tier below the
engine/facade tests.
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
    NodalFieldForcing,
    NodalFieldLinearReaction,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray


def _build_physics(
    bkd: NumpyBkd, ndim: int
) -> Tuple[AdvectionDiffusionReaction[NumpyArray], LagrangeBasis[NumpyArray]]:
    if ndim == 1:
        mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    else:
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
        )
    basis = LagrangeBasis(mesh, degree=1)
    ndofs = basis.ndofs()
    rng = np.random.default_rng(3)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(
            basis, dofs=1.0 + 0.3 * rng.random(ndofs)
        ),
        bkd=bkd,
        reaction=NodalFieldLinearReaction(
            basis, dofs=rng.normal(0.0, 0.5, ndofs)
        ),
        forcing=NodalFieldForcing(basis, dofs=rng.normal(0.0, 1.0, ndofs)),
        boundary_conditions=[DirichletBC(basis, "left", 0.0, bkd)],
    )
    return physics, basis


def _check_field_jacobian(
    bkd: NumpyBkd,
    physics: AdvectionDiffusionReaction[NumpyArray],
    field: object,
    jacobian_np: np.ndarray,
    state: NumpyArray,
) -> None:
    """DerivativeChecker: spatial_residual as a function of field DOFs."""
    base_dofs = np.array(field.dofs(), copy=True)  # type: ignore[attr-defined]
    nstates = physics.nstates()

    def residual_of_dofs(samples: NumpyArray) -> NumpyArray:
        results = []
        for ii in range(samples.shape[1]):
            field.set_dofs(  # type: ignore[attr-defined]
                bkd.to_numpy(samples[:, ii])
            )
            results.append(
                bkd.to_numpy(physics.spatial_residual(state, 0.0)).copy()
            )
        field.set_dofs(base_dofs)  # type: ignore[attr-defined]
        return bkd.asarray(np.stack(results, axis=1))

    def jac_of_dofs(sample: NumpyArray) -> NumpyArray:
        return bkd.asarray(jacobian_np)

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


class TestADRFieldDerivatives:
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_diffusivity_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, ndim)
        rng = np.random.default_rng(5)
        state = bkd.asarray(rng.normal(0.0, 0.5, physics.nstates()))
        analytic = np.asarray(
            physics.residual_diffusivity_jacobian(state).todense()
        )
        _check_field_jacobian(
            bkd, physics, physics.diffusion_function(), analytic, state
        )

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_forcing_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, ndim)
        rng = np.random.default_rng(6)
        state = bkd.asarray(rng.normal(0.0, 0.5, physics.nstates()))
        analytic = np.asarray(physics.residual_forcing_jacobian().todense())
        _check_field_jacobian(
            bkd, physics, physics.forcing_function(), analytic, state
        )

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_reaction_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, ndim)
        rng = np.random.default_rng(7)
        state = bkd.asarray(rng.normal(0.0, 0.5, physics.nstates()))
        analytic = np.asarray(
            physics.residual_reaction_jacobian(state).todense()
        )
        _check_field_jacobian(
            bkd, physics, physics.reaction_function(), analytic, state
        )

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_reaction_state_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """A_r(delta) vs FD of u -> S_r(u) delta (DerivativeChecker)."""
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(9)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        delta = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        delta_np = bkd.to_numpy(delta)

        def sdelta_of_state(samples: NumpyArray) -> NumpyArray:
            results = []
            for ii in range(samples.shape[1]):
                results.append(
                    np.asarray(
                        physics.residual_reaction_jacobian(samples[:, ii])
                        @ delta_np
                    )
                )
            return bkd.asarray(np.stack(results, axis=1))

        def jac_of_state(sample: NumpyArray) -> NumpyArray:
            return bkd.asarray(
                np.asarray(
                    physics.residual_reaction_state_jacobian(
                        delta, sample[:, 0]
                    ).todense()
                )
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=sdelta_of_state,
            jacobian=jac_of_state,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_diffusivity_state_jacobian_vs_fd(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        """A(delta) vs FD of u -> B(u) delta (DerivativeChecker)."""
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(8)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        delta = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        delta_np = bkd.to_numpy(delta)

        def bdelta_of_state(samples: NumpyArray) -> NumpyArray:
            results = []
            for ii in range(samples.shape[1]):
                results.append(
                    np.asarray(
                        physics.residual_diffusivity_jacobian(
                            samples[:, ii]
                        )
                        @ delta_np
                    )
                )
            return bkd.asarray(np.stack(results, axis=1))

        def jac_of_state(sample: NumpyArray) -> NumpyArray:
            return bkd.asarray(
                np.asarray(
                    physics.residual_diffusivity_state_jacobian(
                        delta, sample[:, 0]
                    ).todense()
                )
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=nstates,
            fun=bdelta_of_state,
            jacobian=jac_of_state,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    def test_forcing_set_dofs_changes_residual(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """set_dofs invalidates the forward path (no stale caches)."""
        bkd = numpy_bkd
        physics, basis = _build_physics(bkd, 1)
        state = bkd.asarray(np.zeros(physics.nstates()))
        res_a = bkd.to_numpy(physics.spatial_residual(state, 0.0)).copy()
        forcing = physics.forcing_function()
        assert isinstance(forcing, NodalFieldForcing)
        forcing.set_dofs(np.ones(physics.nstates()))
        res_b = bkd.to_numpy(physics.spatial_residual(state, 0.0))
        assert float(np.max(np.abs(res_a - res_b))) > 1e-8

    def test_forcing_jacobian_requires_nodal_field(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            forcing=lambda x: np.ones(x.shape[1]),
            boundary_conditions=[DirichletBC(basis, "left", 0.0, bkd)],
        )
        with pytest.raises(TypeError, match="NodalFieldForcing"):
            physics.residual_forcing_jacobian()

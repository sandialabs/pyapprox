"""The Galerkin ADR advection term, as assembled from an advection form.

The reference assemblies are the two weak forms written directly as skfem
forms, independent of the form objects: (v.grad u) w for the
non-conservative form and -u v.grad w for the conservative one. The
advection contribution is isolated as the difference between the physics
with and without the velocity.
"""

import pickle

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
from skfem import BilinearForm, asm
from skfem.helpers import dot, grad

from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CoordinateVelocity,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.util.backends.numpy import NumpyBkd

_Arr = NDArray[Any]


def _velocity(x: _Arr) -> _Arr:
    return np.stack([1.0 + x[1] ** 2, np.sin(x[0])])


def _dense(mat: Any) -> _Arr:
    return np.asarray(mat.toarray() if issparse(mat) else mat)


def _nonconservative(u: Any, v: Any, w: Any) -> Any:
    return dot(_velocity(w.x), grad(u)) * v


def _conservative(u: Any, v: Any, w: Any) -> Any:
    return -u * dot(_velocity(w.x), grad(v))


def _physics(
    bkd: NumpyBkd, form: AdvectionFormProtocol, with_velocity: bool
) -> AdvectionDiffusionReaction[_Arr]:
    mesh = StructuredMesh2D(
        nx=5, ny=4, bounds=[[0.0, 1.0], [0.0, 2.0]], bkd=bkd
    )
    return AdvectionDiffusionReaction(
        basis=LagrangeBasis(mesh, degree=2),
        diffusivity=0.3,
        bkd=bkd,
        velocity=CoordinateVelocity(_velocity) if with_velocity else None,
        reaction=0.5,
        advection_form=form,
    )


_CASES = [
    (NonConservativeAdvection(), _nonconservative),
    (ConservativeAdvection(), _conservative),
]
_IDS = ["nonconservative", "conservative"]


class TestAdvectionAssembly:
    @pytest.mark.parametrize("form,reference", _CASES, ids=_IDS)
    def test_jacobian_and_residual_match_reference(
        self,
        numpy_bkd: NumpyBkd,
        form: AdvectionFormProtocol,
        reference: Callable[[Any, Any, Any], Any],
    ) -> None:
        """Advection enters F = b - K u, so it adds -K_adv to the Jacobian
        and -K_adv u to the residual."""
        bkd = numpy_bkd
        with_vel = _physics(bkd, form, True)
        without = _physics(bkd, form, False)
        k_adv = _dense(
            asm(BilinearForm(reference), with_vel.basis().skfem_basis())
        )
        state = bkd.asarray(
            np.random.default_rng(1).normal(size=with_vel.nstates())
        )
        jac_diff = _dense(with_vel.interior_jacobian(state, 0.0)) - _dense(
            without.interior_jacobian(state, 0.0)
        )
        bkd.assert_allclose(
            bkd.asarray(jac_diff), bkd.asarray(-k_adv), rtol=1e-12, atol=1e-13
        )
        res_diff = with_vel.interior_residual(
            state, 0.0
        ) - without.interior_residual(state, 0.0)
        bkd.assert_allclose(
            res_diff,
            bkd.asarray(-k_adv @ bkd.to_numpy(state)),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_forms_agree_on_interior_rows_for_divergence_free_velocity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Integrating by parts, -(u, v.grad w) = (v.grad u, w)
        + (div(v) u, w) - boundary term. For divergence-free v and a test
        function vanishing on the boundary both extra terms vanish, so the
        two stiffnesses agree on interior rows; on boundary rows they
        differ by the boundary term. Derivation-free: it checks the
        conservative sign without restating the form."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=6, ny=6, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)

        def rotation(x: _Arr) -> _Arr:
            return np.stack([-(x[1] - 0.5), x[0] - 0.5])

        stiffness = []
        for form in (NonConservativeAdvection(), ConservativeAdvection()):
            # Diffusion is the same in both, so it cancels in the comparison.
            physics = AdvectionDiffusionReaction(
                basis=basis,
                diffusivity=1.0,
                bkd=bkd,
                velocity=CoordinateVelocity(rotation),
                advection_form=form,
            )
            zero = bkd.zeros((basis.ndofs(),))
            stiffness.append(-_dense(physics.interior_jacobian(zero, 0.0)))
        boundary = np.asarray(basis.skfem_basis().get_dofs().flatten())
        interior = np.setdiff1d(np.arange(basis.ndofs()), boundary)
        bkd.assert_allclose(
            bkd.asarray(stiffness[1][interior]),
            bkd.asarray(stiffness[0][interior]),
            atol=1e-12,
        )
        assert (
            np.abs(stiffness[1][boundary] - stiffness[0][boundary]).max()
            > 1e-3
        )

    def test_rejects_non_form(self, numpy_bkd: NumpyBkd) -> None:
        mesh = StructuredMesh2D(
            nx=2, ny=2, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=numpy_bkd
        )
        with pytest.raises(TypeError, match="AdvectionFormProtocol"):
            AdvectionDiffusionReaction(
                basis=LagrangeBasis(mesh, degree=1),
                diffusivity=1.0,
                bkd=numpy_bkd,
                advection_form=object(),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "form", [NonConservativeAdvection(), ConservativeAdvection()], ids=_IDS
    )
    def test_physics_pickles(
        self, numpy_bkd: NumpyBkd, form: AdvectionFormProtocol
    ) -> None:
        physics = _physics(numpy_bkd, form, True)
        restored = pickle.loads(pickle.dumps(physics))
        state = numpy_bkd.asarray(
            np.random.default_rng(2).normal(size=physics.nstates())
        )
        numpy_bkd.assert_allclose(
            restored.interior_residual(state, 0.0),
            physics.interior_residual(state, 0.0),
            rtol=1e-14,
        )

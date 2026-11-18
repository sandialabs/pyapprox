"""The Galerkin ADR advection term, as assembled from an advection form.

The reference assemblies are the two weak forms written directly as skfem
forms, independent of the form objects: (v.grad u) w for the
non-conservative form, and -u v.grad w plus the boundary term
<(v.n) u, w> on every boundary facet for the conservative one. The
advection contribution is isolated as the difference between the physics
with and without the velocity.

Derivation-free check: integrating by parts, the conservative stiffness
equals the non-conservative one plus (div(v) u, w) on every row,
boundary rows included; exact to round-off on Q2 with the quadratures
used. Wrong-model control: the conservative interior forms alone (no
boundary term) differ from it on boundary rows.
"""

import pickle

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Tuple

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
    TimeIndependent,
)
from pyapprox.pde.elementwise import ElementSet
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


def _conservative_boundary(u: Any, v: Any, w: Any) -> Any:
    return dot(_velocity(w.x), w.n) * u * v


def _reference_stiffness(
    skfem_basis: Any, interior: Callable[..., Any], boundary: Any
) -> _Arr:
    stiffness = _dense(asm(BilinearForm(interior), skfem_basis))
    if boundary is not None:
        stiffness = stiffness + _dense(
            # The physics' facet order on Q2 (maxdeg 4): state and test
            # degrees plus a velocity of the state's degree. The velocity
            # has sin(x), so a different order differs by ~1e-9.
            asm(BilinearForm(boundary), skfem_basis.boundary(intorder=12))
        )
    return stiffness


def _rotation(x: _Arr) -> _Arr:
    return np.stack([-(x[1] - 0.5), x[0] - 0.5])


def _zero_divergence(u: Any, v: Any, w: Any) -> Any:
    return 0.0 * u * v


def _expanding(x: _Arr) -> _Arr:
    return np.stack([x[0] ** 2, x[0] * x[1]])


def _expanding_divergence(u: Any, v: Any, w: Any) -> Any:
    # div(x^2, x y) = 2x + x.
    return 3.0 * w.x[0] * u * v


def _q2_basis(bkd: NumpyBkd) -> LagrangeBasis[_Arr]:
    mesh = StructuredMesh2D(nx=6, ny=6, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    return LagrangeBasis(mesh, degree=2)


def _advection_stiffness(
    bkd: NumpyBkd,
    basis: LagrangeBasis[_Arr],
    velocity: Callable[[_Arr], _Arr],
    form: AdvectionFormProtocol,
) -> _Arr:
    """K_adv, the stiffness with the velocity minus the one without."""
    zero = bkd.zeros((basis.ndofs(),))
    jacobians = [
        _dense(
            AdvectionDiffusionReaction(
                basis=basis,
                diffusivity=1.0,
                bkd=bkd,
                velocity=vel,
                advection_form=form,
            ).interior_jacobian(zero, 0.0)
        )
        for vel in (CoordinateVelocity(TimeIndependent(velocity)), None)
    ]
    return jacobians[1] - jacobians[0]


def _physics(
    bkd: NumpyBkd, form: AdvectionFormProtocol, with_velocity: bool
) -> AdvectionDiffusionReaction[_Arr]:
    mesh = StructuredMesh2D(
        nx=5, ny=4, bounds=[(0.0, 1.0), (0.0, 2.0)], bkd=bkd
    )
    return AdvectionDiffusionReaction(
        basis=LagrangeBasis(mesh, degree=2),
        diffusivity=0.3,
        bkd=bkd,
        velocity=(
            CoordinateVelocity(TimeIndependent(_velocity))
            if with_velocity
            else None
        ),
        reaction=0.5,
        advection_form=form,
    )


_CASES = [
    (NonConservativeAdvection(), (_nonconservative, None)),
    (ConservativeAdvection(), (_conservative, _conservative_boundary)),
]
_IDS = ["nonconservative", "conservative"]


class TestAdvectionAssembly:
    @pytest.mark.parametrize("form,reference", _CASES, ids=_IDS)
    def test_jacobian_and_residual_match_reference(
        self,
        numpy_bkd: NumpyBkd,
        form: AdvectionFormProtocol,
        reference: Tuple[Callable[..., Any], Any],
    ) -> None:
        """Advection enters F = b - K u, so it adds -K_adv to the Jacobian
        and -K_adv u to the residual."""
        bkd = numpy_bkd
        with_vel = _physics(bkd, form, True)
        without = _physics(bkd, form, False)
        k_adv = _reference_stiffness(
            with_vel.basis().skfem_basis(), reference[0], reference[1]
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

    @pytest.mark.parametrize(
        "velocity,divergence",
        [(_rotation, _zero_divergence), (_expanding, _expanding_divergence)],
        ids=["divergence_free", "divergent"],
    )
    def test_forms_differ_by_the_divergence_on_every_row(
        self,
        numpy_bkd: NumpyBkd,
        velocity: Callable[[_Arr], _Arr],
        divergence: Callable[[Any, Any, Any], Any],
    ) -> None:
        """Integrating by parts, -(u, v.grad w) + <(v.n) u, w>
        = (v.grad u, w) + (div(v) u, w) for every test function, so the
        conservative stiffness is the non-conservative one plus the
        divergence mass, boundary rows included. Derivation-free: it
        checks the boundary term's sign and normal without restating
        it."""
        bkd = numpy_bkd
        basis = _q2_basis(bkd)
        stiffness = [
            _advection_stiffness(bkd, basis, velocity, form)
            for form in (NonConservativeAdvection(), ConservativeAdvection())
        ]
        divergence_mass = _dense(
            asm(BilinearForm(divergence), basis.skfem_basis())
        )
        bkd.assert_allclose(
            bkd.asarray(stiffness[1]),
            bkd.asarray(stiffness[0] + divergence_mass),
            atol=1e-12,
        )

    def test_interior_forms_alone_miss_the_boundary_rows(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Wrong-model control: the conservative form without its
        boundary term (its interior forms alone) matches on interior rows
        for divergence-free velocity and misses on boundary rows."""
        bkd = numpy_bkd
        basis = _q2_basis(bkd)
        noncons = _advection_stiffness(
            bkd, basis, _rotation, NonConservativeAdvection()
        )
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            velocity=CoordinateVelocity(TimeIndependent(_rotation)),
            advection_form=ConservativeAdvection(),
        )
        skfem_basis = basis.skfem_basis()
        # The forms are diffusion then advection; the diffusion cancels.
        interior_only = _dense(asm(physics.stiffness_forms()[1], skfem_basis))
        boundary = np.asarray(skfem_basis.get_dofs().flatten())
        interior = np.setdiff1d(np.arange(basis.ndofs()), boundary)
        bkd.assert_allclose(
            bkd.asarray(interior_only[interior]),
            bkd.asarray(noncons[interior]),
            atol=1e-12,
        )
        assert np.abs(interior_only[boundary] - noncons[boundary]).max() > 1e-3

    def test_element_evaluation_refuses_the_boundary_term(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The element-wise path cannot yet attribute the boundary term,
        so it raises rather than return residuals missing it."""
        physics = _physics(numpy_bkd, ConservativeAdvection(), True)
        restricted = physics.restrict(
            ElementSet.all(physics.nelements(), numpy_bkd)
        )
        nper = restricted.element_row_dofs().shape[1]
        local = np.zeros((physics.nelements(), nper))
        with pytest.raises(NotImplementedError, match="boundary terms"):
            restricted.element_residuals(local, 0.0)

    def test_rejects_non_form(self, numpy_bkd: NumpyBkd) -> None:
        mesh = StructuredMesh2D(
            nx=2, ny=2, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=numpy_bkd
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

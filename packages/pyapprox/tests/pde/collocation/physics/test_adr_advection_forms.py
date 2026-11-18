"""The collocation ADR physics under an injected advection form.

- The two forms differ by the product rule only: on 10 Chebyshev nodes
  of [-1, 1] with v = 1 + x and u = x**2 (v u is a cubic, so the
  derivative matrices differentiate it exactly), the non-conservative
  residual minus the conservative one is u v' to 1e-12, and the
  conservative advection term alone is -(v u)' = -(2 x + 3 x**2).
- Both forms' state Jacobians pass DerivativeChecker on a 6 x 6 mesh of
  [-1, 1]^2 with variable diffusion 2 + x y and velocity (1 + y, x / 2).
- The flux carries the form's advective part: -D u' + v u conservative,
  -D u' non-conservative; each Jacobian applied to the (linear) state is
  the flux.
- A constructor given something other than an advection form refuses it.
- Through torch, the gradient of the summed conservative residual with
  respect to the velocity is -(D^T 1) u.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import PhysicsTestBase

if TYPE_CHECKING:
    from pyapprox.util.backends.torch import TorchBkd

_FORMS = [NonConservativeAdvection(), ConservativeAdvection()]


def _physics_1d(
    bkd: Backend[Array], form: AdvectionFormProtocol, diffusion: float = 1.0
) -> Tuple[Array, AdvectionDiffusionReaction[Array]]:
    """The nodes x and the physics with velocity 1 + x."""
    basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
    nodes = basis.nodes()
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=diffusion,
        velocity=[1.0 + nodes],
        advection_form=form,
    )
    return nodes, physics


class TestAdvectionForms(PhysicsTestBase):
    def test_forms_differ_by_the_product_rule(self, bkd: Backend[Array]) -> None:
        x, nonconservative = _physics_1d(bkd, NonConservativeAdvection())
        _, conservative = _physics_1d(bkd, ConservativeAdvection())
        state = x**2
        difference = nonconservative.residual(state, 0.0) - conservative.residual(
            state, 0.0
        )
        bkd.assert_allclose(difference, state * 1.0, atol=1e-12)

    def test_conservative_term(self, bkd: Backend[Array]) -> None:
        basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
        x = basis.nodes()
        physics = AdvectionDiffusionReaction(
            basis, bkd, velocity=[1.0 + x], advection_form=ConservativeAdvection()
        )
        bkd.assert_allclose(
            physics.residual(x**2, 0.0), -(2.0 * x + 3.0 * x**2), atol=1e-12
        )

    @pytest.mark.parametrize("form", _FORMS, ids=repr)
    def test_jacobian(self, bkd: Backend[Array], form: AdvectionFormProtocol) -> None:
        mesh = TransformedMesh2D(6, 6, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        x, y = mesh.points()[0], mesh.points()[1]
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=NodalField(2.0 + x * y),
            velocity=[1.0 + y, x / 2.0],
            reaction=-0.5,
            advection_form=form,
        )
        state = bkd.asarray(np.random.default_rng(2).normal(0.0, 1.0, basis.npts()))
        self.check_jacobian(bkd, physics, state, atol=1e-6)

    @pytest.mark.parametrize(
        "form, advective",
        [(NonConservativeAdvection(), False), (ConservativeAdvection(), True)],
        ids=["non-conservative", "conservative"],
    )
    def test_flux(
        self, bkd: Backend[Array], form: AdvectionFormProtocol, advective: bool
    ) -> None:
        x, physics = _physics_1d(bkd, form, diffusion=2.0)
        state = x**2
        expected = -2.0 * (2.0 * x)
        if advective:
            expected = expected + (1.0 + x) * state
        (flux,) = physics.compute_flux(state, 0.0)
        bkd.assert_allclose(flux, expected, atol=1e-12)
        (jacobian,) = physics.compute_flux_jacobian(state, 0.0)
        bkd.assert_allclose(jacobian @ state, flux, atol=1e-12)

    def test_refuses_a_non_form(self, numpy_bkd: Backend[Array]) -> None:
        basis = ChebyshevBasis1D(TransformedMesh1D(5, numpy_bkd), numpy_bkd)
        with pytest.raises(TypeError, match="AdvectionFormProtocol"):
            AdvectionDiffusionReaction(
                basis, numpy_bkd, diffusion=1.0, advection_form=object()  # type: ignore[arg-type]
            )

    def test_conservative_velocity_gradient(self, torch_bkd: "TorchBkd") -> None:
        import torch

        bkd = torch_bkd
        basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
        x = basis.nodes()
        velocity = (1.0 + x).detach().clone().requires_grad_(True)
        physics = AdvectionDiffusionReaction(
            basis, bkd, velocity=[velocity], advection_form=ConservativeAdvection()
        )
        state = x**2
        (grad,) = torch.autograd.grad(
            torch.sum(physics.residual(state, 0.0)), [velocity]
        )
        derivative = basis.derivative_matrix(1, 0)
        expected = -(derivative.T @ bkd.ones((10,))) * state
        bkd.assert_allclose(grad, expected, rtol=1e-12, atol=1e-12)

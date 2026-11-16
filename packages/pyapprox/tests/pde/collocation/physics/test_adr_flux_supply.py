"""The ADR flux supply reads its coefficients at its own time.

``compute_flux`` and ``compute_flux_jacobian`` evaluate the diffusivity
at the time they are given and the current parameters. They must not
depend on whether, or at what time, the residual was evaluated before.
The expected fluxes are recomputed here from the public derivative
matrices.
"""

from typing import TYPE_CHECKING, Generic, List

import numpy as np
import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.util.backends.torch import TorchBkd

_NPTS_1D = 5


class _GrowingField(Generic[Array]):
    """D(t) = (1 + t) D0: a time-dependent diffusivity."""

    def __init__(self, base: Array) -> None:
        self._base = base

    def __call__(self, time: float) -> Array:
        return (1.0 + time) * self._base


def _setup(
    bkd: Backend[Array],
) -> tuple[ChebyshevBasis2D[Array], AdvectionDiffusionReaction[Array], Array]:
    basis = ChebyshevBasis2D(TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd), bkd)
    physics = AdvectionDiffusionReaction(basis, bkd)
    state = bkd.asarray(np.random.uniform(-1.0, 1.0, (basis.npts(),)))
    return basis, physics, state


def _expected_flux(
    basis: ChebyshevBasis2D[Array], diffusion: Array, state: Array
) -> List[Array]:
    return [
        -diffusion * (basis.derivative_matrix(1, dim) @ state)
        for dim in range(basis.ndim())
    ]


class TestADRFluxSupply:
    def test_flux_before_any_residual(self, bkd: Backend[Array]) -> None:
        """A flux evaluated with no residual evaluation before it reads
        the diffusivity just set, not the construction value."""
        basis, physics, state = _setup(bkd)
        diffusion = bkd.asarray(np.random.uniform(1.5, 2.5, (basis.npts(),)))
        physics.set_diffusion(ConstantInTimeField(diffusion))
        flux = physics.compute_flux(state, 0.0)
        for got, expected in zip(flux, _expected_flux(basis, diffusion, state)):
            bkd.assert_allclose(got, expected, rtol=1e-12)

    def test_flux_at_its_own_time(self, bkd: Backend[Array]) -> None:
        """After a residual at t = 0, a flux at t = 1 uses D(1), and a
        later flux at t = 0 uses D(0) again."""
        basis, physics, state = _setup(bkd)
        base = bkd.asarray(np.random.uniform(1.5, 2.5, (basis.npts(),)))
        physics.set_diffusion(_GrowingField(base))
        physics.residual(state, 0.0)
        for time in (1.0, 0.0):
            flux = physics.compute_flux(state, time)
            expected = _expected_flux(basis, (1.0 + time) * base, state)
            for got, want in zip(flux, expected):
                bkd.assert_allclose(got, want, rtol=1e-12)

    def test_flux_jacobian_at_its_own_time(self, bkd: Backend[Array]) -> None:
        """The flux is linear in the state with no advection, so its
        Jacobian at a time applied to the state is the flux there."""
        _, physics, state = _setup(bkd)
        npts = state.shape[0]
        base = bkd.asarray(np.random.uniform(1.5, 2.5, (npts,)))
        physics.set_diffusion(_GrowingField(base))
        physics.residual(state, 0.0)
        flux = physics.compute_flux(state, 1.0)
        jacs = physics.compute_flux_jacobian(state, 1.0)
        for jac, got in zip(jacs, flux):
            bkd.assert_allclose(jac @ state, got, rtol=1e-12)

    def test_constant_diffusion_residual(self, bkd: Backend[Array]) -> None:
        """Constant D: the residual is D times the Laplacian, and the
        Jacobian applied to the state reproduces it."""
        basis, _, state = _setup(bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=2.0)
        laplacian = basis.derivative_matrix(2, 0) + basis.derivative_matrix(2, 1)
        residual = physics.residual(state, 0.0)
        bkd.assert_allclose(residual, 2.0 * (laplacian @ state), rtol=1e-12)
        bkd.assert_allclose(physics.jacobian(state, 0.0) @ state, residual, rtol=1e-12)

    def test_interface_flux_refuses_function_diffusion(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """The interface flux takes no time, so a diffusivity set as a
        function is refused rather than read at a guessed time."""
        basis, physics, state = _setup(numpy_bkd)
        physics.set_diffusion(ConstantInTimeField(numpy_bkd.full((basis.npts(),), 2.0)))
        with pytest.raises(ValueError, match="takes no time"):
            physics.compute_interface_flux(
                state, numpy_bkd.array([0], dtype=int), numpy_bkd.array([1.0, 0.0])
            )

    def test_flux_preserves_autograd(self, torch_bkd: "TorchBkd") -> None:
        """d(sum flux_0)/dD through autograd equals -(D_0 u)."""
        import torch

        basis, physics, state = _setup(torch_bkd)
        diffusion = torch.full(
            (basis.npts(),), 2.0, dtype=torch.float64, requires_grad=True
        )
        physics.set_diffusion(ConstantInTimeField(diffusion))
        flux = physics.compute_flux(state, 0.0)
        (grad,) = torch.autograd.grad(torch.sum(flux[0]), [diffusion])
        expected = -(basis.derivative_matrix(1, 0) @ state)
        torch_bkd.assert_allclose(grad, expected, rtol=1e-12)

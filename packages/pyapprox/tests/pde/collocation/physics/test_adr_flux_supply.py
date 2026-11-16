"""The ADR flux supply reads its coefficients at its own time.

``compute_flux`` and ``compute_flux_jacobian`` evaluate the diffusivity
at the time they are given and the current parameters. They must not
depend on whether, or at what time, the residual was evaluated before.
The expected fluxes are recomputed here from the public derivative
matrices.
"""

from typing import TYPE_CHECKING, Generic, List, Optional

import numpy as np
import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import (
    NodalCoefficientProtocol,
    NodalField,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.util.backends.torch import TorchBkd

_NPTS_1D = 5


class _GrowingField(Generic[Array]):
    """D(t) = (1 + t) D0: a diffusivity declared time-dependent."""

    def __init__(self, base: Array) -> None:
        self._base = base

    def values(self, time: float) -> Array:
        return (1.0 + time) * self._base

    def is_time_dependent(self) -> bool:
        return True

    def version(self) -> int:
        return 0


def _basis(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    return ChebyshevBasis2D(TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd), bkd)


def _setup(
    bkd: Backend[Array],
    diffusion: Optional[NodalCoefficientProtocol[Array]] = None,
) -> tuple[ChebyshevBasis2D[Array], AdvectionDiffusionReaction[Array], Array]:
    basis = _basis(bkd)
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=diffusion)
    state = bkd.asarray(np.random.uniform(-1.0, 1.0, (basis.npts(),)))
    return basis, physics, state


def _random_diffusion(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.uniform(1.5, 2.5, (_NPTS_1D**2,)))


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
        the diffusivity just written, not the construction value."""
        field = NodalField(bkd.ones((_NPTS_1D**2,)))
        basis, physics, state = _setup(bkd, field)
        diffusion = _random_diffusion(bkd)
        field.set_values(diffusion)
        flux = physics.compute_flux(state, 0.0)
        for got, expected in zip(flux, _expected_flux(basis, diffusion, state)):
            bkd.assert_allclose(got, expected, rtol=1e-12)

    def test_flux_at_its_own_time(self, bkd: Backend[Array]) -> None:
        """After a residual at t = 0, a flux at t = 1 uses D(1), and a
        later flux at t = 0 uses D(0) again."""
        base = _random_diffusion(bkd)
        basis, physics, state = _setup(bkd, _GrowingField(base))
        physics.residual(state, 0.0)
        for time in (1.0, 0.0):
            flux = physics.compute_flux(state, time)
            expected = _expected_flux(basis, (1.0 + time) * base, state)
            for got, want in zip(flux, expected):
                bkd.assert_allclose(got, want, rtol=1e-12)

    def test_flux_jacobian_at_its_own_time(self, bkd: Backend[Array]) -> None:
        """The flux is linear in the state with no advection, so its
        Jacobian at a time applied to the state is the flux there."""
        _, physics, state = _setup(bkd, _GrowingField(_random_diffusion(bkd)))
        physics.residual(state, 0.0)
        flux = physics.compute_flux(state, 1.0)
        jacs = physics.compute_flux_jacobian(state, 1.0)
        for jac, got in zip(jacs, flux):
            bkd.assert_allclose(jac @ state, got, rtol=1e-12)

    def test_constant_diffusion_residual(self, bkd: Backend[Array]) -> None:
        """Constant D: the residual is D times the Laplacian, and the
        Jacobian applied to the state reproduces it."""
        basis = _basis(bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=2.0)
        state = bkd.asarray(np.random.uniform(-1.0, 1.0, (basis.npts(),)))
        laplacian = basis.derivative_matrix(2, 0) + basis.derivative_matrix(2, 1)
        residual = physics.residual(state, 0.0)
        bkd.assert_allclose(residual, 2.0 * (laplacian @ state), rtol=1e-12)
        bkd.assert_allclose(physics.jacobian(state, 0.0) @ state, residual, rtol=1e-12)

    def test_interface_flux_reads_a_time_independent_field(
        self, bkd: Backend[Array]
    ) -> None:
        """D grad(u) . n at the chosen nodes, with D read from the field."""
        diffusion = _random_diffusion(bkd)
        basis, physics, state = _setup(bkd, NodalField(diffusion))
        indices = bkd.array([0, 3], dtype=int)
        got = physics.compute_interface_flux(state, indices, bkd.array([1.0, 0.0]))
        expected = (diffusion * (basis.derivative_matrix(1, 0) @ state))[indices]
        bkd.assert_allclose(got, expected, rtol=1e-12)

    def test_interface_flux_refuses_time_dependent_diffusion(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """The interface flux takes no time, so a diffusivity declared
        time-dependent is refused rather than read at a guessed time."""
        _, physics, state = _setup(
            numpy_bkd, _GrowingField(_random_diffusion(numpy_bkd))
        )
        with pytest.raises(ValueError, match="takes no time"):
            physics.compute_interface_flux(
                state, numpy_bkd.array([0], dtype=int), numpy_bkd.array([1.0, 0.0])
            )

    def test_flux_preserves_autograd(self, torch_bkd: "TorchBkd") -> None:
        """d(sum flux_0)/dD through autograd equals -(D_0 u)."""
        import torch

        diffusion = torch.full(
            (_NPTS_1D**2,), 2.0, dtype=torch.float64, requires_grad=True
        )
        basis, physics, state = _setup(torch_bkd, NodalField(diffusion))
        flux = physics.compute_flux(state, 0.0)
        (grad,) = torch.autograd.grad(torch.sum(flux[0]), [diffusion])
        expected = -(basis.derivative_matrix(1, 0) @ state)
        torch_bkd.assert_allclose(grad, expected, rtol=1e-12)

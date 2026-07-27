"""Diffusivity parameterization for galerkin quasilinear diffusion.

The field-carrying term :math:`a(x) \\kappa(u) \\nabla u` is nonlinear
in the state, so the engine term is wired with genuine callable mixed
second-derivative contractions (the physics's typed
``residual_diffusivity_*`` assemblies) instead of the linearity-derived
slots that suffice for coefficients entering linearly.
"""

from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.galerkin.physics.quasilinear_diffusion import (
    QuasilinearDiffusion,
)
from pyapprox.pde.parameterizations.field_term import (
    MixedHVPAdapter,
    StateJacobianAdapter,
    ToNumpySetter,
    Zero,
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_quasilinear_diffusivity_parameterization(
    physics: QuasilinearDiffusion[Array],
    field_map: FieldMapProtocol[Array],
    bkd: Backend[Array],
) -> "_FieldParameterizationTerm[Array, QuasilinearDiffusion[Array]]":
    """Create a diffusivity parameterization for quasilinear diffusion.

    Maps parameters onto the nodal diffusivity DOFs (positivity enforced
    on apply). The derivative bundle is second order when the field map
    declares a usable HVP. The residual is linear in the field itself,
    so the field-field curvature slot is certified zero.

    Parameters
    ----------
    physics : QuasilinearDiffusion
        The bound physics.
    field_map : FieldMapProtocol
        Map from parameters onto the diffusivity DOFs.
    bkd : Backend
        Computational backend.

    Returns
    -------
    _FieldParameterizationTerm
    """
    if not isinstance(physics, QuasilinearDiffusion):
        raise TypeError(
            "physics must be a QuasilinearDiffusion, got "
            f"{type(physics).__name__}"
        )
    diffusion = physics.diffusion_function()
    return _FieldParameterizationTerm(
        setter=ToNumpySetter(diffusion.set_dofs, bkd),
        physics=physics,
        field_jacobian=StateJacobianAdapter(
            physics.residual_diffusivity_jacobian
        ),
        field_state_hvp=MixedHVPAdapter(
            physics.residual_diffusivity_field_state_hvp
        ),
        state_field_hvp=MixedHVPAdapter(
            physics.residual_diffusivity_state_field_hvp
        ),
        field_field_hvp=Zero(),
        field_map=field_map,
        bkd=bkd,
        nstates=physics.nstates(),
        nfield_dofs=diffusion.ndofs(),
        owned_coefficients=("diffusivity",),
        require_positive=True,
    )

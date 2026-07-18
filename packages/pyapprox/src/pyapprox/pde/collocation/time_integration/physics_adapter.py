"""Adapter to bridge Physics to ODEResidualProtocol.

This module provides fixed-tier adapters that wrap a collocation Physics
object to conform to the ODEResidualProtocol used by time integrators.

Key interface differences:
- Physics: residual(state, time), jacobian(state, time), mass_matrix()
- ODEResidual: __call__(state), jacobian(state), set_time(time), mass_matrix(nstates)

Capability is decided ONCE at construction by
:func:`create_physics_ode_residual`, which None-checks the
parameterization's :class:`ParamDerivatives` bundle to select a
fixed-tier adapter class (never ``hasattr``):

- :class:`PhysicsToODEResidualAdapter` — raw f/J/M, no parameters.
- :class:`PhysicsToODEResidualWithSetParamAdapter` — adds
  ``nparams``/``set_param`` (parameterization without derivative
  capability).
- :class:`PhysicsToODEResidualWithParamJacobianAdapter` — adds
  ``param_jacobian``/``initial_param_jacobian``/
  ``bc_flux_param_sensitivity`` (first-order bundle).
- :class:`PhysicsToODEResidualWithHVPAdapter` — adds the three
  parameterization HVPs (second-order bundle) and ``state_state_hvp``
  (from a physics satisfying ``PhysicsWithStateStateHVPProtocol``).

Each tier class declares its methods unconditionally, so downstream
capability discovery (the ode stepper stack) sees real methods.
"""

from typing import Generic, Optional, overload

from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.protocols.physics import (
    ParameterizationProtocol,
    PhysicsWithStateStateHVPProtocol,
)
from pyapprox.pde.parameterizations.derivatives import (
    BCFluxParamSensitivityFn,
    InitialParamJacobianFn,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.util.backends.protocols import Array, Backend


class PhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from Physics to ODEResidualProtocol (base tier).

    Wraps a collocation Physics object to provide the raw
    ODEResidualProtocol interface expected by time integrators:

    - Stores time internally via set_time()
    - Translates __call__(state) to physics.residual(state, time)

    Boundary conditions are NOT applied here; for transient problems they
    are applied to the Newton residual by the BC-enforcing time residual
    wrapper.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.

    Examples
    --------
    >>> physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
    >>> physics.set_boundary_conditions([bc_left, bc_right])
    >>> ode_residual = create_physics_ode_residual(physics, bkd)
    >>> ode_residual.set_time(0.0)
    >>> f_y = ode_residual(state)
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"physics must satisfy PhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        self._time = 0.0
        self._mass = create_mass_matrix(physics.mass_matrix(), bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate the ODE residual f(y, t).

        Returns the physics residual WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton residual by the time integrator, not to the physics residual.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Physics residual. Shape: (nstates,)
        """
        return self._physics.residual(state, self._time)

    def jacobian(self, state: Array) -> Array:
        """Compute the state Jacobian df/dy.

        Returns the physics Jacobian WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton Jacobian by the time integrator.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Physics Jacobian. Shape: (nstates, nstates)
        """
        return self._physics.jacobian(state, self._time)

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the mass matrix."""
        return self._mass

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"physics={type(self._physics).__name__})"
        )


class PhysicsToODEResidualWithSetParamAdapter(
    PhysicsToODEResidualAdapter[Array]
):
    """Adapter with a parameterization (evaluation-only tier).

    Adds ``nparams``/``set_param`` on top of the base tier. Selected by
    the factory when the parameterization's bundle declares no derivative
    capability.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        super().__init__(physics, bkd)
        self._parameterization = parameterization
        self._current_params_1d: Optional[Array] = None

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """Return the parameterization."""
        return self._parameterization

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def set_param(self, param: Array) -> None:
        """Set parameter values through the parameterization.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,) — the ODEResidual
            protocol convention (strictly validated; callers convert at
            the optimizer/ode boundary).
        """
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D with shape (nparams,), got shape "
                f"{tuple(param.shape)}"
            )
        self._current_params_1d = param
        self._parameterization.apply(self._physics, param)

    def _require_params(self) -> Array:
        """Return the current parameters or raise if set_param not called."""
        if self._current_params_1d is None:
            raise RuntimeError(
                "set_param() must be called before parameter derivatives"
            )
        return self._current_params_1d


class PhysicsToODEResidualWithParamJacobianAdapter(
    PhysicsToODEResidualWithSetParamAdapter[Array]
):
    """Adapter with first-order parameter derivatives.

    Adds ``param_jacobian``/``initial_param_jacobian``/
    ``bc_flux_param_sensitivity`` on top of the evaluation tier. Selected
    by the factory when the bundle has both parameter jacobians. The
    bundle is narrowed ONCE here into always-present private attributes.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        super().__init__(physics, bkd, parameterization)
        derivs = parameterization.param_derivatives()
        param_jacobian = derivs.param_jacobian
        initial_param_jacobian = derivs.initial_param_jacobian
        if param_jacobian is None or initial_param_jacobian is None:
            raise TypeError(
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has param_jacobian and initial_param_jacobian; "
                "use create_physics_ode_residual to select the right tier"
            )
        self._param_jacobian_fn: ParamJacobianFn[Array] = param_jacobian
        self._initial_param_jacobian_fn: InitialParamJacobianFn[Array] = (
            initial_param_jacobian
        )
        # Optional within this tier: absent capability returns None to
        # the (BC-wrapper) consumer, which skips the correction.
        self._bc_flux_fn: Optional[BCFluxParamSensitivityFn[Array]] = (
            derivs.bc_flux_param_sensitivity
        )

    def param_jacobian(self, state: Array) -> Array:
        """Compute the parameter Jacobian df/dp.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._param_jacobian_fn(
            self._physics, state, self._time, self._require_params()
        )

    def initial_param_jacobian(self) -> Array:
        """Compute d(initial_state)/d(params).

        Returns
        -------
        Array
            Initial-condition Jacobian. Shape: (nstates, nparams)
        """
        return self._initial_param_jacobian_fn(
            self._physics, self._require_params()
        )

    def bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Optional[Array]:
        """Compute d(flux·n)/dp at boundary nodes, or None.

        Returns None when the parameterization does not provide the
        capability (the BC wrapper then skips the correction).
        """
        if self._bc_flux_fn is None or self._current_params_1d is None:
            return None
        return self._bc_flux_fn(
            self._physics,
            state,
            time,
            self._current_params_1d,
            bc_indices,
            normals,
        )


class PhysicsToODEResidualWithHVPAdapter(
    PhysicsToODEResidualWithParamJacobianAdapter[Array]
):
    """Adapter with second-order parameter derivatives.

    Adds the three parameterization HVP contractions (from the bundle)
    and ``state_state_hvp`` (from the physics) on top of the sensitivity
    tier. Selected by the factory when the bundle has all three HVPs and
    the physics satisfies ``PhysicsWithStateStateHVPProtocol``.
    """

    def __init__(
        self,
        physics: PhysicsWithStateStateHVPProtocol[Array],
        bkd: Backend[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(physics, PhysicsWithStateStateHVPProtocol):
            raise TypeError(
                f"{type(self).__name__} requires a physics with "
                f"state_state_hvp, got {type(physics).__name__}"
            )
        super().__init__(physics, bkd, parameterization)
        derivs = parameterization.param_derivatives()
        param_param_hvp = derivs.param_param_hvp
        state_param_hvp = derivs.state_param_hvp
        param_state_hvp = derivs.param_state_hvp
        if (
            param_param_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
        ):
            raise TypeError(
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has all three HVP contractions; use "
                "create_physics_ode_residual to select the right tier"
            )
        self._hvp_physics = physics
        self._param_param_hvp_fn: ParamHVPFn[Array] = param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = param_state_hvp

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy^2) w. Shape: (nstates,)."""
        return self._hvp_physics.state_state_hvp(
            state, adj_state, wvec, self._time
        )

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp^2) v. Shape: (nparams,)."""
        return self._param_param_hvp_fn(
            self._physics,
            state,
            self._time,
            self._require_params(),
            adj_state,
            vvec,
        )

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy dp) v. Shape: (nstates,)."""
        return self._state_param_hvp_fn(
            self._physics,
            state,
            self._time,
            self._require_params(),
            adj_state,
            vvec,
        )

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp dy) w. Shape: (nparams,)."""
        return self._param_state_hvp_fn(
            self._physics,
            state,
            self._time,
            self._require_params(),
            adj_state,
            wvec,
        )


@overload
def create_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: None = None,
) -> PhysicsToODEResidualAdapter[Array]: ...


@overload
def create_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: ParameterizationProtocol[Array],
) -> PhysicsToODEResidualWithSetParamAdapter[Array]: ...


def create_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: Optional[ParameterizationProtocol[Array]] = None,
) -> PhysicsToODEResidualAdapter[Array]:
    """Create the widest adapter tier the inputs support.

    Capability enters the stepper stack exactly here: the factory
    None-checks the parameterization's ParamDerivatives bundle (and
    isinstance-checks the physics for ``state_state_hvp``) once, then
    everything above sees unconditional fixed-tier methods.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol, optional
        Maps parameter vectors to physics coefficients.

    Returns
    -------
    PhysicsToODEResidualAdapter
        The widest tier supported by the bundle and physics.
    """
    if parameterization is None:
        return PhysicsToODEResidualAdapter(physics, bkd)
    if not isinstance(parameterization, ParameterizationProtocol):
        raise TypeError(
            f"parameterization must satisfy ParameterizationProtocol, "
            f"got {type(parameterization).__name__}"
        )
    derivs = parameterization.param_derivatives()
    has_first_order = (
        derivs.param_jacobian is not None
        and derivs.initial_param_jacobian is not None
    )
    has_hvps = (
        derivs.param_param_hvp is not None
        and derivs.state_param_hvp is not None
        and derivs.param_state_hvp is not None
    )
    if (
        has_first_order
        and has_hvps
        and isinstance(physics, PhysicsWithStateStateHVPProtocol)
    ):
        return PhysicsToODEResidualWithHVPAdapter(physics, bkd, parameterization)
    if has_first_order:
        return PhysicsToODEResidualWithParamJacobianAdapter(
            physics, bkd, parameterization
        )
    return PhysicsToODEResidualWithSetParamAdapter(
        physics, bkd, parameterization
    )

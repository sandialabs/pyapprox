"""Advection-Diffusion-Reaction physics for spectral collocation.

Implements the scalar ADR equation:
    du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

where:
- D: diffusion coefficient (scalar or field)
- v: velocity field (vector)
- r: reaction coefficient (scalar or field)
- f: forcing/source term (field)
"""

from typing import List, Optional

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.operators.differential import (
    Divergence,
    Gradient,
    Laplacian,
)
from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdvectionDiffusionReaction(AbstractScalarPhysics[Array]):
    """Advection-Diffusion-Reaction physics.

    Implements the scalar ADR equation:
        du/dt = div(D * grad(u)) - v . grad(u) + r*u + f

    In conservative form:
        du/dt = -div(flux) + r*u + f
    where flux = -D * grad(u) + u * v

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    diffusion : float, NodalCoefficientProtocol or callable, optional
        Diffusion coefficient (see ``as_nodal_coefficient``): a constant,
        a ``NodalField`` (required to parameterize it), a ``NodalSupplier``
        or a bare ``f(time)`` (warned, taken as time-dependent). Absent,
        or the constant 0, means no diffusion term.
    velocity : List[Array], optional
        Velocity field components, fixed in time. Each has shape: (npts,).
        Default: None (no advection)
    reaction : float, NodalCoefficientProtocol or callable, optional
        Reaction coefficient, in the forms ``diffusion`` takes. Absent, or
        the constant 0, means no reaction term.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing term, in the forms ``diffusion`` takes. Absent means none.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion: Optional[CoefficientInput[Array]] = None,
        velocity: Optional[List[Array]] = None,
        reaction: Optional[CoefficientInput[Array]] = None,
        forcing: Optional[CoefficientInput[Array]] = None,
    ):
        super().__init__(basis, bkd)
        self._laplacian = Laplacian(basis, bkd)
        self._gradient = Gradient(basis, bkd)
        self._divergence = Divergence(basis, bkd)

        npts = basis.npts()
        ndim = basis.ndim()

        self._diffusion = self._coefficient(diffusion, npts, bkd)

        # Store velocity field
        self._velocity: Optional[List[Array]]
        if velocity is not None:
            if len(velocity) != ndim:
                raise ValueError(
                    f"velocity must have {ndim} components, got {len(velocity)}"
                )
            self._velocity = velocity
        else:
            self._velocity = None

        self._reaction = self._coefficient(reaction, npts, bkd)
        self._forcing = self._coefficient(forcing, npts, bkd)

        # Precompute derivative matrices for efficiency
        self._D_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]
        self._D2_matrix = self._compute_laplacian_matrix()

    def _compute_laplacian_matrix(self) -> Array:
        """Compute the Laplacian matrix (sum of second derivatives)."""
        bkd = self._bkd
        ndim = self._basis.ndim()
        npts = self._basis.npts()
        D2 = bkd.zeros((npts, npts))
        for dim in range(ndim):
            D2_dim = self._basis.derivative_matrix(2, dim)
            D2 = D2 + D2_dim
        return D2

    @staticmethod
    def _coefficient(
        value: Optional[CoefficientInput[Array]], npts: int, bkd: Backend[Array]
    ) -> Optional[NodalCoefficientProtocol[Array]]:
        """The nodal coefficient for ``value``; None for an absent term
        (no value, or the constant 0)."""
        if value is None or (isinstance(value, (int, float)) and value == 0.0):
            return None
        return as_nodal_coefficient(value, npts, bkd)

    def diffusion_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The diffusion coefficient; None without a diffusion term."""
        return self._diffusion

    def reaction_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The reaction coefficient; None without a reaction term."""
        return self._reaction

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The forcing; None without a forcing term."""
        return self._forcing

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        if self._forcing is None:
            return self._bkd.zeros((self.npts(),))
        return self._forcing.values(time)

    def _get_diffusion(self, time: float) -> Array:
        """Get diffusion array at given time.

        Positivity is a property of THIS operator, not of whatever
        produced the field: the diffusion term is elliptic only while
        D > 0, and where it dips negative the solve returns a plausible
        field rather than failing. Checked here, where the operator
        consumes the values, so it holds for a physics built directly,
        a field mutated after construction, and a parameterized one
        alike.

        Evaluated on every call and never cached, so every consumer
        (residual, Jacobian, flux) reads the coefficient at its own time
        and the current parameters, whatever was evaluated before.

        Raises
        ------
        ValueError
            If the physics has no diffusion term, or the diffusivity is
            not positive everywhere.
        """
        if self._diffusion is None:
            raise ValueError("this physics has no diffusion term")
        diffusion = self._diffusion.values(time)
        smallest = self._bkd.to_float(self._bkd.min(diffusion))
        if smallest <= 0.0:
            raise ValueError(
                "diffusivity must be positive everywhere; found "
                f"{smallest:.3e} at time {time:.6g}. A non-positive "
                "diffusivity makes the operator non-elliptic there, "
                "which yields a plausible-looking solution rather than "
                "a solver failure"
            )
        return diffusion

    def _get_reaction(self, time: float) -> Array:
        """Get reaction array at given time, evaluated, never cached.

        Raises
        ------
        ValueError
            If the physics has no reaction term.
        """
        if self._reaction is None:
            raise ValueError("this physics has no reaction term")
        return self._reaction.values(time)

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T (d^2R/du^2) w = 0 (residual linear in state).

        THIS class's coefficients are state-independent nodal
        coefficients, so its residual is linear in the state and the
        contraction is exactly zero. State-dependent coefficients live
        in separate collocation physics (e.g. shallow_ice,
        shallow_shelf, burgers), which must implement their own
        nonzero contraction.
        """
        return self._bkd.zeros((self.npts(),))

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, t)`, no boundary condition applied. Shape: (npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial u`. Shape: (npts, npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Linear in the state: zero curvature."""
        return StateDerivatives.linear(self._bkd)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the diffusion and reaction are declared time-invariant
        (the velocity is fixed in time)."""
        return not any(
            coefficient is not None and coefficient.is_time_dependent()
            for coefficient in (self._diffusion, self._reaction)
        )

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the Jacobian, the forcing is declared
        time-invariant."""
        return self.interior_jacobian_is_time_invariant() and not (
            self._forcing is not None and self._forcing.is_time_dependent()
        )

    # -- full-matrix field-derivative assemblies (engine slots)

    def residual_diffusion_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S(u) = \\partial R/\\partial D_{\\text{field}}`.

        .. math::

            S(u) = \\mathrm{diag}(D_2 u)
            + \\sum_d \\mathrm{diag}(D_d u) D_d

        Applied to a field direction it gives the residual's diffusion
        sensitivity; its transpose contracts adjoints into field space.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        Array
            Assembly. Shape: (npts, npts)
        """
        bkd = self._bkd
        result = bkd.diag(self._D2_matrix @ state)
        for dim in range(self._basis.ndim()):
            result = result + bkd.diag(
                self._D_matrices[dim] @ state
            ) @ self._D_matrices[dim]
        return result

    def residual_diffusion_state_jacobian(
        self, delta: Array, state: Array
    ) -> Array:
        """Assemble the mixed operator
        :math:`A(\\delta) = \\partial/\\partial u \\,[S(u) \\delta]`.

        .. math::

            A(\\delta) = \\mathrm{diag}(\\delta) D_2
            + \\sum_d \\mathrm{diag}(D_d \\delta) D_d

        The diffusion term is bilinear in :math:`(u, D)`, so
        :math:`A(\\delta) w = S(w) \\delta`; its transpose contracts
        adjoints into state space. The assembly is NOT symmetric
        (collocation differentiation matrices are non-symmetric).

        Parameters
        ----------
        delta : Array
            Diffusion-field direction. Shape: (npts,)
        state : Array
            Solution state (unused: the operator is state-independent;
            kept for the engine's mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (npts, npts)
        """
        bkd = self._bkd
        result = bkd.diag(delta) @ self._D2_matrix
        for dim in range(self._basis.ndim()):
            result = result + bkd.diag(
                self._D_matrices[dim] @ delta
            ) @ self._D_matrices[dim]
        return result

    def residual_reaction_jacobian(self, state: Array) -> Array:
        """Assemble :math:`\\partial R/\\partial r_{\\text{field}}
        = \\mathrm{diag}(u)`.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        Array
            Assembly. Shape: (npts, npts)
        """
        return self._bkd.diag(state)

    def residual_reaction_state_jacobian(
        self, delta: Array, state: Array
    ) -> Array:
        """Assemble :math:`\\partial/\\partial u\\,[\\mathrm{diag}(u)
        \\delta] = \\mathrm{diag}(\\delta)`.

        Parameters
        ----------
        delta : Array
            Reaction-field direction. Shape: (npts,)
        state : Array
            Solution state (unused; kept for the engine's
            mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (npts, npts)
        """
        return self._bkd.diag(delta)

    def residual_forcing_jacobian(self) -> Array:
        """Assemble :math:`\\partial R/\\partial f_{\\text{field}} = I`.

        Returns
        -------
        Array
            Identity. Shape: (npts, npts)
        """
        return self._bkd.eye(self.npts())

    def boundary_flux_diffusion_jacobian(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Assemble :math:`\\partial(\\text{flux} \\cdot n)/\\partial
        D_{\\text{field}}` at boundary points.

        The diffusive normal flux :math:`(\\text{flux} \\cdot n)_i =
        -D_i (\\nabla u \\cdot n)_i` depends on the field only through
        its value at boundary point :math:`i`, so row :math:`i` has the
        single nonzero :math:`-(\\nabla u \\cdot n)_i` at column
        ``bc_indices[i]``. Right-multiplying by a field-map Jacobian
        reproduces the ``bc_flux_param_sensitivity`` row convention.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time (unused; kept for signature uniformity).
        bc_indices : Array
            Boundary point indices. Shape: (n_bc,)
        normals : Array
            Outward unit normals. Shape: (n_bc, ndim)

        Returns
        -------
        Array
            Assembly. Shape: (n_bc, npts)
        """
        bkd = self._bkd
        nbnd = bc_indices.shape[0]
        grad_u_dot_n = bkd.zeros((nbnd,))
        for dim in range(self._basis.ndim()):
            grad_u_dim = self._D_matrices[dim] @ state
            grad_u_dot_n = (
                grad_u_dot_n + grad_u_dim[bc_indices] * normals[:, dim]
            )
        result = bkd.copy(bkd.zeros((nbnd, self.npts())))
        for i in range(nbnd):
            result[i, bkd.to_int(bc_indices[i])] = -grad_u_dot_n[i]
        return result

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (npts,)
        """
        bkd = self._bkd
        npts = self.npts()

        # Initialize residual
        residual = bkd.zeros((npts,))

        # Diffusion term: div(D * grad(u)) = D * laplacian(u) + grad(D) . grad(u),
        # in this general form for every D: a coefficient a parameterization
        # rewrites cannot be assumed constant.
        if self._diffusion is not None:
            diff_array = self._get_diffusion(time)
            residual = residual + diff_array * (self._D2_matrix @ state)
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                residual = residual + (D_dim @ diff_array) * (D_dim @ state)

        # Advection term: -v . grad(u)
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                grad_u_dim = D_dim @ state
                residual = residual - self._velocity[dim] * grad_u_dim

        # Reaction term: r * u
        if self._reaction is not None:
            react_array = self._get_reaction(time)
            residual = residual + react_array * state

        # Forcing term
        residual = residual + self._get_forcing(time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (npts, npts)
        """
        bkd = self._bkd
        npts = self.npts()

        # Initialize Jacobian
        jacobian = bkd.zeros((npts, npts))

        # Diffusion term Jacobian: diag(D) @ laplacian + sum_i diag(grad_D_i) @ D_i,
        # formed by row scaling rather than products with diagonal matrices
        if self._diffusion is not None:
            diff_array = self._get_diffusion(time)
            jacobian = jacobian + diff_array[:, None] * self._D2_matrix
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                jacobian = jacobian + (D_dim @ diff_array)[:, None] * D_dim

        # Advection term Jacobian: d/du[-v . grad(u)] = -sum_i diag(v_i) @ D_i
        if self._velocity is not None:
            for dim in range(self._basis.ndim()):
                D_dim = self._D_matrices[dim]
                jacobian = jacobian - self._velocity[dim][:, None] * D_dim

        # Reaction term Jacobian: d/du[r * u] = diag(r)
        if self._reaction is not None:
            react_array = self._get_reaction(time)
            jacobian = jacobian + bkd.diag(react_array)

        return jacobian

    def compute_flux(self, state: Array, time: float) -> List[Array]:
        """Compute total conservative flux at all mesh points.

        Returns the sum of diffusive and advective flux:
          flux_d = -D * du/dx_d + v_d * u  (for each dimension d)

        For pure diffusion (no velocity): flux_d = -D * du/dx_d

        The diffusivity is evaluated at ``time`` and the current
        parameters, independently of any earlier residual evaluation.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Time at which the coefficients are evaluated.

        Returns
        -------
        List[Array]
            Flux components, one per dimension. Each shape: (npts,)
        """
        bkd = self._bkd
        diffusion = (
            self._get_diffusion(time)
            if self._diffusion is not None
            else bkd.zeros((self.npts(),))
        )
        flux = []
        for dim in range(self._basis.ndim()):
            # Diffusive flux: -D * du/dx_d
            flux_dim = -diffusion * (self._D_matrices[dim] @ state)
            # Advective flux: v_d * u
            if self._velocity is not None:
                flux_dim = flux_dim + self._velocity[dim] * state
            flux.append(flux_dim)
        return flux

    def compute_flux_jacobian(self, state: Array, time: float) -> List[Array]:
        """Compute Jacobian of total conservative flux w.r.t. state.

        d(flux_d)/du = -diag(D) @ D_d + diag(v_d), with D evaluated at
        ``time`` and the current parameters.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        time : float
            Time at which the coefficients are evaluated.

        Returns
        -------
        List[Array]
            Jacobian of each flux component. Each shape: (npts, npts)
        """
        bkd = self._bkd
        diffusion = (
            self._get_diffusion(time)
            if self._diffusion is not None
            else bkd.zeros((self.npts(),))
        )
        flux_jacs = []
        for dim in range(self._basis.ndim()):
            jac_dim = -diffusion[:, None] * self._D_matrices[dim]
            if self._velocity is not None:
                jac_dim = jac_dim + bkd.diag(self._velocity[dim])
            flux_jacs.append(jac_dim)
        return flux_jacs

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute diffusive flux at boundary for DtN domain decomposition.

        Computes D * grad(u) · n at the specified boundary points.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Flux at boundary points. Shape: (nboundary,)

        Raises
        ------
        ValueError
            If the diffusivity is declared time-dependent: this flux takes
            no time at which to evaluate it.
        """
        if self._diffusion is None:
            return self._bkd.zeros((boundary_indices.shape[0],))
        if self._diffusion.is_time_dependent():
            raise ValueError(
                "compute_interface_flux takes no time, so it supports only a "
                "diffusivity declared time-independent"
            )
        ndim = self._basis.ndim()
        nboundary = boundary_indices.shape[0]

        # Compute grad(u) · n at boundary points
        flux = self._bkd.zeros((nboundary,))
        for dim in range(ndim):
            D_dim = self._D_matrices[dim]
            grad_u_dim = D_dim @ state
            flux = flux + grad_u_dim[boundary_indices] * self._bkd.to_float(normal[dim])

        # Scale by the diffusion coefficient at the boundary points; it is
        # time-independent, so any time reads it
        return self._get_diffusion(0.0)[boundary_indices] * flux


def create_steady_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion: float,
    forcing: Optional[CoefficientInput[Array]] = None,
) -> AdvectionDiffusionReaction[Array]:
    """Create steady diffusion physics (Poisson equation).

    Solves: -D * laplacian(u) = f  or  D * laplacian(u) + f = 0

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion : float
        Diffusion coefficient (positive).
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source term, in the forms ``AdvectionDiffusionReaction`` takes.

    Returns
    -------
    AdvectionDiffusionReaction
        Physics for steady diffusion.
    """
    return AdvectionDiffusionReaction(basis, bkd, diffusion=diffusion, forcing=forcing)


def create_advection_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion: float,
    velocity: List[Array],
    forcing: Optional[CoefficientInput[Array]] = None,
) -> AdvectionDiffusionReaction[Array]:
    """Create advection-diffusion physics.

    Solves: du/dt = D * laplacian(u) - v . grad(u) + f

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion : float
        Diffusion coefficient.
    velocity : List[Array]
        Velocity field components.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source term, in the forms ``AdvectionDiffusionReaction`` takes.

    Returns
    -------
    AdvectionDiffusionReaction
        Advection-diffusion physics.
    """
    return AdvectionDiffusionReaction(
        basis, bkd, diffusion=diffusion, velocity=velocity, forcing=forcing
    )

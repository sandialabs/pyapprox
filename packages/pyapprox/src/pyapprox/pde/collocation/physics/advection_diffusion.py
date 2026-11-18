"""Advection-Diffusion-Reaction physics for spectral collocation.

Implements the scalar ADR equation:
    du/dt = div(D * grad(u)) - A(v, u) + r*u + f

where:
- D: diffusion coefficient (scalar or field)
- v: velocity field (vector)
- A: the advection term of an injected form, v . grad(u) (non-conservative)
  or div(v u) (conservative)
- r: reaction coefficient (scalar or field)
- f: forcing/source term (field)
"""

from typing import Generic, List, Optional

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.operators.differential import (
    Divergence,
    Gradient,
    Laplacian,
)
from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CheckedValues,
    CoefficientInput,
    NodalCoefficientProtocol,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    NonConservativeAdvection,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdvectionDiffusionReaction(AbstractScalarPhysics[Array]):
    """Advection-Diffusion-Reaction physics.

    Implements the scalar ADR equation:
        du/dt = div(D * grad(u)) - A(v, u) + r*u + f

    with the advection term ``A`` of ``advection_form``: ``v . grad(u)``
    (non-conservative, the default) or ``div(v u)`` (conservative). The
    flux (:meth:`compute_flux`) is the one the form's divergence terms
    carry, ``-D * grad(u)`` plus the form's advective flux (``v u`` for
    the conservative form, none for the non-conservative), so
    ``du/dt = -div(flux) + ... `` holds for either form. A boundary
    condition that prescribes another flux, such as the total flux
    ``-D * grad(u) + v u`` on a non-conservative equation, injects it as
    an :class:`AdvectionDiffusionFlux`.

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
    advection_form : AdvectionFormProtocol, optional
        How the advection term is written. Default:
        ``NonConservativeAdvection()``.

    Raises
    ------
    TypeError
        If ``advection_form`` does not satisfy ``AdvectionFormProtocol``.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion: Optional[CoefficientInput[Array]] = None,
        velocity: Optional[List[Array]] = None,
        reaction: Optional[CoefficientInput[Array]] = None,
        forcing: Optional[CoefficientInput[Array]] = None,
        advection_form: Optional[AdvectionFormProtocol] = None,
    ):
        super().__init__(basis, bkd)
        if advection_form is None:
            advection_form = NonConservativeAdvection()
        if not isinstance(advection_form, AdvectionFormProtocol):
            raise TypeError(
                "advection_form must satisfy AdvectionFormProtocol, got "
                f"{type(advection_form).__name__}"
            )
        self._advection_form = advection_form
        self._laplacian = Laplacian(basis, bkd)
        self._gradient = Gradient(basis, bkd)
        self._divergence = Divergence(basis, bkd)

        npts = basis.npts()
        ndim = basis.ndim()

        self._diffusion = optional_nodal_coefficient(diffusion, npts, bkd)
        self._checked_diffusion: Optional[CheckedValues[Array]] = (
            None
            if self._diffusion is None
            else CheckedValues(self._diffusion, self._check_diffusion)
        )

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

        self._reaction = optional_nodal_coefficient(reaction, npts, bkd)
        self._forcing = optional_nodal_coefficient(forcing, npts, bkd)

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

    def diffusion_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The diffusion coefficient; None without a diffusion term."""
        return self._diffusion

    def velocity(self) -> Optional[List[Array]]:
        """The velocity components, fixed in time; None without advection.
        Each shape: (npts,)"""
        return self._velocity

    def advection_form(self) -> AdvectionFormProtocol:
        """How the advection term is written."""
        return self._advection_form

    def _derivative(self, values: Array, dim: int) -> Array:
        """The derivative along ``dim``, as the advection form applies it:
        to a nodal vector, or to each column of a matrix."""
        return self._D_matrices[dim] @ values

    def _advection(self, state: Array) -> Array:
        """The form's advection term. Shape: (npts,)"""
        if self._velocity is None:
            return self._bkd.zeros((self.npts(),))
        return self._advection_form.strong_term(
            self._velocity, state, self._derivative
        )

    def _advection_jacobian(self) -> Array:
        """The advection term's state Jacobian: the form applied to the
        identity, with each velocity component scaling rows.
        Shape: (npts, npts)"""
        bkd = self._bkd
        if self._velocity is None:
            return bkd.zeros((self.npts(), self.npts()))
        return self._advection_form.strong_term(
            [component[:, None] for component in self._velocity],
            bkd.eye(self.npts()),
            self._derivative,
        )

    def diffusive_flux(self, state: Array, time: float) -> List[Array]:
        """``-D * du/dx_d`` per dimension, D at ``time`` and the current
        parameters; zero without diffusion. Each shape: (npts,)"""
        bkd = self._bkd
        diffusion = (
            self._get_diffusion(time)
            if self._diffusion is not None
            else bkd.zeros((self.npts(),))
        )
        return [-diffusion * (D_dim @ state) for D_dim in self._D_matrices]

    def diffusive_flux_jacobian(self, time: float) -> List[Array]:
        """``-diag(D) @ D_d`` per dimension, the state Jacobian of
        :meth:`diffusive_flux`. Each shape: (npts, npts)"""
        bkd = self._bkd
        diffusion = (
            self._get_diffusion(time)
            if self._diffusion is not None
            else bkd.zeros((self.npts(),))
        )
        return [-diffusion[:, None] * D_dim for D_dim in self._D_matrices]

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
        alike; ``CheckedValues`` repeats the check only when the values
        may have changed.

        Evaluated on every call and never cached, so every consumer
        (residual, Jacobian, flux) reads the coefficient at its own time
        and the current parameters, whatever was evaluated before.

        Raises
        ------
        ValueError
            If the physics has no diffusion term, or the diffusivity is
            not positive everywhere.
        """
        if self._checked_diffusion is None:
            raise ValueError("this physics has no diffusion term")
        return self._checked_diffusion.values(time)

    def _check_diffusion(self, diffusion: Array, time: float) -> None:
        """Raise unless the diffusivity is positive everywhere."""
        smallest = self._bkd.to_float(self._bkd.min(diffusion))
        if smallest <= 0.0:
            raise ValueError(
                "diffusivity must be positive everywhere; found "
                f"{smallest:.3e} at time {time:.6g}. A non-positive "
                "diffusivity makes the operator non-elliptic there, "
                "which yields a plausible-looking solution rather than "
                "a solver failure"
            )

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

    def flux_is_time_dependent(self) -> bool:
        """Whether the diffusion is declared time-dependent: the flux reads
        it and the velocity, which is fixed in time."""
        return not declared_time_invariant(self._diffusion)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the diffusion and reaction are declared time-invariant
        (the velocity is fixed in time)."""
        return declared_time_invariant(self._diffusion, self._reaction)

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the Jacobian, the forcing is declared
        time-invariant."""
        return declared_time_invariant(
            self._diffusion, self._reaction, self._forcing
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

        # Advection term: -A(v, u), v . grad(u) or div(v u) by the form
        if self._velocity is not None:
            residual = residual - self._advection(state)

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

        # Advection term Jacobian: -sum_i diag(v_i) @ D_i (non-conservative)
        # or -sum_i D_i @ diag(v_i) (conservative)
        if self._velocity is not None:
            jacobian = jacobian - self._advection_jacobian()

        # Reaction term Jacobian: d/du[r * u] = diag(r)
        if self._reaction is not None:
            react_array = self._get_reaction(time)
            jacobian = jacobian + bkd.diag(react_array)

        return jacobian

    def compute_flux(self, state: Array, time: float) -> List[Array]:
        """Compute the flux of the equation's divergence terms at all mesh
        points.

        The diffusive flux plus the advection form's advective flux:
          flux_d = -D * du/dx_d + v_d * u  (conservative form)
          flux_d = -D * du/dx_d            (non-conservative form, or no
                                            velocity)

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
        return AdvectionDiffusionFlux(self, self._advection_form).compute_flux(
            state, time
        )

    def compute_flux_jacobian(self, state: Array, time: float) -> List[Array]:
        """Compute Jacobian of :meth:`compute_flux` w.r.t. state.

        d(flux_d)/du = -diag(D) @ D_d, plus diag(v_d) for the conservative
        form, with D evaluated at ``time`` and the current parameters.

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
        return AdvectionDiffusionFlux(
            self, self._advection_form
        ).compute_flux_jacobian(state, time)

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


class AdvectionDiffusionFlux(Generic[Array]):
    r"""The flux of an advection-diffusion physics under a chosen advection
    form, as a boundary condition prescribes it.

    ``flux_d = -D du/dx_d`` plus the form's advective flux: ``v_d u`` for
    :class:`~pyapprox.pde.constitutive.advection.ConservativeAdvection`,
    none for the non-conservative form. With the physics' own form it is
    the physics' flux; with another it states a different condition on
    the same equation, such as the total flux :math:`-D\nabla u\cdot n +
    (v\cdot n) u` (a Danckwerts inflow) on a non-conservative equation.
    Reads the physics' current diffusivity and velocity at each call.

    Satisfies ``FluxProviderProtocol``.

    Parameters
    ----------
    physics : AdvectionDiffusionReaction
        Supplies the diffusivity and velocity.
    advection_form : AdvectionFormProtocol
        Whose advective flux the boundary prescribes.

    Raises
    ------
    TypeError
        If ``advection_form`` does not satisfy ``AdvectionFormProtocol``.
    """

    def __init__(
        self,
        physics: AdvectionDiffusionReaction[Array],
        advection_form: AdvectionFormProtocol,
    ) -> None:
        if not isinstance(advection_form, AdvectionFormProtocol):
            raise TypeError(
                "advection_form must satisfy AdvectionFormProtocol, got "
                f"{type(advection_form).__name__}"
            )
        self._physics = physics
        self._advection_form = advection_form

    def compute_flux(self, state: Array, time: float) -> List[Array]:
        """The flux components at all mesh points. Each shape: (npts,)"""
        flux = self._physics.diffusive_flux(state, time)
        velocity = self._physics.velocity()
        if velocity is None:
            return flux
        advective = self._advection_form.flux(velocity, state)
        if advective is None:
            return flux
        return [q + a for q, a in zip(flux, advective)]

    def compute_flux_jacobian(self, state: Array, time: float) -> List[Array]:
        """The state Jacobian of each flux component.
        Each shape: (npts, npts)"""
        jacobian = self._physics.diffusive_flux_jacobian(time)
        velocity = self._physics.velocity()
        if velocity is None:
            return jacobian
        bkd = self._physics.bkd()
        advective = self._advection_form.flux(
            [component[:, None] for component in velocity],
            bkd.eye(self._physics.npts()),
        )
        if advective is None:
            return jacobian
        return [q + a for q, a in zip(jacobian, advective)]

    def flux_is_time_dependent(self) -> bool:
        """Whether the diffusivity is declared time-dependent (the
        velocity is fixed in time)."""
        return self._physics.flux_is_time_dependent()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._advection_form!r})"


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

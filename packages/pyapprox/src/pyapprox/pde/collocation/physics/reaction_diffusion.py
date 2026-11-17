"""Two-species reaction-diffusion physics for spectral collocation.

Implements the coupled two-species reaction-diffusion system:
    du0/dt = D0 * Laplacian(u0) + R0(u0, u1) + f0
    du1/dt = D1 * Laplacian(u1) + R1(u0, u1) + f1

where:
    u0, u1 = species concentrations (solution variables)
    D0, D1 = diffusion coefficients (can be spatially varying)
    R0, R1 = reaction terms (functions of both species)
    f0, f1 = forcing/source terms
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    # sympy appears only in the reactions' symbolic signatures; importing
    # it here would load it for every user of the physics package
    import sympy as sp


@runtime_checkable
class ReactionProtocol(Protocol, Generic[Array]):
    """Protocol for reaction terms R(u0, u1).

    Reaction terms must return both the reaction values and
    their Jacobian with respect to the species concentrations.
    """

    def __call__(self, u0: Array, u1: Array) -> Tuple[Array, Array]:
        """Evaluate reaction terms.

        Parameters
        ----------
        u0 : Array
            First species concentration. Shape: (npts,)
        u1 : Array
            Second species concentration. Shape: (npts,)

        Returns
        -------
        Tuple[Array, Array]
            (R0, R1) reaction terms for each species. Each shape: (npts,)
        """
        ...

    def jacobian(self, u0: Array, u1: Array) -> Tuple[Array, Array, Array, Array]:
        """Compute Jacobian of reaction terms.

        Parameters
        ----------
        u0 : Array
            First species concentration. Shape: (npts,)
        u1 : Array
            Second species concentration. Shape: (npts,)

        Returns
        -------
        Tuple[Array, Array, Array, Array]
            (dR0/du0, dR0/du1, dR1/du0, dR1/du1). Each shape: (npts,)
        """
        ...


@runtime_checkable
class SymbolicReactionProtocol(ReactionProtocol[Array], Protocol, Generic[Array]):
    """Protocol for reactions with symbolic support.

    Reactions implementing this protocol can be used with manufactured
    solutions for automatic forcing computation. The same reaction object
    can be passed to both physics and manufactured solution classes.

    Subclasses must implement:
    - __call__: Numeric evaluation (from ReactionProtocol)
    - jacobian: Numeric Jacobian (from ReactionProtocol)
    - sympy_expressions: Symbolic evaluation for manufactured solutions
    """

    def sympy_expressions(
        self, u0_expr: sp.Expr, u1_expr: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Return symbolic reaction expressions.

        Parameters
        ----------
        u0_expr : sp.Expr
            Sympy expression for species 0 (the manufactured solution).
        u1_expr : sp.Expr
            Sympy expression for species 1 (the manufactured solution).

        Returns
        -------
        Tuple[sp.Expr, sp.Expr]
            (R0_expr, R1_expr) symbolic reaction expressions with
            u0_expr and u1_expr substituted.
        """
        ...


class LinearReaction(Generic[Array]):
    """Linear reaction: R0 = a00*u0 + a01*u1, R1 = a10*u0 + a11*u1.

    Implements SymbolicReactionProtocol for use with manufactured solutions.
    """

    def __init__(
        self,
        a00: Union[float, Array],
        a01: Union[float, Array],
        a10: Union[float, Array],
        a11: Union[float, Array],
        bkd: Backend[Array],
    ):
        self._a00 = a00
        self._a01 = a01
        self._a10 = a10
        self._a11 = a11
        self._bkd = bkd

    def __call__(self, u0: Array, u1: Array) -> Tuple[Array, Array]:
        R0 = self._a00 * u0 + self._a01 * u1
        R1 = self._a10 * u0 + self._a11 * u1
        return R0, R1

    def jacobian(self, u0: Array, u1: Array) -> Tuple[Array, Array, Array, Array]:
        bkd = self._bkd
        npts = u0.shape[0]

        # For linear reaction, Jacobians are constant
        if isinstance(self._a00, (int, float)):
            dR0_du0 = bkd.full((npts,), float(self._a00))
        else:
            dR0_du0 = self._a00

        if isinstance(self._a01, (int, float)):
            dR0_du1 = bkd.full((npts,), float(self._a01))
        else:
            dR0_du1 = self._a01

        if isinstance(self._a10, (int, float)):
            dR1_du0 = bkd.full((npts,), float(self._a10))
        else:
            dR1_du0 = self._a10

        if isinstance(self._a11, (int, float)):
            dR1_du1 = bkd.full((npts,), float(self._a11))
        else:
            dR1_du1 = self._a11

        return dR0_du0, dR0_du1, dR1_du0, dR1_du1

    def sympy_expressions(
        self, u0_expr: sp.Expr, u1_expr: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Return symbolic linear reaction expressions.

        For linear reaction: R0 = a00*u0 + a01*u1, R1 = a10*u0 + a11*u1
        with u0, u1 substituted with the given expressions.

        Note: Spatially-varying coefficients (arrays) are not supported
        for symbolic expressions. Only scalar coefficients work.
        """
        # Validate that coefficients are scalar for symbolic use
        for name, coef in [
            ("a00", self._a00),
            ("a01", self._a01),
            ("a10", self._a10),
            ("a11", self._a11),
        ]:
            if not isinstance(coef, (int, float)):
                raise ValueError(
                    f"Spatially-varying coefficient {name} not supported "
                    "for symbolic expressions. Use scalar coefficients."
                )

        R0_expr = self._a00 * u0_expr + self._a01 * u1_expr
        R1_expr = self._a10 * u0_expr + self._a11 * u1_expr
        return R0_expr, R1_expr


class FitzHughNagumoReaction(Generic[Array]):
    """FitzHugh-Nagumo reaction kinetics.

    R0 = u0 * (1 - u0) * (u0 - alpha) - u1
    R1 = eps * (beta * u0 - gamma * u1)

    Parameters
    ----------
    alpha : float
        Threshold parameter (typically 0.1).
    eps : float
        Time scale separation (typically 0.01).
    beta : float
        Coupling coefficient (typically 0.5).
    gamma : float
        Recovery rate (typically 1.0).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        alpha: float,
        eps: float,
        beta: float,
        gamma: float,
        bkd: Backend[Array],
    ):
        self._alpha = alpha
        self._eps = eps
        self._beta = beta
        self._gamma = gamma
        self._bkd = bkd

    def __call__(self, u0: Array, u1: Array) -> Tuple[Array, Array]:
        alpha = self._alpha
        eps = self._eps
        beta = self._beta
        gamma = self._gamma

        # Voltage reaction: u0*(1-u0)*(u0-alpha) - u1
        R0 = u0 * (1.0 - u0) * (u0 - alpha) - u1

        # Recovery reaction: eps * (beta*u0 - gamma*u1)
        R1 = eps * (beta * u0 - gamma * u1)

        return R0, R1

    def jacobian(self, u0: Array, u1: Array) -> Tuple[Array, Array, Array, Array]:
        alpha = self._alpha
        eps = self._eps
        beta = self._beta
        gamma = self._gamma
        bkd = self._bkd
        npts = u0.shape[0]

        # dR0/du0 = d/du0[u0*(1-u0)*(u0-alpha)]
        # = (1-u0)*(u0-alpha) + u0*(-(u0-alpha)) + u0*(1-u0)
        # = (1-u0)*(u0-alpha) - u0*(u0-alpha) + u0*(1-u0)
        # = (1-2*u0)*(u0-alpha) + u0*(1-u0)
        # = (u0 - alpha - 2*u0^2 + 2*alpha*u0) + u0 - u0^2
        # = -3*u0^2 + 2*(1+alpha)*u0 - alpha
        dR0_du0 = -3.0 * u0**2 + 2.0 * (1.0 + alpha) * u0 - alpha

        # dR0/du1 = -1
        dR0_du1 = bkd.full((npts,), -1.0)

        # dR1/du0 = eps * beta
        dR1_du0 = bkd.full((npts,), eps * beta)

        # dR1/du1 = -eps * gamma
        dR1_du1 = bkd.full((npts,), -eps * gamma)

        return dR0_du0, dR0_du1, dR1_du0, dR1_du1

    def sympy_expressions(
        self, u0_expr: sp.Expr, u1_expr: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Return symbolic FitzHugh-Nagumo reaction expressions.

        R0 = u0 * (1 - u0) * (u0 - alpha) - u1
        R1 = eps * (beta * u0 - gamma * u1)
        """
        R0_expr = u0_expr * (1 - u0_expr) * (u0_expr - self._alpha) - u1_expr
        R1_expr = self._eps * (self._beta * u0_expr - self._gamma * u1_expr)
        return R0_expr, R1_expr


class TwoSpeciesReactionDiffusionPhysics(AbstractVectorPhysics[Array]):
    """Two-species reaction-diffusion physics.

    Implements the coupled system:
        du0/dt = D0 * Laplacian(u0) + R0(u0, u1) + f0
        du1/dt = D1 * Laplacian(u1) + R1(u0, u1) + f1

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    diffusion0, diffusion1 : float, NodalCoefficientProtocol or callable
        Diffusion coefficient of each species, in the forms
        ``as_nodal_coefficient`` takes; the constant 0 means the species
        does not diffuse.
    reaction : ReactionProtocol
        Reaction term object providing R(u0, u1) and its Jacobian.
    forcing0, forcing1 : float, NodalCoefficientProtocol or callable, optional
        Forcing of each species, in the same forms. Absent means none.

    Raises
    ------
    TypeError
        If ``reaction`` does not satisfy ``ReactionProtocol``.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(30, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> reaction = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)
    >>> physics = TwoSpeciesReactionDiffusionPhysics(
    ...     basis, bkd, diffusion0=1e-3, diffusion1=0.0, reaction=reaction
    ... )
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion0: CoefficientInput[Array],
        diffusion1: CoefficientInput[Array],
        reaction: ReactionProtocol[Array],
        forcing0: Optional[CoefficientInput[Array]] = None,
        forcing1: Optional[CoefficientInput[Array]] = None,
    ):
        super().__init__(basis, bkd, ncomponents=2)
        if not isinstance(reaction, ReactionProtocol):
            raise TypeError(
                "reaction must satisfy ReactionProtocol, got "
                f"{type(reaction).__name__}"
            )

        npts = basis.npts()
        ndim = basis.ndim()
        self._diffusions = (
            optional_nodal_coefficient(diffusion0, npts, bkd),
            optional_nodal_coefficient(diffusion1, npts, bkd),
        )
        self._forcings = (
            optional_nodal_coefficient(forcing0, npts, bkd),
            optional_nodal_coefficient(forcing1, npts, bkd),
        )
        self._reaction = reaction

        # Precompute derivative matrices
        self._D1_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]
        self._laplacian = basis.derivative_matrix(2, 0)
        for dim in range(1, ndim):
            self._laplacian = self._laplacian + basis.derivative_matrix(2, dim)

    def diffusion_functions(
        self,
    ) -> Tuple[
        Optional[NodalCoefficientProtocol[Array]],
        Optional[NodalCoefficientProtocol[Array]],
    ]:
        """The diffusion coefficient of each species; None where the
        species does not diffuse."""
        return self._diffusions

    def forcing_functions(
        self,
    ) -> Tuple[
        Optional[NodalCoefficientProtocol[Array]],
        Optional[NodalCoefficientProtocol[Array]],
    ]:
        """The forcing of each species; None where it has none."""
        return self._forcings

    def reaction(self) -> ReactionProtocol[Array]:
        """The reaction term."""
        return self._reaction

    def _values(
        self, coefficient: Optional[NodalCoefficientProtocol[Array]], time: float
    ) -> Array:
        """A coefficient's values; zeros for an absent term."""
        if coefficient is None:
            return self._bkd.zeros((self.npts(),))
        return coefficient.values(time)

    def _split_state(self, state: Array) -> Tuple[Array, Array]:
        """Split combined state into species components."""
        npts = self.npts()
        return state[:npts], state[npts:]

    def _diffusion_term(
        self, coefficient: Optional[NodalCoefficientProtocol[Array]], u: Array,
        time: float,
    ) -> Array:
        """div(D grad u) = D lap(u) + grad(D) . grad(u); zero without D."""
        if coefficient is None:
            return self._bkd.zeros((self.npts(),))
        diffusion = coefficient.values(time)
        term = diffusion * (self._laplacian @ u)
        for D1 in self._D1_matrices:
            term = term + (D1 @ diffusion) * (D1 @ u)
        return term

    def _diffusion_jacobian(
        self, coefficient: Optional[NodalCoefficientProtocol[Array]], time: float
    ) -> Array:
        """diag(D) lap + sum_i diag(D_i D) D_i, by row scaling."""
        npts = self.npts()
        if coefficient is None:
            return self._bkd.zeros((npts, npts))
        diffusion = coefficient.values(time)
        jacobian = diffusion[:, None] * self._laplacian
        for D1 in self._D1_matrices:
            jacobian = jacobian + (D1 @ diffusion)[:, None] * D1
        return jacobian

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Combined solution state [u0, u1]. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual [res0, res1]. Shape: (2*npts,)
        """
        u0, u1 = self._split_state(state)
        reactions = self._reaction(u0, u1)
        species: List[Array] = []
        for k, u in enumerate((u0, u1)):
            species.append(
                self._diffusion_term(self._diffusions[k], u, time)
                + reactions[k]
                + self._values(self._forcings[k], time)
            )
        return self._bkd.hstack(species)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Combined solution state [u0, u1]. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix [[J00, J01], [J10, J11]], Jij = d(res_i)/d(u_j).
            Shape: (2*npts, 2*npts)
        """
        bkd = self._bkd
        u0, u1 = self._split_state(state)
        dR0_du0, dR0_du1, dR1_du0, dR1_du1 = self._reaction.jacobian(u0, u1)
        J00 = self._diffusion_jacobian(self._diffusions[0], time) + bkd.diag(dR0_du0)
        J11 = self._diffusion_jacobian(self._diffusions[1], time) + bkd.diag(dR1_du1)
        return bkd.vstack(
            [
                bkd.hstack([J00, bkd.diag(dR0_du1)]),
                bkd.hstack([bkd.diag(dR1_du0), J11]),
            ]
        )

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, t)`, no boundary condition applied.
        Shape: (2*npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial u`. Shape: (2*npts, 2*npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """No second derivatives: ``ReactionProtocol`` supplies none, and
        the reaction may be nonlinear."""
        return StateDerivatives.none()

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether both diffusivities are declared time-invariant (the
        reaction does not depend on time)."""
        return declared_time_invariant(*self._diffusions)

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the diffusivities, both forcings are declared
        time-invariant."""
        return declared_time_invariant(*self._diffusions, *self._forcings)

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute diffusive flux at boundary for DtN domain decomposition.

        Computes D_i * grad(u_i) · n for each species at the specified
        boundary points.

        Parameters
        ----------
        state : Array
            Solution state [u0, u1]. Shape: (2*npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Flux [flux0, flux1] at boundary points.
            Shape: (2*nboundary,) with component-stacked ordering.

        Raises
        ------
        ValueError
            If a diffusivity is declared time-dependent: this flux takes no
            time at which to evaluate it.
        """
        bkd = self._bkd
        nboundary = boundary_indices.shape[0]
        fluxes: List[Array] = []
        for coefficient, u in zip(self._diffusions, self._split_state(state)):
            if coefficient is None:
                fluxes.append(bkd.zeros((nboundary,)))
                continue
            if coefficient.is_time_dependent():
                raise ValueError(
                    "compute_interface_flux takes no time, so it supports "
                    "only diffusivities declared time-independent"
                )
            flux = bkd.zeros((nboundary,))
            for dim, D1 in enumerate(self._D1_matrices):
                flux = flux + (D1 @ u)[boundary_indices] * bkd.to_float(normal[dim])
            # time-independent, so any time reads it
            fluxes.append(coefficient.values(0.0)[boundary_indices] * flux)
        # Component-stacked ordering: [flux0_0, ..., flux0_n, flux1_0, ...]
        return bkd.concatenate(fluxes)


def create_two_species_reaction_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion0: CoefficientInput[Array],
    diffusion1: CoefficientInput[Array],
    reaction: ReactionProtocol[Array],
    forcing0: Optional[CoefficientInput[Array]] = None,
    forcing1: Optional[CoefficientInput[Array]] = None,
) -> TwoSpeciesReactionDiffusionPhysics[Array]:
    """Create two-species reaction-diffusion physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion0, diffusion1 : float, NodalCoefficientProtocol or callable
        Diffusion coefficient of each species.
    reaction : ReactionProtocol
        Reaction term object.
    forcing0, forcing1 : float, NodalCoefficientProtocol or callable, optional
        Forcing of each species.

    Returns
    -------
    TwoSpeciesReactionDiffusionPhysics
        Reaction-diffusion physics.
    """
    return TwoSpeciesReactionDiffusionPhysics(
        basis=basis,
        bkd=bkd,
        diffusion0=diffusion0,
        diffusion1=diffusion1,
        reaction=reaction,
        forcing0=forcing0,
        forcing1=forcing1,
    )

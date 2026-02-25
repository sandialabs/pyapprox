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

from typing import Generic, Optional, Callable, Union, Tuple

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics


class ReactionProtocol(Generic[Array]):
    """Protocol for reaction terms R(u0, u1).

    Reaction terms must return both the reaction values and
    their Jacobian with respect to the species concentrations.
    """

    def __call__(
        self, u0: Array, u1: Array
    ) -> Tuple[Array, Array]:
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

    def jacobian(
        self, u0: Array, u1: Array
    ) -> Tuple[Array, Array, Array, Array]:
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


class SymbolicReactionProtocol(ReactionProtocol[Array], Generic[Array]):
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
        raise NotImplementedError


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

    def jacobian(
        self, u0: Array, u1: Array
    ) -> Tuple[Array, Array, Array, Array]:
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
        for name, coef in [("a00", self._a00), ("a01", self._a01),
                           ("a10", self._a10), ("a11", self._a11)]:
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

    def jacobian(
        self, u0: Array, u1: Array
    ) -> Tuple[Array, Array, Array, Array]:
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
        dR0_du0 = -3.0 * u0 ** 2 + 2.0 * (1.0 + alpha) * u0 - alpha

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
    diffusion0 : float or Array
        Diffusion coefficient for species 0.
    diffusion1 : float or Array
        Diffusion coefficient for species 1.
    reaction : ReactionProtocol
        Reaction term object providing R(u0, u1) and its Jacobian.
    forcing0 : Callable[[float], Array], optional
        Forcing term for species 0.
    forcing1 : Callable[[float], Array], optional
        Forcing term for species 1.

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
        diffusion0: Union[float, Array],
        diffusion1: Union[float, Array],
        reaction: ReactionProtocol[Array],
        forcing0: Optional[Callable[[float], Array]] = None,
        forcing1: Optional[Callable[[float], Array]] = None,
    ):
        super().__init__(basis, bkd, ncomponents=2)

        npts = basis.npts()
        ndim = basis.ndim()

        # Store diffusion coefficients
        if isinstance(diffusion0, (int, float)):
            self._diffusion0_value = float(diffusion0)
            self._diffusion0_array = bkd.full((npts,), float(diffusion0))
            self._is_variable_diffusion0 = False
        else:
            self._diffusion0_value = None
            self._diffusion0_array = diffusion0
            self._is_variable_diffusion0 = True

        if isinstance(diffusion1, (int, float)):
            self._diffusion1_value = float(diffusion1)
            self._diffusion1_array = bkd.full((npts,), float(diffusion1))
            self._is_variable_diffusion1 = False
        else:
            self._diffusion1_value = None
            self._diffusion1_array = diffusion1
            self._is_variable_diffusion1 = True

        self._reaction = reaction
        self._forcing0_func = forcing0
        self._forcing1_func = forcing1

        # Precompute derivative matrices
        self._D1_matrices = [basis.derivative_matrix(1, dim) for dim in range(ndim)]
        self._D2_matrices = [basis.derivative_matrix(2, dim) for dim in range(ndim)]

    def diffusion(self) -> Tuple[Union[float, Array], Union[float, Array]]:
        """Return diffusion coefficients."""
        d0 = self._diffusion0_value if not self._is_variable_diffusion0 else self._diffusion0_array
        d1 = self._diffusion1_value if not self._is_variable_diffusion1 else self._diffusion1_array
        return d0, d1

    def _get_forcing(self, time: float) -> Tuple[Array, Array]:
        """Get forcing arrays at given time."""
        npts = self.npts()
        bkd = self._bkd

        if self._forcing0_func is None:
            f0 = bkd.zeros((npts,))
        elif callable(self._forcing0_func):
            f0 = self._forcing0_func(time)
        else:
            f0 = self._forcing0_func

        if self._forcing1_func is None:
            f1 = bkd.zeros((npts,))
        elif callable(self._forcing1_func):
            f1 = self._forcing1_func(time)
        else:
            f1 = self._forcing1_func

        return f0, f1

    def _split_state(self, state: Array) -> Tuple[Array, Array]:
        """Split combined state into species components."""
        npts = self.npts()
        u0 = state[:npts]
        u1 = state[npts:]
        return u0, u1

    def _combine_state(self, u0: Array, u1: Array) -> Array:
        """Combine species components into single state vector."""
        return self._bkd.hstack([u0, u1])

    def _compute_laplacian(self, u: Array) -> Array:
        """Compute Laplacian of a scalar field."""
        laplacian = self._bkd.zeros_like(u)
        for D2 in self._D2_matrices:
            laplacian = laplacian + D2 @ u
        return laplacian

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
        bkd = self._bkd
        u0, u1 = self._split_state(state)

        # Compute Laplacians
        lap_u0 = self._compute_laplacian(u0)
        lap_u1 = self._compute_laplacian(u1)

        # Diffusion terms
        if not self._is_variable_diffusion0:
            diff_term0 = self._diffusion0_value * lap_u0
        else:
            # Variable diffusion: div(D*grad(u)) = D*lap(u) + grad(D)·grad(u)
            diff_term0 = self._diffusion0_array * lap_u0
            for dim, D1 in enumerate(self._D1_matrices):
                grad_D0 = D1 @ self._diffusion0_array
                grad_u0 = D1 @ u0
                diff_term0 = diff_term0 + grad_D0 * grad_u0

        if not self._is_variable_diffusion1:
            diff_term1 = self._diffusion1_value * lap_u1
        else:
            diff_term1 = self._diffusion1_array * lap_u1
            for dim, D1 in enumerate(self._D1_matrices):
                grad_D1 = D1 @ self._diffusion1_array
                grad_u1 = D1 @ u1
                diff_term1 = diff_term1 + grad_D1 * grad_u1

        # Reaction terms
        R0, R1 = self._reaction(u0, u1)

        # Forcing terms
        f0, f1 = self._get_forcing(time)

        # Combine residuals
        res0 = diff_term0 + R0 + f0
        res1 = diff_term1 + R1 + f1

        return self._combine_state(res0, res1)

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
            Jacobian matrix. Shape: (2*npts, 2*npts)
        """
        bkd = self._bkd
        npts = self.npts()
        u0, u1 = self._split_state(state)

        # Initialize Jacobian blocks
        # Jacobian is [[J00, J01], [J10, J11]] where Jij = d(res_i)/d(u_j)
        jacobian = bkd.zeros((2 * npts, 2 * npts))

        # Diffusion contribution to diagonal blocks
        # d(D*lap(u))/du = D*lap operator
        lap_matrix = sum(D2 for D2 in self._D2_matrices)

        if not self._is_variable_diffusion0:
            J00_diff = self._diffusion0_value * lap_matrix
        else:
            # Variable diffusion: more complex Jacobian
            J00_diff = bkd.diag(self._diffusion0_array) @ lap_matrix
            for D1 in self._D1_matrices:
                grad_D0 = D1 @ self._diffusion0_array
                J00_diff = J00_diff + bkd.diag(grad_D0) @ D1

        if not self._is_variable_diffusion1:
            J11_diff = self._diffusion1_value * lap_matrix
        else:
            J11_diff = bkd.diag(self._diffusion1_array) @ lap_matrix
            for D1 in self._D1_matrices:
                grad_D1 = D1 @ self._diffusion1_array
                J11_diff = J11_diff + bkd.diag(grad_D1) @ D1

        # Reaction contribution
        dR0_du0, dR0_du1, dR1_du0, dR1_du1 = self._reaction.jacobian(u0, u1)

        # Assemble Jacobian blocks
        # J00 = diffusion + dR0/du0
        jacobian[:npts, :npts] = J00_diff + bkd.diag(dR0_du0)

        # J01 = dR0/du1
        jacobian[:npts, npts:] = bkd.diag(dR0_du1)

        # J10 = dR1/du0
        jacobian[npts:, :npts] = bkd.diag(dR1_du0)

        # J11 = diffusion + dR1/du1
        jacobian[npts:, npts:] = J11_diff + bkd.diag(dR1_du1)

        return jacobian

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
        """
        bkd = self._bkd
        nboundary = boundary_indices.shape[0]
        ndim = len(self._D1_matrices)

        u0, u1 = self._split_state(state)

        # Compute flux for species 0: D0 * grad(u0) · n
        flux0 = bkd.zeros((nboundary,))
        for dim, D1 in enumerate(self._D1_matrices):
            grad_u0_dim = D1 @ u0
            flux0 = flux0 + grad_u0_dim[boundary_indices] * float(normal[dim])

        # Scale by diffusion coefficient
        if self._is_variable_diffusion0:
            flux0 = self._diffusion0_array[boundary_indices] * flux0
        else:
            flux0 = self._diffusion0_value * flux0

        # Compute flux for species 1: D1 * grad(u1) · n
        flux1 = bkd.zeros((nboundary,))
        for dim, D1 in enumerate(self._D1_matrices):
            grad_u1_dim = D1 @ u1
            flux1 = flux1 + grad_u1_dim[boundary_indices] * float(normal[dim])

        # Scale by diffusion coefficient
        if self._is_variable_diffusion1:
            flux1 = self._diffusion1_array[boundary_indices] * flux1
        else:
            flux1 = self._diffusion1_value * flux1

        # Component-stacked ordering: [flux0_0, ..., flux0_n, flux1_0, ..., flux1_n]
        return bkd.concatenate([flux0, flux1])


def create_two_species_reaction_diffusion(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion0: Union[float, Array],
    diffusion1: Union[float, Array],
    reaction: ReactionProtocol[Array],
    forcing0: Optional[Callable[[float], Array]] = None,
    forcing1: Optional[Callable[[float], Array]] = None,
) -> TwoSpeciesReactionDiffusionPhysics[Array]:
    """Create two-species reaction-diffusion physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion0 : float or Array
        Diffusion coefficient for species 0.
    diffusion1 : float or Array
        Diffusion coefficient for species 1.
    reaction : ReactionProtocol
        Reaction term object.
    forcing0 : Callable or Array, optional
        Forcing for species 0.
    forcing1 : Callable or Array, optional
        Forcing for species 1.

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

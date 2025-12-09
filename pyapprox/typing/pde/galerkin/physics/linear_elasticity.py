"""Linear elasticity physics for Galerkin FEM.

Solves the linear elasticity equations:
    -div(sigma) = f

where sigma is the Cauchy stress tensor:
    sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon

and epsilon is the strain tensor:
    epsilon = (grad(u) + grad(u)^T) / 2

The Lame parameters lambda and mu are related to Young's modulus E
and Poisson's ratio nu by:
    lambda = E * nu / ((1 + nu) * (1 - 2*nu))
    mu = E / (2 * (1 + nu))
"""

from typing import Generic, Optional, Callable, List

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.typing.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis

# Import skfem for assembly
try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.models.poisson import mass as scalar_mass
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class LinearElasticity(Generic[Array]):
    """Linear elasticity physics.

    Solves:
        -div(sigma(u)) = f

    where sigma is the Cauchy stress tensor.

    In weak form (for steady state K*u = b):
        K_ij = integral(sigma(phi_j) : epsilon(phi_i))
        b_i = integral(f . phi_i)

    Parameters
    ----------
    basis : VectorLagrangeBasis
        Vector finite element basis for displacement.
    youngs_modulus : float
        Young's modulus E.
    poisson_ratio : float
        Poisson's ratio nu. Must satisfy -1 < nu < 0.5 for stability.
    bkd : Backend
        Computational backend.
    body_force : Callable, optional
        Body force per unit volume. Takes coordinates (ndim, npts)
        and returns (ndim, npts).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh2D
    >>> from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh2D(
    ...     nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
    ... )
    >>> basis = VectorLagrangeBasis(mesh, degree=1)
    >>> physics = LinearElasticity(
    ...     basis=basis,
    ...     youngs_modulus=1.0,
    ...     poisson_ratio=0.3,
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        basis: VectorLagrangeBasis[Array],
        youngs_modulus: float,
        poisson_ratio: float,
        bkd: Backend[Array],
        body_force: Optional[Callable] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []

        # Validate Poisson ratio
        if not (-1.0 < poisson_ratio < 0.5):
            raise ValueError(
                f"Poisson ratio must satisfy -1 < nu < 0.5, got {poisson_ratio}"
            )

        # Store material parameters
        self._youngs_modulus = youngs_modulus
        self._poisson_ratio = poisson_ratio

        # Compute Lame parameters
        self._lambda, self._mu = lame_parameters(youngs_modulus, poisson_ratio)

        # Store body force
        self._body_force = body_force

        # Cache assembled matrices
        self._stiffness_cached: Optional[Array] = None
        self._mass_cached: Optional[Array] = None
        self._load_cached: Optional[Array] = None

    @property
    def youngs_modulus(self) -> float:
        """Return Young's modulus."""
        return self._youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        """Return Poisson's ratio."""
        return self._poisson_ratio

    @property
    def lame_lambda(self) -> float:
        """Return first Lame parameter."""
        return self._lambda

    @property
    def lame_mu(self) -> float:
        """Return second Lame parameter (shear modulus)."""
        return self._mu

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return total number of DOFs."""
        return self._basis.ndofs()

    def ndim(self) -> int:
        """Return spatial dimension."""
        return self._basis.ncomponents()

    def mass_matrix(self) -> Array:
        """Return the vector mass matrix.

        For elasticity:
            M_ij = integral(phi_i . phi_j)

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        if self._mass_cached is not None:
            return self._mass_cached

        skfem_basis = self._basis.skfem_basis()

        # Vector mass matrix
        def mass_form(u, v, w):
            # u and v are vector-valued
            return sum(u[i] * v[i] for i in range(len(u)))

        mass_np = asm(BilinearForm(mass_form), skfem_basis).toarray()
        self._mass_cached = self._bkd.asarray(mass_np.astype(np.float64))

        return self._mass_cached

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        Parameters
        ----------
        rhs : Array
            Right-hand side vector.

        Returns
        -------
        Array
            Solution x = M^{-1} * rhs.
        """
        M = self.mass_matrix()
        return self._bkd.solve(M, rhs)

    def stiffness_matrix(self) -> Array:
        """Return the elasticity stiffness matrix.

        K_ij = integral(sigma(phi_j) : epsilon(phi_i))

        Returns
        -------
        Array
            Stiffness matrix. Shape: (nstates, nstates)
        """
        if self._stiffness_cached is not None:
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()

        # Use skfem's linear_elasticity
        stiffness_np = asm(
            linear_elasticity(self._lambda, self._mu), skfem_basis
        ).toarray()

        self._stiffness_cached = self._bkd.asarray(stiffness_np.astype(np.float64))

        return self._stiffness_cached

    def load_vector(self) -> Array:
        """Return the load vector from body forces.

        b_i = integral(f . phi_i)

        Returns
        -------
        Array
            Load vector. Shape: (nstates,)
        """
        if self._load_cached is not None:
            return self._load_cached

        skfem_basis = self._basis.skfem_basis()
        ndim = self._basis.ncomponents()

        if self._body_force is None:
            load_np = np.zeros(self.nstates())
        else:
            body_force_func = self._body_force

            def linear_form(v, w):
                # w.x shape: (ndim, nelem, nquad) or (ndim, npts)
                # Convert DiscreteField to numpy array for zeros_like to work
                x = np.asarray(w.x)
                x_shape = x.shape
                if len(x_shape) == 3:
                    n, nelem, nquad = x_shape
                    x_flat = x.reshape(n, -1)
                    force_flat = body_force_func(x_flat)  # (ndim, npts)
                    force = force_flat.reshape(ndim, nelem, nquad)
                else:
                    force = body_force_func(x)

                # Sum over components: f . v
                return sum(force[i] * v[i] for i in range(ndim))

            load_np = asm(LinearForm(linear_form), skfem_basis)

        self._load_cached = self._bkd.asarray(load_np.astype(np.float64))

        return self._load_cached

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F(u, t) = b - K*u.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time (unused for static elasticity).

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        K = self.stiffness_matrix()
        b = self.load_vector()

        return b - K @ state

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du = -K.

        For linear elasticity, the Jacobian is constant.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time (unused).

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        K = self.stiffness_matrix()
        return -K

    def initial_condition(self, func: Callable) -> Array:
        """Create initial condition by interpolating a displacement field.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Takes coordinates (ndim, npts)
            and returns displacement (ndim, npts).

        Returns
        -------
        Array
            DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        return (
            f"LinearElasticity("
            f"nstates={self.nstates()}, "
            f"E={self._youngs_modulus}, "
            f"nu={self._poisson_ratio})"
        )

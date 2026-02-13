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

from typing import Generic, Optional, Callable, List, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    RobinBCProtocol,
)
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
        Body force per unit volume. Takes coordinates (ndim, npts) and
        time (float), returns (ndim, npts). For static forces, use
        ``lambda x, t: f(x)``.
    dirichlet_bcs : list of (str, Callable), optional
        Dirichlet BCs as (boundary_name, value_func) pairs.
        value_func(coords, time) takes (ndim, npts) coords and float time,
        returns (npts, ncomponents) displacement values.

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

        # Pre-compute decomposed stiffness: K = lambda*K_lambda + mu*K_mu
        skfem_basis = self._basis.skfem_basis()
        K_lambda_np = asm(
            linear_elasticity(1.0, 0.0), skfem_basis
        ).toarray().astype(np.float64)
        K_mu_np = asm(
            linear_elasticity(0.0, 1.0), skfem_basis
        ).toarray().astype(np.float64)
        self._K_lambda = self._bkd.asarray(K_lambda_np)
        self._K_mu = self._bkd.asarray(K_mu_np)

        # Cache assembled matrices
        self._stiffness_cached: Optional[Array] = None
        self._mass_cached: Optional[Array] = None

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

    def basis(self) -> VectorLagrangeBasis[Array]:
        """Return the vector finite element basis."""
        return self._basis

    def stiffness_matrix(self) -> Array:
        """Return the elasticity stiffness matrix.

        Reconstructed from decomposition: K = lambda * K_lambda + mu * K_mu.

        Returns
        -------
        Array
            Stiffness matrix. Shape: (nstates, nstates)
        """
        if self._stiffness_cached is not None:
            return self._stiffness_cached

        self._stiffness_cached = (
            self._lambda * self._K_lambda + self._mu * self._K_mu
        )
        return self._stiffness_cached

    def load_vector(self, time: float = 0.0) -> Array:
        """Return the load vector from body forces.

        b_i = integral(f . phi_i)

        Parameters
        ----------
        time : float, default=0.0
            Current time. Passed to body_force(x, time).

        Returns
        -------
        Array
            Load vector. Shape: (nstates,)
        """
        skfem_basis = self._basis.skfem_basis()
        ndim = self._basis.ncomponents()

        if self._body_force is None:
            load_np = np.zeros(self.nstates())
        else:
            body_force_func = self._body_force
            current_time = time

            def linear_form(v, w):
                # w.x shape: (ndim, nelem, nquad) or (ndim, npts)
                # Convert DiscreteField to numpy array for zeros_like to work
                x = np.asarray(w.x)
                x_shape = x.shape
                if len(x_shape) == 3:
                    n, nelem, nquad = x_shape
                    x_flat = x.reshape(n, -1)
                    force_flat = np.asarray(
                        body_force_func(x_flat, current_time)
                    )
                    force = force_flat.reshape(ndim, nelem, nquad)
                else:
                    force = np.asarray(body_force_func(x, current_time))

                # Sum over components: f . v
                return sum(force[i] * v[i] for i in range(ndim))

            load_np = asm(LinearForm(linear_form), skfem_basis)

        return self._bkd.asarray(load_np.astype(np.float64))

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        Computes F = b(t) - K*u. Dirichlet BCs are NOT applied.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        K = self.stiffness_matrix()
        b = self.load_vector(time)
        return b - K @ state

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and their exact values.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices : Array
                DOF indices. Shape: (ndirichlet,)
            dof_values : Array
                Exact values. Shape: (ndirichlet,)
        """
        all_dofs = []
        all_vals = []
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                dofs_np = self._bkd.to_numpy(bc.boundary_dofs())
                vals_np = self._bkd.to_numpy(bc.boundary_values(time))
                all_dofs.append(dofs_np)
                all_vals.append(vals_np)
        if all_dofs:
            return (
                self._bkd.asarray(
                    np.concatenate(all_dofs).astype(np.int64)
                ),
                self._bkd.asarray(
                    np.concatenate(all_vals).astype(np.float64)
                ),
            )
        return (
            self._bkd.asarray(np.array([], dtype=np.int64)),
            self._bkd.asarray(np.array([], dtype=np.float64)),
        )

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F(u, t) = b(t) - K*u with BCs applied.

        Dirichlet BCs replace residual rows with state[dof] - g(x, t).

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        residual = self.spatial_residual(state, time)

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, time)

        return residual

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF/du without Dirichlet enforcement.

        For linear elasticity, dF/du = -K (constant).

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        return -self.stiffness_matrix()

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du = -K with BCs applied.

        Dirichlet BCs replace Jacobian rows with identity.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        K = self.stiffness_matrix()
        jac = -K

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                jac = bc.apply_to_jacobian(jac, state, time)

        return jac

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

    # -----------------------------------------------------------------
    # Parameter sensitivity methods (E, nu)
    # -----------------------------------------------------------------

    def nparams(self) -> int:
        """Return number of material parameters (E, nu)."""
        return 2

    def set_param(self, param: Array) -> None:
        """Update material parameters and invalidate stiffness cache.

        Parameters
        ----------
        param : Array
            Parameter vector [E, nu]. Shape: (2,)
        """
        param_np = self._bkd.to_numpy(param)
        E = float(param_np[0])
        nu = float(param_np[1])

        if not (-1.0 < nu < 0.5):
            raise ValueError(
                f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}"
            )

        self._youngs_modulus = E
        self._poisson_ratio = nu
        self._lambda, self._mu = lame_parameters(E, nu)
        self._stiffness_cached = None

    def param_jacobian(self, state: Array, time: float) -> Array:
        """Compute parameter Jacobian dF/dp where F = b - K*u.

        Since body force does not depend on (E, nu):
            dF/dp = -dK/dp @ u

        Chain rule through Lame parameters:
            dK/dE  = (dLambda/dE) * K_lambda + (dMu/dE) * K_mu
            dK/dnu = (dLambda/dnu) * K_lambda + (dMu/dnu) * K_mu

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, 2)
        """
        E = self._youngs_modulus
        nu = self._poisson_ratio

        # Derivatives of Lame parameters w.r.t. E
        denom1 = (1.0 + nu) * (1.0 - 2.0 * nu)
        dLambda_dE = nu / denom1
        dMu_dE = 1.0 / (2.0 * (1.0 + nu))

        # Derivatives of Lame parameters w.r.t. nu
        # Lambda = E*nu / ((1+nu)*(1-2nu))
        # dLambda/dnu = E*(1 + 2*nu^2) / ((1+nu)*(1-2nu))^2
        dLambda_dnu = E * (1.0 + 2.0 * nu**2) / denom1**2

        # Mu = E / (2*(1+nu))
        # dMu/dnu = -E / (2*(1+nu)^2)
        dMu_dnu = -E / (2.0 * (1.0 + nu)**2)

        # dK/dp @ u
        K_lambda_u = self._K_lambda @ state
        K_mu_u = self._K_mu @ state

        # dF/dE = -(dLambda/dE * K_lambda + dMu/dE * K_mu) @ u
        col_E = -(dLambda_dE * K_lambda_u + dMu_dE * K_mu_u)

        # dF/dnu = -(dLambda/dnu * K_lambda + dMu/dnu * K_mu) @ u
        col_nu = -(dLambda_dnu * K_lambda_u + dMu_dnu * K_mu_u)

        return self._bkd.stack([col_E, col_nu], axis=1)

    def initial_param_jacobian(self) -> Array:
        """Return d(u_0)/dp = 0 (initial condition does not depend on E, nu).

        Returns
        -------
        Array
            Zero matrix. Shape: (nstates, 2)
        """
        n = self.nstates()
        return self._bkd.asarray(np.zeros((n, 2)))

    def __repr__(self) -> str:
        return (
            f"LinearElasticity("
            f"nstates={self.nstates()}, "
            f"E={self._youngs_modulus}, "
            f"nu={self._poisson_ratio})"
        )

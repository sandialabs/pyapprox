"""Subdomain wrapper for domain decomposition.

Wraps an existing PDE physics solver to provide DtN-specific capabilities:
- Setting Dirichlet BCs on interfaces from interface DOF coefficients
- Solving the subdomain problem with current interface values
- Computing normal flux on interfaces (for residual computation)
- Computing flux Jacobian w.r.t. interface DOFs (for Newton solver)
"""

from typing import Dict, Generic, List, Optional

from pyapprox.pde.collocation.boundary.dirichlet import DirichletBC
from pyapprox.pde.collocation.protocols.physics import PhysicsProtocol
from pyapprox.pde.decomposition.protocols.interface import (
    InterfaceProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class SubdomainWrapper(Generic[Array]):
    """Wrapper for subdomain solver in DtN domain decomposition.

    Wraps an existing PDE physics object and provides additional capabilities
    for domain decomposition:

    1. Setting Dirichlet BCs on interfaces from interface DOF coefficients
    2. Solving the subdomain problem with current interface values
    3. Computing normal flux on interfaces (for residual computation)
    4. Computing flux Jacobian w.r.t. interface DOFs (for Newton solver)

    The wrapper stores interfaces and manages interpolation between interface
    basis and subdomain mesh internally.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    subdomain_id : int
        Unique identifier for this subdomain.
    physics : PhysicsProtocol
        PDE physics for this subdomain.
    interfaces : Dict[int, InterfaceProtocol]
        Interfaces adjacent to this subdomain, keyed by interface ID.
    external_bcs : List[DirichletBC], optional
        Dirichlet BCs on external (non-interface) boundaries.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        subdomain_id: int,
        physics: PhysicsProtocol[Array],
        interfaces: Dict[int, InterfaceProtocol[Array]],
        external_bcs: Optional[List[DirichletBC[Array]]] = None,
    ):
        self._bkd = bkd
        self._subdomain_id = subdomain_id
        self._physics = physics
        self._interfaces = interfaces
        self._external_bcs = external_bcs if external_bcs is not None else []

        # Store current interface coefficient values
        self._interface_coeffs: Dict[int, Array] = {}

        # Store interface Dirichlet BCs (created when coefficients are set)
        self._interface_bcs: Dict[int, DirichletBC[Array]] = {}

        # Store boundary indices for each interface
        self._interface_boundary_indices: Dict[int, Array] = {}

        # Store most recent solution
        self._solution: Optional[Array] = None

        # Epsilon for finite difference Jacobian computation
        self._fd_epsilon = 1e-7

        # Current time for time-dependent problems
        self._time: float = 0.0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def subdomain_id(self) -> int:
        """Return unique identifier for this subdomain."""
        return self._subdomain_id

    def interface_ids(self) -> List[int]:
        """Return IDs of interfaces adjacent to this subdomain."""
        return list(self._interfaces.keys())

    def nstates(self) -> int:
        """Return number of DOFs in subdomain solution."""
        return self._physics.nstates()

    def ncomponents(self) -> int:
        """Return number of solution components.

        Delegates to the underlying physics object.
        """
        # Check if physics has ncomponents method
        if hasattr(self._physics, "ncomponents"):
            return self._physics.ncomponents()
        return 1  # Default to scalar

    def set_interface_boundary_indices(
        self, interface_id: int, boundary_indices: Array
    ) -> None:
        """Set subdomain mesh indices corresponding to an interface.

        This defines which mesh points are on the interface and will have
        Dirichlet BCs applied when interface values are set.

        Parameters
        ----------
        interface_id : int
            ID of the interface.
        boundary_indices : Array
            Indices of mesh points on this interface. Shape: (n_boundary,)
        """
        if interface_id not in self._interfaces:
            raise ValueError(
                f"interface_id {interface_id} not in interfaces "
                f"{list(self._interfaces.keys())}"
            )
        self._interface_boundary_indices[interface_id] = boundary_indices

    def set_interface_dirichlet(
        self, interface_id: int, interface_coeffs: Array
    ) -> None:
        """Set Dirichlet BC on interface from interface DOF coefficients.

        The interface coefficients are interpolated to subdomain boundary
        nodes and used to create a Dirichlet boundary condition.

        For vector-valued PDEs, coefficients use component-stacked ordering
        and the BC is applied to all components.

        Parameters
        ----------
        interface_id : int
            ID of the interface.
        interface_coeffs : Array
            Interface DOF coefficients.
            Shape: (interface_ndofs * ncomponents,) with component-stacked ordering.
        """
        if interface_id not in self._interfaces:
            raise ValueError(
                f"interface_id {interface_id} not in interfaces "
                f"{list(self._interfaces.keys())}"
            )

        if interface_id not in self._interface_boundary_indices:
            raise ValueError(
                f"Boundary indices not set for interface {interface_id}. "
                "Call set_interface_boundary_indices first."
            )

        # Store coefficients
        self._interface_coeffs[interface_id] = interface_coeffs

        # Get interface and interpolate to subdomain boundary
        interface = self._interfaces[interface_id]
        boundary_indices = self._interface_boundary_indices[interface_id]
        ncomponents = self.ncomponents()

        # Evaluate interface function at interface points
        interface_values = interface.evaluate(interface_coeffs)

        # Interpolate to subdomain boundary nodes
        boundary_values = interface.interpolate_to_subdomain(
            self._subdomain_id, interface_values
        )

        if ncomponents == 1:
            # Scalar case - use boundary indices directly
            self._interface_bcs[interface_id] = DirichletBC(
                self._bkd, boundary_indices, boundary_values
            )
        else:
            # Vector case - build component-stacked indices
            # boundary_indices are per-component mesh indices
            # For vector physics with state [u_0,...,u_n, v_0,...,v_n],
            # we need indices: [i, i+npts, i+2*npts, ...] for each i in boundary_indices
            nboundary = boundary_indices.shape[0]
            npts = self._physics.npts()

            # Build full index array for all components
            all_indices = []
            all_values = []
            for c in range(ncomponents):
                # Component c: indices are boundary_indices + c * npts
                comp_indices = boundary_indices + c * npts
                comp_values = boundary_values[c * nboundary : (c + 1) * nboundary]
                all_indices.append(comp_indices)
                all_values.append(comp_values)

            full_indices = self._bkd.concatenate(all_indices)
            full_values = self._bkd.concatenate(all_values)

            self._interface_bcs[interface_id] = DirichletBC(
                self._bkd, full_indices, full_values
            )

    def solve(self) -> Array:
        """Solve subdomain problem with current boundary conditions.

        Uses Newton iteration to solve the nonlinear system.
        The current time can be set via set_time() for time-dependent problems.

        Returns
        -------
        Array
            Subdomain solution. Shape: (nstates,)
        """
        bkd = self._bkd
        nstates = self.nstates()
        time = self._time

        # Initial guess (could be improved with previous solution)
        if self._solution is not None:
            state = bkd.copy(self._solution)
        else:
            state = bkd.zeros((nstates,))

        # Newton iteration
        max_iters = 20
        tol = 1e-10

        for _ in range(max_iters):
            # Compute residual and Jacobian
            residual = self._physics.residual(state, time)
            jacobian = self._physics.jacobian(state, time)

            # Apply boundary conditions
            residual, jacobian = self._apply_all_bcs(residual, jacobian, state, time)

            # Check convergence
            res_norm = bkd.to_float(bkd.norm(residual))
            if res_norm < tol:
                break

            # Newton step
            delta = bkd.solve(jacobian, -residual)
            state = state + delta

        self._solution = state
        return state

    def _apply_all_bcs(
        self, residual: Array, jacobian: Array, state: Array, time: float
    ) -> tuple:
        """Apply all boundary conditions (external and interface)."""
        # Apply external BCs
        for bc in self._external_bcs:
            residual = bc.apply_to_residual(residual, state, time)
            jacobian = bc.apply_to_jacobian(jacobian, state, time)

        # Apply interface BCs
        for interface_id, bc in self._interface_bcs.items():
            residual = bc.apply_to_residual(residual, state, time)
            jacobian = bc.apply_to_jacobian(jacobian, state, time)

        return residual, jacobian

    def solution(self) -> Array:
        """Return most recent subdomain solution.

        Returns
        -------
        Array
            Subdomain solution. Shape: (nstates,)
        """
        if self._solution is None:
            raise ValueError("No solution computed yet. Call solve() first.")
        return self._solution

    def set_initial_solution(self, state: Array) -> None:
        """Set initial solution for Newton iteration or direct use.

        This is useful for:
        1. Providing a good initial guess for Newton iteration
        2. Setting a known solution directly (e.g., quiescent state for
           shallow water equations)

        Parameters
        ----------
        state : Array
            Initial state. Shape: (nstates,)
        """
        nstates = self.nstates()
        if state.shape[0] != nstates:
            raise ValueError(
                f"state shape {state.shape} doesn't match nstates {nstates}"
            )
        self._solution = self._bkd.copy(state)

    def set_time(self, time: float) -> None:
        """Set the current time for time-dependent problems.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def get_time(self) -> float:
        """Get the current time.

        Returns
        -------
        float
            Current time.
        """
        return self._time

    def compute_interface_flux(self, interface_id: int) -> Array:
        """Compute normal flux on interface.

        The flux computation is physics-dependent:
        - Diffusion: k * (grad u) · n
        - Elasticity: σ · n (traction vector)
        - Reaction-diffusion: D_i * (grad u_i) · n per species

        If the underlying physics object implements compute_interface_flux(),
        that method is used. Otherwise, a default gradient-based flux is computed.

        Parameters
        ----------
        interface_id : int
            ID of the interface.

        Returns
        -------
        Array
            Normal flux at interface points.
            Shape: (interface_npts * ncomponents,) with component-stacked ordering.
        """
        if interface_id not in self._interfaces:
            raise ValueError(
                f"interface_id {interface_id} not in interfaces "
                f"{list(self._interfaces.keys())}"
            )

        if self._solution is None:
            raise ValueError("No solution computed yet. Call solve() first.")

        interface = self._interfaces[interface_id]
        boundary_indices = self._interface_boundary_indices[interface_id]

        # Get outward normal for this subdomain
        normal = interface.normal(self._subdomain_id)

        # Check if physics implements compute_interface_flux
        if hasattr(self._physics, "compute_interface_flux"):
            # Delegate to physics-specific flux computation
            boundary_flux = self._physics.compute_interface_flux(
                self._solution, boundary_indices, normal
            )
        else:
            # Default: compute grad(u) · n for each component
            boundary_flux = self._compute_default_flux(boundary_indices, normal)

        # Restrict from subdomain boundary to interface points
        interface_flux = interface.restrict_from_subdomain(
            self._subdomain_id, boundary_flux
        )

        return interface_flux

    def _compute_default_flux(self, boundary_indices: Array, normal: Array) -> Array:
        """Compute default gradient-based flux.

        Computes grad(u) · n for each component (without diffusion coefficient).

        Parameters
        ----------
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Flux at boundary points.
            Shape: (nboundary * ncomponents,) with component-stacked ordering.
        """
        basis = self._physics.basis()
        ndim = basis.ndim()
        ncomponents = self.ncomponents()
        nboundary = boundary_indices.shape[0]

        if ncomponents == 1:
            # Scalar case: existing behavior
            grad_u_at_boundary = []
            for dim in range(ndim):
                D_dim = basis.derivative_matrix(1, dim)
                grad_u_dim = D_dim @ self._solution
                grad_u_at_boundary.append(grad_u_dim[boundary_indices])

            flux = self._bkd.zeros((nboundary,))
            for dim in range(ndim):
                flux = flux + grad_u_at_boundary[dim] * self._bkd.to_float(normal[dim])
            return flux

        # Vector case: compute flux for each component
        npts = self._physics.npts()
        all_flux = self._bkd.zeros((ncomponents * nboundary,))

        for c in range(ncomponents):
            # Extract component from solution
            comp_solution = self._solution[c * npts : (c + 1) * npts]

            # Compute gradient for this component
            grad_at_boundary = []
            for dim in range(ndim):
                D_dim = basis.derivative_matrix(1, dim)
                grad_dim = D_dim @ comp_solution
                grad_at_boundary.append(grad_dim[boundary_indices])

            # Compute flux = grad(u_c) · n
            comp_flux = self._bkd.zeros((nboundary,))
            for dim in range(ndim):
                n_d = self._bkd.to_float(normal[dim])
                comp_flux = comp_flux + grad_at_boundary[dim] * n_d

            all_flux[c * nboundary : (c + 1) * nboundary] = comp_flux

        return all_flux

    def compute_flux_jacobian_column(
        self, interface_id: int, perturbed_interface_id: int, dof_index: int
    ) -> Array:
        """Compute flux Jacobian column for one interface DOF.

        Computes d(flux_interface_id) / d(interface_coeff[dof_index])
        by solving the subdomain with a perturbed interface value.

        For vector-valued PDEs, dof_index runs over all components:
        dof_index = component * ndofs_per_component + local_dof_index

        Parameters
        ----------
        interface_id : int
            ID of interface where flux is computed.
        perturbed_interface_id : int
            ID of interface where DOF is perturbed.
        dof_index : int
            Index of perturbed DOF in interface basis (0 to total_ndofs-1).

        Returns
        -------
        Array
            Flux Jacobian column.
            Shape: (interface_npts * ncomponents,) with component-stacked ordering.
        """
        if interface_id not in self._interfaces:
            raise ValueError(
                f"interface_id {interface_id} not in interfaces "
                f"{list(self._interfaces.keys())}"
            )
        if perturbed_interface_id not in self._interfaces:
            raise ValueError(
                f"perturbed_interface_id {perturbed_interface_id} not in "
                f"interfaces {list(self._interfaces.keys())}"
            )

        # Save original interface coefficients
        original_coeffs = {}
        for iid, coeffs in self._interface_coeffs.items():
            original_coeffs[iid] = self._bkd.copy(coeffs)

        # Compute base flux
        base_flux = self.compute_interface_flux(interface_id)

        # Perturb interface DOF
        perturbed_coeffs = self._bkd.copy(
            self._interface_coeffs[perturbed_interface_id]
        )
        perturbed_coeffs[dof_index] = perturbed_coeffs[dof_index] + self._fd_epsilon

        # Set perturbed interface values and solve
        self.set_interface_dirichlet(perturbed_interface_id, perturbed_coeffs)
        self.solve()

        # Compute perturbed flux
        perturbed_flux = self.compute_interface_flux(interface_id)

        # Restore original interface values and solve
        for iid, coeffs in original_coeffs.items():
            self.set_interface_dirichlet(iid, coeffs)
        self.solve()

        # Finite difference approximation
        flux_jacobian_col = (perturbed_flux - base_flux) / self._fd_epsilon

        return flux_jacobian_col

    def __repr__(self) -> str:
        return (
            f"SubdomainWrapper(id={self._subdomain_id}, "
            f"nstates={self.nstates()}, interfaces={list(self._interfaces.keys())})"
        )

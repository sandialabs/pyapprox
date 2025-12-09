"""Protocol for subdomain solvers in domain decomposition.

Defines SubdomainSolverProtocol which wraps existing PDE solvers
to provide DtN-specific capabilities:
- Setting Dirichlet BCs on interfaces
- Computing fluxes on interfaces
- Computing flux Jacobians for Newton iteration
"""

from typing import Protocol, Generic, runtime_checkable, List

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class SubdomainSolverProtocol(Protocol, Generic[Array]):
    """Protocol for subdomain solvers in DtN domain decomposition.

    A subdomain solver wraps an existing PDE solver (e.g., collocation physics)
    and provides additional capabilities for domain decomposition:

    1. Setting Dirichlet boundary conditions on interfaces from interface DOFs
    2. Solving the subdomain problem with current interface values
    3. Computing normal flux on interfaces (for residual computation)
    4. Computing flux Jacobian w.r.t. interface DOFs (for Newton solver)

    The solver handles interpolation between interface basis and subdomain mesh
    internally via the Interface objects.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def subdomain_id(self) -> int:
        """Return unique identifier for this subdomain."""
        ...

    def interface_ids(self) -> List[int]:
        """Return IDs of interfaces adjacent to this subdomain."""
        ...

    def nstates(self) -> int:
        """Return number of DOFs in subdomain solution.

        For vector-valued PDEs, this is npts * ncomponents.
        """
        ...

    def ncomponents(self) -> int:
        """Return number of solution components.

        For scalar PDEs, this returns 1.
        For vector PDEs (e.g., elasticity, reaction-diffusion), this returns
        the number of solution components.
        """
        ...

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
        ...

    def solve(self) -> Array:
        """Solve subdomain problem with current boundary conditions.

        Returns
        -------
        Array
            Subdomain solution. Shape: (nstates,)
        """
        ...

    def solution(self) -> Array:
        """Return most recent subdomain solution.

        Returns
        -------
        Array
            Subdomain solution. Shape: (nstates,)
        """
        ...

    def compute_interface_flux(self, interface_id: int) -> Array:
        """Compute normal flux on interface.

        The flux computation is physics-dependent:
        - Diffusion: k * (grad u) · n
        - Elasticity: σ · n (traction vector)
        - Reaction-diffusion: D_i * (grad u_i) · n per species

        The underlying physics object must implement compute_interface_flux().

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
        ...

    def compute_flux_jacobian_column(
        self, interface_id: int, perturbed_interface_id: int, dof_index: int
    ) -> Array:
        """Compute flux Jacobian column for one interface DOF.

        Computes d(flux_interface_id) / d(interface_coeff[dof_index])
        by solving the subdomain with a perturbed interface value.

        For vector-valued PDEs, dof_index runs over all components:
        dof_index = component * ndofs_per_component + local_dof_index

        For exact Jacobian computation, this requires:
        1. Save current interface values
        2. Perturb interface DOF by epsilon
        3. Solve subdomain
        4. Compute flux difference / epsilon
        5. Restore original interface values

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
        ...

"""Protocol for domain decomposition setup.

Defines DomainDecompositionProtocol which manages the full decomposition:
- Collection of subdomain solvers
- Collection of interfaces
- DOF indexing for global interface vector
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.decomposition.protocols.interface import (
    InterfaceProtocol,
)
from pyapprox.typing.pde.decomposition.protocols.subdomain import (
    SubdomainSolverProtocol,
)


@runtime_checkable
class DomainDecompositionProtocol(Protocol, Generic[Array]):
    """Protocol for domain decomposition setup.

    Manages the full decomposition including:
    - Subdomain solvers
    - Interfaces between subdomains
    - Global interface DOF indexing

    The global interface DOF vector concatenates DOFs from all interfaces:
    [interface_0_dofs, interface_1_dofs, ..., interface_n_dofs]

    The interface_dof_offsets array stores the starting index for each
    interface in this global vector.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nsubdomains(self) -> int:
        """Return number of subdomains."""
        ...

    def ninterfaces(self) -> int:
        """Return number of interfaces."""
        ...

    def subdomain_solver(self, subdomain_id: int) -> SubdomainSolverProtocol[Array]:
        """Return subdomain solver for given subdomain.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.

        Returns
        -------
        SubdomainSolverProtocol
            Subdomain solver.
        """
        ...

    def interface(self, interface_id: int) -> InterfaceProtocol[Array]:
        """Return interface for given interface ID.

        Parameters
        ----------
        interface_id : int
            ID of interface.

        Returns
        -------
        InterfaceProtocol
            Interface object.
        """
        ...

    def interface_dof_offsets(self) -> Array:
        """Return starting index for each interface in global DOF vector.

        Returns
        -------
        Array
            DOF offsets. Shape: (ninterfaces + 1,)
            offsets[i] is starting index for interface i
            offsets[ninterfaces] is total number of DOFs
        """
        ...

    def total_interface_dofs(self) -> int:
        """Return total number of interface DOFs across all interfaces."""
        ...

    def extract_interface_dofs(
        self, global_dofs: Array, interface_id: int
    ) -> Array:
        """Extract DOFs for one interface from global DOF vector.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_interface_dofs,)
        interface_id : int
            ID of interface.

        Returns
        -------
        Array
            DOFs for this interface. Shape: (interface_ndofs,)
        """
        ...

    def set_interface_dofs(
        self, global_dofs: Array, interface_id: int, interface_dofs: Array
    ) -> Array:
        """Set DOFs for one interface in global DOF vector.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_interface_dofs,)
        interface_id : int
            ID of interface.
        interface_dofs : Array
            DOFs to set. Shape: (interface_ndofs,)

        Returns
        -------
        Array
            Modified global DOF vector. Shape: (total_interface_dofs,)
        """
        ...

"""DtN residual computation for domain decomposition.

The residual is the flux mismatch across all interfaces:
    R = sum_{interfaces} (flux_left + flux_right)

At convergence, R = 0 (flux conservation).
"""

from typing import Dict, Generic, List

from pyapprox.pde.decomposition.protocols.interface import InterfaceProtocol
from pyapprox.pde.decomposition.subdomain.wrapper import SubdomainWrapper
from pyapprox.util.backends.protocols import Array, Backend


class DtNResidual(Generic[Array]):
    """Computes DtN residual (flux mismatch) for Newton solver.

    The residual measures the flux mismatch across all interfaces.
    Given interface DOF coefficients λ:

    1. Set Dirichlet BCs on all interfaces using λ
    2. Solve all subdomain problems independently
    3. Compute normal flux from each side of each interface
    4. Return flux sum (should be zero for conservation)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    interfaces : Dict[int, InterfaceProtocol]
        All interfaces in the decomposition, keyed by interface ID.
    subdomain_solvers : Dict[int, SubdomainWrapper]
        All subdomain solvers, keyed by subdomain ID.
    interface_dof_offsets : Array
        Starting DOF index for each interface. Shape: (n_interfaces + 1,)
        interface_dof_offsets[i] is the first DOF index for interface i.
        interface_dof_offsets[-1] is the total number of interface DOFs.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        interfaces: Dict[int, InterfaceProtocol[Array]],
        subdomain_solvers: Dict[int, SubdomainWrapper[Array]],
        interface_dof_offsets: Array,
    ):
        self._bkd = bkd
        self._interfaces = interfaces
        self._subdomain_solvers = subdomain_solvers
        self._interface_dof_offsets = interface_dof_offsets

        # Build interface ID list (sorted for consistent ordering)
        self._interface_ids = sorted(interfaces.keys())
        self._n_interfaces = len(self._interface_ids)

        # Total interface DOFs
        self._total_dofs = self._bkd.to_int(interface_dof_offsets[-1])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def total_dofs(self) -> int:
        """Return total number of interface DOFs."""
        return self._total_dofs

    def interface_ids(self) -> List[int]:
        """Return ordered list of interface IDs."""
        return self._interface_ids

    def unpack_interface_dofs(self, global_dofs: Array) -> Dict[int, Array]:
        """Unpack global DOF vector into per-interface coefficients.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)

        Returns
        -------
        Dict[int, Array]
            Interface coefficients keyed by interface ID.
        """
        interface_coeffs = {}
        for i, interface_id in enumerate(self._interface_ids):
            start = self._bkd.to_int(self._interface_dof_offsets[i])
            end = self._bkd.to_int(self._interface_dof_offsets[i + 1])
            interface_coeffs[interface_id] = global_dofs[start:end]
        return interface_coeffs

    def pack_interface_dofs(self, interface_coeffs: Dict[int, Array]) -> Array:
        """Pack per-interface coefficients into global DOF vector.

        Parameters
        ----------
        interface_coeffs : Dict[int, Array]
            Interface coefficients keyed by interface ID.

        Returns
        -------
        Array
            Global interface DOF vector. Shape: (total_dofs,)
        """
        bkd = self._bkd
        global_dofs = bkd.zeros((self._total_dofs,))
        for i, interface_id in enumerate(self._interface_ids):
            start = self._bkd.to_int(self._interface_dof_offsets[i])
            self._bkd.to_int(self._interface_dof_offsets[i + 1])
            coeffs = interface_coeffs[interface_id]
            for j, val in enumerate(coeffs):
                global_dofs[start + j] = val
        return global_dofs

    def set_interface_dirichlet(self, global_dofs: Array) -> None:
        """Set Dirichlet BCs on all interfaces.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)
        """
        interface_coeffs = self.unpack_interface_dofs(global_dofs)

        # Set Dirichlet BCs on all subdomain solvers
        for solver in self._subdomain_solvers.values():
            for interface_id in solver.interface_ids():
                if interface_id in interface_coeffs:
                    solver.set_interface_dirichlet(
                        interface_id, interface_coeffs[interface_id]
                    )

    def solve_all_subdomains(self) -> Dict[int, Array]:
        """Solve all subdomain problems.

        Returns
        -------
        Dict[int, Array]
            Subdomain solutions keyed by subdomain ID.
        """
        solutions = {}
        for subdomain_id, solver in self._subdomain_solvers.items():
            solutions[subdomain_id] = solver.solve()
        return solutions

    def compute_interface_fluxes(
        self,
    ) -> Dict[int, tuple[Any, ...]]:
        """Compute fluxes on both sides of all interfaces.

        Returns
        -------
        Dict[int, tuple]
            For each interface: (flux_left, flux_right) where left/right
            correspond to subdomain_ids[0] and subdomain_ids[1].
        """
        fluxes = {}
        for interface_id, interface in self._interfaces.items():
            subdomain_ids = interface.subdomain_ids()
            left_id, right_id = subdomain_ids

            # Get flux from left subdomain
            flux_left = self._subdomain_solvers[left_id].compute_interface_flux(
                interface_id
            )

            # Get flux from right subdomain
            flux_right = self._subdomain_solvers[right_id].compute_interface_flux(
                interface_id
            )

            fluxes[interface_id] = (flux_left, flux_right)

        return fluxes

    def __call__(self, global_dofs: Array) -> Array:
        """Compute DtN residual (flux mismatch).

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)

        Returns
        -------
        Array
            Residual vector (flux mismatch). Shape: (total_dofs,)
        """
        # Set interface values and solve subdomains
        self.set_interface_dirichlet(global_dofs)
        self.solve_all_subdomains()

        # Compute flux mismatch on each interface
        fluxes = self.compute_interface_fluxes()

        # Pack into residual vector
        residual = self._bkd.zeros((self._total_dofs,))
        for i, interface_id in enumerate(self._interface_ids):
            start = self._bkd.to_int(self._interface_dof_offsets[i])
            end = self._bkd.to_int(self._interface_dof_offsets[i + 1])

            flux_left, flux_right = fluxes[interface_id]
            # Flux mismatch: should be zero for conservation
            flux_sum = flux_left + flux_right
            for j in range(end - start):
                residual[start + j] = flux_sum[j]

        return residual

    def residual_norm(self, global_dofs: Array) -> float:
        """Compute L2 norm of residual.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)

        Returns
        -------
        float
            L2 norm of residual.
        """
        residual = self(global_dofs)
        return self._bkd.to_float(self._bkd.norm(residual))

    def set_time(self, time: float) -> None:
        """Set the current time for time-dependent problems.

        Propagates to all subdomain solvers.

        Parameters
        ----------
        time : float
            Current time.
        """
        for solver in self._subdomain_solvers.values():
            solver.set_time(time)

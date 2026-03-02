"""DtN Jacobian computation for domain decomposition.

Computes the exact Jacobian of the DtN residual w.r.t. interface DOFs.
Each column requires solving subdomain problems with perturbed interface values.
"""

from typing import Generic

from pyapprox.pde.decomposition.solver.dtn_residual import DtNResidual
from pyapprox.util.backends.protocols import Array, Backend


class DtNJacobian(Generic[Array]):
    """Computes exact DtN Jacobian via finite differences.

    For each interface DOF j, computes d(residual)/d(lambda_j) by:
    1. Perturbing lambda_j by epsilon
    2. Solving all affected subdomains
    3. Computing flux derivatives

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    residual : DtNResidual
        Residual object (provides subdomain solvers and interfaces).
    epsilon : float, optional
        Finite difference step size. Default: 1e-7
    """

    def __init__(
        self,
        bkd: Backend[Array],
        residual: DtNResidual[Array],
        epsilon: float = 1e-7,
    ):
        self._bkd = bkd
        self._residual = residual
        self._epsilon = epsilon

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, global_dofs: Array) -> Array:
        """Compute DtN Jacobian via finite differences.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (total_dofs, total_dofs)
        """
        bkd = self._bkd
        n = self._residual.total_dofs()

        # Compute base residual
        base_residual = self._residual(global_dofs)

        # Compute Jacobian column by column
        jacobian = bkd.zeros((n, n))
        for j in range(n):
            # Perturb DOF j
            perturbed_dofs = bkd.copy(global_dofs)
            perturbed_dofs[j] = perturbed_dofs[j] + self._epsilon

            # Compute perturbed residual
            perturbed_residual = self._residual(perturbed_dofs)

            # Finite difference
            col = (perturbed_residual - base_residual) / self._epsilon
            for i in range(n):
                jacobian[i, j] = col[i]

        # Restore original state
        self._residual(global_dofs)

        return jacobian


class DtNJacobianExact(Generic[Array]):
    """Computes exact DtN Jacobian using subdomain flux Jacobians.

    More efficient than full finite differences when subdomain solvers
    provide flux Jacobians directly.

    For each interface DOF j, the Jacobian column is computed by:
    1. Computing how perturbing lambda_j affects subdomain solutions
    2. Computing how subdomain solutions affect interface fluxes
    3. Combining to get d(flux_mismatch)/d(lambda_j)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    residual : DtNResidual
        Residual object (provides subdomain solvers and interfaces).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        residual: DtNResidual[Array],
    ):
        self._bkd = bkd
        self._residual = residual

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, global_dofs: Array) -> Array:
        """Compute DtN Jacobian using flux Jacobians from subdomain solvers.

        Parameters
        ----------
        global_dofs : Array
            Global interface DOF vector. Shape: (total_dofs,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (total_dofs, total_dofs)
        """
        bkd = self._bkd
        n = self._residual.total_dofs()
        interfaces = self._residual._interfaces
        subdomain_solvers = self._residual._subdomain_solvers
        interface_ids = self._residual.interface_ids()
        offsets = self._residual._interface_dof_offsets

        # Ensure current state is set
        self._residual.set_interface_dirichlet(global_dofs)
        self._residual.solve_all_subdomains()

        # Initialize Jacobian
        jacobian = bkd.zeros((n, n))

        # For each interface where flux is computed
        for i_row, interface_id_row in enumerate(interface_ids):
            row_start = bkd.to_int(offsets[i_row])
            bkd.to_int(offsets[i_row + 1])
            interface = interfaces[interface_id_row]
            left_id, right_id = interface.subdomain_ids()

            # For each interface DOF that is perturbed
            for i_col, interface_id_col in enumerate(interface_ids):
                col_start = bkd.to_int(offsets[i_col])
                col_end = bkd.to_int(offsets[i_col + 1])
                n_dofs_col = col_end - col_start

                for local_dof in range(n_dofs_col):
                    global_dof = col_start + local_dof

                    # Compute flux Jacobian from left subdomain
                    flux_jac_left = subdomain_solvers[
                        left_id
                    ].compute_flux_jacobian_column(
                        interface_id_row, interface_id_col, local_dof
                    )

                    # Compute flux Jacobian from right subdomain
                    flux_jac_right = subdomain_solvers[
                        right_id
                    ].compute_flux_jacobian_column(
                        interface_id_row, interface_id_col, local_dof
                    )

                    # Sum of flux Jacobians
                    flux_jac_sum = flux_jac_left + flux_jac_right

                    # Fill Jacobian column
                    for k, val in enumerate(flux_jac_sum):
                        jacobian[row_start + k, global_dof] = val

        return jacobian


def create_jacobian(
    bkd: Backend[Array],
    residual: DtNResidual[Array],
    method: str = "finite_difference",
    epsilon: float = 1e-7,
):
    """Create a DtN Jacobian object.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    residual : DtNResidual
        Residual object.
    method : str, optional
        Jacobian computation method. Options:
        - "finite_difference": Column-by-column finite differences
        - "exact": Use subdomain flux Jacobians
        Default: "finite_difference"
    epsilon : float, optional
        Finite difference step size. Default: 1e-7

    Returns
    -------
    DtNJacobian or DtNJacobianExact
        Jacobian computation object.
    """
    if method == "finite_difference":
        return DtNJacobian(bkd, residual, epsilon)
    elif method == "exact":
        return DtNJacobianExact(bkd, residual)
    else:
        raise ValueError(f"Unknown Jacobian method: {method}")

"""Protocols for interface representation in domain decomposition.

Defines:
- InterfaceBasisProtocol: polynomial basis for interface function representation
- InterfaceProtocol: interface between two subdomains
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class InterfaceBasisProtocol(Protocol, Generic[Array]):
    """Protocol for interface function representation.

    The interface basis represents functions on an interface between subdomains
    using polynomial expansion. Corners are excluded from DOFs to avoid
    conflicts at multi-interface junctions.

    The basis maps from reference coordinates [-1, 1]^(ndim-1) to the
    physical interface geometry.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndofs(self) -> int:
        """Return number of interface DOFs per component.

        This excludes corner points to avoid conflicts at interfaces.
        For 1D interfaces (lines), corners are the endpoints.
        """
        ...

    def ncomponents(self) -> int:
        """Return number of solution components.

        For scalar PDEs, this returns 1.
        For vector PDEs (e.g., elasticity, reaction-diffusion), this returns
        the number of solution components (e.g., 2 for 2D elasticity).
        """
        ...

    def total_ndofs(self) -> int:
        """Return total number of interface DOFs across all components.

        Returns ndofs() * ncomponents().
        """
        ...

    def npts(self) -> int:
        """Return number of collocation/quadrature points on interface."""
        ...

    def reference_points(self) -> Array:
        """Return collocation points in reference coordinates.

        Returns
        -------
        Array
            Reference coordinates. Shape: (ndim_interface, npts)
            For 1D interface: (1, npts)
        """
        ...

    def physical_points(self) -> Array:
        """Return collocation points in physical coordinates.

        Returns
        -------
        Array
            Physical coordinates. Shape: (ndim, npts)
            For interface in 2D domain: (2, npts)
        """
        ...

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function at collocation points.

        For vector-valued PDEs, coefficients use component-stacked ordering:
        [comp0_dof0, ..., comp0_dofN, comp1_dof0, ..., comp1_dofN, ...]

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients. Shape: (total_ndofs,) = (ndofs * ncomponents,)

        Returns
        -------
        Array
            Function values at collocation points. Shape: (npts * ncomponents,)
            Component-stacked: [comp0_pt0, ..., comp0_ptN, comp1_pt0, ...]
        """
        ...

    def quadrature_weights(self) -> Array:
        """Return quadrature weights for integration on interface.

        Returns
        -------
        Array
            Quadrature weights. Shape: (npts,)
        """
        ...


@runtime_checkable
class InterfaceProtocol(Protocol, Generic[Array]):
    """Protocol for interface between two subdomains.

    An interface connects exactly two subdomains and manages:
    - Polynomial basis for interface function representation
    - Interpolation between interface basis and subdomain meshes
    - Normal vectors for each side of the interface
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def interface_id(self) -> int:
        """Return unique identifier for this interface."""
        ...

    def basis(self) -> InterfaceBasisProtocol[Array]:
        """Return the interface function basis."""
        ...

    def ncomponents(self) -> int:
        """Return number of solution components.

        Delegates to basis.ncomponents().
        """
        ...

    def total_ndofs(self) -> int:
        """Return total number of interface DOFs across all components.

        Delegates to basis.total_ndofs().
        """
        ...

    def subdomain_ids(self) -> Tuple[int, int]:
        """Return IDs of subdomains sharing this interface.

        Returns
        -------
        Tuple[int, int]
            (subdomain_id_0, subdomain_id_1) where subdomain_id_0 < subdomain_id_1
        """
        ...

    def physical_bounds(self) -> Tuple[Array, Array]:
        """Return start and end coordinates of interface.

        Returns
        -------
        Tuple[Array, Array]
            (start_coord, end_coord) each with shape (ndim,)
        """
        ...

    def normal(self, subdomain_id: int) -> Array:
        """Return outward normal vector for given subdomain.

        The normal points outward from the specified subdomain,
        i.e., into the adjacent subdomain.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain (must be one of subdomain_ids()).

        Returns
        -------
        Array
            Unit normal vector. Shape: (ndim,)
        """
        ...

    def interpolate_to_subdomain(
        self, subdomain_id: int, interface_values: Array
    ) -> Array:
        """Interpolate interface values to subdomain boundary nodes.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of target subdomain.
        interface_values : Array
            Values at interface collocation points.
            Shape: (npts * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at subdomain boundary nodes.
            Shape: (nboundary_pts * ncomponents,) with component-stacked ordering.
        """
        ...

    def restrict_from_subdomain(
        self, subdomain_id: int, boundary_values: Array
    ) -> Array:
        """Restrict subdomain boundary values to interface points.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of source subdomain.
        boundary_values : Array
            Values at subdomain boundary nodes.
            Shape: (nboundary_pts * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface collocation points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        ...

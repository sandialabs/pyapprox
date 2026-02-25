"""Interface class for domain decomposition.

Represents an interface between two subdomains, managing:
- Polynomial basis for interface function representation
- Interpolation between interface basis and subdomain meshes
- Normal vectors for flux computation
"""

from typing import TYPE_CHECKING, Dict, Generic, Optional, Tuple

from pyapprox.pde.decomposition.interface.interpolation import (
    InterpolationOperator,
    RestrictionOperator,
)
from pyapprox.pde.decomposition.protocols.interface import (
    InterfaceBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.pde.decomposition.interface.basis import (
        LegendreInterfaceBasis2D,
    )


class Interface1D(Generic[Array]):
    """Interface between two 1D subdomains (a single point).

    For 1D problems, the interface is a single point where two subdomains
    meet. There is 1 DOF per component (the value at the interface point).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    interface_id : int
        Unique identifier for this interface.
    subdomain_ids : Tuple[int, int]
        IDs of subdomains sharing this interface (left, right).
    interface_point : float
        Physical coordinate of the interface point.
    ncomponents : int, optional
        Number of solution components. Default 1 for scalar PDEs.
        For vector PDEs (e.g., elasticity), set to number of components.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        interface_id: int,
        subdomain_ids: Tuple[int, int],
        interface_point: float,
        ncomponents: int = 1,
    ):
        self._bkd = bkd
        self._id = interface_id
        self._subdomain_ids = subdomain_ids
        self._interface_point = interface_point
        self._ncomponents = ncomponents

        # Store interpolation operators per subdomain
        self._interp_to_subdomain: Dict[int, InterpolationOperator[Array]] = {}
        self._restrict_from_subdomain: Dict[int, RestrictionOperator[Array]] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def interface_id(self) -> int:
        """Return unique identifier for this interface."""
        return self._id

    def ndofs(self) -> int:
        """Return number of interface DOFs per component."""
        return 1

    def ncomponents(self) -> int:
        """Return number of solution components."""
        return self._ncomponents

    def total_ndofs(self) -> int:
        """Return total number of interface DOFs across all components."""
        return self.ndofs() * self._ncomponents

    def npts(self) -> int:
        """Return number of interface points."""
        return 1

    def subdomain_ids(self) -> Tuple[int, int]:
        """Return IDs of subdomains sharing this interface."""
        return self._subdomain_ids

    def physical_point(self) -> float:
        """Return physical coordinate of interface point."""
        return self._interface_point

    def physical_points(self) -> Array:
        """Return interface point as array.

        Returns
        -------
        Array
            Physical coordinate. Shape: (1, 1)
        """
        return self._bkd.asarray([[self._interface_point]])

    def normal(self, subdomain_id: int) -> Array:
        """Return outward normal for given subdomain.

        In 1D, the normal is +1 or -1.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.

        Returns
        -------
        Array
            Unit normal. Shape: (1,)
        """
        if subdomain_id not in self._subdomain_ids:
            raise ValueError(
                f"subdomain_id {subdomain_id} not in {self._subdomain_ids}"
            )

        # Convention: subdomain_ids[0] is on the left, normal points right (+1)
        # subdomain_ids[1] is on the right, normal points left (-1)
        if subdomain_id == self._subdomain_ids[0]:
            return self._bkd.asarray([1.0])  # Left domain, normal points right
        else:
            return self._bkd.asarray([-1.0])  # Right domain, normal points left

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function (for 1D, just return the coefficient).

        For vector-valued PDEs, coefficients use component-stacked ordering:
        [comp0_dof0, comp1_dof0, ...] -> [comp0_val, comp1_val, ...]

        Parameters
        ----------
        coeffs : Array
            Interface DOFs. Shape: (ncomponents,) for vector, (1,) for scalar.

        Returns
        -------
        Array
            Values at interface point. Shape: (ncomponents,) for vector, (1,) for
            scalar.
        """
        return coeffs

    def set_subdomain_boundary_points(
        self, subdomain_id: int, boundary_pts: Array
    ) -> None:
        """Set subdomain boundary points for interpolation.

        For 1D interface (single point), this creates trivial interpolation.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.
        boundary_pts : Array
            Physical coordinates of boundary points. Shape: (n_boundary,)
        """
        interface_pt = self._bkd.asarray([self._interface_point])

        self._interp_to_subdomain[subdomain_id] = InterpolationOperator(
            interface_pt, boundary_pts, self._bkd
        )
        self._restrict_from_subdomain[subdomain_id] = RestrictionOperator(
            interface_pt, boundary_pts, self._bkd
        )

    def interpolate_to_subdomain(
        self, subdomain_id: int, interface_values: Array
    ) -> Array:
        """Interpolate interface values to subdomain boundary.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of target subdomain.
        interface_values : Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at subdomain boundary nodes.
            Shape: (nboundary * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._interp_to_subdomain:
            raise ValueError(
                f"No interpolation set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points first."
            )
        if self._ncomponents == 1:
            return self._interp_to_subdomain[subdomain_id].apply(interface_values)

        # Handle multiple components
        npts = self.npts()
        interp = self._interp_to_subdomain[subdomain_id]
        n_target = interp.n_target()
        result = self._bkd.zeros((self._ncomponents * n_target,))

        for c in range(self._ncomponents):
            comp_vals = interface_values[c * npts : (c + 1) * npts]
            result[c * n_target : (c + 1) * n_target] = interp.apply(comp_vals)

        return result

    def restrict_from_subdomain(
        self, subdomain_id: int, boundary_values: Array
    ) -> Array:
        """Restrict subdomain boundary values to interface.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of source subdomain.
        boundary_values : Array
            Values at subdomain boundary nodes.
            Shape: (nboundary * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._restrict_from_subdomain:
            raise ValueError(
                f"No restriction set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points first."
            )

        if self._ncomponents == 1:
            return self._restrict_from_subdomain[subdomain_id].apply(boundary_values)

        # Handle multiple components
        npts = self.npts()
        restrict = self._restrict_from_subdomain[subdomain_id]
        n_source = restrict.n_source()
        result = self._bkd.zeros((self._ncomponents * npts,))

        for c in range(self._ncomponents):
            comp_vals = boundary_values[c * n_source : (c + 1) * n_source]
            result[c * npts : (c + 1) * npts] = restrict.apply(comp_vals)

        return result

    def __repr__(self) -> str:
        return (
            f"Interface1D(id={self._id}, subdomains={self._subdomain_ids}, "
            f"point={self._interface_point}, ncomponents={self._ncomponents})"
        )


class Interface(Generic[Array]):
    """Interface between two subdomains with polynomial basis.

    General interface class that wraps an InterfaceBasisProtocol and
    manages interpolation to/from subdomain boundaries.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    interface_id : int
        Unique identifier for this interface.
    subdomain_ids : Tuple[int, int]
        IDs of subdomains sharing this interface.
    basis : InterfaceBasisProtocol
        Polynomial basis for interface function representation.
    normal_direction : int
        Direction of interface normal (for computing outward normals).
        0 = x-direction, 1 = y-direction, etc.
    ambient_dim : int, optional
        Dimension of the ambient space. If None, inferred from basis.
        Use this for 2D/3D problems with lower-dimensional interfaces
        (e.g., 1D line interface in 2D domain).
    ncomponents : int, optional
        Number of solution components. Default 1 for scalar PDEs.
        For vector PDEs (e.g., elasticity), set to number of components.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        interface_id: int,
        subdomain_ids: Tuple[int, int],
        basis: InterfaceBasisProtocol[Array],
        normal_direction: int = 0,
        ambient_dim: Optional[int] = None,
        ncomponents: int = 1,
    ):
        self._bkd = bkd
        self._id = interface_id
        self._subdomain_ids = subdomain_ids
        self._basis = basis
        self._normal_direction = normal_direction
        self._ncomponents = ncomponents

        # Ambient dimension: from parameter or basis
        if ambient_dim is not None:
            self._ambient_dim = ambient_dim
        else:
            self._ambient_dim = basis.physical_points().shape[0]

        # Store interpolation operators per subdomain
        self._interp_to_subdomain: Dict[int, InterpolationOperator[Array]] = {}
        self._restrict_from_subdomain: Dict[int, RestrictionOperator[Array]] = {}

        # Store normal signs per subdomain
        self._normal_signs: Dict[int, float] = {
            subdomain_ids[0]: 1.0,  # First subdomain: normal points "positive"
            subdomain_ids[1]: -1.0,  # Second subdomain: normal points "negative"
        }

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def interface_id(self) -> int:
        """Return unique identifier for this interface."""
        return self._id

    def basis(self) -> InterfaceBasisProtocol[Array]:
        """Return the interface function basis."""
        return self._basis

    def ndofs(self) -> int:
        """Return number of interface DOFs per component."""
        return self._basis.ndofs()

    def ncomponents(self) -> int:
        """Return number of solution components."""
        return self._ncomponents

    def total_ndofs(self) -> int:
        """Return total number of interface DOFs across all components."""
        return self.ndofs() * self._ncomponents

    def npts(self) -> int:
        """Return number of interface points."""
        return self._basis.npts()

    def subdomain_ids(self) -> Tuple[int, int]:
        """Return IDs of subdomains sharing this interface."""
        return self._subdomain_ids

    def physical_points(self) -> Array:
        """Return physical coordinates of interface points.

        Returns
        -------
        Array
            Physical coordinates. Shape: (ndim, npts)
        """
        return self._basis.physical_points()

    def normal(self, subdomain_id: int) -> Array:
        """Return outward normal for given subdomain.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.

        Returns
        -------
        Array
            Unit normal vector. Shape: (ambient_dim,)
        """
        if subdomain_id not in self._subdomain_ids:
            raise ValueError(
                f"subdomain_id {subdomain_id} not in {self._subdomain_ids}"
            )

        # Build normal vector in the normal_direction
        normal = self._bkd.zeros((self._ambient_dim,))
        normal[self._normal_direction] = self._normal_signs[subdomain_id]
        return normal

    def set_normal_sign(self, subdomain_id: int, sign: float) -> None:
        """Override normal sign for a subdomain.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.
        sign : float
            Normal sign (+1 or -1).
        """
        if subdomain_id not in self._subdomain_ids:
            raise ValueError(
                f"subdomain_id {subdomain_id} not in {self._subdomain_ids}"
            )
        self._normal_signs[subdomain_id] = sign

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function at collocation points.

        For vector-valued PDEs, coefficients use component-stacked ordering:
        [comp0_dof0, ..., comp0_dofN, comp1_dof0, ..., comp1_dofN, ...]

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients.
            Shape: (ndofs * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        if self._ncomponents == 1:
            return self._basis.evaluate(coeffs)

        # Handle multiple components
        ndofs = self.ndofs()
        npts = self.npts()
        result = self._bkd.zeros((self._ncomponents * npts,))

        for c in range(self._ncomponents):
            comp_coeffs = coeffs[c * ndofs : (c + 1) * ndofs]
            result[c * npts : (c + 1) * npts] = self._basis.evaluate(comp_coeffs)

        return result

    def set_subdomain_boundary_points(
        self, subdomain_id: int, boundary_pts: Array
    ) -> None:
        """Set subdomain boundary points for interpolation.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.
        boundary_pts : Array
            Physical coordinates of boundary points along interface.
            Shape: (n_boundary,) for 1D interface parameter.
        """
        # Get interface points (in 1D parameter space for the interface)
        interface_pts = self._basis.physical_points()

        # For 1D interface (line in 2D), extract the varying coordinate
        # Assumes interface_pts shape is (ndim, npts)
        # and boundary_pts is the same coordinate extracted from subdomain mesh
        interface_1d = interface_pts[0, :]  # First coordinate (may need generalization)

        self._interp_to_subdomain[subdomain_id] = InterpolationOperator(
            interface_1d, boundary_pts, self._bkd
        )
        self._restrict_from_subdomain[subdomain_id] = RestrictionOperator(
            interface_1d, boundary_pts, self._bkd
        )

    def interpolate_to_subdomain(
        self, subdomain_id: int, interface_values: Array
    ) -> Array:
        """Interpolate interface values to subdomain boundary.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of target subdomain.
        interface_values : Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at subdomain boundary nodes.
            Shape: (nboundary * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._interp_to_subdomain:
            raise ValueError(
                f"No interpolation set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points first."
            )

        if self._ncomponents == 1:
            return self._interp_to_subdomain[subdomain_id].apply(interface_values)

        # Handle multiple components
        npts = self.npts()
        interp = self._interp_to_subdomain[subdomain_id]
        n_target = interp.n_target()
        result = self._bkd.zeros((self._ncomponents * n_target,))

        for c in range(self._ncomponents):
            comp_vals = interface_values[c * npts : (c + 1) * npts]
            result[c * n_target : (c + 1) * n_target] = interp.apply(comp_vals)

        return result

    def restrict_from_subdomain(
        self, subdomain_id: int, boundary_values: Array
    ) -> Array:
        """Restrict subdomain boundary values to interface.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of source subdomain.
        boundary_values : Array
            Values at subdomain boundary nodes.
            Shape: (nboundary * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._restrict_from_subdomain:
            raise ValueError(
                f"No restriction set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points first."
            )

        if self._ncomponents == 1:
            return self._restrict_from_subdomain[subdomain_id].apply(boundary_values)

        # Handle multiple components
        npts = self.npts()
        restrict = self._restrict_from_subdomain[subdomain_id]
        n_source = restrict.n_source()
        result = self._bkd.zeros((self._ncomponents * npts,))

        for c in range(self._ncomponents):
            comp_vals = boundary_values[c * n_source : (c + 1) * n_source]
            result[c * npts : (c + 1) * npts] = restrict.apply(comp_vals)

        return result

    def __repr__(self) -> str:
        return (
            f"Interface(id={self._id}, subdomains={self._subdomain_ids}, "
            f"ndofs={self.ndofs()}, ncomponents={self._ncomponents})"
        )


class Interface2D(Generic[Array]):
    """Interface between two 3D subdomains with 2D polynomial basis.

    For 3D domain decomposition, interfaces are 2D surfaces. This class
    handles tensor product interpolation between the 2D interface basis
    and the 2D boundary face of each subdomain.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    interface_id : int
        Unique identifier for this interface.
    subdomain_ids : Tuple[int, int]
        IDs of subdomains sharing this interface.
    basis : LegendreInterfaceBasis2D
        2D polynomial basis for interface function representation.
    normal_direction : int
        Direction of interface normal (0=x, 1=y, 2=z).
    ncomponents : int, optional
        Number of solution components. Default 1 for scalar PDEs.
        For vector PDEs (e.g., elasticity), set to number of components.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        interface_id: int,
        subdomain_ids: Tuple[int, int],
        basis: "LegendreInterfaceBasis2D[Array]",
        normal_direction: int = 0,
        ncomponents: int = 1,
    ):
        self._bkd = bkd
        self._id = interface_id
        self._subdomain_ids = subdomain_ids
        self._basis = basis
        self._normal_direction = normal_direction
        self._ncomponents = ncomponents
        self._ambient_dim = 3  # Always 3D for Interface2D

        # Store interpolation matrices per subdomain
        self._interp_to_subdomain: Dict[int, Array] = {}
        self._restrict_from_subdomain: Dict[int, Array] = {}

        # Store normal signs per subdomain
        self._normal_signs: Dict[int, float] = {
            subdomain_ids[0]: 1.0,
            subdomain_ids[1]: -1.0,
        }

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def interface_id(self) -> int:
        """Return unique identifier for this interface."""
        return self._id

    def basis(self) -> "LegendreInterfaceBasis2D[Array]":
        """Return the interface function basis."""
        return self._basis

    def ndofs(self) -> int:
        """Return number of interface DOFs per component."""
        return self._basis.ndofs()

    def ncomponents(self) -> int:
        """Return number of solution components."""
        return self._ncomponents

    def total_ndofs(self) -> int:
        """Return total number of interface DOFs across all components."""
        return self.ndofs() * self._ncomponents

    def npts(self) -> int:
        """Return number of interface points."""
        return self._basis.npts()

    def subdomain_ids(self) -> Tuple[int, int]:
        """Return IDs of subdomains sharing this interface."""
        return self._subdomain_ids

    def physical_points(self) -> Array:
        """Return physical coordinates of interface points.

        Returns
        -------
        Array
            Physical coordinates. Shape: (2, npts)
            Row 0: y-coordinates, Row 1: z-coordinates
        """
        return self._basis.physical_points()

    def normal(self, subdomain_id: int) -> Array:
        """Return outward normal for given subdomain.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.

        Returns
        -------
        Array
            Unit normal vector. Shape: (3,)
        """
        if subdomain_id not in self._subdomain_ids:
            raise ValueError(
                f"subdomain_id {subdomain_id} not in {self._subdomain_ids}"
            )

        normal = self._bkd.zeros((self._ambient_dim,))
        normal[self._normal_direction] = self._normal_signs[subdomain_id]
        return normal

    def set_normal_sign(self, subdomain_id: int, sign: float) -> None:
        """Override normal sign for a subdomain."""
        if subdomain_id not in self._subdomain_ids:
            raise ValueError(
                f"subdomain_id {subdomain_id} not in {self._subdomain_ids}"
            )
        self._normal_signs[subdomain_id] = sign

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function at collocation points.

        For vector-valued PDEs, coefficients use component-stacked ordering:
        [comp0_dof0, ..., comp0_dofN, comp1_dof0, ..., comp1_dofN, ...]

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients.
            Shape: (ndofs * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        if self._ncomponents == 1:
            return self._basis.evaluate(coeffs)

        # Handle multiple components
        ndofs = self.ndofs()
        npts = self.npts()
        result = self._bkd.zeros((self._ncomponents * npts,))

        for c in range(self._ncomponents):
            comp_coeffs = coeffs[c * ndofs : (c + 1) * ndofs]
            result[c * npts : (c + 1) * npts] = self._basis.evaluate(comp_coeffs)

        return result

    def set_subdomain_boundary_points_2d(
        self,
        subdomain_id: int,
        boundary_pts_dim1: Array,
        boundary_pts_dim2: Array,
    ) -> None:
        """Set subdomain boundary points for 2D interpolation.

        Parameters
        ----------
        subdomain_id : int
            ID of subdomain.
        boundary_pts_dim1 : Array
            First coordinate (e.g. y) of boundary points. Shape: (n_dim1,)
        boundary_pts_dim2 : Array
            Second coordinate (e.g. z) of boundary points. Shape: (n_dim2,)

        The subdomain boundary is assumed to be a tensor product grid
        of n_dim1 x n_dim2 points.
        """
        # Get interpolation matrix from basis to subdomain boundary
        interp_matrix = self._basis.interpolation_matrix_to_grid(
            boundary_pts_dim1, boundary_pts_dim2
        )
        self._interp_to_subdomain[subdomain_id] = interp_matrix

        # Restriction: pseudo-inverse for least squares fit
        # (M^T M)^-1 M^T
        MTM = interp_matrix.T @ interp_matrix
        self._restrict_from_subdomain[subdomain_id] = (
            self._bkd.inv(MTM) @ interp_matrix.T
        )

    def interpolate_to_subdomain(
        self, subdomain_id: int, interface_values: Array
    ) -> Array:
        """Interpolate interface values to subdomain boundary.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of target subdomain.
        interface_values : Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at subdomain boundary nodes.
            Shape: (n_boundary * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._interp_to_subdomain:
            raise ValueError(
                f"No interpolation set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points_2d first."
            )

        if self._ncomponents == 1:
            return self._interp_to_subdomain[subdomain_id] @ interface_values

        # Handle multiple components
        npts = self.npts()
        interp_matrix = self._interp_to_subdomain[subdomain_id]
        n_target = interp_matrix.shape[0]
        result = self._bkd.zeros((self._ncomponents * n_target,))

        for c in range(self._ncomponents):
            comp_vals = interface_values[c * npts : (c + 1) * npts]
            result[c * n_target : (c + 1) * n_target] = interp_matrix @ comp_vals

        return result

    def restrict_from_subdomain(
        self, subdomain_id: int, boundary_values: Array
    ) -> Array:
        """Restrict subdomain boundary values to interface.

        For vector-valued PDEs, handles each component separately using
        component-stacked ordering.

        Parameters
        ----------
        subdomain_id : int
            ID of source subdomain.
        boundary_values : Array
            Values at subdomain boundary nodes.
            Shape: (n_boundary * ncomponents,) with component-stacked ordering.

        Returns
        -------
        Array
            Values at interface points.
            Shape: (npts * ncomponents,) with component-stacked ordering.
        """
        if subdomain_id not in self._restrict_from_subdomain:
            raise ValueError(
                f"No restriction set up for subdomain {subdomain_id}. "
                "Call set_subdomain_boundary_points_2d first."
            )

        if self._ncomponents == 1:
            return self._restrict_from_subdomain[subdomain_id] @ boundary_values

        # Handle multiple components
        npts = self.npts()
        restrict_matrix = self._restrict_from_subdomain[subdomain_id]
        n_source = restrict_matrix.shape[1]
        result = self._bkd.zeros((self._ncomponents * npts,))

        for c in range(self._ncomponents):
            comp_vals = boundary_values[c * n_source : (c + 1) * n_source]
            result[c * npts : (c + 1) * npts] = restrict_matrix @ comp_vals

        return result

    def __repr__(self) -> str:
        return (
            f"Interface2D(id={self._id}, subdomains={self._subdomain_ids}, "
            f"ndofs={self.ndofs()}, ncomponents={self._ncomponents})"
        )

"""Polynomial basis for interface function representation.

Provides LegendreInterfaceBasis1D which uses Legendre polynomials
with Gauss-Legendre quadrature points (strictly interior to [-1,1]).
"""

from typing import Generic, Tuple

from pyapprox.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
)
from pyapprox.util.backends.protocols import Array, Backend


class LegendreInterfaceBasis1D(Generic[Array]):
    """Legendre polynomial basis for 1D interface functions.

    Uses Gauss-Legendre quadrature points which are strictly interior to
    [-1, 1], avoiding corner points. This provides optimal polynomial
    integration accuracy.

    For n Gauss-Legendre points:
    - Integrates polynomials of degree 2n-1 exactly
    - All points strictly interior (no corners)
    - n DOFs for nodal basis

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    degree : int
        Maximum polynomial degree. Uses degree+1 Gauss points for
        exact integration of polynomials up to degree 2*degree+1.
        Must be >= 1 for at least 1 interior DOF.
    physical_bounds : Tuple[float, float]
        Physical coordinate bounds (start, end) of the interface.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        degree: int,
        physical_bounds: Tuple[float, float],
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1 for at least 1 DOF, got {degree}")

        self._bkd = bkd
        self._degree = degree
        self._physical_bounds = physical_bounds

        # Create Legendre polynomial basis
        self._legendre = LegendrePolynomial1D(bkd)
        self._legendre.set_nterms(degree + 1)

        # Get Gauss-Legendre points (all strictly interior to [-1, 1])
        # n points integrates polynomials of degree 2n-1 exactly
        npts = degree + 1
        gauss_pts, gauss_wts = self._legendre.gauss_quadrature_rule(npts)

        # Reference points: gauss_pts has shape (1, npoints)
        # gauss_wts has shape (npoints, 1)
        self._ref_points = gauss_pts[0, :]  # Shape: (npts,)
        self._ref_weights = gauss_wts[:, 0]  # Shape: (npts,)

        self._ndofs = npts  # All points are DOFs (no excluded corners)
        self._npts = npts

        # Map to physical coordinates
        start, end = physical_bounds
        self._physical_pts = self._map_to_physical(self._ref_points)

        # Scale weights for physical domain
        # The reference weights sum to 1 (probability normalization on [-1,1])
        # For integration, we need weights that sum to domain length (end - start)
        # So multiply by (end - start) to get physical integration weights
        self._physical_weights = self._ref_weights * (end - start)

        # Build Lagrange interpolation matrix for nodal basis
        # Each row is the Lagrange basis function evaluated at that point
        # For nodal basis, this is identity matrix
        self._lagrange_matrix = self._build_lagrange_matrix()

    def _map_to_physical(self, ref_coords: Array) -> Array:
        """Map reference coordinates [-1, 1] to physical coordinates."""
        start, end = self._physical_bounds
        return start + (ref_coords + 1.0) * (end - start) / 2.0

    def _map_to_reference(self, physical_coords: Array) -> Array:
        """Map physical coordinates to reference coordinates [-1, 1]."""
        start, end = self._physical_bounds
        return 2.0 * (physical_coords - start) / (end - start) - 1.0

    def _build_lagrange_matrix(self) -> Array:
        """Build Lagrange interpolation matrix.

        For nodal basis at Gauss-Lobatto interior points, compute
        L[i, j] = l_j(x_i) where l_j is the j-th Lagrange basis function.

        For nodal DOFs, this is simply the identity matrix.
        """
        return self._bkd.eye(self._ndofs)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndofs(self) -> int:
        """Return number of interface DOFs (excludes corners)."""
        return self._ndofs

    def ncomponents(self) -> int:
        """Return number of solution components (always 1 for basis)."""
        return 1

    def total_ndofs(self) -> int:
        """Return total number of DOFs (same as ndofs for basis)."""
        return self._ndofs

    def npts(self) -> int:
        """Return number of collocation points on interface."""
        return self._npts

    def degree(self) -> int:
        """Return polynomial degree."""
        return self._degree

    def reference_points(self) -> Array:
        """Return collocation points in reference coordinates [-1, 1].

        Returns
        -------
        Array
            Reference coordinates. Shape: (1, npts)
        """
        return self._ref_points[None, :]

    def physical_points(self) -> Array:
        """Return collocation points in physical coordinates.

        Returns
        -------
        Array
            Physical coordinates. Shape: (1, npts)
            Note: First dimension is 1 for 1D interface.
        """
        return self._physical_pts[None, :]

    def quadrature_weights(self) -> Array:
        """Return quadrature weights for integration on interface.

        Weights are scaled for physical domain integration.

        Returns
        -------
        Array
            Quadrature weights. Shape: (npts,)
        """
        return self._physical_weights

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function at collocation points.

        For nodal basis, coefficients are function values at nodes,
        so evaluation simply returns the coefficients.

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients (nodal values). Shape: (ndofs,)

        Returns
        -------
        Array
            Function values at collocation points. Shape: (npts,)
        """
        if coeffs.shape[0] != self._ndofs:
            raise ValueError(
                f"coeffs shape {coeffs.shape} must match ndofs ({self._ndofs},)"
            )
        # For nodal basis with collocation at nodes, evaluation is identity
        return coeffs

    def evaluate_at_points(self, coeffs: Array, physical_pts: Array) -> Array:
        """Evaluate interface function at arbitrary physical points.

        Uses Lagrange interpolation to evaluate the interface function
        represented by nodal coefficients at arbitrary points.

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients (nodal values). Shape: (ndofs,)
        physical_pts : Array
            Physical coordinates for evaluation. Shape: (neval,)

        Returns
        -------
        Array
            Function values at evaluation points. Shape: (neval,)
        """
        if coeffs.shape[0] != self._ndofs:
            raise ValueError(
                f"coeffs shape {coeffs.shape} must match ndofs ({self._ndofs},)"
            )

        # Map to reference coordinates
        ref_pts = self._map_to_reference(physical_pts)

        # Evaluate Lagrange basis at these points
        # L[i, j] = l_j(ref_pts[i]) where l_j is j-th Lagrange basis
        neval = physical_pts.shape[0]
        lagrange_vals = self._bkd.zeros((neval, self._ndofs))

        # Lagrange basis functions:
        # l_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
        for j in range(self._ndofs):
            l_j = self._bkd.ones((neval,))
            for k in range(self._ndofs):
                if k != j:
                    l_j = (
                        l_j
                        * (ref_pts - self._ref_points[k])
                        / (self._ref_points[j] - self._ref_points[k])
                    )
            lagrange_vals[:, j] = l_j

        # Interpolated values: sum_j coeffs[j] * l_j(x)
        return lagrange_vals @ coeffs

    def interpolation_matrix_to_points(self, physical_pts: Array) -> Array:
        """Compute interpolation matrix from DOFs to arbitrary points.

        Returns matrix M such that values = M @ coeffs.

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates for interpolation. Shape: (neval,)

        Returns
        -------
        Array
            Interpolation matrix. Shape: (neval, ndofs)
        """
        ref_pts = self._map_to_reference(physical_pts)
        neval = physical_pts.shape[0]
        interp_matrix = self._bkd.zeros((neval, self._ndofs))

        for j in range(self._ndofs):
            l_j = self._bkd.ones((neval,))
            for k in range(self._ndofs):
                if k != j:
                    l_j = (
                        l_j
                        * (ref_pts - self._ref_points[k])
                        / (self._ref_points[j] - self._ref_points[k])
                    )
            interp_matrix[:, j] = l_j

        return interp_matrix

    def integrate(self, values: Array) -> float:
        """Integrate function values using quadrature.

        Parameters
        ----------
        values : Array
            Function values at collocation points. Shape: (npts,)

        Returns
        -------
        float
            Integral over interface.
        """
        return self._bkd.to_float(self._bkd.sum(values * self._physical_weights))

    def __repr__(self) -> str:
        return (
            f"LegendreInterfaceBasis1D(degree={self._degree}, "
            f"ndofs={self._ndofs}, bounds={self._physical_bounds})"
        )


class LegendreInterfaceBasis2D(Generic[Array]):
    """Tensor product Legendre polynomial basis for 2D interface functions.

    Uses Gauss-Legendre quadrature points in each direction which are strictly
    interior to [-1, 1], avoiding corner and edge points.

    For n_y x n_z points:
    - Total DOFs: n_y * n_z
    - All points strictly interior (no edges/corners)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    degree_y : int
        Maximum polynomial degree in y-direction. Uses degree+1 points.
    degree_z : int
        Maximum polynomial degree in z-direction. Uses degree+1 points.
    physical_bounds_y : Tuple[float, float]
        Physical coordinate bounds (start, end) in y-direction.
    physical_bounds_z : Tuple[float, float]
        Physical coordinate bounds (start, end) in z-direction.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        degree_y: int,
        degree_z: int,
        physical_bounds_y: Tuple[float, float],
        physical_bounds_z: Tuple[float, float],
    ):
        if degree_y < 1 or degree_z < 1:
            raise ValueError(
                f"degrees must be >= 1, got degree_y={degree_y}, degree_z={degree_z}"
            )

        self._bkd = bkd
        self._degree_y = degree_y
        self._degree_z = degree_z
        self._physical_bounds_y = physical_bounds_y
        self._physical_bounds_z = physical_bounds_z

        # Create 1D bases
        self._basis_y = LegendreInterfaceBasis1D(bkd, degree_y, physical_bounds_y)
        self._basis_z = LegendreInterfaceBasis1D(bkd, degree_z, physical_bounds_z)

        self._npts_y = self._basis_y.npts()
        self._npts_z = self._basis_z.npts()
        self._npts = self._npts_y * self._npts_z
        self._ndofs = self._npts  # Nodal basis

        # Build tensor product grid
        pts_y = self._basis_y.physical_points()[0, :]  # Shape: (npts_y,)
        pts_z = self._basis_z.physical_points()[0, :]  # Shape: (npts_z,)

        # Physical points: (2, npts_y * npts_z) in (y, z) order
        # Ordering: z varies fastest (j*npts_z + k for point (y_j, z_k))
        self._physical_pts = bkd.zeros((2, self._npts))
        for j in range(self._npts_y):
            for k in range(self._npts_z):
                idx = j * self._npts_z + k
                self._physical_pts[0, idx] = pts_y[j]
                self._physical_pts[1, idx] = pts_z[k]

        # Quadrature weights (tensor product)
        wts_y = self._basis_y.quadrature_weights()
        wts_z = self._basis_z.quadrature_weights()
        self._physical_weights = bkd.zeros((self._npts,))
        for j in range(self._npts_y):
            for k in range(self._npts_z):
                idx = j * self._npts_z + k
                self._physical_weights[idx] = wts_y[j] * wts_z[k]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndofs(self) -> int:
        """Return number of interface DOFs."""
        return self._ndofs

    def ncomponents(self) -> int:
        """Return number of solution components (always 1 for basis)."""
        return 1

    def total_ndofs(self) -> int:
        """Return total number of DOFs (same as ndofs for basis)."""
        return self._ndofs

    def npts(self) -> int:
        """Return number of collocation points on interface."""
        return self._npts

    def npts_y(self) -> int:
        """Return number of points in y-direction."""
        return self._npts_y

    def npts_z(self) -> int:
        """Return number of points in z-direction."""
        return self._npts_z

    def physical_points(self) -> Array:
        """Return collocation points in physical coordinates.

        Returns
        -------
        Array
            Physical coordinates. Shape: (2, npts)
            Row 0: y-coordinates, Row 1: z-coordinates
        """
        return self._physical_pts

    def quadrature_weights(self) -> Array:
        """Return quadrature weights for integration on interface.

        Returns
        -------
        Array
            Quadrature weights. Shape: (npts,)
        """
        return self._physical_weights

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate interface function at collocation points.

        For nodal basis, coefficients are function values at nodes.

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients (nodal values). Shape: (ndofs,)

        Returns
        -------
        Array
            Function values at collocation points. Shape: (npts,)
        """
        if coeffs.shape[0] != self._ndofs:
            raise ValueError(
                f"coeffs shape {coeffs.shape} must match ndofs ({self._ndofs},)"
            )
        return coeffs

    def evaluate_at_points(
        self, coeffs: Array, physical_pts_y: Array, physical_pts_z: Array
    ) -> Array:
        """Evaluate interface function at arbitrary physical points.

        Uses tensor product Lagrange interpolation.

        Parameters
        ----------
        coeffs : Array
            Interface DOF coefficients (nodal values). Shape: (ndofs,)
        physical_pts_y : Array
            Y-coordinates for evaluation. Shape: (n_eval_y,)
        physical_pts_z : Array
            Z-coordinates for evaluation. Shape: (n_eval_z,)

        Returns
        -------
        Array
            Function values at tensor product of eval points.
            Shape: (n_eval_y * n_eval_z,)
        """
        n_eval_y = physical_pts_y.shape[0]
        n_eval_z = physical_pts_z.shape[0]
        n_eval = n_eval_y * n_eval_z

        # Get 1D interpolation matrices
        interp_y = self._basis_y.interpolation_matrix_to_points(physical_pts_y)
        interp_z = self._basis_z.interpolation_matrix_to_points(physical_pts_z)

        # Reshape coeffs to 2D grid: (npts_y, npts_z)
        coeffs_2d = coeffs.reshape((self._npts_y, self._npts_z))

        # Apply tensor product interpolation
        # First interpolate in y: result shape (n_eval_y, npts_z)
        temp = interp_y @ coeffs_2d
        # Then interpolate in z: result shape (n_eval_y, n_eval_z)
        result_2d = temp @ interp_z.T

        return result_2d.reshape((n_eval,))

    def interpolation_matrix_to_grid(
        self, target_pts_y: Array, target_pts_z: Array
    ) -> Array:
        """Compute interpolation matrix from DOFs to tensor product grid.

        Parameters
        ----------
        target_pts_y : Array
            Y-coordinates for interpolation. Shape: (n_target_y,)
        target_pts_z : Array
            Z-coordinates for interpolation. Shape: (n_target_z,)

        Returns
        -------
        Array
            Interpolation matrix. Shape: (n_target_y * n_target_z, ndofs)
        """
        bkd = self._bkd
        n_target_y = target_pts_y.shape[0]
        n_target_z = target_pts_z.shape[0]
        n_target = n_target_y * n_target_z

        # Get 1D interpolation matrices
        interp_y = self._basis_y.interpolation_matrix_to_points(target_pts_y)
        interp_z = self._basis_z.interpolation_matrix_to_points(target_pts_z)

        # Build full matrix: M[i, j] where i = ty*n_target_z + tz, j = sy*npts_z + sz
        # M = kron(interp_y, interp_z) for z-varies-fastest ordering
        full_matrix = bkd.zeros((n_target, self._ndofs))
        for ty in range(n_target_y):
            for tz in range(n_target_z):
                target_idx = ty * n_target_z + tz
                for sy in range(self._npts_y):
                    for sz in range(self._npts_z):
                        source_idx = sy * self._npts_z + sz
                        full_matrix[target_idx, source_idx] = (
                            interp_y[ty, sy] * interp_z[tz, sz]
                        )

        return full_matrix

    def __repr__(self) -> str:
        return (
            f"LegendreInterfaceBasis2D(degree_y={self._degree_y}, "
            f"degree_z={self._degree_z}, ndofs={self._ndofs})"
        )

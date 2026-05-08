"""Affine (scale and translate) coordinate transforms.

Provides linear mappings from reference domain [-1, 1]^d to physical domains.
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class AffineTransform1D(Generic[Array]):
    """Affine transform for 1D domains.

    Maps reference interval [-1, 1] to physical interval [a, b].

    Parameters
    ----------
    physical_bounds : Tuple[float, float]
        Physical domain bounds (a, b).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physical_bounds: Tuple[float, float],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._a, self._b = physical_bounds
        # Scale and shift: x = (b-a)/2 * xi + (a+b)/2
        self._scale = (self._b - self._a) / 2.0
        self._shift = (self._a + self._b) / 2.0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 1

    def physical_bounds(self) -> Tuple[float, float]:
        """Return physical domain bounds."""
        return (self._a, self._b)

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference [-1, 1] to physical [a, b].

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Physical coordinates. Shape: (1, npts)
        """
        return self._scale * reference_pts + self._shift

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical [a, b] to reference [-1, 1].

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Reference coordinates. Shape: (1, npts)
        """
        return (physical_pts - self._shift) / self._scale

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix (constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 1, 1)
        """
        npts = reference_pts.shape[1]
        jac = self._bkd.full((npts, 1, 1), self._scale)
        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant (constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        npts = reference_pts.shape[1]
        return self._bkd.full((npts,), self._scale)

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors (constant for affine).

        For affine transforms, scale factors are the diagonal Jacobian entries.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 1)
        """
        npts = reference_pts.shape[1]
        return self._bkd.full((npts, 1), self._scale)

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit curvilinear basis vectors (identity for Cartesian).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 1, 1)
        """
        npts = reference_pts.shape[1]
        return self._bkd.ones((npts, 1, 1))

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute gradient factors for coordinate transformation.

        For affine transforms, gradient factors are 1/scale on the diagonal.
        This converts reference derivatives to physical derivatives:
            d/dx_phys = (1/scale) * d/d_xi_ref

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (1, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, 1, 1)
            gradient_factors[i, d, j] = factor to multiply j-th reference
            derivative to get d-th physical derivative at point i.
        """
        npts = reference_pts.shape[1]
        return self._bkd.full((npts, 1, 1), 1.0 / self._scale)


class AffineTransform2D(Generic[Array]):
    """Affine transform for 2D domains.

    Maps reference square [-1, 1]^2 to physical rectangle [ax, bx] x [ay, by].

    Parameters
    ----------
    physical_bounds : Tuple[float, float, float, float]
        Physical domain bounds (ax, bx, ay, by).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physical_bounds: Tuple[float, float, float, float],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._ax, self._bx, self._ay, self._by = physical_bounds
        # Scales per dimension
        self._scale_x = (self._bx - self._ax) / 2.0
        self._scale_y = (self._by - self._ay) / 2.0
        self._shift_x = (self._ax + self._bx) / 2.0
        self._shift_y = (self._ay + self._by) / 2.0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2

    def physical_bounds(self) -> Tuple[float, float, float, float]:
        """Return physical domain bounds (ax, bx, ay, by)."""
        return (self._ax, self._bx, self._ay, self._by)

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference [-1, 1]^2 to physical domain.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Physical coordinates. Shape: (2, npts)
        """
        result = self._bkd.zeros(reference_pts.shape)
        result[0, :] = self._scale_x * reference_pts[0, :] + self._shift_x
        result[1, :] = self._scale_y * reference_pts[1, :] + self._shift_y
        return result

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical domain to reference [-1, 1]^2.

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Reference coordinates. Shape: (2, npts)
        """
        result = self._bkd.zeros(physical_pts.shape)
        result[0, :] = (physical_pts[0, :] - self._shift_x) / self._scale_x
        result[1, :] = (physical_pts[1, :] - self._shift_y) / self._scale_y
        return result

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix (diagonal, constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 2, 2)
        """
        npts = reference_pts.shape[1]
        jac = self._bkd.zeros((npts, 2, 2))
        jac[:, 0, 0] = self._scale_x
        jac[:, 1, 1] = self._scale_y
        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant (constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        npts = reference_pts.shape[1]
        det = self._scale_x * self._scale_y
        return self._bkd.full((npts,), det)

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors (constant for affine).

        For affine transforms, scale factors are the diagonal Jacobian entries.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 2)
        """
        npts = reference_pts.shape[1]
        scales = self._bkd.zeros((npts, 2))
        scales[:, 0] = self._scale_x
        scales[:, 1] = self._scale_y
        return scales

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit curvilinear basis vectors (identity for Cartesian).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 2, 2)
        """
        npts = reference_pts.shape[1]
        basis = self._bkd.zeros((npts, 2, 2))
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        return basis

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute gradient factors for coordinate transformation.

        For affine transforms, gradient factors are 1/scale on the diagonal.
        This converts reference derivatives to physical derivatives:
            d/dx_phys = (1/scale_x) * d/d_xi_ref
            d/dy_phys = (1/scale_y) * d/d_eta_ref

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (2, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, 2, 2)
            gradient_factors[i, d, j] = factor to multiply j-th reference
            derivative to get d-th physical derivative at point i.
        """
        npts = reference_pts.shape[1]
        factors = self._bkd.zeros((npts, 2, 2))
        factors[:, 0, 0] = 1.0 / self._scale_x
        factors[:, 1, 1] = 1.0 / self._scale_y
        return factors


class AffineTransform3D(Generic[Array]):
    """Affine transform for 3D domains.

    Maps reference cube [-1, 1]^3 to physical box
    [ax, bx] x [ay, by] x [az, bz].

    Parameters
    ----------
    physical_bounds : Tuple[float, float, float, float, float, float]
        Physical domain bounds (ax, bx, ay, by, az, bz).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physical_bounds: Tuple[float, float, float, float, float, float],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        (
            self._ax,
            self._bx,
            self._ay,
            self._by,
            self._az,
            self._bz,
        ) = physical_bounds
        # Scales per dimension
        self._scale_x = (self._bx - self._ax) / 2.0
        self._scale_y = (self._by - self._ay) / 2.0
        self._scale_z = (self._bz - self._az) / 2.0
        self._shift_x = (self._ax + self._bx) / 2.0
        self._shift_y = (self._ay + self._by) / 2.0
        self._shift_z = (self._az + self._bz) / 2.0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 3

    def physical_bounds(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Return physical domain bounds (ax, bx, ay, by, az, bz)."""
        return (
            self._ax,
            self._bx,
            self._ay,
            self._by,
            self._az,
            self._bz,
        )

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference [-1, 1]^3 to physical domain.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Physical coordinates. Shape: (3, npts)
        """
        result = self._bkd.zeros(reference_pts.shape)
        result[0, :] = self._scale_x * reference_pts[0, :] + self._shift_x
        result[1, :] = self._scale_y * reference_pts[1, :] + self._shift_y
        result[2, :] = self._scale_z * reference_pts[2, :] + self._shift_z
        return result

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical domain to reference [-1, 1]^3.

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Reference coordinates. Shape: (3, npts)
        """
        result = self._bkd.zeros(physical_pts.shape)
        result[0, :] = (physical_pts[0, :] - self._shift_x) / self._scale_x
        result[1, :] = (physical_pts[1, :] - self._shift_y) / self._scale_y
        result[2, :] = (physical_pts[2, :] - self._shift_z) / self._scale_z
        return result

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix (diagonal, constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 3, 3)
        """
        npts = reference_pts.shape[1]
        jac = self._bkd.zeros((npts, 3, 3))
        jac[:, 0, 0] = self._scale_x
        jac[:, 1, 1] = self._scale_y
        jac[:, 2, 2] = self._scale_z
        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant (constant for affine).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        npts = reference_pts.shape[1]
        det = self._scale_x * self._scale_y * self._scale_z
        return self._bkd.full((npts,), det)

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors (constant for affine).

        For affine transforms, scale factors are the diagonal Jacobian entries.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, 3)
        """
        npts = reference_pts.shape[1]
        scales = self._bkd.zeros((npts, 3))
        scales[:, 0] = self._scale_x
        scales[:, 1] = self._scale_y
        scales[:, 2] = self._scale_z
        return scales

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit curvilinear basis vectors (identity for Cartesian).

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, 3, 3)
        """
        npts = reference_pts.shape[1]
        basis = self._bkd.zeros((npts, 3, 3))
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        basis[:, 2, 2] = 1.0
        return basis

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute gradient factors for coordinate transformation.

        For affine transforms, gradient factors are 1/scale on the diagonal.
        This converts reference derivatives to physical derivatives:
            d/dx_phys = (1/scale_x) * d/d_xi_ref
            d/dy_phys = (1/scale_y) * d/d_eta_ref
            d/dz_phys = (1/scale_z) * d/d_zeta_ref

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (3, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, 3, 3)
            gradient_factors[i, d, j] = factor to multiply j-th reference
            derivative to get d-th physical derivative at point i.
        """
        npts = reference_pts.shape[1]
        factors = self._bkd.zeros((npts, 3, 3))
        factors[:, 0, 0] = 1.0 / self._scale_x
        factors[:, 1, 1] = 1.0 / self._scale_y
        factors[:, 2, 2] = 1.0 / self._scale_z
        return factors

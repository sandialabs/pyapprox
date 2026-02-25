"""Chained coordinate transforms for composing multiple transforms.

A ChainedTransform composes multiple transforms in sequence:
    physical = T_n(T_{n-1}(...T_1(reference)...))

The Jacobian of the composition is the product of individual Jacobians.
"""

from typing import Generic, List

from pyapprox.util.backends.protocols import Array, Backend


class ChainedTransform(Generic[Array]):
    """Compose multiple transforms in sequence.

    Given transforms [T1, T2, ..., Tn], maps:
        reference -> T1 -> T2 -> ... -> Tn -> physical

    The forward map is: physical = Tn(Tn-1(...T1(reference)...))
    The inverse map is: reference = T1^{-1}(T2^{-1}(...Tn^{-1}(physical)...))

    Parameters
    ----------
    transforms : List
        List of transform objects to compose. Each must have:
        - map_to_physical(pts) method
        - map_to_reference(pts) method
        - jacobian_matrix(pts) method
        - jacobian_determinant(pts) method
        - ndim() method
        All transforms must have the same ndim.
    bkd : Backend
        Computational backend.

    Notes
    -----
    The Jacobian of the composition is the product of individual Jacobians:
        J_total = J_n @ J_{n-1} @ ... @ J_1

    For curvilinear methods (scale_factors, unit_curvilinear_basis),
    the composition multiplies factors and transforms bases.
    """

    def __init__(
        self,
        transforms: List,
        bkd: Backend[Array],
    ):
        if len(transforms) == 0:
            raise ValueError("At least one transform is required")

        ndim = transforms[0].ndim()
        for ii, transform in enumerate(transforms):
            if transform.ndim() != ndim:
                raise ValueError(
                    f"Transform {ii} has ndim={transform.ndim()}, expected {ndim}"
                )

        self._transforms = transforms
        self._bkd = bkd
        self._ndim = ndim

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._ndim

    def transforms(self) -> List:
        """Return the list of transforms."""
        return self._transforms

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference to physical through all transforms.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Physical coordinates. Shape: (ndim, npts)
        """
        pts = reference_pts
        for transform in self._transforms:
            pts = transform.map_to_physical(pts)
        return pts

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical to reference through all transforms (reversed).

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Reference coordinates. Shape: (ndim, npts)
        """
        pts = physical_pts
        for transform in reversed(self._transforms):
            pts = transform.map_to_reference(pts)
        return pts

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the composed mapping.

        J_total = J_n @ J_{n-1} @ ... @ J_1

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, ndim, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self._ndim

        # Start with identity
        jac_total = self._bkd.zeros((npts, ndim, ndim))
        for ii in range(ndim):
            jac_total[:, ii, ii] = 1.0

        pts = reference_pts
        for transform in self._transforms:
            jac_i = transform.jacobian_matrix(pts)
            # J_total = J_i @ J_total (matrix multiplication per point)
            jac_total = self._bkd.einsum("nij,njk->nik", jac_i, jac_total)
            pts = transform.map_to_physical(pts)

        return jac_total

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant of the composed mapping.

        det(J_total) = det(J_n) * det(J_{n-1}) * ... * det(J_1)

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        npts = reference_pts.shape[1]
        det_total = self._bkd.ones((npts,))

        pts = reference_pts
        for transform in self._transforms:
            det_total = det_total * transform.jacobian_determinant(pts)
            pts = transform.map_to_physical(pts)

        return det_total

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for the composed transform.

        The scale factors multiply: h_total_i = h1_i * h2_i * ... * hn_i

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self._ndim
        scale_total = self._bkd.ones((npts, ndim))

        pts = reference_pts
        for transform in self._transforms:
            if hasattr(transform, "scale_factors"):
                scale_total = scale_total * transform.scale_factors(pts)
            pts = transform.map_to_physical(pts)

        return scale_total

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit vectors in curvilinear coordinates.

        For composed transforms, the bases multiply:
            basis_total = basis_n @ basis_{n-1} @ ... @ basis_1

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, ndim, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self._ndim

        # Start with identity
        basis_total = self._bkd.zeros((npts, ndim, ndim))
        for ii in range(ndim):
            basis_total[:, ii, ii] = 1.0

        pts = reference_pts
        for transform in self._transforms:
            if hasattr(transform, "unit_curvilinear_basis"):
                basis_i = transform.unit_curvilinear_basis(pts)
                # basis_total = basis_i @ basis_total
                basis_total = self._bkd.einsum("nij,njk->nik", basis_i, basis_total)
            pts = transform.map_to_physical(pts)

        return basis_total

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute factors for transforming gradients.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, ndim, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self._ndim

        # Start with identity
        grad_factors = self._bkd.zeros((npts, ndim, ndim))
        for ii in range(ndim):
            grad_factors[:, ii, ii] = 1.0

        pts = reference_pts
        for transform in self._transforms:
            if hasattr(transform, "gradient_factors"):
                factors_i = transform.gradient_factors(pts)
                grad_factors = self._bkd.einsum("nij,njk->nik", factors_i, grad_factors)
            pts = transform.map_to_physical(pts)

        return grad_factors

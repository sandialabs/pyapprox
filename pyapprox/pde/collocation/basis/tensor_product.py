"""Tensor product basis for multi-dimensional collocation.

Constructs multi-D derivative matrices from 1D components using
Kronecker products. This is the standard approach for tensor-product
grids (Cartesian meshes with tensor-product node distributions).

Supports coordinate transformations via mesh gradient factors. When a
mesh provides non-identity gradient_factors(), physical derivative matrices
are computed by applying gradient factors to scale reference derivatives.
"""

from typing import Generic, List, Tuple, Optional, Dict

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.protocols.basis import (
    DerivativeMatrix1DProtocol,
)
from pyapprox.pde.collocation.protocols.mesh import (
    MeshWithTransformProtocol,
)


class TensorProductBasis(Generic[Array]):
    """Tensor product basis assembling 1D components via Kronecker products.

    Given a mesh (with reference nodes and gradient factors) and a derivative
    matrix computer, this class constructs:
    - 1D reference derivative matrices for each dimension
    - Full multi-D reference derivative matrices via Kronecker products
    - Physical derivative matrices scaled by gradient factors

    The Kronecker product structure is:
    - 1D: D_x
    - 2D: D_x = I_y ⊗ D_x, D_y = D_y ⊗ I_x
    - 3D: D_x = I_z ⊗ I_y ⊗ D_x, D_y = I_z ⊗ D_y ⊗ I_x, D_z = D_z ⊗ I_y ⊗ I_x

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing reference nodes, physical nodes, and gradient factors.
    deriv_matrix_computer : DerivativeMatrix1DProtocol
        Computer for 1D derivative matrices (same for all dimensions).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        mesh: MeshWithTransformProtocol[Array],
        deriv_matrix_computer: DerivativeMatrix1DProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._mesh = mesh
        self._npts_per_dim = mesh.npts_per_dim()
        self._ndim = mesh.ndim()

        # Get 1D reference nodes from mesh
        self._nodes_1d: List[Array] = [
            mesh.reference_nodes_1d(d) for d in range(self._ndim)
        ]

        # Compute 1D derivative matrices for each dimension
        self._deriv_matrices_1d: List[Array] = []
        for nodes in self._nodes_1d:
            self._deriv_matrices_1d.append(deriv_matrix_computer.compute(nodes))

        # Get gradient factors from mesh
        self._gradient_factors = mesh.gradient_factors()

        # Cache for reference derivative matrices (order, dim) -> matrix
        self._ref_deriv_cache: Dict[Tuple[int, int], Array] = {}

        # Cache for physical derivative matrices (order, dim) -> matrix
        self._phys_deriv_cache: Dict[Tuple[int, int], Array] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> MeshWithTransformProtocol[Array]:
        """Return the mesh."""
        return self._mesh

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._ndim

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points in each dimension."""
        return self._npts_per_dim

    def npts(self) -> int:
        """Return total number of points."""
        total = 1
        for n in self._npts_per_dim:
            total *= n
        return total

    def nodes_1d(self, dim: int) -> Array:
        """Return 1D reference nodes for specified dimension.

        Parameters
        ----------
        dim : int
            Dimension index (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            1D nodes. Shape: (npts_dim,)
        """
        if dim < 0 or dim >= self._ndim:
            raise ValueError(f"dim must be in [0, {self._ndim}), got {dim}")
        return self._nodes_1d[dim]

    def derivative_matrix_1d(self, dim: int) -> Array:
        """Return 1D reference derivative matrix for specified dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Array
            1D derivative matrix. Shape: (npts_dim, npts_dim)
        """
        if dim < 0 or dim >= self._ndim:
            raise ValueError(f"dim must be in [0, {self._ndim}), got {dim}")
        return self._deriv_matrices_1d[dim]

    def reference_derivative_matrix(self, order: int, dim: int) -> Array:
        """Return full reference derivative matrix via Kronecker product.

        This is the derivative matrix in reference coordinates (before
        applying gradient factors for coordinate transformation).

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Reference dimension (0, 1, or 2).

        Returns
        -------
        Array
            Full derivative matrix. Shape: (npts, npts)
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if dim < 0 or dim >= self._ndim:
            raise ValueError(f"dim must be in [0, {self._ndim}), got {dim}")

        cache_key = (order, dim)
        if cache_key in self._ref_deriv_cache:
            return self._ref_deriv_cache[cache_key]

        # Build first-order derivative matrix if not cached
        if (1, dim) not in self._ref_deriv_cache:
            D_full = self._build_kronecker_derivative(dim)
            self._ref_deriv_cache[(1, dim)] = D_full
        else:
            D_full = self._ref_deriv_cache[(1, dim)]

        # For higher orders, compute matrix power
        if order == 1:
            result = D_full
        else:
            result = D_full
            for _ in range(order - 1):
                result = result @ D_full
            self._ref_deriv_cache[cache_key] = result

        return result

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Return full physical derivative matrix.

        Computes the derivative matrix in physical coordinates by applying
        gradient factors to the reference derivative matrices:

            D_phys[d] = sum_j G[:, d, j] * D_ref[j]

        where G is the gradient factors matrix from the mesh.

        For identity transforms (Cartesian meshes on [-1,1]^d), this equals
        the reference derivative matrix.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Physical dimension (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            Full physical derivative matrix. Shape: (npts, npts)
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if dim < 0 or dim >= self._ndim:
            raise ValueError(f"dim must be in [0, {self._ndim}), got {dim}")

        # For first order, apply gradient factors
        if order == 1:
            cache_key = (1, dim)
            if cache_key in self._phys_deriv_cache:
                return self._phys_deriv_cache[cache_key]

            D_phys = self._build_physical_derivative(dim)
            self._phys_deriv_cache[cache_key] = D_phys
            return D_phys

        # For higher orders, compose first-order physical derivatives
        cache_key = (order, dim)
        if cache_key in self._phys_deriv_cache:
            return self._phys_deriv_cache[cache_key]

        D1 = self.derivative_matrix(1, dim)
        result = D1
        for _ in range(order - 1):
            result = result @ D1
        self._phys_deriv_cache[cache_key] = result
        return result

    def _build_kronecker_derivative(self, dim: int) -> Array:
        """Build full reference derivative matrix using Kronecker products.

        For dimension `dim`, we place the 1D derivative matrix in position
        `dim` and identity matrices elsewhere.

        In 2D (npts = [n_x, n_y]):
          - dim=0: kron(I_{n_y}, D_x)
          - dim=1: kron(D_y, I_{n_x})

        In 3D (npts = [n_x, n_y, n_z]):
          - dim=0: kron(I_{n_z}, kron(I_{n_y}, D_x))
          - dim=1: kron(I_{n_z}, kron(D_y, I_{n_x}))
          - dim=2: kron(D_z, kron(I_{n_y}, I_{n_x}))
        """
        bkd = self._bkd

        # Start with the rightmost factor (dimension 0)
        # Work from right to left in the Kronecker product
        result: Optional[Array] = None

        for d in range(self._ndim):
            n_d = self._npts_per_dim[d]
            if d == dim:
                factor = self._deriv_matrices_1d[d]
            else:
                factor = bkd.eye(n_d)

            if result is None:
                result = factor
            else:
                result = bkd.kron(factor, result)

        return result

    def _build_physical_derivative(self, dim: int) -> Array:
        """Build physical derivative matrix by applying gradient factors.

        Physical derivative in direction `dim` is:
            D_phys[dim] = sum_j G[:, dim, j] * D_ref[j]

        where G[:, dim, j] is the gradient factor for converting reference
        derivative j to physical derivative dim.
        """
        bkd = self._bkd
        npts = self.npts()

        D_phys = bkd.zeros((npts, npts))
        for ref_dim in range(self._ndim):
            # Get gradient factor for this reference -> physical mapping
            # Shape: (npts,)
            g = self._gradient_factors[:, dim, ref_dim]

            # Get reference derivative matrix
            D_ref = self.reference_derivative_matrix(1, ref_dim)

            # Add contribution: diag(g) @ D_ref
            # This scales each row of D_ref by the corresponding g value
            D_phys = D_phys + bkd.diag(g) @ D_ref

        return D_phys

"""Tensor product basis for multi-dimensional collocation.

Constructs multi-D derivative matrices from 1D components using
Kronecker products. This is the standard approach for tensor-product
grids (Cartesian meshes with tensor-product node distributions).
"""

from typing import Generic, List, Tuple, Optional, Dict

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.basis import (
    NodesGenerator1DProtocol,
    DerivativeMatrix1DProtocol,
)


class TensorProductBasis(Generic[Array]):
    """Tensor product basis assembling 1D components via Kronecker products.

    Given 1D node generators and derivative matrix computers for each
    dimension, this class constructs:
    - 1D nodes for each dimension
    - 1D derivative matrices for each dimension
    - Full multi-D derivative matrices via Kronecker products

    The Kronecker product structure is:
    - 1D: D_x
    - 2D: D_x = I_y ⊗ D_x, D_y = D_y ⊗ I_x
    - 3D: D_x = I_z ⊗ I_y ⊗ D_x, D_y = I_z ⊗ D_y ⊗ I_x, D_z = D_z ⊗ I_y ⊗ I_x

    Parameters
    ----------
    nodes_generators : List[NodesGenerator1DProtocol]
        1D node generators for each dimension.
    deriv_matrix_computers : List[DerivativeMatrix1DProtocol]
        1D derivative matrix computers for each dimension.
    npts_per_dim : Tuple[int, ...]
        Number of nodes in each dimension.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        nodes_generators: List[NodesGenerator1DProtocol[Array]],
        deriv_matrix_computers: List[DerivativeMatrix1DProtocol[Array]],
        npts_per_dim: Tuple[int, ...],
        bkd: Backend[Array],
    ):
        ndim = len(npts_per_dim)
        if len(nodes_generators) != ndim:
            raise ValueError(
                f"Expected {ndim} node generators, got {len(nodes_generators)}"
            )
        if len(deriv_matrix_computers) != ndim:
            raise ValueError(
                f"Expected {ndim} deriv matrix computers, "
                f"got {len(deriv_matrix_computers)}"
            )

        self._bkd = bkd
        self._npts_per_dim = npts_per_dim
        self._ndim = ndim

        # Generate 1D nodes for each dimension
        self._nodes_1d: List[Array] = []
        for dim, (gen, n) in enumerate(zip(nodes_generators, npts_per_dim)):
            self._nodes_1d.append(gen.generate(n))

        # Compute 1D derivative matrices for each dimension
        self._deriv_matrices_1d: List[Array] = []
        for dim, (comp, nodes) in enumerate(
            zip(deriv_matrix_computers, self._nodes_1d)
        ):
            self._deriv_matrices_1d.append(comp.compute(nodes))

        # Cache for full derivative matrices (order, dim) -> matrix
        self._deriv_cache: Dict[Tuple[int, int], Array] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

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
        """Return 1D nodes for specified dimension.

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
        """Return 1D derivative matrix for specified dimension.

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

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """Return full derivative matrix via Kronecker product.

        For a tensor product basis, derivatives in different directions
        use the Kronecker product structure. For example, in 2D with
        n_x nodes in x and n_y nodes in y (total n = n_x * n_y):

        - D_x has shape (n, n) = kron(I_{n_y}, D_x^{1D})
        - D_y has shape (n, n) = kron(D_y^{1D}, I_{n_x})

        Higher order derivatives are computed by matrix powers of the
        first derivative.

        Parameters
        ----------
        order : int
            Derivative order (1 for first derivative, 2 for second, etc.)
        dim : int
            Spatial dimension (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            Full derivative matrix. Shape: (npts, npts)
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if dim < 0 or dim >= self._ndim:
            raise ValueError(f"dim must be in [0, {self._ndim}), got {dim}")

        # Check cache
        cache_key = (order, dim)
        if cache_key in self._deriv_cache:
            return self._deriv_cache[cache_key]

        # Build first-order derivative matrix if not cached
        if (1, dim) not in self._deriv_cache:
            D_full = self._build_kronecker_derivative(dim)
            self._deriv_cache[(1, dim)] = D_full
        else:
            D_full = self._deriv_cache[(1, dim)]

        # For higher orders, compute matrix power
        if order == 1:
            result = D_full
        else:
            result = D_full
            for _ in range(order - 1):
                result = result @ D_full
            self._deriv_cache[cache_key] = result

        return result

    def _build_kronecker_derivative(self, dim: int) -> Array:
        """Build full derivative matrix using Kronecker products.

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
                # kron(new_factor, existing_result)
                # We build from right to left: result = kron(factor, result)
                result = bkd.kron(factor, result)

        return result

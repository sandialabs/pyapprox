"""Linear algebra utilities for PyApprox typing module."""

from pyapprox.util.linalg.cholesky_factor import (
    CholeskyFactor,
)
from pyapprox.util.linalg.indexing import (
    extract_submatrix,
)
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
    InnerProductProtocol,
    MassInnerProduct,
    m_orthonormality_drift,
)
from pyapprox.util.linalg.pivoted_lu import (
    PivotedLUFactorizer,
    get_final_pivots_from_sequential_pivots,
    swap_rows,
)
from pyapprox.util.linalg.pivoted_qr import (
    PivotedQRFactorizer,
)
from pyapprox.util.linalg.protocols import (
    IncrementalFactorizerProtocol,
    PivotedFactorizerProtocol,
)
from pyapprox.util.linalg.randomized import (
    DenseMatVecOperator,
    DenseSymmetricMatVecOperator,
    FunctionMatVecOperator,
    FunctionSymmetricMatVecOperator,
    MatVecOperator,
    RandomizedSVD,
    SymmetricMatVecOperator,
    SymmetricRandomizedSVD,
    TwoPassRandomizedSVD,
    adjust_sign_svd,
    get_low_rank_matrix,
    randomized_symmetric_eigendecomposition,
)
from pyapprox.util.linalg.sparse_dispatch import (
    solve_maybe_sparse,
    sparse_or_dense_solve,
)
from pyapprox.util.linalg.truncated_pivoted_qr import (
    TruncatedPivotedQRFactorizer,
)

__all__ = [
    # Indexing
    "extract_submatrix",
    # Cholesky
    "CholeskyFactor",
    # Protocols
    "PivotedFactorizerProtocol",
    "IncrementalFactorizerProtocol",
    # Pivoted LU
    "PivotedLUFactorizer",
    "swap_rows",
    "get_final_pivots_from_sequential_pivots",
    # Pivoted QR
    "PivotedQRFactorizer",
    "TruncatedPivotedQRFactorizer",
    # Sparse dispatch
    "solve_maybe_sparse",
    "sparse_or_dense_solve",
    # Inner products (the metric a projection is taken in)
    "InnerProductProtocol",
    "EuclideanInnerProduct",
    "DiagonalInnerProduct",
    "MassInnerProduct",
    "m_orthonormality_drift",
    # Randomized
    "MatVecOperator",
    "SymmetricMatVecOperator",
    "DenseMatVecOperator",
    "DenseSymmetricMatVecOperator",
    "FunctionMatVecOperator",
    "FunctionSymmetricMatVecOperator",
    "RandomizedSVD",
    "TwoPassRandomizedSVD",
    "SymmetricRandomizedSVD",
    "randomized_symmetric_eigendecomposition",
    "adjust_sign_svd",
    "get_low_rank_matrix",
]

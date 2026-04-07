"""Linear algebra utilities for PyApprox typing module."""

from pyapprox.util.linalg.cholesky_factor import (
    CholeskyFactor,
)
from pyapprox.util.linalg.indexing import (
    extract_submatrix,
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
from pyapprox.util.linalg.sparse_dispatch import (
    solve_maybe_sparse,
    sparse_or_dense_solve,
)
from pyapprox.util.linalg.randomized import (
    DenseMatVecOperator,
    DenseSymmetricMatVecOperator,
    DoublePassRandomizedSVD,
    FunctionMatVecOperator,
    FunctionSymmetricMatVecOperator,
    MatVecOperator,
    RandomizedSVD,
    SinglePassRandomizedSVD,
    SymmetricMatVecOperator,
    adjust_sign_svd,
    get_low_rank_matrix,
    randomized_symmetric_eigendecomposition,
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
    # Sparse dispatch
    "solve_maybe_sparse",
    "sparse_or_dense_solve",
    # Randomized
    "MatVecOperator",
    "SymmetricMatVecOperator",
    "DenseMatVecOperator",
    "DenseSymmetricMatVecOperator",
    "FunctionMatVecOperator",
    "FunctionSymmetricMatVecOperator",
    "RandomizedSVD",
    "SinglePassRandomizedSVD",
    "DoublePassRandomizedSVD",
    "randomized_symmetric_eigendecomposition",
    "adjust_sign_svd",
    "get_low_rank_matrix",
]

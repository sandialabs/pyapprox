"""Linear algebra utilities for PyApprox typing module."""

from pyapprox.typing.util.linalg.cholesky_factor import (
    CholeskyFactor,
)

from pyapprox.typing.util.linalg.protocols import (
    PivotedFactorizerProtocol,
    IncrementalFactorizerProtocol,
)

from pyapprox.typing.util.linalg.pivoted_lu import (
    PivotedLUFactorizer,
    swap_rows,
    get_final_pivots_from_sequential_pivots,
)

from pyapprox.typing.util.linalg.pivoted_qr import (
    PivotedQRFactorizer,
)

from pyapprox.typing.util.linalg.randomized import (
    MatVecOperator,
    SymmetricMatVecOperator,
    DenseMatVecOperator,
    DenseSymmetricMatVecOperator,
    FunctionMatVecOperator,
    FunctionSymmetricMatVecOperator,
    RandomizedSVD,
    SinglePassRandomizedSVD,
    DoublePassRandomizedSVD,
    randomized_symmetric_eigendecomposition,
    adjust_sign_svd,
    get_low_rank_matrix,
)

__all__ = [
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

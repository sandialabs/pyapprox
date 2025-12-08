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
]

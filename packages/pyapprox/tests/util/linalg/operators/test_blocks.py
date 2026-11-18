r"""Block vectors, block operators and block lower-triangular solves, on
every stack.

Components of sizes 3 and 2 (and 3, 2, 4 for the three-block solves).
Every result is compared with the same computation on the dense matrix
assembled from the blocks: the product space's operations component by
component, the block operator's action and transpose (a zero block
included), forward substitution against a dense solve with the assembled
lower-triangular matrix, and backward substitution against a dense solve
with its transpose. On the mock stack the components cannot be looked
inside, so the block layer is shown to use only the component spaces and
operators.
"""

from tests._helpers.linalg.operator_suites.blocks import (
    BlockLowerTriangularHandleSuite,
    BlockOperatorSuite,
    BlockVectorSpaceSuite,
)


class TestBlockVectorSpace(BlockVectorSpaceSuite):
    pass


class TestBlockOperator(BlockOperatorSuite):
    pass


class TestBlockLowerTriangularHandle(BlockLowerTriangularHandleSuite):
    pass

"""The format witness makes a storage mismatch a static type error.

A consumer that needs CSR storage declares
``ActionOperatorProtocol[NDArray, CSRFormat]``. mypy must accept an
identity built in CSR format and refuse a matrix-free operator. Checked
by running mypy on a snippet, since the property under test is what the
type checker reports, not what runs.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SNIPPET = textwrap.dedent(
    """
    from typing import Any, Optional

    from numpy.typing import NDArray

    from pyapprox.util.linalg.operators import (
        ActionOperatorProtocol,
        SpaceDescriptor,
        MatrixFree,
        OperatorProperties,
        ScaledSum,
        as_matrix_free,
    )
    from pyapprox.util.linalg.scipy_stack import (
        CSRFormat,
        NumpyVectorSpace,
        ScipyCSROperator,
        ScipySparseLU,
    )


    def factor(op: ActionOperatorProtocol[NDArray[Any], CSRFormat]) -> None:
        pass


    class Shift:
        def domain(self) -> SpaceDescriptor:
            raise NotImplementedError

        def range(self) -> SpaceDescriptor:
            raise NotImplementedError

        def apply(
            self, x: NDArray[Any], out: Optional[NDArray[Any]] = None
        ) -> NDArray[Any]:
            raise NotImplementedError

        def apply_transpose(
            self, y: NDArray[Any], out: Optional[NDArray[Any]] = None
        ) -> NDArray[Any]:
            raise NotImplementedError

        def properties(self) -> OperatorProperties:
            raise NotImplementedError

        def format(self) -> MatrixFree:
            return MatrixFree()


    u = SpaceDescriptor("u", 3)
    space = NumpyVectorSpace(u)
    csr = ScipyCSROperator.identity(u)
    factor(csr)  # PAIRS
    factor(Shift())  # REFUSED
    factor(ScaledSum([(1.0, csr), (2.0, csr)], space))  # PAIRS
    ScaledSum([(1.0, as_matrix_free(csr)), (2.0, csr)], space)  # REFUSED
    ScipySparseLU().setup(csr)  # PAIRS
    ScipySparseLU().setup(Shift())  # REFUSED
    """
)


def _lines_of(marker: str) -> list[int]:
    return [
        number
        for number, line in enumerate(_SNIPPET.splitlines(), start=1)
        if marker in line
    ]


def test_matrix_free_operator_cannot_pair_with_a_csr_consumer(
    tmp_path: Path, pytestconfig: pytest.Config
) -> None:
    pytest.importorskip("mypy")
    snippet = tmp_path / "pairing.py"
    snippet.write_text(_SNIPPET)
    # A persistent cache: a cold mypy run over the imported modules takes
    # tens of seconds, a warm one a few.
    assert pytestconfig.cache is not None
    cache_dir = pytestconfig.cache.mkdir("mypy-format-pairing")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--no-color-output",
            "--cache-dir",
            str(cache_dir),
            "--follow-imports=silent",
            str(snippet),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    errors = sorted(
        {
            int(match.group(1))
            for match in re.finditer(r"pairing\.py:(\d+): error", result.stdout)
        }
    )
    assert errors == _lines_of("# REFUSED"), result.stdout

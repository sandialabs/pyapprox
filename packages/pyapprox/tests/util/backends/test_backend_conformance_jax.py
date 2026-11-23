"""The experimental ``JaxBkd`` against the conformance table in
``tests/_helpers/backend_conformance.py``: every row of a method it
implements (one not in ``NOT_YET_ON_JAX``) runs under ``jax.jit``, or
eagerly when the row is ``eager_only``.

JAX is optional, so the whole module is skipped without it.
"""

from typing import Any, Callable, Iterator, List, Tuple

import pytest

# Skip the module without jax; the imports below are then ordinary ones.
pytest.importorskip("jax")

import jax

from pyapprox.util.backends.jax import JaxBkd
from tests._helpers.backend_conformance import (
    CASES,
    NOT_YET_ON_JAX,
    Case,
    case_id,
    run_case,
)

_ON_JAX = [case for case in CASES if case.method not in NOT_YET_ON_JAX]


def _leaves(value: Any) -> List[jax.Array]:
    """The JAX arrays in ``value``, also inside tuples."""
    if isinstance(value, tuple):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value] if isinstance(value, jax.Array) else []


def _rebuild(value: Any, leaves: Iterator[jax.Array]) -> Any:
    """``value`` with its JAX arrays replaced, in order, by ``leaves``."""
    if isinstance(value, tuple):
        return tuple(_rebuild(v, leaves) for v in value)
    return next(leaves) if isinstance(value, jax.Array) else value


def _jitted(method: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    # The arrays are traced; shapes, axes, strings and scalars are closed
    # over, so they are static.
    def traced(*arrays: jax.Array) -> Any:
        return method(*_rebuild(args, iter(arrays)))

    return jax.jit(traced)(*_leaves(args))


@pytest.fixture
def jax_bkd() -> JaxBkd:
    return JaxBkd()


@pytest.mark.parametrize("case", _ON_JAX, ids=case_id)
def test_conforms_on_jax(jax_bkd: JaxBkd, case: Case) -> None:
    run_case(jax_bkd, case, call=None if case.eager_only else _jitted)

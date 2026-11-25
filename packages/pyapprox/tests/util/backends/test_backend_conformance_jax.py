"""The experimental ``JaxBkd``: every row of the conformance table in
``tests/_helpers/backend_conformance.py`` of a method it implements (one
not in ``NOT_YET_ON_JAX``), run eagerly, as the general ``bkd`` fixture
will once ``JaxBkd`` joins it; compiled rows run in
``test_backend_conformance.py``. Also what ``JaxBkd.jit`` does beyond
the other backends: ``static_argnames`` and its cache across pickling.

JAX is optional, so the whole module is skipped without it.
"""

import pickle
from typing import Callable

import pytest

# Skip the module without jax; the imports below are then ordinary ones.
pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from pyapprox.util.backends.jax import JaxBkd
from pyapprox.util.backends.precision import Precision
from tests._helpers.backend_conformance import (
    CASES,
    NOT_YET_ON_JAX,
    Case,
    case_id,
    run_case,
)

_ON_JAX = [case for case in CASES if case.method not in NOT_YET_ON_JAX]

# Annotated alias: jax.config.update is untyped in jax's stubs.
_update_config: Callable[[str, bool], None] = jax.config.update


@pytest.fixture
def jax_bkd() -> JaxBkd:
    # 64-bit types are the caller's to enable; JaxBkd only checks them.
    _update_config("jax_enable_x64", True)
    return JaxBkd()


@pytest.mark.parametrize("case", _ON_JAX, ids=case_id)
def test_conforms_on_jax(jax_bkd: JaxBkd, case: Case) -> None:
    run_case(jax_bkd, case)


def _ones_sum(scale: jax.Array, *, bkd: JaxBkd, size: int) -> jax.Array:
    # size sets a shape, so it must be static under jax.jit.
    return bkd.sum(bkd.ones((size,)) * scale)


class TestJaxJit:
    def test_static_argnames_select_a_shape(self, jax_bkd: JaxBkd) -> None:
        compiled = jax_bkd.jit(_ones_sum, static_argnames=("size",))
        scale = jax_bkd.asarray(2.0)
        jax_bkd.assert_allclose(compiled(scale, size=3), jax_bkd.asarray(6.0))
        jax_bkd.assert_allclose(compiled(scale, size=5), jax_bkd.asarray(10.0))

    def test_a_shape_argument_must_be_static(self, jax_bkd: JaxBkd) -> None:
        compiled = jax_bkd.jit(_ones_sum)
        with pytest.raises(TypeError):
            compiled(jax_bkd.asarray(2.0), size=3)

    def test_cached_per_kernel_and_options(self, jax_bkd: JaxBkd) -> None:
        first = jax_bkd.jit(_ones_sum, static_argnames=("size",))
        assert jax_bkd.jit(_ones_sum, static_argnames=("size",)) is first
        assert (
            jax_bkd.jit(_ones_sum, static_argnames=("size",), dynamic=True) is not first
        )

    def test_pickles_without_its_cache(self, jax_bkd: JaxBkd) -> None:
        jax_bkd.jit(_ones_sum, static_argnames=("size",))
        restored = pickle.loads(pickle.dumps(jax_bkd))
        compiled = restored.jit(_ones_sum, static_argnames=("size",))
        restored.assert_allclose(
            compiled(restored.asarray(1.5), size=2), restored.asarray(3.0)
        )


class TestConfiguration:
    """What JAX's process-wide 64-bit switch adds; the configurations
    themselves run through config_bkd in the other backend tests."""

    def test_double_needs_64_bit_types(self) -> None:
        with jax.enable_x64(False):
            with pytest.raises(ValueError, match="jax_enable_x64"):
                JaxBkd(precision=Precision.DOUBLE)

    def test_single_runs_without_64_bit_types(self) -> None:
        with jax.enable_x64(False):
            bkd = JaxBkd(precision=Precision.SINGLE)
            assert bkd.zeros((2,)).dtype == jnp.float32
            assert bkd.arange(3).dtype == bkd.index_dtype() == jnp.int32
            assert bkd.asarray([0.1]).dtype == jnp.float32

    def test_construction_leaves_the_switch_alone(self) -> None:
        with jax.enable_x64(False):
            JaxBkd(precision=Precision.SINGLE)
            assert not jax.config.jax_enable_x64

    def test_precision_change_keeps_the_trace(self, jax_bkd: JaxBkd) -> None:
        single = jax_bkd.replace(precision=Precision.SINGLE)

        def through_single(x: jax.Array) -> jax.Array:
            narrowed = single.from_other(x, jax_bkd)
            return jax_bkd.sum(jax_bkd.from_other(narrowed * 2.0, single))

        gradient = jax.grad(through_single)(jax_bkd.asarray([1.0, 2.0, 3.0]))
        jax_bkd.assert_allclose(gradient, 2.0 * jax_bkd.ones((3,)))

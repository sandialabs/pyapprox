"""Test helper: a Galerkin physics composes its natural-BC terms once.

The spatial operator is ``F = F_Omega + F_Gamma``: the physics supplies the
interior ``F_Omega`` and the natural-BC terms ``F_Gamma = sum_k c_k`` are
added once, by the composition. ``check_natural_bc_composition`` checks:

1. the interior does not see the boundary conditions: building the physics
   with the natural BCs leaves ``interior_residual`` and
   ``interior_jacobian`` unchanged;
2. the boundary terms are added exactly once: the spatial residual with the
   BCs minus without them equals ``sum_k c_k``, and likewise for the
   Jacobian.

A physics that adds a term itself (inside its interior) fails check 1; one
that drops or doubles a term fails check 2.

What it assumes, and checks where it can:

- ``make_physics`` builds the same physics apart from the BC list. Checked
  only in part: the bare physics must hold no natural BCs and the composed
  one exactly ``natural_bcs`` (the same objects). Different coefficients
  would show up as a spurious failure of check 1.
- The physics exposes ``interior_residual``/``interior_jacobian``
  (``GalerkinInteriorOperatorProtocol``). Checked.
- ``state`` is admissible for the physics (e.g. no inverted elements for
  hyperelasticity). Not checked; NaNs fail the comparison.

What it does not check: that each term is mathematically right. The terms'
own ``apply_to_residual``/``apply_to_jacobian`` are the reference here; their
correctness is tested independently (sign-convention and limit tests).

Transitional: once physics constructors take no BCs (the composed system
owns them), check 1 holds by construction.
"""

from typing import Any, Callable, List, Sequence

from pyapprox.pde.boundary import WeakFormBCProtocol
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinInteriorOperatorProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from scipy.sparse import issparse


def _dense(matrix: Any, bkd: Backend[Array]) -> Array:
    """A Jacobian as a dense backend array (Jacobians may be sparse)."""
    if issparse(matrix):
        return bkd.asarray(matrix.toarray())
    dense: Array = matrix
    return dense


def check_natural_bc_composition(
    make_physics: Callable[[List[Any]], Any],
    natural_bcs: Sequence[WeakFormBCProtocol[Array]],
    state: Array,
    time: float = 0.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Assert that a physics composes its natural-BC terms exactly once.

    Parameters
    ----------
    make_physics : Callable
        Builds the physics from a boundary-condition list. Called once
        with no BCs and once with ``natural_bcs``; everything else about
        the physics must be the same.
    natural_bcs : Sequence[WeakFormBCProtocol]
        Natural (Neumann, Robin) BCs to compose; at least one.
    state : Array
        An admissible state at which to compare. Shape: (nstates,)
    time : float
        Time at which to compare.
    rtol, atol : float
        Tolerances of the comparisons.
    """
    if not natural_bcs:
        raise ValueError("natural_bcs must contain at least one BC")
    bare = make_physics([])
    composed = make_physics(list(natural_bcs))
    for physics in (bare, composed):
        if not isinstance(physics, GalerkinInteriorOperatorProtocol):
            raise TypeError(
                f"{type(physics).__name__} does not expose interior_residual"
                "/interior_jacobian (GalerkinInteriorOperatorProtocol)"
            )
    # The factory must vary only the BC list.
    assert bare.weak_form_bcs() == [], "the bare physics holds natural BCs"
    held = composed.weak_form_bcs()
    assert len(held) == len(natural_bcs) and all(
        a is b for a, b in zip(held, natural_bcs)
    ), "the composed physics does not hold exactly the given natural BCs"

    bkd: Backend[Array] = bare.bkd()
    nstates = bare.nstates()

    # 1. The interior never sees the boundary conditions.
    bkd.assert_allclose(
        composed.interior_residual(state, time),
        bare.interior_residual(state, time),
        rtol=rtol,
        atol=atol,
    )
    bkd.assert_allclose(
        _dense(composed.interior_jacobian(state, time), bkd),
        _dense(bare.interior_jacobian(state, time), bkd),
        rtol=rtol,
        atol=atol,
    )

    # 2. The terms are added exactly once.
    expected_residual = bkd.zeros((nstates,))
    expected_jacobian = bkd.zeros((nstates, nstates))
    for bc in natural_bcs:
        expected_residual = bc.apply_to_residual(expected_residual, state, time)
        expected_jacobian = bc.apply_to_jacobian(expected_jacobian, state, time)
    bkd.assert_allclose(
        composed.spatial_residual(state, time)
        - bare.spatial_residual(state, time),
        expected_residual,
        rtol=rtol,
        atol=atol,
    )
    bkd.assert_allclose(
        _dense(composed.spatial_jacobian(state, time), bkd)
        - _dense(bare.spatial_jacobian(state, time), bkd),
        _dense(expected_jacobian, bkd),
        rtol=rtol,
        atol=atol,
    )

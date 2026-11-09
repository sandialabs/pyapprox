r"""Finite-difference and dot tests of a derivatives bundle, on arrays.

The pieces every bundle checker shares, whatever the bundle's owner (a
residual, a stepper, a functional's terms): the finite-difference errors
of one declared derivative along a random direction, with their
predicted rounding; the dot test
:math:`\langle w, A v\rangle = \langle A^\top w, v\rangle`, the only
check of a transpose no finite difference reaches; the symmetry check of
an operator declaring itself symmetric; and the report collecting them.
"""

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
    FDCurve,
    predicted_rounding,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import LinearMapProtocol

T = TypeVar("T")


@dataclass(frozen=True)
class BundleCheckReport:
    """Result of a bundle check.

    Attributes
    ----------
    fd_checks : Mapping[str, FDCurve]
        For each term checked, its finite-difference errors over the
        steps with their predicted rounding, judged by :meth:`failures`:
        each error relative to the operator's action, or absolute where
        the action is zero (a stated zero).
    transpose_errors : Mapping[str, float]
        For each operator, the relative dot-test error; for an operator
        declaring itself symmetric, also ``"<name> (symmetric)"``, the
        relative difference between its action and its transpose's.
    structure_errors : Mapping[str, float]
        For each structural declaration checked (for instance a quantity
        affine in its argument, or a derivative independent of time), the
        relative defect along one large step: round-off when the
        declaration holds.
    """

    fd_checks: Mapping[str, FDCurve]
    transpose_errors: Mapping[str, float]
    structure_errors: Mapping[str, float] = field(default_factory=dict)

    def failures(
        self,
        fd_tol: float = 1e-6,
        transpose_tol: float = 1e-12,
        structure_tol: float = 1e-6,
        order_tol: float = 0.25,
        max_bottom_step: Optional[float] = None,
        rounding_factor: float = 100.0,
    ) -> List[str]:
        """The names of the checks that fail: a finite-difference check
        whose smallest error exceeds ``fd_tol`` or whose errors do not form
        a correct derivative's V (judged by :meth:`FDCurve.v_shape` with
        ``order_tol``, ``max_bottom_step`` and ``rounding_factor``), and a
        dot test or structural defect exceeding its tolerance.

        The defaults leave correct derivatives a wide margin; loosen a
        tolerance here for numerical noise rather than change a checker.
        """

        def fd_passes(curve: FDCurve) -> bool:
            shape = curve.v_shape(order_tol, max_bottom_step, rounding_factor)
            return curve.min_error() <= fd_tol and shape.passed

        return (
            [name for name, curve in self.fd_checks.items() if not fd_passes(curve)]
            + [
                name
                for name, error in self.transpose_errors.items()
                if not error <= transpose_tol
            ]
            + [
                name
                for name, error in self.structure_errors.items()
                if not error <= structure_tol
            ]
        )


def refetched(field: Optional[T], name: str) -> T:
    """A bundle field fetched again after the owner's parameters moved.

    Raises
    ------
    ValueError
        If the field is absent: which fields a bundle holds must not
        depend on the parameter values.
    """
    if field is None:
        raise ValueError(
            f"a bundle fetched after set_param lacks '{name}', which the first "
            "fetch had; which fields are present must not depend on the "
            "parameter values"
        )
    return field


class _PointTerm(Generic[Array]):
    """A quantity of one 1-D point and its derivative's action, in the
    batch form :class:`FunctionWithJVPFromCallable` takes."""

    def __init__(
        self,
        bkd: Backend[Array],
        value: Callable[[Array], Array],
        action: Callable[[Array, Array], Array],
    ) -> None:
        self._bkd = bkd
        self._value, self._action = value, action

    def values(self, samples: Array) -> Array:
        return self._bkd.stack(
            [self._value(samples[:, k]) for k in range(samples.shape[1])], axis=1
        )

    def jvp(self, sample: Array, vec: Array) -> Array:
        return self._action(sample[:, 0], vec)


def fd_check(
    bkd: Backend[Array],
    point: Array,
    value: Callable[[Array], Array],
    action: Callable[[Array, Array], Array],
    rng: np.random.Generator,
    fd_eps: Optional[Array] = None,
    central: bool = False,
) -> FDCurve:
    """The finite-difference errors of one term along a random direction,
    by :class:`DerivativeChecker` on the term as a function carrying its
    ``jvp`` in its bundle, with their predicted rounding.

    ``value`` maps a 1-D point to a 1-D quantity and ``action`` a 1-D
    point and a direction block to a block; both are evaluated at
    ``point`` first, so a term that moves parameters starts there. A zero
    action is compared in absolute terms, so a false stated zero fails.
    The steps default to ``1e-1`` down to ``1e-13``.
    """
    nvars = point.shape[0]
    base = value(point)
    nqoi = base.shape[0]
    term = _PointTerm(bkd, value, action)
    direction = bkd.asarray(rng.normal(size=(nvars, 1)))
    direction = direction / bkd.norm(direction)
    checker = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi, nvars, term.values, term.jvp, bkd)
    )
    steps = bkd.flip(bkd.logspace(-13, -1, 13)) if fd_eps is None else fd_eps
    rounding = predicted_rounding(
        float(np.finfo(backend_dtype_name(bkd)).eps),
        bkd.to_float(bkd.norm(base)),
        bkd.to_float(bkd.norm(action(point, direction))),
        bkd.to_float(bkd.norm(point)),
        [bkd.to_float(steps[ii]) for ii in range(steps.shape[0])],
    )
    errors = checker.check_derivatives(
        bkd.reshape(point, (nvars, 1)),
        fd_eps=steps,
        direction=direction,
        central=central,
    )[0]
    return checker.curve(errors, steps, central, rounding)


def random_column(bkd: Backend[Array], size: int, rng: np.random.Generator) -> Array:
    """A standard-normal column of ``size`` entries."""
    return bkd.asarray(rng.normal(size=(size, 1)))


def dot_test_error(
    bkd: Backend[Array], op: LinearMapProtocol[Array], rng: np.random.Generator
) -> float:
    r"""The relative difference of :math:`\langle w, A v\rangle` and
    :math:`\langle A^\top w, v\rangle` for random ``v`` and ``w``."""
    v = random_column(bkd, op.domain().size, rng)
    w = random_column(bkd, op.range().size, rng)
    av, atw = op.apply(v), op.apply_transpose(w)
    scale = max(
        bkd.to_float(bkd.norm(w) * bkd.norm(av)),
        bkd.to_float(bkd.norm(v) * bkd.norm(atw)),
    )
    difference = bkd.to_float(bkd.sum(w * av) - bkd.sum(atw * v))
    return 0.0 if scale == 0.0 else abs(difference) / scale


def symmetry_error(
    bkd: Backend[Array], op: LinearMapProtocol[Array], rng: np.random.Generator
) -> float:
    r"""The relative difference of :math:`A v` and :math:`A^\top v` for a
    random ``v``."""
    v = random_column(bkd, op.domain().size, rng)
    av, atv = op.apply(v), op.apply_transpose(v)
    scale = max(bkd.to_float(bkd.norm(av)), bkd.to_float(bkd.norm(atv)))
    difference = bkd.to_float(bkd.norm(av - atv))
    return 0.0 if scale == 0.0 else difference / scale


def transpose_errors(
    bkd: Backend[Array],
    operators: Sequence[tuple[str, LinearMapProtocol[Array]]],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Each operator's dot-test error, and its symmetry error under
    ``"<name> (symmetric)"`` when it declares itself symmetric."""
    errors: Dict[str, float] = {}
    for name, op in operators:
        errors[name] = dot_test_error(bkd, op, rng)
        if op.properties().symmetric:
            errors[f"{name} (symmetric)"] = symmetry_error(bkd, op, rng)
    return errors

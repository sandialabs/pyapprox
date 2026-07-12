"""Frozen bundle of optional derivative capabilities.

Absence of a capability is ``None``, never a missing attribute. In any
bundle, ``jacobian`` means d/d(this function's input), unqualified: a GP
loss's bundle differentiates w.r.t. hyperparameters because hyperparameters
ARE that function's input; a surrogate's bundle differentiates w.r.t. x.
Accessors that produce bundles for other derivative variables carry the
"wrt what" in their own name (e.g. ``param_derivatives()`` /
``input_derivatives(X2)`` on kernels) and close over any extra context so
the field arities below always hold.

Shape contracts (single-sample optimizer-facing forms; the strict contract
attaches to ``ObjectiveProtocol.derivatives()`` — kernel-style accessors
document their own conventions):

- ``JacobianFn``:      (nvars, 1) -> (nqoi, nvars)
- ``JacobianBatchFn``: (nvars, n) -> (n, nqoi, nvars)
- ``JVPFn``:           (sample, vec) -> (nqoi, 1)
- ``HVPFn``:           (sample, vec) -> (nvars, 1)   [nqoi == 1 Hessian]
- ``WHVPFn``:          (sample, vec, weights) -> (nvars, 1)
                       [weights (nqoi, 1); (sum_i w_i * hess f_i) @ vec]
- ``HessianFn``:       (nvars, 1) -> (nvars, nvars)  [nqoi == 1 only]
- ``HessianBatchFn``:  (nvars, n) -> (n, nvars, nvars) [nqoi == 1 only]
- ``HVPBatchFn``:      (samples (nvars, n), vecs (nvars, n)) -> (n, nvars)
                       [nqoi == 1 only; SCALAR-IMPLICIT: unlike
                       jacobian_batch there is NO nqoi axis in the output —
                       a recurring trap when reshaping]
- ``WHVPBatchFn``:     (samples, vecs, weights (nqoi, 1)) -> (n, nvars)
                       [one weight vector applied to every sample]
- ``InexactValueFn`` / ``InexactJacobianFn``: (sample, tol) -> (nqoi, 1) /
                       (nqoi, nvars)

``hvp`` and ``whvp`` describe contractions of the SAME Hessian tensor.
Producers implement whichever is natural; consumers never read the raw
pair — they call :meth:`Derivatives.resolved_hvp` /
:meth:`Derivatives.resolved_whvp`, which lift/synthesize exactly (w = [1])
when ``nqoi == 1``. The resolvers deliberately do NOT synthesize an hvp by
slicing a materialized ``hessian``/``hessian_batch``: that would silently
convert a matrix-free algorithm into an O(nvars^2)-memory one. If such a
conversion is ever wanted it must be an explicit helper the caller opts
into. No finite-difference fallback exists anywhere in this module or in
producers: what to do about an absent capability is the consumer's
decision (scipy receives ``jac=None`` and applies its own FD; ROL falls
back to its internal secant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union

from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

JacobianFn = Callable[[Array], Array]
JacobianBatchFn = Callable[[Array], Array]
JVPFn = Callable[[Array, Array], Array]
HVPFn = Callable[[Array, Array], Array]
WHVPFn = Callable[[Array, Array, Array], Array]
HessianFn = Callable[[Array], Array]
HessianBatchFn = Callable[[Array], Array]
HVPBatchFn = Callable[[Array, Array], Array]
WHVPBatchFn = Callable[[Array, Array, Array], Array]
InexactValueFn = Callable[[Array, float], Array]
InexactJacobianFn = Callable[[Array, float], Array]

A = TypeVar("A", bound=ArrayProtocol)


class _Unset:
    """Sentinel distinguishing 'not overridden' from an explicit None."""

    __slots__ = ()


_UNSET = _Unset()


@dataclass(frozen=True)
class InexactSuite(Generic[Array]):
    """Tolerance-aware evaluation (ROL inexact computation interface).

    ``value`` is required; ``jacobian`` is optional. Both take
    ``(sample, tol)``.
    """

    value: InexactValueFn[Array]
    jacobian: Optional[InexactJacobianFn[Array]] = None

    def __post_init__(self) -> None:
        if not callable(self.value):
            raise TypeError(
                "InexactSuite field 'value' must be callable; got "
                f"{type(self.value).__name__}"
            )
        if self.jacobian is not None and not callable(self.jacobian):
            raise TypeError(
                "InexactSuite field 'jacobian' must be callable or None; "
                f"got {type(self.jacobian).__name__}"
            )


@dataclass(frozen=True)
class Derivatives(Generic[Array]):
    """Bundle of optional derivative capabilities.

    Fields hold callables (typically bound methods). Classes keep their
    real ``def jacobian(...)`` / ``def hvp(...)`` methods — the bundle
    references them. Bundles are frozen; capability is fixed per
    ``bind()``/``minimize()`` run. If capability changes (e.g. active
    hyperparameters are toggled via ``hyp_list().set_active_values()``),
    rebuild the producer and rebind — never mutate a bundle.

    Storing the bundle in ``__init__`` is safe under ``copy.deepcopy`` and
    pickling even when fields are bound methods of the owning object
    (the cycle is re-pointed to the clone; verified by the Phase-0 spike).
    """

    jacobian: Optional[JacobianFn[Array]] = None
    jacobian_batch: Optional[JacobianBatchFn[Array]] = None
    jvp: Optional[JVPFn[Array]] = None
    hvp: Optional[HVPFn[Array]] = None
    whvp: Optional[WHVPFn[Array]] = None
    hessian: Optional[HessianFn[Array]] = None
    hessian_batch: Optional[HessianBatchFn[Array]] = None
    hvp_batch: Optional[HVPBatchFn[Array]] = None
    whvp_batch: Optional[WHVPBatchFn[Array]] = None
    inexact: Optional[InexactSuite[Array]] = None

    def __post_init__(self) -> None:
        for name in (
            "jacobian",
            "jacobian_batch",
            "jvp",
            "hvp",
            "whvp",
            "hessian",
            "hessian_batch",
            "hvp_batch",
            "whvp_batch",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"Derivatives field '{name}' must be callable or None; "
                    f"got {type(value).__name__}. Did you pass a computed "
                    "value (e.g. jacobian(x0)) instead of the function "
                    "itself?"
                )
        if self.inexact is not None and not isinstance(
            self.inexact, InexactSuite
        ):
            raise TypeError(
                "Derivatives field 'inexact' must be an InexactSuite or "
                f"None; got {type(self.inexact).__name__}"
            )

    @staticmethod
    def none() -> "Derivatives[A]":
        """Bundle with no capabilities."""
        return Derivatives()

    @staticmethod
    def first_order(
        jacobian: JacobianFn[A],
        *,
        jvp: Optional[JVPFn[A]] = None,
        jacobian_batch: Optional[JacobianBatchFn[A]] = None,
    ) -> "Derivatives[A]":
        """First-order capability: jacobian required."""
        if jacobian is None:
            raise TypeError("first_order requires a jacobian callable")
        return Derivatives(
            jacobian=jacobian, jvp=jvp, jacobian_batch=jacobian_batch
        )

    @staticmethod
    def second_order(
        jacobian: JacobianFn[A],
        hvp: HVPFn[A],
        *,
        jvp: Optional[JVPFn[A]] = None,
        jacobian_batch: Optional[JacobianBatchFn[A]] = None,
        hvp_batch: Optional[HVPBatchFn[A]] = None,
        hessian_batch: Optional[HessianBatchFn[A]] = None,
    ) -> "Derivatives[A]":
        """Scalar C^2 capability: jacobian AND hvp required (nqoi == 1).

        Unusual combinations (e.g. a materialized single-sample
        ``hessian``, or hvp AND whvp together) use the raw constructor.
        """
        if jacobian is None or hvp is None:
            raise TypeError(
                "second_order requires both jacobian and hvp callables; "
                "use first_order if hvp is unavailable"
            )
        return Derivatives(
            jacobian=jacobian,
            hvp=hvp,
            jvp=jvp,
            jacobian_batch=jacobian_batch,
            hvp_batch=hvp_batch,
            hessian_batch=hessian_batch,
        )

    @staticmethod
    def second_order_weighted(
        jacobian: JacobianFn[A],
        whvp: WHVPFn[A],
        *,
        jvp: Optional[JVPFn[A]] = None,
        jacobian_batch: Optional[JacobianBatchFn[A]] = None,
        whvp_batch: Optional[WHVPBatchFn[A]] = None,
    ) -> "Derivatives[A]":
        """Vector-valued adjoint-Hessian capability: jacobian AND whvp."""
        if jacobian is None or whvp is None:
            raise TypeError(
                "second_order_weighted requires both jacobian and whvp "
                "callables; use first_order if whvp is unavailable"
            )
        return Derivatives(
            jacobian=jacobian,
            whvp=whvp,
            jvp=jvp,
            jacobian_batch=jacobian_batch,
            whvp_batch=whvp_batch,
        )

    def with_(
        self,
        *,
        jacobian: Union[Optional[JacobianFn[Array]], _Unset] = _UNSET,
        jacobian_batch: Union[
            Optional[JacobianBatchFn[Array]], _Unset
        ] = _UNSET,
        jvp: Union[Optional[JVPFn[Array]], _Unset] = _UNSET,
        hvp: Union[Optional[HVPFn[Array]], _Unset] = _UNSET,
        whvp: Union[Optional[WHVPFn[Array]], _Unset] = _UNSET,
        hessian: Union[Optional[HessianFn[Array]], _Unset] = _UNSET,
        hessian_batch: Union[
            Optional[HessianBatchFn[Array]], _Unset
        ] = _UNSET,
        hvp_batch: Union[Optional[HVPBatchFn[Array]], _Unset] = _UNSET,
        whvp_batch: Union[Optional[WHVPBatchFn[Array]], _Unset] = _UNSET,
        inexact: Union[Optional[InexactSuite[Array]], _Unset] = _UNSET,
    ) -> "Derivatives[Array]":
        """Return a copy with the given fields overridden.

        Passing None explicitly REMOVES a capability; omitted fields are
        kept unchanged. Built by direct construction (not
        ``dataclasses.replace``) so mypy checks every field value in the
        body, not only at the call site.
        """
        return Derivatives(
            jacobian=self.jacobian
            if isinstance(jacobian, _Unset)
            else jacobian,
            jacobian_batch=self.jacobian_batch
            if isinstance(jacobian_batch, _Unset)
            else jacobian_batch,
            jvp=self.jvp if isinstance(jvp, _Unset) else jvp,
            hvp=self.hvp if isinstance(hvp, _Unset) else hvp,
            whvp=self.whvp if isinstance(whvp, _Unset) else whvp,
            hessian=self.hessian
            if isinstance(hessian, _Unset)
            else hessian,
            hessian_batch=self.hessian_batch
            if isinstance(hessian_batch, _Unset)
            else hessian_batch,
            hvp_batch=self.hvp_batch
            if isinstance(hvp_batch, _Unset)
            else hvp_batch,
            whvp_batch=self.whvp_batch
            if isinstance(whvp_batch, _Unset)
            else whvp_batch,
            inexact=self.inexact
            if isinstance(inexact, _Unset)
            else inexact,
        )

    def resolved_hvp(
        self, nqoi: int, bkd: Backend[Array]
    ) -> Optional[HVPFn[Array]]:
        """Plain Hessian-vector product, synthesized from whvp when nqoi==1.

        Prefers ``hvp``; when only ``whvp`` is populated and nqoi == 1 the
        weighted form with w = [1] IS the plain hvp. ``nqoi`` must be the
        OWNING object's ``nqoi()``, never another participant's. The
        synthesized form is a module-level callable object so it (and any
        bundle holding it) stays picklable, e.g. for multiprocessing.
        """
        if self.hvp is not None:
            return self.hvp
        whvp = self.whvp
        if whvp is None or nqoi != 1:
            return None
        return _HVPFromWHVP(whvp, bkd.ones((1, 1)))

    def resolved_whvp(self, nqoi: int) -> Optional[WHVPFn[Array]]:
        """Weighted (adjoint) Hessian-vector product, lifted from hvp.

        Prefers ``whvp``; when only ``hvp`` is populated and nqoi == 1,
        the weighted form is w[0] * hvp. ``nqoi`` must be the OWNING
        object's ``nqoi()`` (a constraint's own nqoi, not the
        objective's). The lifted form is a module-level callable object so
        it stays picklable.
        """
        if self.whvp is not None:
            return self.whvp
        hvp = self.hvp
        if hvp is None or nqoi != 1:
            return None
        return _WHVPFromHVP(hvp)


@dataclass(frozen=True)
class _HVPFromWHVP(Generic[Array]):
    """Picklable synthesis: plain hvp = whvp with w = [1] (nqoi == 1)."""

    whvp: WHVPFn[Array]
    weights: Array

    def __call__(self, sample: Array, vec: Array) -> Array:
        return self.whvp(sample, vec, self.weights)


@dataclass(frozen=True)
class _WHVPFromHVP(Generic[Array]):
    """Picklable lift: whvp = w[0] * hvp (nqoi == 1)."""

    hvp: HVPFn[Array]

    def __call__(self, sample: Array, vec: Array, weights: Array) -> Array:
        return weights[0, 0] * self.hvp(sample, vec)


def with_shape_validation(
    d: Derivatives[Array], nvars: int, nqoi: int
) -> Derivatives[Array]:
    """Wrap each populated field with boundary shape checks (opt-in).

    Composed via ``with_`` so mypy checks every wrapped callable against
    its field type. Concrete classes keep their internal shape checks;
    this is an extra boundary guard for e.g. debugging user-provided
    bundles. The wrappers are module-level callable objects
    (:mod:`pyapprox.interface.functions._shape_checked`), not closures, so
    validated bundles stay picklable, e.g. for multiprocessing.
    """
    # local import: _shape_checked imports the validators, whose package
    # __init__ leads back to this module
    from pyapprox.interface.functions import _shape_checked as checked

    out = d
    jacobian = d.jacobian
    if jacobian is not None:
        out = out.with_(
            jacobian=checked.CheckedJacobian(jacobian, nvars, nqoi)
        )
    hvp = d.hvp
    if hvp is not None:
        out = out.with_(hvp=checked.CheckedHVP(hvp, nvars))
    whvp = d.whvp
    if whvp is not None:
        out = out.with_(whvp=checked.CheckedWHVP(whvp, nvars, nqoi))
    hessian = d.hessian
    if hessian is not None:
        out = out.with_(hessian=checked.CheckedHessian(hessian, nvars))
    hessian_batch = d.hessian_batch
    if hessian_batch is not None:
        out = out.with_(
            hessian_batch=checked.CheckedHessianBatch(hessian_batch, nvars)
        )
    hvp_batch = d.hvp_batch
    if hvp_batch is not None:
        out = out.with_(hvp_batch=checked.CheckedHVPBatch(hvp_batch, nvars))
    whvp_batch = d.whvp_batch
    if whvp_batch is not None:
        out = out.with_(
            whvp_batch=checked.CheckedWHVPBatch(whvp_batch, nvars, nqoi)
        )
    return out


def _check_shape(
    label: str, got: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if got != expected:
        raise ValueError(f"{label}: expected shape {expected}, got {got}")

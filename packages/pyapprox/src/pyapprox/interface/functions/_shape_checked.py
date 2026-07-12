"""Picklable shape-checking wrappers for Derivatives bundle fields.

Module-level frozen dataclass callables (not closures) so a
shape-validated bundle survives pickling, e.g. for multiprocessing.
Built by :func:`pyapprox.interface.functions.derivatives.with_shape_validation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from pyapprox.interface.functions.derivatives import (
    HessianBatchFn,
    HessianFn,
    HVPBatchFn,
    HVPFn,
    JacobianFn,
    WHVPBatchFn,
    WHVPFn,
    _check_shape,
)
from pyapprox.interface.functions.protocols.validation import (
    validate_jacobian,
    validate_sample,
    validate_samples,
    validate_vector_for_apply,
)
from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class CheckedJacobian(Generic[Array]):
    inner: JacobianFn[Array]
    nvars: int
    nqoi: int

    def __call__(self, sample: Array) -> Array:
        validate_sample(self.nvars, sample)
        result = self.inner(sample)
        validate_jacobian(self.nqoi, self.nvars, result)
        return result


@dataclass(frozen=True)
class CheckedHVP(Generic[Array]):
    inner: HVPFn[Array]
    nvars: int

    def __call__(self, sample: Array, vec: Array) -> Array:
        validate_sample(self.nvars, sample)
        validate_vector_for_apply(self.nvars, vec)
        result = self.inner(sample, vec)
        _check_shape("hvp output", result.shape, (self.nvars, 1))
        return result


@dataclass(frozen=True)
class CheckedWHVP(Generic[Array]):
    inner: WHVPFn[Array]
    nvars: int
    nqoi: int

    def __call__(self, sample: Array, vec: Array, weights: Array) -> Array:
        validate_sample(self.nvars, sample)
        validate_vector_for_apply(self.nvars, vec)
        _check_shape("whvp weights", weights.shape, (self.nqoi, 1))
        result = self.inner(sample, vec, weights)
        _check_shape("whvp output", result.shape, (self.nvars, 1))
        return result


@dataclass(frozen=True)
class CheckedHessian(Generic[Array]):
    inner: HessianFn[Array]
    nvars: int

    def __call__(self, sample: Array) -> Array:
        validate_sample(self.nvars, sample)
        result = self.inner(sample)
        _check_shape(
            "hessian output", result.shape, (self.nvars, self.nvars)
        )
        return result


@dataclass(frozen=True)
class CheckedHessianBatch(Generic[Array]):
    inner: HessianBatchFn[Array]
    nvars: int

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars, samples)
        result = self.inner(samples)
        _check_shape(
            "hessian_batch output",
            result.shape,
            (samples.shape[1], self.nvars, self.nvars),
        )
        return result


@dataclass(frozen=True)
class CheckedHVPBatch(Generic[Array]):
    inner: HVPBatchFn[Array]
    nvars: int

    def __call__(self, samples: Array, vecs: Array) -> Array:
        validate_samples(self.nvars, samples)
        _check_shape("hvp_batch vecs", vecs.shape, samples.shape)
        result = self.inner(samples, vecs)
        # scalar-implicit: (nsamples, nvars), no nqoi axis
        _check_shape(
            "hvp_batch output",
            result.shape,
            (samples.shape[1], self.nvars),
        )
        return result


@dataclass(frozen=True)
class CheckedWHVPBatch(Generic[Array]):
    inner: WHVPBatchFn[Array]
    nvars: int
    nqoi: int

    def __call__(self, samples: Array, vecs: Array, weights: Array) -> Array:
        validate_samples(self.nvars, samples)
        _check_shape("whvp_batch vecs", vecs.shape, samples.shape)
        _check_shape("whvp_batch weights", weights.shape, (self.nqoi, 1))
        result = self.inner(samples, vecs, weights)
        _check_shape(
            "whvp_batch output",
            result.shape,
            (samples.shape[1], self.nvars),
        )
        return result

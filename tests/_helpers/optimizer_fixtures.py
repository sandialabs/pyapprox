"""Legacy-style producers for scipy-optimizer regression tests.

Written against the CURRENT capability convention (real ``def jacobian`` /
``def hvp`` methods discovered by the optimizers) and exercised purely
through the public bind()/minimize() API, so these tests survive the
Derivatives-bundle consumer rewrite unchanged and protect it.
"""

from typing import Any, Generic, List

from pyapprox.util.backends.protocols import Array, Backend


class QuadraticNoDerivatives(Generic[Array]):
    """f(x) = sum_i (x_i - c_i)^2 — value only (finite-difference path)."""

    def __init__(self, bkd: Backend[Array], center: List[float]) -> None:
        self._bkd = bkd
        self._center = bkd.asarray(center)[:, None]
        self._nvars = len(center)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        shifted = samples - self._center
        return self._bkd.sum(shifted * shifted, axis=0)[None, :]


class QuadraticWithJacobian(QuadraticNoDerivatives[Array]):
    """Adds an analytic jacobian and counts its invocations."""

    def __init__(self, bkd: Backend[Array], center: List[float]) -> None:
        super().__init__(bkd, center)
        self.njacobian_calls = 0

    def jacobian(self, sample: Array) -> Array:
        self.njacobian_calls += 1
        return ((sample - self._center) * 2.0).T


class QuadraticWithJacobianAndHVP(QuadraticWithJacobian[Array]):
    """Adds an analytic hvp; records the dtype of every incoming vec.

    scipy's trust-constr probes hessp with an int8 vector during setup;
    the optimizer stack must coerce it to double BEFORE it reaches the
    producer. ``foreign_vec_dtypes`` stays empty when that holds.
    """

    def __init__(self, bkd: Backend[Array], center: List[float]) -> None:
        super().__init__(bkd, center)
        self.nhvp_calls = 0
        self.foreign_vec_dtypes: List[Any] = []

    def hvp(self, sample: Array, vec: Array) -> Array:
        self.nhvp_calls += 1
        if vec.dtype != self._bkd.double_dtype():
            self.foreign_vec_dtypes.append(vec.dtype)
        return vec * 2.0


class SumConstraint(Generic[Array]):
    """Constraint lb <= sum_i x_i <= ub — value only."""

    def __init__(
        self, bkd: Backend[Array], nvars: int, lb: float, ub: float
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        # constraint bounds are 1D (nqoi,) per the existing convention
        self._lb = bkd.asarray([lb])
        self._ub = bkd.asarray([ub])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._bkd.sum(samples, axis=0)[None, :]

    def lb(self) -> Array:
        return self._lb

    def ub(self) -> Array:
        return self._ub


class SumConstraintWithJacobian(SumConstraint[Array]):
    """Adds the analytic (constant) jacobian."""

    def __init__(
        self, bkd: Backend[Array], nvars: int, lb: float, ub: float
    ) -> None:
        super().__init__(bkd, nvars, lb, ub)
        self.njacobian_calls = 0

    def jacobian(self, sample: Array) -> Array:
        self.njacobian_calls += 1
        return self._bkd.ones((1, self._nvars))


class SumConstraintWithJacobianAndWHVP(SumConstraintWithJacobian[Array]):
    """Adds the (zero) weighted Hessian-vector product of a linear map and
    records the dtypes of incoming vec/weights (int8-probe regression)."""

    def __init__(
        self, bkd: Backend[Array], nvars: int, lb: float, ub: float
    ) -> None:
        super().__init__(bkd, nvars, lb, ub)
        self.nwhvp_calls = 0
        self.foreign_dtypes: List[Any] = []

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        self.nwhvp_calls += 1
        for array in (vec, weights):
            if array.dtype != self._bkd.double_dtype():
                self.foreign_dtypes.append(array.dtype)
        return vec * 0.0

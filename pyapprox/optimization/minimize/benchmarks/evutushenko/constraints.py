from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
    validate_vector_for_apply,
    validate_hvp,
    validate_values,
    validate_jacobian,
)


class EvtushenkoNonLinearConstraint(Generic[Array]):
    """
    Nonlinear constraint of the constrained optimization benchmark from Evtushenko.

    The constraint is defined as:
        c(z) = 6z_2 + 4z_3 - z_1^3 - 3

    Parameters
    ----------
    backend : Backend[Array]
        Backend for numerical computations.
    """

    def __init__(self, backend: Backend[Array]):
        self._bkd = backend
        self._lb = backend.asarray([0.0])
        self._ub = backend.asarray([np.inf])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return 3

    def lb(self) -> Array:
        return self._lb

    def ub(self) -> Array:
        return self._ub

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        vals = (6 * samples[1] + 4 * samples[2] - samples[0] ** 3 - 3)[None, :]
        validate_values(self.nqoi(), samples, vals)
        return vals

    def jacobian(self, sample: Array) -> Array:
        validate_sample(self.nvars(), sample)
        jac = self._bkd.stack(
            (
                -3.0 * sample[0] ** 2,
                self._bkd.array([6.0]),
                self._bkd.array([4.0]),
            ),
            axis=1,
        )
        validate_jacobian(self.nqoi(), self.nvars(), jac)
        return jac

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_vector_for_apply(self.nvars(), vec)
        validate_sample(self.nqoi(), weights)
        return self._bkd.hstack(
            [-6 * sample[0] * vec[0] * weights[0], self._bkd.zeros((2,))]
        )[:, None]

    def weighted_hessian(self, sample: Array, weights: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_sample(self.nqoi(), weights)
        hess = self._bkd.zeros((3, 3))
        hess[0, 0] = -6 * sample[0, 0] * weights[0, 0]
        return hess

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

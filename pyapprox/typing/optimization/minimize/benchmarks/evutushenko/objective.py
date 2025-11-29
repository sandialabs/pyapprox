from typing import Generic

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
    validate_vector_for_apply,
    validate_hvp,
    validate_values,
    validate_jacobian,
)


class EvtushenkoObjective(Generic[Array]):
    """
    Objective of the constrained optimization benchmark from Evtushenko.

    The objective function is defined as:
        f(z) = (z_1 + 3z_2 + z_3)^2 + 4(z_1 - z_2)^2

    Parameters
    ----------
    backend : Backend[Array]
        Backend for numerical computations.
    """

    def __init__(self, backend: Backend[Array]):
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return 3

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        vals = (
            (samples[0] + 3 * samples[1] + samples[2]) ** 2
            + 4 * (samples[0] - samples[1]) ** 2
        )[None, :]
        validate_values(self.nqoi(), samples, vals)
        return vals

    def jacobian(self, sample: Array) -> Array:
        validate_sample(self.nvars(), sample)
        jac = self._bkd.stack(
            (
                2 * (sample[0] + 3 * sample[1] + sample[2])
                + 8 * (sample[0] - sample[1]),
                6 * (sample[0] + 3 * sample[1] + sample[2])
                - 8 * (sample[0] - sample[1]),
                2 * (sample[0] + 3 * sample[1] + sample[2]),
            ),
            axis=1,
        )
        validate_jacobian(self.nqoi(), self.nvars(), jac)
        return jac

    def hvp(self, sample: Array, vec: Array) -> Array:
        validate_sample(self.nvars(), sample)
        validate_vector_for_apply(self.nvars(), vec)
        hvp = self._bkd.stack(
            (
                10 * vec[0] - 2 * vec[1] + 2 * vec[2],
                -2 * vec[0] + 26 * vec[1] + 6 * vec[2],
                2 * vec[0] + 6 * vec[1] + 2 * vec[2],
            ),
            axis=0,
        )
        validate_hvp(self.nvars(), hvp)
        return hvp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

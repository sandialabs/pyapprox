"""Forrester multi-fidelity test function ensemble.

High fidelity: f_h(x) = (6x - 2)^2 sin(12x - 4)
Low fidelity:  f_l(x) = A * f_h(x) + B * (x - 0.5) + C

Default parameters (Forrester et al., 2008): A = 0.5, B = 10, C = -5.
Domain: x in [0, 1].
"""

from typing import Generic, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class ForresterModelFunction(Generic[Array]):
    """Single Forrester model function.

    High-fidelity form: f(x) = (6x - 2)^2 sin(12x - 4)
    Low-fidelity form:  f(x) = A * f_h(x) + B * (x - 0.5) + C

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    A : float or None
        Scaling of the HF function. None means this IS the HF function.
    B : float
        Linear trend coefficient.
    C : float
        Constant shift.
    """
    # TODO: add references section with citation Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling: a practical guide. Wiley. and html link

    def __init__(
        self,
        bkd: Backend[Array],
        A: float = None,
        B: float = 0.0,
        C: float = 0.0,
    ) -> None:
        self._bkd = bkd
        self._A = A
        self._B = B
        self._C = C

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the Forrester function.

        Parameters
        ----------
        samples : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (1, nsamples).
        """
        x = samples[0:1, :]
        fh = (6.0 * x - 2.0) ** 2 * self._bkd.sin(12.0 * x - 4.0)
        if self._A is None:
            return fh
        return self._A * fh + self._B * (x - 0.5) + self._C


class ForresterEnsemble(Generic[Array]):
    """Two-model Forrester ensemble.

    Model 0 (high fidelity): f_h(x) = (6x - 2)^2 sin(12x - 4)
    Model 1 (low fidelity):  f_l(x) = A * f_h(x) + B * (x - 0.5) + C

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    A : float
        Scaling of HF function in the LF model.
    B : float
        Linear trend coefficient in the LF model.
    C : float
        Constant shift in the LF model.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        A: float = 0.5,
        B: float = 10.0,
        C: float = -5.0,
    ) -> None:
        self._bkd = bkd
        self._models = [
            ForresterModelFunction(bkd),
            ForresterModelFunction(bkd, A=A, B=B, C=C),
        ]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nmodels(self) -> int:
        return 2

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return 1

    def models(self) -> Sequence[ForresterModelFunction[Array]]:
        return self._models

    def __getitem__(self, idx: int) -> ForresterModelFunction[Array]:
        return self._models[idx]

    def costs(self) -> Array:
        return self._bkd.array([1.0, 0.1])

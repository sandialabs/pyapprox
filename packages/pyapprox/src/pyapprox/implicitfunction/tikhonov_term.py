r"""A Tikhonov regularization of the parameters, a parameter term of any
functional: of a steady solution or of a trajectory."""

from typing import Generic, Optional

from pyapprox.implicitfunction.functional_terms import (
    BackendScalarGradient,
    ParameterTermDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend


class TikhonovTerm(Generic[Array]):
    r"""The regularization :math:`(\alpha/2)\, p^\top W p` of the
    parameters last set (:math:`W = I` by default), a parameter term.

    Parameters
    ----------
    bkd : Backend
        The backend.
    alpha : float
        :math:`\alpha`.
    weight : Array, optional
        :math:`W`, symmetric, shape ``(nparams, nparams)``.

    Raises
    ------
    ValueError
        If ``weight`` is not square or not symmetric.
    """

    def __init__(
        self, bkd: Backend[Array], alpha: float, weight: Optional[Array] = None
    ) -> None:
        if weight is not None:
            if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"weight must be square, got shape {tuple(weight.shape)}"
                )
            asymmetry = bkd.to_float(bkd.max(bkd.abs(weight - weight.T)))
            if asymmetry > 1e-12 * max(bkd.to_float(bkd.max(bkd.abs(weight))), 1.0):
                raise ValueError("weight must be symmetric")
        self._bkd, self._alpha, self._weight = bkd, alpha, weight
        self._param: Optional[Array] = None
        self._derivatives: ParameterTermDerivatives[Array, Array] = (
            ParameterTermDerivatives.second_order(self._wrt_param, self._param_param)
        )

    def set_param(self, param: Array) -> None:
        """Set :math:`p`, a column."""
        if param.ndim != 2 or param.shape[1] != 1:
            raise ValueError(
                f"param must be a column (nparams, 1), got {tuple(param.shape)}"
            )
        self._param = param

    def _weighted(self, v: Array) -> Array:
        return v if self._weight is None else self._bkd.dot(self._weight, v)

    def _current(self) -> Array:
        if self._param is None:
            raise ValueError("set_param must be called before the term is read")
        return self._param

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        param = self._current()
        return (0.5 * self._alpha) * self._bkd.dot(param.T, self._weighted(param))

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return self._derivatives

    def _wrt_param(self) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(
            self._alpha * self._weighted(self._current()), self._bkd
        )

    def _param_param(self, weights: Array, v: Array) -> Array:
        return self._alpha * self._weighted(v) * weights

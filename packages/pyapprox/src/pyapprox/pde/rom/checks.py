r"""Checks that a cheaper reduced residual agrees with a reference.

An affine or hyper-reduced residual is an evaluation strategy for the
same mathematical object as the exact projection, so it can be checked
against it. A check needs only the fitted residual and a reference, so
it runs on a fit's result rather than inside the fit: which points,
which quantities and what tolerance stay the caller's choice, any
reference can be used, and the cost of building the exact projection
is visible where it is paid. :class:`AgreementCheck` compares values
and Jacobians.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.pde.rom.projection import ReducedResidualProtocol
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class ReducedResidualCheckProtocol(Protocol, Generic[Array]):
    """Compares a fitted reduced residual with a reference one."""

    def check(
        self,
        fitted: ReducedResidualProtocol[Array],
        reference: ReducedResidualProtocol[Array],
    ) -> None:
        """Return if they agree; raise ``ValueError`` otherwise."""
        ...


class AgreementCheck(Generic[Array]):
    r"""Residual and state Jacobian agree at given parameters and codes.

    For a residual affine in the codes, agreement of the value and the
    Jacobian at one code fixes it everywhere, so the default code is
    zero (the offset). For anything else, pass several codes.

    Parameters
    ----------
    params : Array
        Parameters to compare at. Shape: (nparams, k).
    codes : Array, optional
        Codes to compare at. Shape: (ncodes, m). Default: zero only.
    rtol : float
        Tolerance on the largest absolute difference, relative to the
        largest absolute entry of the reference.
    time : float
        Time to compare at.
    """

    def __init__(
        self,
        params: Array,
        codes: Optional[Array] = None,
        rtol: float = 1e-8,
        time: float = 0.0,
    ) -> None:
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (nparams, k), got ndim={params.ndim}")
        if codes is not None and codes.ndim != 2:
            raise ValueError(f"codes must be 2D (ncodes, m), got ndim={codes.ndim}")
        self._params = params
        self._codes = codes
        self._rtol = rtol
        self._time = time

    def check(
        self,
        fitted: ReducedResidualProtocol[Array],
        reference: ReducedResidualProtocol[Array],
    ) -> None:
        """Raise ``ValueError`` at the first quantity that disagrees."""
        bkd = reference.bkd()
        codes = (
            bkd.zeros((reference.ncodes(), 1)) if self._codes is None else self._codes
        )
        for jj in range(int(self._params.shape[1])):
            params = self._params[:, jj]
            for kk in range(int(codes.shape[1])):
                code = codes[:, kk]
                for name, fitted_value, reference_value in (
                    (
                        "residual",
                        fitted.residual(code, self._time, params),
                        reference.residual(code, self._time, params),
                    ),
                    (
                        "state Jacobian",
                        fitted.state_jacobian(code, self._time, params),
                        reference.state_jacobian(code, self._time, params),
                    ),
                ):
                    scale = bkd.to_float(bkd.max(bkd.abs(reference_value)))
                    gap = bkd.to_float(bkd.max(bkd.abs(fitted_value - reference_value)))
                    if gap > self._rtol * scale:
                        raise ValueError(
                            f"fitted {name} departs from the reference by "
                            f"{gap:.2e} (scale {scale:.2e}) at parameter "
                            f"column {jj}, code column {kk}"
                        )

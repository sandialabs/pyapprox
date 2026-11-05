"""The parameters a parameterization was last applied with, and how often.

Two systems built over one parameterization, for instance the system a
transient problem is solved with and the unconstrained system its
boundary flux is read from, must read the same parameters, and report
the same parameters version: derivative sweeps check a trajectory
against a stepper's version, and a step operator is reused under it. A
:class:`ParameterState` is that one source. Every system built on it
reads its parameters and its version, so one ``set_param`` moves them
all; no system holds a second copy that could go stale.
"""

from typing import Generic, Optional

from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array


class ParameterState(Generic[Array]):
    """A parameterization, the parameters last applied through it, and
    the number of applications so far.

    Parameters
    ----------
    parameterization : ParameterizationProtocol
        Maps parameter vectors to coefficients of the objects it targets.

    Raises
    ------
    TypeError
        If ``parameterization`` does not satisfy
        :class:`~pyapprox.pde.parameterizations.protocol.ParameterizationProtocol`.
    """

    def __init__(self, parameterization: ParameterizationProtocol[Array]) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                "parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._parameterization = parameterization
        self._current: Optional[Array] = None
        self._version = 0

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """The parameterization."""
        return self._parameterization

    def nparams(self) -> int:
        """The number of parameters."""
        return self._parameterization.nparams()

    def set_param(self, param: Array) -> None:
        """Apply ``param`` through the parameterization.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,)

        Raises
        ------
        ValueError
            If ``param`` is not 1-D.
        """
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D with shape (nparams,), got shape "
                f"{tuple(param.shape)}"
            )
        self._current = param
        self._parameterization.apply(param)
        self._version += 1

    def parameters_version(self) -> int:
        """The number of ``set_param`` calls so far."""
        return self._version

    def current(self) -> Array:
        """The parameters last applied. Shape: (nparams,)

        Raises
        ------
        RuntimeError
            If ``set_param`` has not been called.
        """
        if self._current is None:
            raise RuntimeError(
                "set_param() must be called before parameter derivatives"
            )
        return self._current

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparams={self.nparams()}, "
            f"version={self._version})"
        )

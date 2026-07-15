"""View a distribution's logpdf as an optimizer-bindable function.

A distribution object spends its single function identity (``__call__``
plus ``derivatives()``) on the pdf; the logpdf is a second scalar
function on the same object, exposed through the differently-named
``logpdf_derivatives()`` accessor. Protocol-speaking machinery
(optimizers, ``DerivativeChecker``) binds exactly one
``__call__``/``derivatives()`` pair, so handing them the logpdf requires
this view, which re-points the identity: ``__call__ = logpdf`` and
``derivatives() = logpdf_derivatives()``.

Typical uses: MAP-point optimization over the input space, and
finite-difference validation of a distribution's logpdf bundle.
"""

from typing import Generic

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.probability.protocols.distribution import (
    DistributionHasLogpdfDerivativesProtocol,
    DistributionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class LogpdfFunction(Generic[Array]):
    """Present ``distribution.logpdf`` as an ``ObjectiveProtocol`` function.

    Parameters
    ----------
    distribution : DistributionProtocol[Array]
        Distribution providing ``logpdf`` and ``logpdf_derivatives()``.
    """

    def __init__(self, distribution: DistributionProtocol[Array]) -> None:
        if not isinstance(distribution, DistributionProtocol):
            raise TypeError(
                "distribution must satisfy DistributionProtocol, got "
                f"{type(distribution).__name__}"
            )
        if not isinstance(
            distribution, DistributionHasLogpdfDerivativesProtocol
        ):
            raise TypeError(
                "distribution must expose logpdf_derivatives(), got "
                f"{type(distribution).__name__}"
            )
        self._distribution = distribution

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._distribution.bkd()

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._distribution.nvars()

    def nqoi(self) -> int:
        """Return the number of quantities of interest (always 1)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the logpdf.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        return self._distribution.logpdf(samples)

    def derivatives(self) -> Derivatives[Array]:
        """Return the distribution's logpdf derivative bundle."""
        return self._distribution.logpdf_derivatives()

    def __repr__(self) -> str:
        return f"LogpdfFunction({self._distribution!r})"

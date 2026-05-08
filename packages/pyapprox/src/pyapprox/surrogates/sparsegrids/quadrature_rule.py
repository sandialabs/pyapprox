"""Sparse grid quadrature rule satisfying ParameterizedQuadratureRuleProtocol.

Wraps IsotropicSparseGridFitter to provide a level-parameterized quadrature
rule that returns combined samples and weights at unique positions.
"""

from typing import Callable, Generic, Tuple, cast

from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    SubspaceFactoryProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ParameterizedIsotropicSparseGridQuadratureRule(Generic[Array]):
    """Sparse grid quadrature rule parameterized by level.

    Given a level, creates an :class:`IsotropicSparseGridFitter` and
    returns the combined quadrature samples and weights at unique positions.

    Satisfies :class:`ParameterizedQuadratureRuleProtocol`.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    factory : SubspaceFactoryProtocol[Array]
        Factory for creating tensor product subspaces.
    pnorm : float, optional
        p-norm for the hyperbolic cross index set. Default: 1.0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        factory: SubspaceFactoryProtocol[Array],
        pnorm: float = 1.0,
    ) -> None:
        self._bkd = bkd
        self._factory = factory
        self._pnorm = pnorm

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._factory.nvars_physical()

    def __call__(self, level: int) -> Tuple[Array, Array]:
        """Generate sparse grid quadrature rule for given level.

        Parameters
        ----------
        level : int
            Sparse grid level (higher = more accurate).

        Returns
        -------
        Tuple[Array, Array]
            ``(samples, weights)`` with shapes
            ``(nvars, n_unique)`` and ``(n_unique,)``.
        """
        fitter = IsotropicSparseGridFitter(
            self._bkd, self._factory, level, self._pnorm,
        )
        # nconfig_vars=0 (default), so get_samples() returns Array
        samples = cast(Array, fitter.get_samples())
        return samples, fitter.get_quadrature_weights()

    def integrate(
        self,
        func: Callable[[Array], Array],
        level: int,
    ) -> Array:
        """Integrate a function at given quadrature level.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate. Takes samples ``(nvars, nsamples)``
            and returns values ``(nsamples, nqoi)``.
        level : int
            Quadrature level.

        Returns
        -------
        Array
            Integral estimate of shape ``(nqoi,)``.
        """
        samples, weights = self(level)
        values = func(samples)
        return self._bkd.sum(weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return (
            f"ParameterizedIsotropicSparseGridQuadratureRule("
            f"nvars={self.nvars()}, pnorm={self._pnorm})"
        )

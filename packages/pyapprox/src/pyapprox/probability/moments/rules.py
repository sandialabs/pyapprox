"""Weighted point sets that moment sources integrate with.

``WeightedRuleProtocol`` is the narrowest rule a source needs: a fixed set
of points and weights. Fixed-size multivariate quadrature rules satisfy it
as they are; the adapters here make samplers and level-parameterized rules
satisfy it too.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.protocols.quadrature import ParameterizedQuadratureRuleProtocol
from pyapprox.util.protocols.sampling import QuadratureSamplerProtocol


@runtime_checkable
class WeightedRuleProtocol(Protocol, Generic[Array]):
    """A fixed set of points and weights."""

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def nvars(self) -> int:
        """Number of variables."""
        ...

    def __call__(self) -> Tuple[Array, Array]:
        """Points and weights.

        Returns
        -------
        Tuple[Array, Array]
            Points of shape (nvars, npoints) and weights of shape
            (npoints,), as the existing quadrature rules return them.
        """
        ...


class SampledRule(Generic[Array]):
    """A sampler's points drawn once, as a fixed rule.

    The points are drawn at construction, so every call returns the same
    rule; the sampler is not reset, and its state advances only once.

    Parameters
    ----------
    sampler : QuadratureSamplerProtocol[Array]
        Monte Carlo, quasi-Monte Carlo or other sampler.
    nsamples : int
        Number of points to draw.
    """

    def __init__(
        self, sampler: QuadratureSamplerProtocol[Array], nsamples: int
    ) -> None:
        if not isinstance(sampler, QuadratureSamplerProtocol):
            raise TypeError(
                "sampler must satisfy QuadratureSamplerProtocol, got "
                f"{type(sampler).__name__}"
            )
        if nsamples < 1:
            raise ValueError(f"nsamples must be positive, got {nsamples}")
        self._bkd = sampler.bkd()
        self._nvars = sampler.nvars()
        self._points, self._weights = sampler.sample(nsamples)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of variables."""
        return self._nvars

    def __call__(self) -> Tuple[Array, Array]:
        """The drawn points (nvars, nsamples) and weights (nsamples,)."""
        return self._points, self._weights


class AtLevel(Generic[Array]):
    """A level-parameterized rule fixed at one level.

    Parameters
    ----------
    rule : ParameterizedQuadratureRuleProtocol[Array]
        For example an isotropic sparse grid or a tensor-product rule.
    level : int
        The level to evaluate it at.
    """

    def __init__(
        self, rule: ParameterizedQuadratureRuleProtocol[Array], level: int
    ) -> None:
        if not isinstance(rule, ParameterizedQuadratureRuleProtocol):
            raise TypeError(
                "rule must satisfy ParameterizedQuadratureRuleProtocol, got "
                f"{type(rule).__name__}"
            )
        self._rule = rule
        self._level = level

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._rule.bkd()

    def nvars(self) -> int:
        """Number of variables."""
        return self._rule.nvars()

    def __call__(self) -> Tuple[Array, Array]:
        """Points (nvars, npoints) and weights (npoints,) at the level."""
        return self._rule(self._level)

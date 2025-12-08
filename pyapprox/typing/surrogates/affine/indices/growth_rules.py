"""Index growth rules for tensor-product approximations.

Growth rules map polynomial levels to the number of basis functions,
controlling how approximations grow with increasing level.

Common rules:
- Linear: n(l) = scale * l + shift
- Double plus one: n(l) = 2^l + 1 (nested Clenshaw-Curtis)
"""

from abc import ABC, abstractmethod


class IndexGrowthRule(ABC):
    """Abstract base class for index growth rules.

    Growth rules determine how many basis functions are used at each
    polynomial level in tensor-product approximations.
    """

    @abstractmethod
    def __call__(self, level: int) -> int:
        """Return the number of basis functions at the given level.

        Parameters
        ----------
        level : int
            Polynomial level.

        Returns
        -------
        int
            Number of basis functions at this level.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class LinearGrowthRule(IndexGrowthRule):
    """Linear growth rule: n(l) = scale * l + shift for l > 0.

    At level 0, always returns 1.

    Parameters
    ----------
    scale : int
        Multiplicative factor.
    shift : int
        Additive shift.

    Examples
    --------
    >>> rule = LinearGrowthRule(scale=2, shift=1)
    >>> [rule(l) for l in range(5)]
    [1, 3, 5, 7, 9]
    """

    def __init__(self, scale: int, shift: int):
        self._scale = scale
        self._shift = shift

    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return self._scale * level + self._shift

    def __repr__(self) -> str:
        return f"LinearGrowthRule(scale={self._scale}, shift={self._shift})"


class DoublePlusOneGrowthRule(IndexGrowthRule):
    """Double plus one growth rule: n(l) = 2^l + 1 for l > 0.

    At level 0, returns 1.
    This rule corresponds to nested Clenshaw-Curtis quadrature.

    Examples
    --------
    >>> rule = DoublePlusOneGrowthRule()
    >>> [rule(l) for l in range(5)]
    [1, 3, 5, 9, 17]
    """

    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return 2**level + 1

    def __repr__(self) -> str:
        return "DoublePlusOneGrowthRule()"


class ConstantGrowthRule(IndexGrowthRule):
    """Constant growth rule: n(l) = value for all levels.

    Parameters
    ----------
    value : int
        Number of basis functions at every level.
    """

    def __init__(self, value: int):
        self._value = value

    def __call__(self, level: int) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"ConstantGrowthRule(value={self._value})"


class ExponentialGrowthRule(IndexGrowthRule):
    """Exponential growth rule: n(l) = base^l.

    Parameters
    ----------
    base : int
        Base of the exponential.

    Examples
    --------
    >>> rule = ExponentialGrowthRule(base=2)
    >>> [rule(l) for l in range(5)]
    [1, 2, 4, 8, 16]
    """

    def __init__(self, base: int):
        self._base = base

    def __call__(self, level: int) -> int:
        return self._base**level

    def __repr__(self) -> str:
        return f"ExponentialGrowthRule(base={self._base})"

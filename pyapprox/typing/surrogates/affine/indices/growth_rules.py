"""Index growth rules for tensor-product approximations.

Growth rules map polynomial levels to the number of basis functions,
controlling how approximations grow with increasing level.

Common rules:
- Linear: n(l) = scale * l + shift
- Clenshaw-Curtis: n(l) = 2^l + 1 (nested Clenshaw-Curtis)
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


class ClenshawCurtisGrowthRule(IndexGrowthRule):
    """Clenshaw-Curtis nested growth rule: n(l) = 2^l + 1 for l > 0.

    At level 0, returns 1.
    This rule produces nested point sets compatible with Clenshaw-Curtis
    quadrature, where points at level l are a subset of points at level l+1.

    Examples
    --------
    >>> rule = ClenshawCurtisGrowthRule()
    >>> [rule(l) for l in range(5)]
    [1, 3, 5, 9, 17]
    """

    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return 2**level + 1

    def __repr__(self) -> str:
        return "ClenshawCurtisGrowthRule()"


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


class CubicNestedGrowthRule(IndexGrowthRule):
    """Nested growth rule for cubic piecewise polynomials: n(l) = 3 * 2^(l-1) + 1.

    At level 0, returns 1.

    This rule produces nested equidistant points compatible with cubic
    piecewise polynomial bases, which require (n - 4) % 3 == 0.

    The sequence is: 1, 4, 7, 13, 25, 49, 97, ...

    For l > 0: n(l) - 1 = 3 * 2^(l-1), which ensures:
    - n(l) - 4 = 3 * (2^(l-1) - 1), divisible by 3 ✓
    - n(l+1) - 1 = 2 * (n(l) - 1), so equidistant points are nested ✓

    Examples
    --------
    >>> rule = CubicNestedGrowthRule()
    >>> [rule(l) for l in range(6)]
    [1, 4, 7, 13, 25, 49]
    """

    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return 3 * (2 ** (level - 1)) + 1

    def __repr__(self) -> str:
        return "CubicNestedGrowthRule()"


def inverse_growth_rule(degree: int, growth_rule: IndexGrowthRule) -> int:
    """Find minimum level l such that growth_rule(l) > degree.

    This is the inverse of the growth rule: given a polynomial degree d,
    find the minimum sparse grid level needed to represent polynomials
    of that degree. Since Lagrange interpolation with n points can
    exactly represent polynomials of degree n-1, we need the first level
    where growth_rule(l) > degree.

    Parameters
    ----------
    degree : int
        Polynomial degree to represent.
    growth_rule : IndexGrowthRule
        Growth rule mapping level to number of points.

    Returns
    -------
    int
        Minimum level l such that growth_rule(l) > degree.

    Raises
    ------
    ValueError
        If degree is negative or if no level found within search limit.

    Examples
    --------
    >>> rule = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1
    >>> inverse_growth_rule(0, rule)  # need n > 0, so n=1 at level 0
    0
    >>> inverse_growth_rule(1, rule)  # need n > 1, so n=2 at level 1
    1
    >>> inverse_growth_rule(3, rule)  # need n > 3, so n=4 at level 3
    3

    >>> rule = LinearGrowthRule(scale=2, shift=1)  # n(l) = 2*l + 1 for l>0
    >>> inverse_growth_rule(2, rule)  # need n > 2, so n=3 at level 1
    1
    >>> inverse_growth_rule(4, rule)  # need n > 4, so n=5 at level 2
    2
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")

    for level in range(1000):
        if growth_rule(level) > degree:
            return level

    raise ValueError(
        f"Could not find level for degree {degree} within search limit"
    )

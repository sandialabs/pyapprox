"""Differentiable approximations to non-smooth functions.

This module provides smooth (infinitely differentiable) approximations to
non-differentiable functions, enabling their use in gradient-based optimization.

Key functions approximated:
- max(0, x): The positive part / ReLU function
- H(-x): Left Heaviside step function (1 if x <= 0, else 0)
- H(x): Right Heaviside step function (1 if x >= 0, else 0)

These approximations are parameterized by epsilon (eps), controlling smoothness:
- As eps -> 0, approximations converge to the original non-smooth functions
- Larger eps gives smoother but less accurate approximations

Primary use cases:
- Stochastic dominance constraints (FSD, SSD)
- Any optimization involving indicator or max functions
"""

from abc import abstractmethod
from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DifferentiableApproximationProtocol(Protocol, Generic[Array]):
    """Protocol for differentiable approximations to non-smooth functions.

    Implementations provide the function value and its first two derivatives,
    enabling use with second-order optimization methods.
    """

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        ...

    def eps(self) -> float:
        """Return smoothing parameter epsilon."""
        ...

    def __call__(self, x: Array) -> Array:
        """Evaluate the smooth approximation.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Approximated values. Shape: (ndim, nsamples)
        """
        ...

    def first_derivative(self, x: Array) -> Array:
        """First derivative of the smooth approximation.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Derivative values. Shape: (ndim, nsamples)
        """
        ...

    def second_derivative(self, x: Array) -> Array:
        """Second derivative of the smooth approximation.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Second derivative values. Shape: (ndim, nsamples)
        """
        ...


class DifferentiableApproximationBase(Generic[Array]):
    """Base class for differentiable approximations.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float
        Smoothing parameter. Smaller values give sharper approximations.
    threshold : float, optional
        Threshold for numerical stability. Default: 100.0.
        Values |x/eps| > threshold use asymptotic formulas.
    shift : float, optional
        Shift parameter applied to input. Default: 0.0.
        The input x is transformed to (x + shift) before evaluation.
        Useful for numerical stability near the transition region.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        eps: float,
        threshold: float = 100.0,
        shift: float = 0.0,
    ):
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self._bkd = bkd
        self._eps = eps
        self._threshold = threshold
        self._shift = shift

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def eps(self) -> float:
        """Return smoothing parameter epsilon."""
        return self._eps

    def threshold(self) -> float:
        """Return numerical stability threshold."""
        return self._threshold

    def shift(self) -> float:
        """Return shift parameter."""
        return self._shift

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Evaluate the smooth approximation."""
        raise NotImplementedError

    @abstractmethod
    def first_derivative(self, x: Array) -> Array:
        """First derivative of the smooth approximation."""
        raise NotImplementedError

    @abstractmethod
    def second_derivative(self, x: Array) -> Array:
        """Second derivative of the smooth approximation."""
        raise NotImplementedError


class SmoothLogBasedMaxFunction(DifferentiableApproximationBase[Array]):
    """Smooth approximation to max(0, x).

    Approximation: (x + shift) + eps * log(1 + exp(-(x + shift)/eps))

    This is also known as the softplus function.

    As eps -> 0:
    - For x > -shift: approaches x + shift
    - For x < -shift: approaches 0
    - For x = -shift: approaches 0

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float
        Smoothing parameter.
    threshold : float, optional
        Threshold for numerical stability. Default: 100.0.
    shift : float, optional
        Shift parameter applied to input. Default: 0.0.
    """

    def __call__(self, x: Array) -> Array:
        """Evaluate smooth max(0, x + shift).

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Approximated max(0, x + shift). Shape: (ndim, nsamples)
        """
        bkd = self._bkd
        x_shifted = x + self._shift
        x_div_eps = x_shifted / self._eps

        # Initialize output
        vals = bkd.zeros(x.shape)

        # Middle region: use full formula (avoid overflow)
        middle = (x_div_eps < self._threshold) & (x_div_eps > -self._threshold)
        middle_idx = bkd.where(middle)
        vals[middle_idx] = x_shifted[middle_idx] + self._eps * bkd.log(
            1 + bkd.exp(-x_div_eps[middle_idx])
        )

        # Large positive x: asymptote to x + shift
        large_pos = x_div_eps >= self._threshold
        large_pos_idx = bkd.where(large_pos)
        vals[large_pos_idx] = x_shifted[large_pos_idx]

        # Large negative x: stays at 0 (already initialized)

        return vals

    def first_derivative(self, x: Array) -> Array:
        """First derivative: sigmoid function 1/(1 + exp(-(x+shift)/eps)).

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Derivative values (sigmoid). Shape: (ndim, nsamples)
        """
        bkd = self._bkd
        x_shifted = x + self._shift
        x_div_eps = x_shifted / self._eps

        # Initialize to zero
        deriv = bkd.zeros(x.shape)

        # Middle region
        middle = (x_div_eps < self._threshold) & (x_div_eps > -self._threshold)
        middle_idx = bkd.where(middle)
        deriv[middle_idx] = 1.0 / (1 + bkd.exp(-x_div_eps[middle_idx]))

        # Large positive x: derivative is 1
        large_pos = x_div_eps >= self._threshold
        large_pos_idx = bkd.where(large_pos)
        deriv[large_pos_idx] = 1.0

        # Large negative x: derivative is 0 (already initialized)

        return deriv

    def second_derivative(self, x: Array) -> Array:
        """Second derivative: sigmoid derivative.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Second derivative values. Shape: (ndim, nsamples)
        """
        bkd = self._bkd
        x_shifted = x + self._shift
        x_div_eps = x_shifted / self._eps

        # Initialize to zero
        deriv2 = bkd.zeros(x.shape)

        # Middle region
        middle = (x_div_eps < self._threshold) & (x_div_eps > -self._threshold)
        middle_idx = bkd.where(middle)
        exp_x = bkd.exp(x_div_eps[middle_idx])
        deriv2[middle_idx] = exp_x / (self._eps * (exp_x + 1) ** 2)

        # Asymptotic regions: second derivative is 0 (already initialized)

        return deriv2

    def third_derivative(self, x: Array) -> Array:
        """Third derivative for use by Heaviside second derivative.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Third derivative values. Shape: (ndim, nsamples)
        """
        bkd = self._bkd
        x_shifted = x + self._shift
        x_div_eps = x_shifted / self._eps

        # Initialize to zero
        deriv3 = bkd.zeros(x.shape)

        # Middle region
        middle = (x_div_eps < self._threshold) & (x_div_eps > -self._threshold)
        middle_idx = bkd.where(middle)
        exp_x = bkd.exp(x_div_eps[middle_idx])
        # d/dx [exp(x/eps) / (eps * (1 + exp(x/eps))^2)]
        # = (exp(x/eps) * (1 - exp(x/eps))) / (eps^2 * (1 + exp(x/eps))^3)
        deriv3[middle_idx] = (
            exp_x * (1 - exp_x) / (self._eps**2 * (exp_x + 1) ** 3)
        )

        return deriv3


class SmoothLogBasedRightHeavisideFunction(DifferentiableApproximationBase[Array]):
    """Smooth approximation to right Heaviside H(x) = 1 if x >= 0, else 0.

    This is the first derivative of the smooth max function.
    Approximation: 1 / (1 + exp(-(x+shift)/eps)) (sigmoid function)

    As eps -> 0:
    - For x > -shift: approaches 1
    - For x < -shift: approaches 0
    - For x = -shift: approaches 0.5

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float
        Smoothing parameter.
    threshold : float, optional
        Threshold for numerical stability. Default: 100.0.
    shift : float, optional
        Shift parameter applied to input. Default: 0.0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        eps: float,
        threshold: float = 100.0,
        shift: float = 0.0,
    ):
        super().__init__(bkd, eps, threshold, shift)
        self._max_func = SmoothLogBasedMaxFunction(bkd, eps, threshold, shift)

    def __call__(self, x: Array) -> Array:
        """Evaluate smooth right Heaviside H(x).

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Approximated H(x). Shape: (ndim, nsamples)
        """
        return self._max_func.first_derivative(x)

    def first_derivative(self, x: Array) -> Array:
        """First derivative of smooth right Heaviside.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Derivative values. Shape: (ndim, nsamples)
        """
        return self._max_func.second_derivative(x)

    def second_derivative(self, x: Array) -> Array:
        """Second derivative of smooth right Heaviside.

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Second derivative values. Shape: (ndim, nsamples)
        """
        return self._max_func.third_derivative(x)


class SmoothLogBasedLeftHeavisideFunction(DifferentiableApproximationBase[Array]):
    """Smooth approximation to left Heaviside H(-x) = 1 if x <= 0, else 0.

    Approximation: 1 / (1 + exp((x+shift)/eps))

    This is the smooth right Heaviside evaluated at -x.

    As eps -> 0:
    - For x < -shift: approaches 1
    - For x > -shift: approaches 0
    - For x = -shift: approaches 0.5

    Used in First-order Stochastic Dominance (FSD) constraints.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float
        Smoothing parameter.
    threshold : float, optional
        Threshold for numerical stability. Default: 100.0.
    shift : float, optional
        Shift parameter applied to input. Default: 0.0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        eps: float,
        threshold: float = 100.0,
        shift: float = 0.0,
    ):
        super().__init__(bkd, eps, threshold, shift)
        self._right_heaviside = SmoothLogBasedRightHeavisideFunction(
            bkd, eps, threshold, shift
        )

    def __call__(self, x: Array) -> Array:
        """Evaluate smooth left Heaviside H(-x).

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Approximated H(-x). Shape: (ndim, nsamples)
        """
        return self._right_heaviside(-x)

    def first_derivative(self, x: Array) -> Array:
        """First derivative of smooth left Heaviside.

        Note: d/dx H(-x) = -H'(-x)

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Derivative values. Shape: (ndim, nsamples)
        """
        return -self._right_heaviside.first_derivative(-x)

    def second_derivative(self, x: Array) -> Array:
        """Second derivative of smooth left Heaviside.

        Note: d^2/dx^2 H(-x) = H''(-x)

        Parameters
        ----------
        x : Array
            Input values. Shape: (ndim, nsamples)

        Returns
        -------
        Array
            Second derivative values. Shape: (ndim, nsamples)
        """
        return self._right_heaviside.second_derivative(-x)

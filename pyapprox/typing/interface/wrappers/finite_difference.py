"""Finite difference wrapper for computing derivatives.

This module provides a wrapper that adds finite difference jacobian
and hessian methods to any function implementing FunctionProtocol.
"""

import math
from typing import Generic, Literal

from pyapprox.typing.util.backends.protocols import Array, Backend


class FiniteDifferenceWrapper(Generic[Array]):
    """Add finite difference derivatives to any function.

    This wrapper computes jacobian and hessian using finite differences.
    Supports forward, backward, and centered difference schemes.

    Parameters
    ----------
    model : FunctionProtocol[Array]
        The model to wrap. Must have bkd(), nvars(), nqoi(), and __call__.
    method : {"forward", "backward", "centered"}, optional
        The finite difference method. Default is "centered".
    step : float, optional
        The finite difference step size. Default is 2*sqrt(machine_epsilon).

    Notes
    -----
    - Forward difference: O(h) accuracy, n+1 function evaluations for jacobian
    - Backward difference: O(h) accuracy, n+1 function evaluations for jacobian
    - Centered difference: O(h^2) accuracy, 2n function evaluations for jacobian

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.interface.functions.fromcallable.function import (
    ...     FunctionFromCallable,
    ... )
    >>> bkd = NumpyBkd()
    >>> def quadratic(samples):
    ...     x = samples[0:1, :]
    ...     return x ** 2
    >>> model = FunctionFromCallable(nqoi=1, nvars=1, fun=quadratic, bkd=bkd)
    >>> fd_model = FiniteDifferenceWrapper(model, method="centered")
    >>> sample = bkd.asarray([[2.0]])
    >>> fd_model.jacobian(sample)  # Should be close to 4.0
    array([[4.]])
    """

    _DEFAULT_STEP = 2 * math.sqrt(2.220446049250313e-16)  # 2 * sqrt(eps)

    def __init__(
        self,
        model: "FunctionProtocol[Array]",  # type: ignore[name-defined]
        method: Literal["forward", "backward", "centered"] = "centered",
        step: float = _DEFAULT_STEP,
    ) -> None:
        """Initialize the finite difference wrapper.

        Parameters
        ----------
        model : FunctionProtocol[Array]
            The model to wrap.
        method : {"forward", "backward", "centered"}, optional
            The finite difference method. Default is "centered".
        step : float, optional
            The finite difference step size.
        """
        self._model = model
        self._method = method
        self._step = step

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._model.bkd()  # type: ignore[no-any-return]

    def nvars(self) -> int:
        """Return the number of variables."""
        return int(self._model.nvars())

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return int(self._model.nqoi())

    def wrapped(self) -> "FunctionProtocol[Array]":  # type: ignore[name-defined]
        """Return the wrapped model."""
        return self._model

    def set_step(self, step: float) -> None:
        """Set the finite difference step size.

        Parameters
        ----------
        step : float
            The new step size.
        """
        self._step = step

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model (pass-through).

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Output values of shape (nqoi, nsamples).
        """
        result: Array = self._model(samples)
        return result

    def jacobian(self, sample: Array) -> Array:
        """Compute the Jacobian using finite differences.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (nqoi, nvars).
        """
        if self._method == "forward":
            return self._forward_jacobian(sample)
        elif self._method == "backward":
            return self._backward_jacobian(sample)
        else:  # centered
            return self._centered_jacobian(sample)

    def _forward_jacobian(self, sample: Array) -> Array:
        """Compute Jacobian using forward difference."""
        bkd = self.bkd()
        nvars = self.nvars()
        h = self._step

        # Evaluate at base point
        f0 = self._model(sample)  # (nqoi, 1)

        # Create perturbed samples: add h to each variable
        perturbed = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        for ii in range(nvars):
            perturbed[ii, ii] += h

        # Evaluate at all perturbed points
        f_perturbed = self._model(perturbed)  # (nqoi, nvars)

        # Compute jacobian: (f(x+h) - f(x)) / h
        jacobian: Array = (f_perturbed - f0) / h  # (nqoi, nvars)
        return jacobian

    def _backward_jacobian(self, sample: Array) -> Array:
        """Compute Jacobian using backward difference."""
        bkd = self.bkd()
        nvars = self.nvars()
        h = self._step

        # Evaluate at base point
        f0 = self._model(sample)  # (nqoi, 1)

        # Create perturbed samples: subtract h from each variable
        perturbed = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        for ii in range(nvars):
            perturbed[ii, ii] -= h

        # Evaluate at all perturbed points
        f_perturbed = self._model(perturbed)  # (nqoi, nvars)

        # Compute jacobian: (f(x) - f(x-h)) / h
        jacobian: Array = (f0 - f_perturbed) / h  # (nqoi, nvars)
        return jacobian

    def _centered_jacobian(self, sample: Array) -> Array:
        """Compute Jacobian using centered difference."""
        bkd = self.bkd()
        nvars = self.nvars()
        h = self._step

        # Create +h and -h perturbed samples
        perturbed_plus = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        perturbed_minus = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        for ii in range(nvars):
            perturbed_plus[ii, ii] += h
            perturbed_minus[ii, ii] -= h

        # Evaluate at all perturbed points
        f_plus = self._model(perturbed_plus)  # (nqoi, nvars)
        f_minus = self._model(perturbed_minus)  # (nqoi, nvars)

        # Compute jacobian: (f(x+h) - f(x-h)) / (2h)
        jacobian: Array = (f_plus - f_minus) / (2 * h)  # (nqoi, nvars)
        return jacobian

    def jvp(self, sample: Array, vec: Array) -> Array:
        """Compute Jacobian-vector product using finite differences.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian-vector product of shape (nqoi, 1).
        """
        if self._method == "forward":
            return self._forward_jvp(sample, vec)
        elif self._method == "backward":
            return self._backward_jvp(sample, vec)
        else:  # centered
            return self._centered_jvp(sample, vec)

    def _forward_jvp(self, sample: Array, vec: Array) -> Array:
        """Compute JVP using forward difference."""
        h = self._step
        f0 = self._model(sample)
        f_perturbed = self._model(sample + h * vec)
        result: Array = (f_perturbed - f0) / h
        return result

    def _backward_jvp(self, sample: Array, vec: Array) -> Array:
        """Compute JVP using backward difference."""
        h = self._step
        f0 = self._model(sample)
        f_perturbed = self._model(sample - h * vec)
        result: Array = (f0 - f_perturbed) / h
        return result

    def _centered_jvp(self, sample: Array, vec: Array) -> Array:
        """Compute JVP using centered difference."""
        h = self._step
        f_plus = self._model(sample + h * vec)
        f_minus = self._model(sample - h * vec)
        result: Array = (f_plus - f_minus) / (2 * h)
        return result

    def hessian(self, sample: Array) -> Array:
        """Compute the Hessian using finite differences.

        This method is only valid for scalar functions (nqoi == 1).

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian matrix of shape (nvars, nvars).

        Raises
        ------
        ValueError
            If nqoi != 1.
        """
        if self.nqoi() != 1:
            raise ValueError(
                f"Hessian only defined for nqoi=1, got nqoi={self.nqoi()}"
            )

        if self._method == "centered":
            return self._centered_hessian(sample)
        else:
            return self._forward_hessian(sample)

    def _forward_hessian(self, sample: Array) -> Array:
        """Compute Hessian using forward differences on jacobian."""
        bkd = self.bkd()
        nvars = self.nvars()
        h = self._step

        # Get base jacobian
        jac0 = self.jacobian(sample)  # (1, nvars)

        # Create perturbed samples
        perturbed = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        for ii in range(nvars):
            perturbed[ii, ii] += h

        # Compute jacobian at each perturbed point
        jac_perturbed = bkd.zeros((nvars, nvars))
        for ii in range(nvars):
            jac_i = self.jacobian(perturbed[:, ii : ii + 1])  # (1, nvars)
            jac_perturbed[ii, :] = jac_i[0, :]

        # Hessian: (J(x+h_i) - J(x)) / h
        hessian = (jac_perturbed - jac0) / h  # (nvars, nvars)
        return hessian

    def _centered_hessian(self, sample: Array) -> Array:
        """Compute Hessian using centered differences on jacobian."""
        bkd = self.bkd()
        nvars = self.nvars()
        h = self._step

        # Create +h and -h perturbed samples
        perturbed_plus = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        perturbed_minus = bkd.tile(sample, (nvars,))  # (nvars, nvars)
        for ii in range(nvars):
            perturbed_plus[ii, ii] += h
            perturbed_minus[ii, ii] -= h

        # Compute jacobian at each perturbed point
        jac_plus = bkd.zeros((nvars, nvars))
        jac_minus = bkd.zeros((nvars, nvars))
        for ii in range(nvars):
            jac_p = self.jacobian(perturbed_plus[:, ii : ii + 1])  # (1, nvars)
            jac_m = self.jacobian(perturbed_minus[:, ii : ii + 1])  # (1, nvars)
            jac_plus[ii, :] = jac_p[0, :]
            jac_minus[ii, :] = jac_m[0, :]

        # Hessian: (J(x+h_i) - J(x-h_i)) / (2h)
        hessian = (jac_plus - jac_minus) / (2 * h)  # (nvars, nvars)
        return hessian

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product using finite differences.

        This method is only valid for scalar functions (nqoi == 1).

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1).

        Raises
        ------
        ValueError
            If nqoi != 1.
        """
        if self.nqoi() != 1:
            raise ValueError(
                f"HVP only defined for nqoi=1, got nqoi={self.nqoi()}"
            )

        if self._method == "centered":
            return self._centered_hvp(sample, vec)
        else:
            return self._forward_hvp(sample, vec)

    def _forward_hvp(self, sample: Array, vec: Array) -> Array:
        """Compute HVP using forward difference on jacobian."""
        h = self._step
        jac0 = self.jacobian(sample)  # (1, nvars)
        jac_perturbed = self.jacobian(sample + h * vec)  # (1, nvars)
        # H @ v = (J(x+hv) - J(x)) / h, transposed to (nvars, 1)
        hvp = ((jac_perturbed - jac0) / h).T
        return hvp

    def _centered_hvp(self, sample: Array, vec: Array) -> Array:
        """Compute HVP using centered difference on jacobian."""
        h = self._step
        jac_plus = self.jacobian(sample + h * vec)  # (1, nvars)
        jac_minus = self.jacobian(sample - h * vec)  # (1, nvars)
        # H @ v = (J(x+hv) - J(x-hv)) / (2h), transposed to (nvars, 1)
        hvp = ((jac_plus - jac_minus) / (2 * h)).T
        return hvp

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FiniteDifferenceWrapper({self._model!r}, "
            f"method={self._method!r}, step={self._step})"
        )

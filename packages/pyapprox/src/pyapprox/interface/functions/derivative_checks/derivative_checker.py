"""Finite-difference checking of Derivatives-bundle capabilities.

The checkers iterate whatever a function's bundle declares
(``resolve_bundle`` requires the ``derivatives()`` accessor):
jacobian/jvp are checked directly; hvp is checked as the derivative of
the gradient; whvp as the derivative of the weighted gradient w^T f.
"""

import math
from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
)

from pyapprox.interface.functions.derivative_checks.base import (
    JVPChecker,
)
from pyapprox.interface.functions.derivative_checks.resolve import (
    resolve_bundle,
)
from pyapprox.interface.functions.derivative_checks.wrappers import (
    FunctionWithJVP,
    FunctionWithJVPFromHVP,
    SingleSampleFromBatchHessian,
    SingleSampleFromBatchJacobian,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class VShapeReport:
    """Result of ``DerivativeChecker.check_v_shape``.

    Attributes
    ----------
    order : float
        Fitted convergence order above the minimum; NaN if too few steps
        lie in the fitting window.
    expected_order : float
        1 for forward differences, 2 for central.
    bottom_step : float
        The step with the smallest error. The order is fitted over the
        steps from 10 to 1000 times this.
    max_bottom_step : float
        Largest step at which the minimum counts as set by rounding.
    rounding_side_rises : bool
        Whether the error rises again at steps below the minimum.
    bottom_is_rounding_limited : bool
        Whether the minimum is at a step small enough to be set by rounding.
    order_matches : bool
        Whether ``order`` is close to ``expected_order``.
    passed : bool
        All three checks hold.
    """

    order: float
    expected_order: float
    bottom_step: float
    max_bottom_step: float
    rounding_side_rises: bool
    bottom_is_rounding_limited: bool
    order_matches: bool
    passed: bool


class DerivativeChecker(Generic[Array]):
    def __init__(self, function: FunctionProtocol[Array]):
        self._derivs = self._validate_function(function)
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def _validate_function(
        self, function: FunctionProtocol[Array]
    ) -> Derivatives[Array]:
        derivs = resolve_bundle(function)
        if derivs.jacobian is None and derivs.jvp is None:
            raise ValueError(
                "The provided function must declare a jacobian or jvp in "
                "its Derivatives bundle. "
                f"Got an object of type {type(function).__name__}."
            )
        return derivs

    def check_derivatives(
        self,
        sample: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
        weights: Optional[Array] = None,
        central: bool = False,
    ) -> List[Array]:
        """Finite-difference errors of each declared derivative.

        ``central=True`` uses central differences (see ``JVPChecker``),
        which need ``sample - h * direction`` to be a valid input.
        """
        jacobian_checker = JVPChecker(
            FunctionWithJVP(self._fun),
            "J",
            fd_eps,
            direction,
            relative,
            verbosity,
            central=central,
        )
        errors = [jacobian_checker.check(sample)]
        if self._derivs.hvp is None and self._derivs.whvp is None:
            return errors
        if weights is None and self._derivs.hvp is None:
            weights = self.bkd().ones((self._fun.nqoi(), 1))
        hessian_checker = JVPChecker(
            FunctionWithJVPFromHVP(self._fun, weights),
            "H",
            fd_eps,
            direction,
            relative,
            verbosity,
            central=central,
        )
        errors.append(hessian_checker.check(sample))
        return errors

    def convergence_order(self, errors: Array, fd_eps: Array) -> Array:
        """Rate at which the finite-difference error falls above its minimum.

        Against a correct derivative the finite-difference error falls like
        ``h**p`` (``p = 1`` forward, ``p = 2`` central) until rounding takes
        over at small ``h``: a V in ``log error`` against ``log h``. This
        fits ``p`` by least squares on the truncation side, over the steps
        from 10 to 1000 times the step with the smallest error. One decade
        of clearance keeps rounding near the minimum out of the fit, and
        the upper limit keeps out higher-order terms at large steps.

        It assumes the smallest error is the bottom of a V. A wrong
        derivative whose error cancels the truncation error at one step
        makes a dip that also falls at order ``p``, so use
        ``check_v_shape``, which also tests that the V has a rounding side
        and that its bottom is where rounding puts it.

        Parameters
        ----------
        errors : Array
            One error array from ``check_derivatives``. Shape: (nsteps,)
        fd_eps : Array
            The steps those errors were computed with. Shape: (nsteps,)

        Returns
        -------
        Array
            The fitted order ``p``. Shape: ()

        Raises
        ------
        ValueError
            If fewer than two steps lie in the window, which means the
            smallest error is at the largest steps; widen ``fd_eps``.
        """
        order = self._window_order(errors, fd_eps)
        if order is None:
            raise ValueError(
                "fewer than two steps lie 10 to 1000 times above the step with "
                "the smallest error, so there is no truncation regime to fit; "
                "widen fd_eps to larger steps"
            )
        return order

    def check_v_shape(
        self,
        errors: Array,
        fd_eps: Array,
        central: bool = False,
        order_tol: float = 0.25,
        max_bottom_step: Optional[float] = None,
    ) -> "VShapeReport":
        """Test that the finite-difference errors form the V of a correct
        derivative.

        Three checks, all of which a correct derivative passes:

        1. **Rounding side.** Some step smaller than the one with the
           smallest error has an error at least 10 times that minimum, so
           the minimum is the bottom of a V, not the end of a plateau.
           ``fd_eps`` must reach below the rounding scale (about ``1e-9``
           forward, ``1e-6`` central).
        2. **Rounding-limited bottom.** The minimum lies at a step no larger
           than ``max_bottom_step``. A correct derivative's minimum is set
           by rounding, near ``sqrt(eps)`` forward and ``eps**(1/3)``
           central for inputs of order one. A wrong derivative can cancel
           the truncation error at a much larger step, giving a dip.
        3. **Order.** ``convergence_order`` is within ``order_tol`` of 1
           (forward) or 2 (central).

        Use it alongside ``error_ratio``, which measures the V's depth.

        Parameters
        ----------
        errors : Array
            One error array from ``check_derivatives``. Shape: (nsteps,)
        fd_eps : Array
            The steps those errors were computed with. Shape: (nsteps,)
        central : bool
            Whether ``errors`` came from central differences.
        order_tol : float
            Allowed distance of the fitted order from the expected one.
        max_bottom_step : float, optional
            Largest step allowed for the minimum. Default ``1e-5`` forward,
            ``1e-3`` central, which assumes inputs of order one.
        """
        bkd = self.bkd()
        self._check_shapes(errors, fd_eps)
        if max_bottom_step is None:
            max_bottom_step = 1e-3 if central else 1e-5
        expected = 2.0 if central else 1.0
        imin = int(bkd.to_float(bkd.argmin(errors)))
        min_error = bkd.to_float(errors[imin])
        h_min = bkd.to_float(fd_eps[imin])
        smaller = [
            bkd.to_float(errors[ii])
            for ii in range(fd_eps.shape[0])
            if bkd.to_float(fd_eps[ii]) < h_min
        ]
        rounding_side_rises = bool(smaller) and max(smaller) >= 10.0 * min_error
        bottom_is_rounding_limited = h_min <= max_bottom_step
        window_order = self._window_order(errors, fd_eps)
        order = math.nan if window_order is None else bkd.to_float(window_order)
        order_matches = abs(order - expected) <= order_tol
        return VShapeReport(
            order=order,
            expected_order=expected,
            bottom_step=h_min,
            max_bottom_step=max_bottom_step,
            rounding_side_rises=rounding_side_rises,
            bottom_is_rounding_limited=bottom_is_rounding_limited,
            order_matches=order_matches,
            passed=(
                rounding_side_rises and bottom_is_rounding_limited and order_matches
            ),
        )

    def _check_shapes(self, errors: Array, fd_eps: Array) -> None:
        if errors.shape != fd_eps.shape:
            raise ValueError(
                f"errors and fd_eps must have the same shape, got "
                f"{tuple(errors.shape)} and {tuple(fd_eps.shape)}"
            )

    def _window_order(self, errors: Array, fd_eps: Array) -> Optional[Array]:
        """Least-squares slope of log error on log step over the steps from
        10 to 1000 times the step with the smallest error; None if fewer
        than two steps lie there."""
        bkd = self.bkd()
        self._check_shapes(errors, fd_eps)
        h_min = bkd.to_float(fd_eps[int(bkd.to_float(bkd.argmin(errors)))])
        window = [
            ii
            for ii in range(fd_eps.shape[0])
            if 10.0 * h_min <= bkd.to_float(fd_eps[ii]) <= 1000.0 * h_min
        ]
        if len(window) < 2:
            return None
        log_h = bkd.log(fd_eps[window])
        log_e = bkd.log(errors[window])
        dh = log_h - bkd.mean(log_h)
        de = log_e - bkd.mean(log_e)
        return bkd.sum(dh * de) / bkd.sum(dh * dh)

    def error_ratio(self, errors: Array) -> Array:
        """Ratio of smallest to largest finite-difference error.

        Takes **one** error array, not the list ``check_derivatives``
        returns. That method yields one array per capability checked, so
        a caller passes the element it means -- ``errors[0]`` for the
        jacobian check.

        TODO: passing the list itself works on NumPy, whose ``min``
        accepts a sequence, and raises on Torch with "min(): argument
        'input' must be Tensor, not list". So a caller who gets this
        wrong sees it only on one backend, and any test written that way
        is NumPy-only by accident. Resolve by having this reduce over a
        sequence, or by returning an array when a single capability was
        checked -- either way it wants settling alongside the batched
        checker rather than piecemeal.
        """
        return self.bkd().min(errors) / self.bkd().max(errors)


class BatchDerivativeChecker(Generic[Array]):
    """Check derivatives for functions declaring batch capabilities.

    This checker validates batch derivative fields (jacobian_batch,
    hessian_batch) by wrapping them to expose single-sample interfaces and
    using DerivativeChecker on each sample individually.

    Parameters
    ----------
    function : FunctionProtocol[Array]
        Function whose Derivatives bundle declares jacobian_batch (and
        optionally hessian_batch).
    samples : Array
        Samples at which to evaluate. Shape: (nvars, nsamples)

    Examples
    --------
    >>> from pyapprox.surrogates.affine import create_pce
    >>> pce = create_pce(bases_1d, max_level, bkd)
    >>> pce.set_coefficients(coef)
    >>> samples = bkd.asarray([[-0.5, 0.3], [0.2, -0.1]])  # (nvars=2, nsamples=2)
    >>> checker = BatchDerivativeChecker(pce, samples)
    >>> errors = checker.check_jacobian_batch(verbosity=1)
    >>> ratio = checker.error_ratio(errors)  # Should be ~0.25
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        samples: Array,
    ):
        self._derivs = resolve_bundle(function)
        self._fun = function
        self._samples = samples

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def check_jacobian_batch(
        self,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ) -> Array:
        """Check jacobian_batch implementation.

        Parameters
        ----------
        fd_eps : Array, optional
            Finite difference step sizes for error estimation.
        direction : Array, optional
            Direction vector for JVP checks.
        relative : bool
            Whether to use relative errors.
        verbosity : int
            Verbosity level (0=silent, 1=print results).

        Returns
        -------
        Array
            Finite difference errors. Shape: (nsamples, n_eps)
        """
        all_errors = []
        nsamples = self._samples.shape[1]
        # Wrap to expose jacobian from jacobian_batch
        wrapped = SingleSampleFromBatchJacobian(self._fun)
        for ii in range(nsamples):
            sample = self._samples[:, ii : ii + 1]  # (nvars, 1)
            checker: DerivativeChecker[Array] = DerivativeChecker(wrapped)
            errors = checker.check_derivatives(
                sample, fd_eps, direction, relative, verbosity
            )
            all_errors.append(errors[0])
        return self.bkd().stack(all_errors, axis=0)

    def check_hessian_batch(
        self,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ) -> Array:
        """Check hessian_batch implementation.

        Only available for functions with nqoi=1.

        Parameters
        ----------
        fd_eps : Array, optional
            Finite difference step sizes for error estimation.
        direction : Array, optional
            Direction vector for HVP checks.
        relative : bool
            Whether to use relative errors.
        verbosity : int
            Verbosity level (0=silent, 1=print results).

        Returns
        -------
        Array
            Finite difference errors. Shape: (nsamples, n_eps)
        """
        if self._derivs.hessian_batch is None:
            raise ValueError(
                "Function does not declare hessian_batch in its "
                f"Derivatives bundle. Got {type(self._fun).__name__}."
            )
        all_errors = []
        nsamples = self._samples.shape[1]
        # Wrap to expose hessian from hessian_batch
        wrapped = SingleSampleFromBatchHessian(self._fun)
        for ii in range(nsamples):
            sample = self._samples[:, ii : ii + 1]  # (nvars, 1)
            checker: DerivativeChecker[Array] = DerivativeChecker(wrapped)
            errors = checker.check_derivatives(
                sample, fd_eps, direction, relative, verbosity
            )
            all_errors.append(errors[1])  # hessian errors
        return self.bkd().stack(all_errors, axis=0)

    def check_derivatives(
        self,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ) -> List[Array]:
        """Check all declared batch derivative fields.

        Returns
        -------
        List[Array]
            List of error arrays: [jacobian_batch_errors, hessian_batch_errors]
            hessian_batch_errors only included if hessian_batch is declared.
        """
        errors = [self.check_jacobian_batch(fd_eps, direction, relative, verbosity)]
        if self._derivs.hessian_batch is not None:
            errors.append(
                self.check_hessian_batch(fd_eps, direction, relative, verbosity)
            )
        return errors

    def error_ratio(self, errors: Array) -> Array:
        """Compute error ratio to assess convergence.

        For correct derivatives, ratio should be ~0.25 (second-order convergence).

        Parameters
        ----------
        errors : Array
            Error array. Shape: (nsamples, n_eps) or (n_eps,)

        Returns
        -------
        Array
            Worst-case error ratio across all samples.
        """
        if errors.ndim == 1:
            return self.bkd().min(errors) / self.bkd().max(errors)
        # For batch, compute ratio per sample then take worst case
        ratios = self.bkd().min(errors, axis=1) / self.bkd().max(errors, axis=1)
        return self.bkd().min(ratios)

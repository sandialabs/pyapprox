"""Finite-difference checking of Derivatives-bundle capabilities.

The checkers iterate whatever a function's bundle declares (via the
migration shim ``as_derivatives``): jacobian/jvp are checked directly;
hvp is checked as the derivative of the gradient; whvp as the derivative
of the weighted gradient w^T f.
"""

from typing import (
    Generic,
    List,
    Optional,
)

from pyapprox.interface.functions.derivative_checks.base import (
    JVPChecker,
)
from pyapprox.interface.functions.derivative_checks.wrappers import (
    FunctionWithJVP,
    FunctionWithJVPFromHVP,
    SingleSampleFromBatchHessian,
    SingleSampleFromBatchJacobian,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.legacy_adapter import as_derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class DerivativeChecker(Generic[Array]):
    def __init__(self, function: FunctionProtocol[Array]):
        self._derivs = self._validate_function(function)
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def _validate_function(
        self, function: FunctionProtocol[Array]
    ) -> Derivatives[Array]:
        derivs = as_derivatives(function)
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
    ) -> List[Array]:
        jacobian_checker = JVPChecker(
            FunctionWithJVP(self._fun),
            "J",
            fd_eps,
            direction,
            relative,
            verbosity,
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
        )
        errors.append(hessian_checker.check(sample))
        return errors

    def error_ratio(self, errors: Array) -> Array:
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
        self._derivs = as_derivatives(function)
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

from typing import (
    Protocol,
    Optional,
    runtime_checkable,
    Generic,
    Union,
    List,
    cast,
)

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.interface.functions.protocols.jacobian import (
    function_has_jacobian_or_jvp,
    FunctionWithJacobianOrJVPProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithHVPAndJacobianOrJVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
    function_has_hvp_and_jacobian_or_jvp,
)
from pyapprox.interface.functions.derivative_checks.base import (
    JVPChecker,
)
from pyapprox.interface.functions.derivative_checks.wrappers import (
    FunctionWithJVP,
    FunctionWithJVPFromHVP,
    SingleSampleFromBatchJacobian,
    SingleSampleFromBatchHessian,
    BatchJacobianProtocol,
    BatchHessianProtocol,
)


class DerivativeChecker(Generic[Array]):
    def __init__(self, function: FunctionWithJacobianOrJVPProtocol[Array]):
        self._validate_function(function)
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def _validate_function(
        self,
        function: Union[
            FunctionWithJacobianOrJVPProtocol[Array],
            FunctionWithHVPAndJacobianOrJVPProtocol[Array],
            FunctionWithJacobianAndWHVPProtocol[Array],
        ],
    ) -> None:
        if not function_has_hvp_and_jacobian_or_jvp(
            function
        ) and not function_has_jacobian_or_jvp(function):
            raise ValueError(
                "The provided function must satisfy either "
                "'FunctionWithJacobianOrJVPProtocol. "
                f"Got an object of type {type(function).__name__}."
            )

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
        if not function_has_hvp_and_jacobian_or_jvp(self._fun):
            return errors
        # use cast because type checker cannot determine that
        # self._fun is guaranteed to be this type if execution makes it here
        if weights is None and not hasattr(self._fun, "hvp"):
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
    """Check derivatives for functions with batch Jacobian/Hessian methods.

    This checker validates batch derivative methods (jacobian_batch, hessian_batch)
    by wrapping them to expose single-sample interfaces and using DerivativeChecker
    on each sample individually.

    Parameters
    ----------
    function : BatchJacobianProtocol[Array] or BatchHessianProtocol[Array]
        Function with jacobian_batch (and optionally hessian_batch) method.
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
        function: Union[BatchJacobianProtocol[Array], BatchHessianProtocol[Array]],
        samples: Array,
    ):
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
            checker = DerivativeChecker(wrapped)
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
        if not hasattr(self._fun, "hessian_batch"):
            raise ValueError(
                "Function does not have hessian_batch method. "
                f"Got {type(self._fun).__name__}."
            )
        all_errors = []
        nsamples = self._samples.shape[1]
        # Wrap to expose hessian from hessian_batch
        wrapped = SingleSampleFromBatchHessian(self._fun)  # type: ignore
        for ii in range(nsamples):
            sample = self._samples[:, ii : ii + 1]  # (nvars, 1)
            checker = DerivativeChecker(wrapped)
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
        """Check all available batch derivative methods.

        Returns
        -------
        List[Array]
            List of error arrays: [jacobian_batch_errors, hessian_batch_errors]
            hessian_batch_errors only included if hessian_batch is available.
        """
        errors = [
            self.check_jacobian_batch(fd_eps, direction, relative, verbosity)
        ]
        if hasattr(self._fun, "hessian_batch"):
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

"""
Conditional Independent Joint distribution.

Provides a joint distribution of independent conditionals that all share
the same conditioning variable.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.probability.conditional.protocols import (
    ConditionalDistributionProtocol,
)


class ConditionalIndependentJoint(Generic[Array]):
    """
    Joint distribution of independent conditionals.

    p(y1, y2, ... | x) = p(y1 | x) * p(y2 | x) * ...

    All conditionals share the same conditioning variable x.

    Parameters
    ----------
    conditionals : List[ConditionalDistributionProtocol[Array]]
        List of conditional distributions. All must have the same nvars().
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.conditional import (
    ...     ConditionalGaussian, ConditionalIndependentJoint
    ... )
    >>>
    >>> bkd = NumpyBkd()
    >>> # Create two conditional Gaussians
    >>> cond_y1 = ConditionalGaussian(mean_func_1, log_stdev_func_1, bkd)
    >>> cond_y2 = ConditionalGaussian(mean_func_2, log_stdev_func_2, bkd)
    >>>
    >>> joint = ConditionalIndependentJoint([cond_y1, cond_y2], bkd)
    >>> x = bkd.array([[0.5, 0.7]])  # Shape: (nvars=1, nsamples=2)
    >>> y = bkd.array([[1.0, 2.0], [0.5, 0.8]])  # Shape: (nqoi=2, nsamples=2)
    >>> log_probs = joint.logpdf(x, y)  # Shape: (1, 2)
    """

    def __init__(
        self,
        conditionals: List[ConditionalDistributionProtocol[Array]],
        bkd: Backend[Array],
    ):
        if not conditionals:
            raise ValueError("Must provide at least one conditional")

        self._conditionals = conditionals
        self._bkd = bkd

        # Validate all have same nvars
        nvars = conditionals[0].nvars()
        for i, cond in enumerate(conditionals[1:], start=1):
            if cond.nvars() != nvars:
                raise ValueError(
                    f"All conditionals must have same nvars. "
                    f"Conditional 0 has {nvars}, conditional {i} has {cond.nvars()}"
                )

        # Store qoi counts for splitting y
        self._qoi_counts = [c.nqoi() for c in conditionals]
        self._total_nqoi = sum(self._qoi_counts)

        # Setup optional methods based on capabilities
        self._setup_methods()

    def _setup_methods(self) -> None:
        """Bind optional methods based on component capabilities."""
        # Combine hyp_lists if all conditionals have them
        if all(hasattr(c, "hyp_list") for c in self._conditionals):
            self._hyp_list = self._conditionals[0].hyp_list()
            for c in self._conditionals[1:]:
                self._hyp_list = self._hyp_list + c.hyp_list()
            self.hyp_list = self._get_hyp_list
            self.nparams = self._get_nparams

        # Bind jacobian_wrt_x if all conditionals support it
        if all(hasattr(c, "logpdf_jacobian_wrt_x") for c in self._conditionals):
            self.logpdf_jacobian_wrt_x = self._logpdf_jacobian_wrt_x

        # Bind jacobian_wrt_params if all conditionals support it
        if all(hasattr(c, "logpdf_jacobian_wrt_params") for c in self._conditionals):
            self.logpdf_jacobian_wrt_params = self._logpdf_jacobian_wrt_params

    def _get_hyp_list(self) -> HyperParameterList:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def _get_nparams(self) -> int:
        """Return the total number of parameters."""
        return self._hyp_list.nparams()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of conditioning variables."""
        return self._conditionals[0].nvars()

    def nqoi(self) -> int:
        """Return the total number of output variables."""
        return self._total_nqoi

    def _validate_inputs(self, x: Array, y: Array) -> None:
        """Validate input shapes."""
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got {y.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )
        if y.shape[0] != self._total_nqoi:
            raise ValueError(
                f"y first dimension must be {self._total_nqoi}, got {y.shape[0]}"
            )
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have same number of samples, "
                f"got {x.shape[1]} and {y.shape[1]}"
            )

    def _split_y(self, y: Array) -> List[Array]:
        """Split y into per-conditional chunks."""
        result = []
        idx = 0
        for count in self._qoi_counts:
            result.append(y[idx : idx + count, :])
            idx += count
        return result

    def logpdf(self, x: Array, y: Array) -> Array:
        """
        Evaluate the log probability density function.

        Log PDF is the sum of component log PDFs.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (nqoi, nsamples)
            y is split into chunks for each conditional.

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)

        ys = self._split_y(y)
        log_probs = [c.logpdf(x, yi) for c, yi in zip(self._conditionals, ys)]

        # Sum log probabilities across conditionals
        stacked = self._bkd.vstack(log_probs)  # (nconditionals, nsamples)
        return self._bkd.sum(stacked, axis=0, keepdims=True)  # (1, nsamples)

    def rvs(self, x: Array) -> Array:
        """
        Generate random samples given conditioning variable.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Random samples. Shape: (nqoi, nsamples)
            Stacked samples from each conditional.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )

        samples = [c.rvs(x) for c in self._conditionals]
        return self._bkd.vstack(samples)  # (nqoi, nsamples)

    def _logpdf_jacobian_wrt_x(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. conditioning variable x.

        Sum of component Jacobians.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, 1)
        y : Array
            Output variable values. Shape: (nqoi, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        self._validate_inputs(x, y)

        ys = self._split_y(y)
        jacs = [c.logpdf_jacobian_wrt_x(x, yi) for c, yi in zip(self._conditionals, ys)]

        # Sum Jacobians across conditionals
        stacked = self._bkd.vstack(jacs)  # (nconditionals, nvars)
        return self._bkd.sum(stacked, axis=0, keepdims=True)  # (1, nvars)

    def _logpdf_jacobian_wrt_params(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. active parameters.

        Concatenates Jacobians from each conditional along parameter axis.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (nqoi, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nactive_params)
            Parameters are ordered: [cond1_params, cond2_params, ...]
        """
        self._validate_inputs(x, y)

        ys = self._split_y(y)
        jacs = [
            c.logpdf_jacobian_wrt_params(x, yi)
            for c, yi in zip(self._conditionals, ys)
        ]

        # Concatenate along parameter axis
        return self._bkd.hstack(jacs)  # (nsamples, total_nparams)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalIndependentJoint(nvars={self.nvars()}, "
            f"nqoi={self.nqoi()}, nconditionals={len(self._conditionals)})"
        )

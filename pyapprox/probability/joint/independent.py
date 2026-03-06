"""
Independent joint distributions.

Provides joint distributions where all marginals are independent.
The joint PDF is the product of marginal PDFs.
"""

from typing import Generic, List, Optional, Sequence

import numpy as np

from pyapprox.interface.functions.plot.plot1d import Plotter1D
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
)
from pyapprox.probability.protocols import MarginalProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class IndependentJoint(Generic[Array]):
    """
    Joint distribution with independent marginals.

    The joint PDF is the product of marginal PDFs:
        p(x_1, ..., x_n) = p_1(x_1) * ... * p_n(x_n)

    The joint log-PDF is the sum of marginal log-PDFs:
        log p(x) = sum_i log p_i(x_i)

    Parameters
    ----------
    marginals : Sequence[MarginalProtocol[Array]]
        List of marginal distributions.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import norm, beta
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import ScipyContinuousMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [
    ...     ScipyContinuousMarginal(norm(0, 1), bkd),
    ...     ScipyContinuousMarginal(beta(2, 5), bkd),
    ... ]
    >>> joint = IndependentJoint(marginals, bkd)
    >>> samples = joint.rvs(100)  # Shape: (2, 100)
    """

    def __init__(
        self,
        marginals: Sequence[MarginalProtocol[Array]],
        bkd: Backend[Array],
    ):
        if len(marginals) == 0:
            raise ValueError("Must provide at least one marginal distribution")
        self._bkd = bkd
        self._marginals = list(marginals)
        self._nvars = len(marginals)
        self._setup_hyperparameter_list()
        self._setup_derivative_methods()

    def _setup_hyperparameter_list(self) -> None:
        """
        Setup the aggregated hyperparameter list from marginals.

        Only available if all marginals have hyp_list() method.
        """
        self._hyp_list: Optional[HyperParameterList] = None
        if all(hasattr(m, "hyp_list") for m in self._marginals):
            hyp_lists = [m.hyp_list() for m in self._marginals]  # type: ignore
            if hyp_lists:
                combined = hyp_lists[0]
                for hyp_list in hyp_lists[1:]:
                    combined = combined + hyp_list
                self._hyp_list = combined

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally add Jacobian methods based on marginal capabilities.

        Uses dynamic method binding: methods are only available if ALL marginals
        support the required operations.
        """
        # Check if all marginals have logpdf_jacobian
        if all(hasattr(m, "logpdf_jacobian") for m in self._marginals):
            self.logpdf_jacobian = self._logpdf_jacobian  # type: ignore
            self.logpdf_jacobian_batch = self._logpdf_jacobian_batch  # type: ignore

        # Check if all marginals have pdf_jacobian
        if all(hasattr(m, "pdf_jacobian") for m in self._marginals):
            self.jacobian = self._jacobian  # type: ignore
            self.jacobian_batch = self._jacobian_batch  # type: ignore

        # Check if all marginals have logpdf_jacobian_wrt_params
        if all(hasattr(m, "logpdf_jacobian_wrt_params") for m in self._marginals):
            self.logpdf_jacobian_wrt_params = (  # type: ignore
                self._logpdf_jacobian_wrt_params
            )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables (dimension of the distribution).
        """
        return self._nvars

    def hyp_list(self) -> HyperParameterList:
        """
        Return the aggregated hyperparameter list.

        Only available if all marginals have hyp_list() method.

        Returns
        -------
        HyperParameterList
            Combined hyperparameter list from all marginals.

        Raises
        ------
        RuntimeError
            If not all marginals have hyp_list() method.
        """
        if self._hyp_list is None:
            raise RuntimeError(
                "hyp_list() not available: not all marginals have hyp_list() method"
            )
        return self._hyp_list

    def nparams(self) -> int:
        """
        Return the total number of distribution parameters.

        Only available if all marginals have hyp_list() method.

        Returns
        -------
        int
            Total number of parameters across all marginals.

        Raises
        ------
        RuntimeError
            If not all marginals have hyp_list() method.
        """
        return self.hyp_list().nparams()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest.

        For a joint PDF, this is always 1 (the probability density is scalar).

        Returns
        -------
        int
            Always returns 1.
        """
        return 1

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """
        Return the marginal distributions.

        Returns
        -------
        List[MarginalProtocol]
            List of marginal distributions.
        """
        return self._marginals

    def marginal(self, index: int) -> MarginalProtocol[Array]:
        """
        Return a specific marginal distribution.

        Parameters
        ----------
        index : int
            Index of the marginal.

        Returns
        -------
        MarginalProtocol
            The marginal distribution at given index.
        """
        return self._marginals[index]

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the joint distribution.

        Samples each marginal independently and stacks them.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        samples_list = []
        for marginal in self._marginals:
            # Each marginal returns shape (1, nsamples)
            marg_samples = marginal.rvs(nsamples)
            # Flatten to (nsamples,) for stacking
            samples_list.append(self._bkd.flatten(marg_samples))
        return self._bkd.stack(samples_list, axis=0)

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {samples.shape[0]}"
            )

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the joint log probability density function.

        For independent marginals:
            log p(x) = sum_i log p_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(samples)

        nsamples = samples.shape[1]
        logpdf_total = self._bkd.zeros((nsamples,))

        for i, marginal in enumerate(self._marginals):
            # Slice out row i and reshape to (1, nsamples) for marginal
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            marg_logpdf = marginal.logpdf(row_2d)  # Returns (1, nsamples)
            logpdf_total = logpdf_total + marg_logpdf[0]

        return self._bkd.reshape(logpdf_total, (1, -1))

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability density function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        return self._bkd.exp(self.logpdf(samples))

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the joint probability density function.

        Alias for `pdf()` to satisfy `FunctionProtocol`.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            PDF values. Shape: (1, nsamples)
        """
        return self.pdf(samples)

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the joint cumulative distribution function.

        For independent marginals:
            F(x) = prod_i F_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(samples)

        nsamples = samples.shape[1]
        cdf_total = self._bkd.ones((nsamples,))

        for i, marginal in enumerate(self._marginals):
            # Slice out row i and reshape to (1, nsamples) for marginal
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            marg_cdf = marginal.cdf(row_2d)  # Returns (1, nsamples)
            cdf_total = cdf_total * marg_cdf[0]

        return self._bkd.reshape(cdf_total, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF component-wise.

        For independent marginals, applies inverse CDF to each dimension.

        Parameters
        ----------
        probs : Array
            Probability values. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Quantile values. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(probs)

        samples_list = []
        for i, marginal in enumerate(self._marginals):
            # Slice out row i and reshape to (1, nsamples) for marginal
            row_2d = self._bkd.reshape(probs[i], (1, -1))
            marg_invcdf = marginal.invcdf(row_2d)  # Returns (1, nsamples)
            samples_list.append(marg_invcdf[0])  # Flatten to 1D for stacking

        return self._bkd.stack(samples_list, axis=0)

    def correlation_matrix(self) -> Array:
        """
        Return the correlation matrix.

        For independent marginals, this is the identity matrix.

        Returns
        -------
        Array
            Identity correlation matrix. Shape: (nvars, nvars)
        """
        return self._bkd.eye(self._nvars)

    def mean(self) -> Array:
        """
        Return the mean of the joint distribution.

        Returns
        -------
        Array
            Mean vector. Shape: (nvars,)
        """
        means = []
        for marginal in self._marginals:
            if hasattr(marginal, "mean_value"):
                means.append(marginal.mean_value())
            else:
                # Fallback: use sampling
                samples = marginal.rvs(10000)
                means.append(self._bkd.to_float(self._bkd.sum(samples) / 10000))
        return self._bkd.asarray(means)

    def variance(self) -> Array:
        """
        Return the variance of each marginal.

        Returns
        -------
        Array
            Variance vector. Shape: (nvars,)
        """
        variances = []
        for marginal in self._marginals:
            if hasattr(marginal, "variance"):
                variances.append(marginal.variance())
            else:
                # Fallback: use sampling
                samples = marginal.rvs(10000)
                mean = self._bkd.to_float(self._bkd.sum(samples) / 10000)
                var = self._bkd.sum((samples - mean) ** 2) / 10000
                variances.append(self._bkd.to_float(var))
        return self._bkd.asarray(variances)

    def covariance(self) -> Array:
        """
        Return the covariance matrix.

        For independent marginals, this is diagonal with variances.

        Returns
        -------
        Array
            Diagonal covariance matrix. Shape: (nvars, nvars)
        """
        return self._bkd.diag(self.variance())

    def is_bounded(self) -> bool:
        """
        Check if all marginals have bounded support.

        Returns
        -------
        bool
            True if all marginals are bounded.
        """
        for marginal in self._marginals:
            if hasattr(marginal, "is_bounded"):
                if not marginal.is_bounded():
                    return False
            else:
                # Assume unbounded if method not available
                return False
        return True

    def bounds(self) -> Array:
        """
        Return the bounds of the joint distribution.

        Returns
        -------
        Array
            Bounds array. Shape: (2, nvars) where [0, :] is lower, [1, :] is upper.
        """
        lower_bounds = []
        upper_bounds = []

        for marginal in self._marginals:
            if hasattr(marginal, "interval"):
                interval = marginal.interval(1.0)  # Returns shape (1, 2)
                interval_flat = self._bkd.flatten(interval)
                lower_bounds.append(self._bkd.to_float(interval_flat[0]))
                upper_bounds.append(self._bkd.to_float(interval_flat[1]))
            else:
                # Default to unbounded
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)

        return self._bkd.stack(
            [self._bkd.asarray(lower_bounds), self._bkd.asarray(upper_bounds)],
            axis=0,
        )

    def domain(self) -> Array:
        """
        Return the domain of the joint distribution.

        Returns bounds as (nvars, 2) per CLAUDE.md conventions.
        For bounded marginals, uses interval(1.0). For unbounded, uses [-inf, inf].

        Returns
        -------
        Array
            Domain bounds. Shape: (nvars, 2) where [:, 0] is lower, [:, 1] is upper.
        """
        domain_list = []
        for marginal in self._marginals:
            if hasattr(marginal, "is_bounded") and marginal.is_bounded():
                if hasattr(marginal, "interval"):
                    interval = marginal.interval(1.0)  # Returns shape (1, 2)
                    domain_list.append(self._bkd.flatten(interval))
                else:
                    domain_list.append(self._bkd.asarray([-np.inf, np.inf]))
            else:
                domain_list.append(self._bkd.asarray([-np.inf, np.inf]))
        return self._bkd.stack(domain_list, axis=0)

    def plotter(
        self,
        plot_limits: Optional[Array] = None,
        reducer: Optional[object] = None,
        variable_names: Optional[List[str]] = None,
    ) -> object:
        """
        Create a plotter for visualizing the joint PDF.

        For ``nvars <= 2``, returns a ``Plotter1D`` or
        ``Plotter2DRectangularDomain``.  For ``nvars > 2``, returns a
        ``PairPlotter`` using independent marginals directly (no
        quadrature needed).  An optional *reducer* overrides the
        default independent-marginal construction.

        Parameters
        ----------
        plot_limits : Optional[Array]
            Plot limits.  For 1D: ``[xmin, xmax]``; for 2D:
            ``[xmin, xmax, ymin, ymax]``; for nvars > 2: shape
            ``(nvars, 2)`` with ``[lb, ub]`` per variable.
            Required if the distribution is unbounded.
        reducer : Optional[DimensionReducerProtocol[Array]]
            Dimension reducer for building pair-plot panels.  If *None*
            and ``nvars > 2``, independent marginals are used (no
            quadrature).
        variable_names : Optional[List[str]]
            Axis labels for the pair plot.

        Returns
        -------
        Union[Plotter1D, Plotter2DRectangularDomain, PairPlotter]
            A plotter object.

        Raises
        ------
        ValueError
            If distribution is unbounded and plot_limits is not provided.
        """
        if not self.is_bounded() and plot_limits is None:
            raise ValueError(
                "Must provide plot_limits because distribution is unbounded"
            )
        if self._nvars <= 2:
            if plot_limits is None:
                plot_limits = self._bkd.flatten(self.domain())
            if self._nvars == 1:
                return Plotter1D(self, plot_limits)
            return Plotter2DRectangularDomain(self, plot_limits)

        # Lazy imports to avoid circular dependency
        from pyapprox.interface.functions.marginalize import (
            ReducedFunction,
        )
        from pyapprox.interface.functions.plot.pair_plot import (
            PairPlotter,
        )

        # nvars > 2
        if plot_limits is None:
            plot_limits = self.domain()
        if reducer is not None:
            return PairPlotter(reducer, plot_limits, self._bkd, variable_names)

        # Build PairPlotter from independent marginals (no quadrature)
        bkd = self._bkd

        def _make_1d_fn(idx: int) -> ReducedFunction[Array]:
            marginal = self._marginals[idx]

            def fn(samples: Array) -> Array:
                return marginal.pdf(samples)

            return ReducedFunction(1, 1, fn, bkd)

        def _make_2d_fn(row_idx: int, col_idx: int) -> ReducedFunction[Array]:
            marg_row = self._marginals[row_idx]
            marg_col = self._marginals[col_idx]

            def fn(samples: Array) -> Array:
                return marg_col.pdf(samples[0:1]) * marg_row.pdf(samples[1:2])

            return ReducedFunction(2, 1, fn, bkd)

        functions_1d = [_make_1d_fn(i) for i in range(self._nvars)]
        functions_2d = {
            (i, j): _make_2d_fn(i, j) for i in range(self._nvars) for j in range(i)
        }
        return PairPlotter.from_functions(
            functions_1d, functions_2d, plot_limits, bkd, variable_names
        )

    def _logpdf_jacobian(self, sample: Array) -> Array:
        """
        Compute Jacobian of joint log-PDF for a single sample.

        For independent marginals:
            d/dx_i [sum_j log(p_j)] = d/dx_i [log(p_i)] = logpdf_jacobian_i

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nvars)
        """
        self._validate_input(sample)
        jac_values = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(sample[i], (1, -1))
            # marginal.logpdf_jacobian returns (1, 1) for single sample
            marg_jac = marginal.logpdf_jacobian(row_2d)  # type: ignore
            jac_values.append(marg_jac[0, 0])
        return self._bkd.reshape(self._bkd.asarray(jac_values), (1, -1))

    def _logpdf_jacobian_batch(self, samples: Array) -> Array:
        """
        Compute Jacobian of joint log-PDF for multiple samples.

        For independent marginals:
            d/dx_i [sum_j log(p_j)] = d/dx_i [log(p_i)] = logpdf_jacobian_i

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (nsamples, 1, nvars)
        """
        self._validate_input(samples)
        nsamples = samples.shape[1]
        jac_list = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            # marginal.logpdf_jacobian returns (1, nsamples)
            marg_jac = marginal.logpdf_jacobian(row_2d)  # type: ignore
            jac_list.append(marg_jac[0])  # Shape: (nsamples,)
        # Stack to (nvars, nsamples) then transpose to (nsamples, nvars)
        jac_2d = self._bkd.stack(jac_list, axis=0).T
        # Reshape to (nsamples, 1, nvars)
        return self._bkd.reshape(jac_2d, (nsamples, 1, self._nvars))

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute Jacobian of joint PDF for a single sample.

        Uses product rule: d/dx_i [prod_j p_j] = p'_i * prod_{j!=i} p_j

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nvars)
        """
        self._validate_input(sample)
        # Compute all marginal PDFs
        pdf_vals = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(sample[i], (1, -1))
            pdf_vals.append(marginal(row_2d)[0, 0])  # Scalar

        jac_values = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(sample[i], (1, -1))
            # marginal.pdf_jacobian returns (1, 1) for single sample
            pdf_jac_i = marginal.pdf_jacobian(row_2d)[0, 0]  # type: ignore
            # Product of all other PDFs
            other_product = 1.0
            for j, pdf_val in enumerate(pdf_vals):
                if j != i:
                    other_product = other_product * pdf_val
            jac_values.append(pdf_jac_i * other_product)

        return self._bkd.reshape(self._bkd.asarray(jac_values), (1, -1))

    def _jacobian_batch(self, samples: Array) -> Array:
        """
        Compute Jacobian of joint PDF for multiple samples.

        Uses product rule: d/dx_i [prod_j p_j] = p'_i * prod_{j!=i} p_j

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (nsamples, 1, nvars)
        """
        self._validate_input(samples)
        nsamples = samples.shape[1]

        # Compute all marginal PDFs: shape (nvars, nsamples)
        pdf_vals = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            pdf_vals.append(marginal(row_2d)[0])  # Shape: (nsamples,)
        pdf_stack = self._bkd.stack(pdf_vals, axis=0)  # Shape: (nvars, nsamples)

        # Compute Jacobian for each variable using product rule
        jac_list = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            # marginal.pdf_jacobian returns (1, nsamples)
            pdf_jac_i = marginal.pdf_jacobian(row_2d)[0]  # type: ignore
            # Product of all other PDFs
            other_product = self._bkd.ones((nsamples,))
            for j in range(self._nvars):
                if j != i:
                    other_product = other_product * pdf_stack[j]
            jac_list.append(pdf_jac_i * other_product)

        # Stack to (nvars, nsamples) then transpose to (nsamples, nvars)
        jac_2d = self._bkd.stack(jac_list, axis=0).T
        # Reshape to (nsamples, 1, nvars)
        return self._bkd.reshape(jac_2d, (nsamples, 1, self._nvars))

    def _logpdf_jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of joint log-PDF w.r.t. all distribution parameters.

        For independent marginals with parameters theta = (theta_1, ..., theta_n):
            d/d(theta_i) [sum_j log(p_j)] = d/d(theta_i) [log(p_i)]

        Each marginal's parameter gradients are concatenated.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (nsamples, total_nparams)
            where total_nparams = sum of nparams across all marginals.
        """
        self._validate_input(samples)
        samples.shape[1]

        jac_list = []
        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            # Each marginal returns shape (nsamples, marginal_nparams)
            marg_jac = marginal.logpdf_jacobian_wrt_params(row_2d)  # type: ignore
            jac_list.append(marg_jac)

        # Concatenate along parameter dimension: shape (nsamples, total_nparams)
        return self._bkd.hstack(jac_list)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IndependentJoint(nvars={self._nvars})"

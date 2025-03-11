from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy import stats

from pyapprox.variables.marginals import (
    get_unique_variables,
    get_distribution_info,
    is_bounded_continuous_variable,
    get_truncated_range,
)
from pyapprox.variables._nataf import (
    nataf_joint_density,
    generate_x_samples_using_gaussian_copula,
    transform_correlations,
    scipy_gauss_hermite_pts_wts_1D,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array


class JointVariable(ABC):
    r"""
    Base class for multivariate variables.
    """

    def __init__(self, backend: LinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def rvs(self, nsamples: int) -> Array:
        """
        Generate samples from a random variable.

        Parameters
        ----------
        nsamples : integer
            The number of samples to generate

        Returns
        -------
        samples : Array (nvars, nsamples)
            Independent samples from the target distribution
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def __repr__(self) -> str:
        return self.__str__()

    def pdf_jacobian_implemented(self) -> bool:
        return False

    def pdf_jacobian(self, samples: Array) -> Array:
        raise NotImplementedError

    def pdf_hessian_implemented(self) -> bool:
        # if true then both log pdf and pdf hessian are implemented
        return False

    def pdf_hessian(self, samples: Array) -> Array:
        # if true then both log pdf and pdf hessian are implemented
        raise NotImplementedError


class IndependentMarginalsVariable(JointVariable):
    """
    Class representing independent random variables

    Examples
    --------
    >>> from pyapprox.variables.joint import IndependentMarginalsVariable
    >>> from scipy.stats import norm, beta
    >>> marginals = [norm(0,1),beta(0,1),norm()]
    >>> variable = IndependentMarginalsVariable(marginals)
    >>> print(variable)
    I.I.D. Variable
    Number of variables: 3
    Unique variables and global id:
        norm(loc=0,scale=1): z0, z2
        beta(a=0,b=1,loc=0,scale=1): z1
    """

    def __init__(
        self,
        unique_marginals: List,
        unique_variable_indices: Array = None,
        variable_labels=None,
        backend=NumpyLinAlgMixin,
    ):
        """
        Constructor method
        """
        super().__init__(backend)
        self._bkd = backend
        if unique_variable_indices is None:
            self._unique_marginals, self._unique_variable_indices = (
                get_unique_variables(unique_marginals)
            )
        else:
            self._unique_marginals = unique_marginals.copy()
            self._unique_variable_indices = unique_variable_indices.copy()
        self._nunique_vars = len(self._unique_marginals)
        assert self._nunique_vars == len(self._unique_variable_indices)
        self._nvars = 0
        for ii in range(self._nunique_vars):
            self._unique_variable_indices[ii] = np.asarray(
                self._unique_variable_indices[ii]
            )
            self._nvars += self._unique_variable_indices[ii].shape[0]
        if unique_variable_indices is None:
            assert self._nvars == len(unique_marginals)
        self._variable_labels = variable_labels

    def nvars(self) -> int:
        """
        Return The number of independent 1D variables

        Returns
        -------
        nvars : integer
            The number of independent 1D variables
        """
        return self._nvars

    def marginals(self) -> list:
        """
        Return a list of all the 1D scipy.stats random variables.

        Returns
        -------
        variables : list
            List of :class:`scipy.stats.dist` variables
        """
        all_variables = [None for ii in range(self.nvars())]
        for ii in range(self._nunique_vars):
            for jj in self._unique_variable_indices[ii]:
                all_variables[jj] = self._unique_marginals[ii]
        return all_variables

    def get_statistics(self, function_name: str, *args, **kwargs) -> Array:
        """
        Get a statistic from each univariate random variable.

        Parameters
        ----------
        function_name : str
            The function name of the scipy random variable statistic of
            interest

        kwargs : kwargs
            The arguments to the scipy statistic function

        Returns
        -------
        stat : Array
            The output of the stat function

        Examples
        --------
        >>> import pyapprox as pya
        >>> from scipy.stats import uniform
        >>> nvars = 2
        >>> variable = pya.IndependentMarginalsVariable([uniform(-2, 3)], [np.arange(nvars)])
        >>> variable.get_statistics("interval", confidence=1)
        array([[-2.,  1.],
               [-2.,  1.]])
        >>> variable.get_statistics("pdf",x=np.linspace(-2, 1, 3))
        array([[0.33333333, 0.33333333, 0.33333333],
               [0.33333333, 0.33333333, 0.33333333]])

        """
        for ii in range(self._nunique_vars):
            var = self._unique_marginals[ii]
            indices = self._unique_variable_indices[ii]
            stats_ii = self._bkd.atleast1d(
                self._bkd.asarray(getattr(var, function_name)(*args, **kwargs))
            )
            assert stats_ii.ndim == 1
            if ii == 0:
                stats = self._bkd.empty((self.nvars(), stats_ii.shape[0]))
            stats[indices] = stats_ii
        return stats

    def evaluate(self, function_name: str, x: Array) -> Array:
        """
        Evaluate a frunction for each univariate random variable.

        Parameters
        ----------
        function_name : string
            The function name of the scipy random variable statistic of
            interest

        x : np.ndarray (nsamples)
            The input to the scipy statistic function

        Returns
        -------
        stat : np.ndarray (nsamples, nqoi)
            The outputs of the stat function for each variable
        """
        stats = None
        for ii in range(self._nunique_vars):
            var = self._unique_marginals[ii]
            indices = self._unique_variable_indices[ii]
            for jj in indices:
                stats_jj = self._bkd.atleast1d(
                    self._bkd.asarray(getattr(var, function_name)(x[jj, :]))
                )
                assert stats_jj.ndim == 1
                if stats is None:
                    stats = self._bkd.empty((self.nvars(), stats_jj.shape[0]))
                stats[jj] = stats_jj
        return stats

    def pdf(self, x: Array, log: bool = False) -> Array:
        """
        Evaluate the joint probability distribution function.

        Parameters
        ----------
        x : np.ndarray (nvars, nsamples)
            Values in the domain of the random variable X

        log : boolean
            True - return the natural logarithm of the PDF values
            False - return the PDF values

        Returns
        -------
        values : np.ndarray (nsamples, 1)
            The values of the PDF at x
        """
        assert x.shape[0] == self.nvars()
        if log is False:
            marginal_vals = self.evaluate("pdf", x)
            return self._bkd.prod(marginal_vals, axis=0)[:, None]

        marginal_vals = self.evaluate("logpdf", x)
        return self._bkd.sum(marginal_vals, axis=0)[:, None]

    def _evaluate(self, function_name: str, x: Array):
        """
        Evaluate a frunction for each univariate random variable using rv.dist
        This is faster than evaluate because it avoids error checks.
        Use with caution. dist is only for canonical distribution so
        user must transform variables before using this function

        Parameters
        ----------
        function_name : string
            The function name of the scipy random variable statistic of
            interest

        x : np.ndarray (nsamples)
            The input to the scipy statistic function

        Returns
        -------
        stat : np.ndarray (nsamples, nqoi)
            The outputs of the stat function for each variable
        """
        stats = None
        for ii in range(self._nunique_vars):
            var = self._unique_marginals[ii]
            indices = self._unique_variable_indices[ii]
            for jj in indices:
                stats_jj = self._bkd.atleast1d(
                    getattr(var.dist, function_name)(x[jj, :])
                )
                assert stats_jj.ndim == 1
                if stats is None:
                    stats = self._bkd.empty((self.nvars(), stats_jj.shape[0]))
                stats[jj] = stats_jj
        return stats

    def _pdf(self, x: Array, log: bool = False) -> Array:
        if not log:
            marginal_vals = self._evaluate("_pdf", x)
            return self._bkd.prod(marginal_vals, axis=0)[:, None]

        marginal_vals = self._evaluate("_logpdf", x)
        return self._bkd.sum(marginal_vals, axis=0)[:, None]

    def __str__(self) -> str:
        variable_labels = self._variable_labels
        if variable_labels is None:
            variable_labels = ["z%d" % ii for ii in range(self.nvars())]
        string = "Independent Marginal Variable\n"
        string += f"Number of variables: {self.nvars()}\n"
        string += "Unique variables and global id:\n"
        for ii in range(self._nunique_vars):
            var = self._unique_marginals[ii]
            indices = self._unique_variable_indices[ii]
            name, scales, shapes = get_distribution_info(var)
            shape_string = ",".join(
                [f"{name}={val}" for name, val in shapes.items()]
            )
            scales_string = ",".join(
                [f"{name}={val}" for name, val in scales.items()]
            )
            string += "    " + var.dist.name + "("
            if len(shapes) > 0:
                string += ",".join([shape_string, scales_string])
            else:
                string += scales_string
            string += "): "
            string += ", ".join([variable_labels[idx] for idx in indices])
            if ii < self._nunique_vars - 1:
                string += "\n"
        return string

    def is_bounded_continuous_variable(self) -> bool:
        """
        Are all 1D variables are continuous and bounded.

        Returns
        -------
        is_bounded : boolean
            True - all 1D variables are continuous and bounded
            False - otherwise
        """
        for rv in self._unique_marginals:
            if not is_bounded_continuous_variable(rv):
                return False
        return True

    def rvs(self, num_samples: int, random_states=None) -> Array:
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (nvars, num_samples)
            Independent samples from the target distribution
        """
        num_samples = int(num_samples)
        samples = self._bkd.empty((self.nvars(), num_samples), dtype=float)
        if random_states is not None:
            assert len(random_states) == self.nvars()
        else:
            random_states = [None] * self.nvars()
        for ii in range(self._nunique_vars):
            var = self._unique_marginals[ii]
            indices = self._unique_variable_indices[ii]
            # samples[indices, :] = self._bkd.asarray(
            samples = self._bkd.up(
                samples,
                indices,
                self._bkd.asarray(
                    var.rvs(
                        size=(indices.shape[0], num_samples),
                        random_state=random_states[ii],
                    )
                ),
                axis=0,
            )
        return samples


class GaussCopulaVariable(JointVariable):
    """
    Multivariate random variable with Gaussian correlation and arbitrary
    marginals
    """

    def __init__(
        self,
        marginals: List,
        x_correlation: Array,
        bisection_opts: dict = {},
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._nvars = len(marginals)
        self._marginals = marginals
        self._x_correlation = x_correlation
        self._x_marginal_means = self._bkd.array(
            [m.mean() for m in self._marginals]
        )
        self._x_marginal_stdevs = self._bkd.array(
            [m.std() for m in self._marginals]
        )
        self._x_marginal_pdfs = [m.pdf for m in self._marginals]
        self._x_marginal_cdfs = [m.cdf for m in self._marginals]
        self._x_marginal_inv_cdfs = [m.ppf for m in self._marginals]

        quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
        self._z_correlation = transform_correlations(
            self._x_correlation,
            self._x_marginal_inv_cdfs,
            self._x_marginal_means,
            self._x_marginal_stdevs,
            quad_rule,
            bisection_opts,
        )
        self._z_variable = stats.multivariate_normal(
            mean=self._bkd.zeros((self._nvars)), cov=self._z_correlation
        )

    def z_joint_density(self, z_samples: Array) -> Array:
        return self._z_variable.pdf(z_samples.T)

    def pdf(self, x_samples: Array, log: bool = False) -> Array:
        vals = nataf_joint_density(
            x_samples,
            self._x_marginal_cdfs,
            self._x_marginal_pdfs,
            self.z_joint_density,
            self._bkd,
        )
        if not log:
            return vals
        return self._bkd.log(vals)

    def nvars(self) -> int:
        return self._nvars

    def rvs(self, nsamples: int, return_all: bool = False) -> Array:
        out = generate_x_samples_using_gaussian_copula(
            self.nvars(),
            self._z_correlation,
            self._x_marginal_inv_cdfs,
            nsamples,
            bkd=self._bkd,
        )
        if not return_all:
            return out[0]
        return out

    def marginals(self) -> list:
        return self._marginals


def define_iid_random_variable(rv, nvars: int) -> IndependentMarginalsVariable:
    """
    Create independent identically distributed variables

    Parameters
    ----------
    rv : :class:`scipy.stats.dist`
        A 1D random variable object

    nvars : integer
        The number of 1D variables

    Returns
    -------
    variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
        The multivariate random variable
    """
    unique_marginals = [rv]
    unique_var_indices = [np.arange(nvars)]
    return IndependentMarginalsVariable(unique_marginals, unique_var_indices)


def get_truncated_ranges(
    variable: JointVariable,
    unbounded_alpha: float = 0.99,
    bounded_alpha: float = 1.0,
) -> Array:
    r"""
    Get truncated ranges for independent random variables or Copulas

    Parameters
    ----------
    variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
        Variable

    unbounded_alpha : float
        fraction in (0, 1) of probability captured by ranges for unbounded
        random variables

    bounded_alpha : float
        fraction in (0, 1) of probability captured by ranges for bounded
        random variables. bounded_alpha < 1 is useful when variable is
        bounded but is used in a copula

    Returns
    -------
    ranges : np.ndarray (2*nvars)
        The finite (possibly truncated) ranges of the random variables
        [lb0, ub0, lb1, ub1, ...]
    """
    ranges = []
    if isinstance(variable, GaussCopulaVariable) and (bounded_alpha == 1):
        bounded_alpha = unbounded_alpha

    for rv in variable.marginals():
        ranges += get_truncated_range(rv, unbounded_alpha, bounded_alpha)
    return np.array(ranges)


def combine_uncertain_and_bounded_design_variables(
    random_variable, design_variable, random_variable_indices=None
):
    """
    Convert design variables to random variables defined over them
    optimization bounds.

    Parameters
    ----------
    random_variable_indices : np.ndarray
        The variable numbers of the random variables in the new combined
        variable.
    """

    if random_variable_indices is None:
        random_variable_indices = np.arange(random_variable.nvars())

    if len(random_variable_indices) != random_variable.nvars():
        raise ValueError

    nvars = random_variable.nvars() + design_variable.nvars()
    design_variable_indices = np.setdiff1d(
        np.arange(nvars), random_variable_indices
    )

    variable_list = [None for ii in range(nvars)]
    all_random_variables = random_variable.marginals()
    for ii in range(random_variable.nvars()):
        variable_list[random_variable_indices[ii]] = all_random_variables[ii]
    for ii in range(design_variable.nvars()):
        lb = design_variable.bounds.lb[ii]
        ub = design_variable.bounds.ub[ii]
        if not np.isfinite(lb) or not np.isfinite(ub):
            raise ValueError(f"Design variable {ii} is not bounded")
        rv = stats.uniform(lb, ub - lb)
        variable_list[design_variable_indices[ii]] = rv
    return IndependentMarginalsVariable(variable_list)


class DesignVariable:
    """
    Design variables with no probability information
    """

    def __init__(self, bounds):
        """
        Constructor method

        Parameters
        ----------
        bounds : array_like
            Lower and upper bounds for each variable [lb0,ub0, lb1, ub1, ...]
        """
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be 2d array with two columns")
        self._bounds = bounds

    def nvars(self):
        """
        Return The number of independent 1D variables

        Returns
        -------
        nvars : integer
            The number of independent 1D variables
        """
        return self.bounds().shape[0]

    def bounds(self):
        """Return the bounds of the design variable"""
        return self._bounds


class FiniteSamplesVariable(JointVariable):
    def __init__(
        self,
        samples,
        randomness="replacement",
        weights=None,
        backend=NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._samples = samples.copy()
        self._nvars = samples.shape[0]
        self._weights = weights
        randomness_names = ["none", "replacement"]
        if randomness not in randomness_names:
            raise ValueError(
                "randomness must be one of {0}".format(randomness_names)
            )
        self._randomness = randomness
        self._sample_cnt = 0
        if randomness == "replacement" and weights is not None:
            raise ValueError(
                "weights must be none when randomly sampling with replacement"
            )

    def _rvs_deterministic(self, nsamples):
        if self._sample_cnt + nsamples > self._samples.shape[1]:
            msg = "Too many samples requested when randomness is None. "
            msg += f"self._sample+cnt_nsamples={self._sample_cnt+nsamples}"
            msg += f" but only {self._samples.shape[1]} samples available"
            msg += " This can be overidden by reseting self._sample_cnt=0"
            raise ValueError(msg)
        indices = np.arange(
            self._sample_cnt, self._sample_cnt + nsamples, dtype=int
        )
        self._sample_cnt += nsamples
        return self._samples[:, indices], indices

    def _rvs(self, nsamples: int) -> Array:
        if self._randomness == "none":
            return self._rvs_deterministic(nsamples)

        indices = np.random.choice(
            np.arange(self._samples.shape[1]),
            nsamples,
            p=self._weights,
            replace=True,
        )
        return self._samples[:, indices], indices

    def rvs(self, nsamples: int) -> Array:
        """
        Randomly sample with replacement from all available samples
        if weights is None uniform weights are applied to each sample
        otherwise sample according to weights
        """
        return self._rvs(nsamples)[0]

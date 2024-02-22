import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

from pyapprox.variables.marginals import (
    get_unique_variables, get_distribution_info,
    is_bounded_continuous_variable, get_truncated_range
)
from pyapprox.variables.nataf import (
    nataf_joint_density, generate_x_samples_using_gaussian_copula,
    transform_correlations, scipy_gauss_hermite_pts_wts_1D
)


class JointVariable(ABC):
    r"""
    Base class for multivariate variables.
    """

    @abstractmethod
    def rvs(self, num_samples):
        """
        Generate samples from a random variable.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (num_vars, num_samples)
            Independent samples from the target distribution
        """
        raise NotImplementedError()

    def __str__(self):
        return "JointVariable"


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
    def __init__(self, unique_variables, unique_variable_indices=None,
                 variable_labels=None):
        """
        Constructor method
        """
        if unique_variable_indices is None:
            self.unique_variables, self.unique_variable_indices =\
                get_unique_variables(unique_variables)
        else:
            self.unique_variables = unique_variables.copy()
            self.unique_variable_indices = unique_variable_indices.copy()
        self.nunique_vars = len(self.unique_variables)
        assert self.nunique_vars == len(self.unique_variable_indices)
        self.nvars = 0
        for ii in range(self.nunique_vars):
            self.unique_variable_indices[ii] = np.asarray(
                self.unique_variable_indices[ii])
            self.nvars += self.unique_variable_indices[ii].shape[0]
        if unique_variable_indices is None:
            assert self.nvars == len(unique_variables)
        self.variable_labels = variable_labels

    def num_vars(self):
        """
        Return The number of independent 1D variables

        Returns
        -------
        nvars : integer
            The number of independent 1D variables
        """
        return self.nvars

    def marginals(self):
        """
        Return a list of all the 1D scipy.stats random variables.

        Returns
        -------
        variables : list
            List of :class:`scipy.stats.dist` variables
        """
        all_variables = [None for ii in range(self.nvars)]
        for ii in range(self.nunique_vars):
            for jj in self.unique_variable_indices[ii]:
                all_variables[jj] = self.unique_variables[ii]
        return all_variables

    def get_statistics(self, function_name, *args, **kwargs):
        """
        Get a statistic from each univariate random variable.

        Parameters
        ----------
        function_name : string
            The function name of the scipy random variable statistic of
            interest

        kwargs : kwargs
            The arguments to the scipy statistic function

        Returns
        -------
        stat : np.ndarray
            The output of the stat function

        Examples
        --------
        >>> import pyapprox as pya
        >>> from scipy.stats import uniform
        >>> num_vars = 2
        >>> variable = pya.IndependentMarginalsVariable([uniform(-2, 3)], [np.arange(num_vars)])
        >>> variable.get_statistics("interval", confidence=1)
        array([[-2.,  1.],
               [-2.,  1.]])
        >>> variable.get_statistics("pdf",x=np.linspace(-2, 1, 3))
        array([[0.33333333, 0.33333333, 0.33333333],
               [0.33333333, 0.33333333, 0.33333333]])

        """
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            stats_ii = np.atleast_1d(getattr(var, function_name)(
                *args, **kwargs))
            assert stats_ii.ndim == 1
            if ii == 0:
                stats = np.empty((self.num_vars(), stats_ii.shape[0]))
            stats[indices] = stats_ii
        return stats

    def evaluate(self, function_name, x):
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
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            for jj in indices:
                stats_jj = np.atleast_1d(getattr(var, function_name)(x[jj, :]))
                assert stats_jj.ndim == 1
                if stats is None:
                    stats = np.empty((self.num_vars(), stats_jj.shape[0]))
                stats[jj] = stats_jj
        return stats

    def pdf(self, x, log=False):
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
        assert x.shape[0] == self.num_vars()
        if log is False:
            marginal_vals = self.evaluate("pdf", x)
            return np.prod(marginal_vals, axis=0)[:, None]

        marginal_vals = self.evaluate("logpdf", x)
        return np.sum(marginal_vals, axis=0)[:, None]

    def _evaluate(self, function_name, x):
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
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            for jj in indices:
                stats_jj = np.atleast_1d(
                    getattr(var.dist, function_name)(x[jj, :]))
                assert stats_jj.ndim == 1
                if stats is None:
                    stats = np.empty((self.num_vars(), stats_jj.shape[0]))
                stats[jj] = stats_jj
        return stats

    def _pdf(self, x, log=False):
        if not log:
            marginal_vals = self._evaluate("_pdf", x)
            return np.prod(marginal_vals, axis=0)[:, None]

        marginal_vals = self._evaluate("_logpdf", x)
        return np.sum(marginal_vals, axis=0)[:, None]

    def __str__(self):
        variable_labels = self.variable_labels
        if variable_labels is None:
            variable_labels = ["z%d" % ii for ii in range(self.num_vars())]
        string = "Independent Marginal Variable\n"
        string += f"Number of variables: {self.num_vars()}\n"
        string += "Unique variables and global id:\n"
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            name, scales, shapes = get_distribution_info(var)
            shape_string = ",".join(
                [f"{name}={val}" for name, val in shapes.items()])
            scales_string = ",".join(
                [f"{name}={val}" for name, val in scales.items()])
            string += "    "+var.dist.name + "("
            if len(shapes) > 0:
                string += ",".join([shape_string, scales_string])
            else:
                string += scales_string
            string += "): "
            string += ", ".join(
                [variable_labels[idx] for idx in indices])
            if ii < self.nunique_vars-1:
                string += "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def is_bounded_continuous_variable(self):
        """
        Are all 1D variables are continuous and bounded.

        Returns
        -------
        is_bounded : boolean
            True - all 1D variables are continuous and bounded
            False - otherwise
        """
        for rv in self.unique_variables:
            if not is_bounded_continuous_variable(rv):
                return False
        return True

    def rvs(self, num_samples, random_states=None):
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (num_vars, num_samples)
            Independent samples from the target distribution
        """
        num_samples = int(num_samples)
        samples = np.empty((self.num_vars(), num_samples), dtype=float)
        if random_states is not None:
            assert len(random_states) == self.num_vars()
        else:
            random_states = [None]*self.num_vars()
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            samples[indices, :] = var.rvs(
                size=(indices.shape[0], num_samples),
                random_state=random_states[ii])
        return samples


class GaussCopulaVariable(JointVariable):
    """
    Multivariate random variable with Gaussian correlation and arbitrary
    marginals
    """

    def __init__(self, marginals, x_correlation, bisection_opts={}):
        self.nvars = len(marginals)
        self._marginals = marginals
        self.x_correlation = x_correlation
        self.x_marginal_means = np.array([m.mean() for m in self._marginals])
        self.x_marginal_stdevs = np.array([m.std() for m in self._marginals])
        self.x_marginal_pdfs = [m.pdf for m in self._marginals]
        self.x_marginal_cdfs = [m.cdf for m in self._marginals]
        self.x_marginal_inv_cdfs = [m.ppf for m in self._marginals]

        quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
        self.z_correlation = transform_correlations(
            self.x_correlation, self.x_marginal_inv_cdfs,
            self.x_marginal_means, self.x_marginal_stdevs, quad_rule,
            bisection_opts)
        self.z_variable = stats.multivariate_normal(
            mean=np.zeros((self.nvars)), cov=self.z_correlation)

    def z_joint_density(self, z_samples):
        return self.z_variable.pdf(z_samples.T)

    def pdf(self, x_samples, log=False):
        vals = nataf_joint_density(
            x_samples, self.x_marginal_cdfs, self.x_marginal_pdfs,
            self.z_joint_density)
        if not log:
            return vals
        return np.log(vals)

    def num_vars(self):
        return self.nvars

    def rvs(self, nsamples, return_all=False):
        out = generate_x_samples_using_gaussian_copula(
            self.nvars, self.z_correlation, self.x_marginal_inv_cdfs, nsamples)
        if not return_all:
            return out[0]
        return out

    def marginals(self):
        return self._marginals


def define_iid_random_variable(rv, num_vars):
    """
    Create independent identically distributed variables

    Parameters
    ----------
    rv : :class:`scipy.stats.dist`
        A 1D random variable object

    num_vars : integer
        The number of 1D variables

    Returns
    -------
    variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
        The multivariate random variable
    """
    unique_variables = [rv]
    unique_var_indices = [np.arange(num_vars)]
    return IndependentMarginalsVariable(
        unique_variables, unique_var_indices)


def get_truncated_ranges(variable, unbounded_alpha=0.99, bounded_alpha=1.0):
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
    if (type(variable) == GaussCopulaVariable) and (bounded_alpha == 1):
        bounded_alpha = unbounded_alpha

    for rv in variable.marginals():
        ranges += get_truncated_range(rv, unbounded_alpha, bounded_alpha)
    return np.array(ranges)


def combine_uncertain_and_bounded_design_variables(
        random_variable, design_variable, random_variable_indices=None):
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
        random_variable_indices = np.arange(random_variable.num_vars())

    if len(random_variable_indices) != random_variable.num_vars():
        raise ValueError

    nvars = random_variable.num_vars() + design_variable.num_vars()
    design_variable_indices = np.setdiff1d(
        np.arange(nvars), random_variable_indices)

    variable_list = [None for ii in range(nvars)]
    all_random_variables = random_variable.marginals()
    for ii in range(random_variable.num_vars()):
        variable_list[random_variable_indices[ii]] = all_random_variables[ii]
    for ii in range(design_variable.num_vars()):
        lb = design_variable.bounds.lb[ii]
        ub = design_variable.bounds.ub[ii]
        if not np.isfinite(lb) or not np.isfinite(ub):
            raise ValueError(f"Design variable {ii} is not bounded")
        rv = stats.uniform(lb, ub-lb)
        variable_list[design_variable_indices[ii]] = rv
    return IndependentMarginalsVariable(variable_list)


class DesignVariable(object):
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
        self.bounds = bounds

    def num_vars(self):
        """
        Return The number of independent 1D variables

        Returns
        -------
        nvars : integer
            The number of independent 1D variables
        """
        return len(self.bounds.lb)

from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.variables.marginals import (
    Marginal,
    ContinuousMarginalMixin,
    ContinuousScipyMarginal,
    DiscreteScipyMarginal,
    parse_marginal,
)
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.template import BackendMixin, Array


class JointVariable(ABC):
    r"""
    Base class for multivariate variables.
    """

    def __init__(self, backend: BackendMixin):
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

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    def pdf(self, samples: Array) -> Array:
        raise NotImplementedError

    def logpdf(self, samples: Array) -> Array:
        raise NotImplementedError

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

    def get_plot_axis(self, figsize=(8, 6), surface=False):
        if self.nvars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def _plot_pdf_1d(self, ax, npts_1d: Array, plot_limits: Array, **kwargs):
        plot_xx = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        ax.plot(plot_xx[0], self.pdf(plot_xx), **kwargs)

    def meshgrid_samples(
        self, plot_limits: Array, npts_1d: Union[Array, int] = 51
    ) -> Array:
        if self.nvars() != 2:
            raise RuntimeError("nvars !=2.")
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        return X, Y, pts

    def plot_pdf(
        self, ax, plot_limits: Array, npts_1d: Union[Array, int] = 51, **kwargs
    ):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot PDF when nvars >= 3.")
        if len(plot_limits) != self.nvars() * 2:
            raise ValueError("plot_limits has the wrong shape")
        if not isinstance(npts_1d, list):
            npts_1d = [npts_1d] * self.nvars()
        if self.nvars() == 1:
            return self._plot_pdf_1d(ax, npts_1d, plot_limits, **kwargs)
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        Z = self._bkd.reshape(self.pdf(pts), X.shape)
        if kwargs.get("levels", None) is None:
            if ax.name != "3d":
                raise ValueError(
                    "levels not specified so trying to plot surface but not"
                    " given 3d axis"
                )
            return ax.plot_surface(X, Y, Z, **kwargs)
        return ax.contourf(X, Y, Z, **kwargs)


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
        marginals: Union[
            List[Marginal], List[Union[stats.rv_continuous, stats.rv_discrete]]
        ],
        unique_indices: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Constructor method
        """
        super().__init__(backend)
        self._bkd = backend

        original_marginals = marginals
        marginals = [
            parse_marginal(marginal, backend)
            for marginal in original_marginals
        ]
        if unique_indices is None:
            self._unique_marginals, self._unique_indices = (
                self._get_unique_marginals(marginals)
            )
        else:
            self._unique_marginals = marginals
            self._unique_indices = unique_indices
        self._nunique_vars = len(self._unique_marginals)
        self._nvars = 0
        for ii in range(self._nunique_vars):
            self._unique_indices[ii] = np.asarray(self._unique_indices[ii])
            self._nvars += self._unique_indices[ii].shape[0]
        if unique_indices is None and self._nvars != len(marginals):
            raise ValueError("unique_indices and marginals are inconsistent")

    def _get_unique_marginals(
        self, marginals: List[Marginal]
    ) -> Tuple[Marginal, Array]:
        """
        Get the unique 1D marginals from a list of marginals.
        """
        nvars = len(marginals)
        unique_marginals = [marginals[0]]
        unique_var_indices = [[0]]
        for ii in range(1, nvars):
            found = False
            for jj in range(len(unique_marginals)):
                if marginals[ii] == unique_marginals[jj]:
                    unique_var_indices[jj].append(ii)
                    found = True
                    break
            if not found:
                unique_marginals.append(marginals[ii])
                unique_var_indices.append([ii])
        return unique_marginals, unique_var_indices

    def nvars(self) -> int:
        """
        Return The number of independent 1D marginals

        Returns
        -------
        nvars : integer
            The number of independent 1D marginals
        """
        return self._nvars

    def marginals(self) -> List[Marginal]:
        """
        Return a list of all the 1D Marginals.

        Returns
        -------
        marginals : list
            List of 1D Marginals
        """
        all_variables = [None for ii in range(self.nvars())]
        for ii in range(self._nunique_vars):
            for jj in self._unique_indices[ii]:
                all_variables[jj] = self._unique_marginals[ii]
        return all_variables

    def mean(self) -> Array:
        return self._bkd.asarray(
            [marginal.mean() for marginal in self.marginals()]
        )[:, None]

    def std(self) -> Array:
        return self._bkd.asarray(
            [marginal.std() for marginal in self.marginals()]
        )[:, None]

    def var(self) -> Array:
        return self._bkd.asarray(
            [marginal.var() for marginal in self.marginals()]
        )[:, None]

    def covariance(self) -> Array:
        return self._bkd.diag(self.var()[:, 0])

    def median(self) -> Array:
        return self._bkd.asarray(
            [marginal.median() for marginal in self.marginals()]
        )[:, None]

    def interval(self, alpha: float) -> Array:
        return self._bkd.stack(
            [marginal.interval(alpha) for marginal in self.marginals()], axis=0
        )

    def truncated_ranges(self, alpha: float = None) -> Array:
        return self._bkd.stack(
            [marginal.truncated_range(alpha) for marginal in self.marginals()],
            axis=0,
        )

    def _check_samples(self, samples: Array):
        if samples.ndim != 2 or samples.shape[0] != self.nvars():
            raise ValueError("samples has the wrong shape")

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability distribution function.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Values in the domain of the random variable X

        Returns
        -------
        values : np.ndarray (nsamples, 1)
            The values of the PDF at x
        """
        self._check_samples(samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.pdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.prod(marginal_vals, axis=0)[:, None]

    def logpdf(self, samples: Array, log: bool = False) -> Array:
        self._check_samples(samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.logpdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.sum(marginal_vals, axis=0)[:, None]

    def ppf(self, samples: Array) -> Array:
        """
        Compute inverse cdf on each marginal independently
        """
        return self._bkd.stack(
            [
                marginal.ppf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )

    def __repr__(self) -> str:
        if self.nvars() > 5:
            return "{0}(nvars={1})".format(
                self.__class__.__name__, self.nvars()
            )
        return "{0}(nvars={1}, {2})".format(
            self.__class__.__name__,
            self.nvars(),
            ", ".join([str(m) for m in self.marginals()]),
        )

    def is_bounded_continuous_variable(self) -> bool:
        """
        Are all 1D variables are continuous and bounded.

        Returns
        -------
        is_bounded : boolean
            True - all 1D variables are continuous and bounded
            False - otherwise
        """
        for marginal in self._unique_marginals:
            if not marginal.is_bounded() or not isinstance(
                marginal, ContinuousMarginalMixin
            ):
                return False
        return True

    def rvs(self, nsamples: int) -> Array:
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        nsamples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (nvars, nsamples)
            Independent samples from the target distribution
        """
        marginal_samples = [
            marginal.rvs(nsamples) for marginal in self.marginals()
        ]
        return self._bkd.stack(marginal_samples, axis=0)

    def _rvs_given_random_states(
        self, nsamples: int, random_states: List
    ) -> Array:
        # needed to avoid race conditions on seed when running rvs
        # on multiple processors
        nsamples = int(nsamples)
        samples = self._bkd.empty((self.nvars(), nsamples), dtype=float)
        if random_states is not None:
            assert len(random_states) == self.nvars()
        else:
            random_states = [None] * self.nvars()
        for ii, marginal in enumerate(self.marginals()):
            samples[ii, :] = self._bkd.asarray(
                marginal._scipy_rv.rvs(
                    nsamples, random_state=random_states[ii]
                )
            )
        return samples

    def kl_divergence(self, other: "IndependentMarginalsVariable"):
        for marginal in self.marginals():
            if not marginal.kl_divergence_implemented():
                raise NotImplementedError(
                    f"{marginal} does not support KL divergence so divergence "
                    "of joint cannot be computed"
                )
        other_marginals = other.marginals()
        return sum(
            [
                marginal.kl_divergence(other_marginals[ii])
                for ii, marginal in enumerate(self.marginals())
            ]
        )


def define_iid_random_variable(
    marginal: Union[Marginal, stats.rv_continuous, stats.rv_discrete],
    nvars: int,
    backend: BackendMixin = NumpyMixin,
) -> IndependentMarginalsVariable:
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
    unique_marginals = [parse_marginal(marginal, backend)]
    unique_indices = [backend.arange(nvars)]
    return IndependentMarginalsVariable(
        unique_marginals, unique_indices, backend=backend
    )


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
        samples: Array,
        randomness: str = "replacement",
        weights: Array = None,
        backend: BackendMixin = NumpyMixin,
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

    def nvars(self) -> int:
        return self._nvars

    def _rvs_deterministic(self, nsamples: int) -> Array:
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


class RejectionSamplingVariable:
    def __init__(
        self,
        target: JointVariable,
        proposal: JointVariable,
        envelope_factor: float,
        verbosity: int = False,
        batch_size: int = None,
    ) -> Array:
        """
        Obtain samples from a density f(x) using samples from a proposal
        distribution g(x).

        Parameters
        ----------
        target : JointVariable
            The target density f(x)

        proposal : JointVariable
            The proposal density g(x)

        envelope_factor : float
            Factor M that satifies f(x)<=Mg(x). Set M such that inequality is
            close to equality as possible

        verbosity: integer
            Flag specifying the amount of diagnostic information printed

        batch_size : integer
            The number of evaluations of each density to be performed in a batch.
            Almost always we should set batch_size=nsamples
        """
        self._bkd = target._bkd
        self._target = target
        self._proposal = proposal
        self._envelope_factor = envelope_factor
        self._verbosity = verbosity
        self._batch_size = batch_size
        self._nvars = self._proposal.nvars()

    def rvs(self, nsamples: int) -> Array:
        """
        Parameters
        ----------

        nsamples : integer
            The number of samples required

        Returns
        -------
        samples : Array (nvars, nsamples)
            Independent samples from the target distribution
        """
        nsamples = int(nsamples)
        if self._batch_size is None:
            batch_size = nsamples
        else:
            batch_size = self._batch_size

        cntr = 0
        nproposal_samples = 0
        samples = self._bkd.empty((self._nvars, nsamples), dtype=float)
        while cntr < nsamples:
            proposal_samples = self._proposal.rvs(batch_size)
            target_vals = self._target.pdf(proposal_samples)[:, 0]
            proposal_vals = self._proposal.pdf(proposal_samples)[:, 0]
            urand = self._bkd.asarray(
                np.random.uniform(0.0, 1.0, (batch_size))
            )

            # ensure envelop_factor is large enough
            if self._bkd.any(
                target_vals > (self._envelope_factor * proposal_vals)
            ):
                idx = self._bkd.argmax(
                    target_vals / (self._envelope_factor * proposal_vals)
                )
                msg = "proposal_density*envelop factor does not bound target "
                msg += "density: %f,%f" % (
                    target_vals[idx],
                    (self._envelope_factor * proposal_vals)[idx],
                )
                raise ValueError(msg)

            idx = self._bkd.where(
                urand < target_vals / (self._envelope_factor * proposal_vals)
            )[0]

            nbatch_samples_accepted = min(idx.shape[0], nsamples - cntr)
            idx = idx[:nbatch_samples_accepted]
            samples[:, cntr : cntr + nbatch_samples_accepted] = (
                proposal_samples[:, idx]
            )
            cntr += nbatch_samples_accepted
            nproposal_samples += batch_size

        if self._verbosity > 0:
            print(("num accepted", nsamples))
            print(("num rejected", nproposal_samples - nsamples))
            print(("inverse envelope factor", 1 / self._envelope_factor))
            print(
                (
                    "acceptance probability",
                    float(nsamples) / float(nproposal_samples),
                )
            )
        return samples

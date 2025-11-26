from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Generic, Union

import numpy as np
from scipy.stats import _continuous_distns, _discrete_distns
from matplotlib.axes import Axes

from pyapprox.typing.util.backend import Array, Backend, validate_backend


FrozenScipyConinousVariable = (
    "scipy.stats._distn_infrastructure.rv_continuous_frozen"
)
FrozenScipyDiscreteVariable = (
    "scipy.stats._distn_infrastructure.rv_discrete_frozen"
)
FrozenScipyVariable = Union[
    FrozenScipyConinousVariable, FrozenScipyDiscreteVariable
]


class ScipyRandomVariable1D(ABC, Generic[Array]):
    """
    Marginal distribution based on SciPy random variables.

    Parameters
    ----------
    scipy_rv : rv_continuous or rv_discrete
        The SciPy random variable representing the marginal distribution.
    backend : BackendMixin
        The backend to use for computations.
    """

    def __init__(
        self,
        scipy_rv: FrozenScipyVariable,
        bkd: Backend,
    ):
        validate_backend(bkd)
        self._bkd = bkd
        self._validate_marginal(scipy_rv)
        self._scipy_rv = scipy_rv
        self._name, self._scales, self._shapes = self.get_distribution_info()

    def interval(self, alpha: float) -> Array:
        """
        Compute the interval containing the given probability mass.

        Parameters
        ----------
        alpha : float
            The probability mass to include in the interval.

        Returns
        -------
        interval : Array
            The interval containing the given probability mass.
        """
        return self._bkd.asarray(self._scipy_rv.interval(alpha))

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            The points at which to evaluate the PDF.

        Returns
        -------
        pdf_vals : Array
            The evaluated PDF values.
        """
        return self._bkd.asarray(
            self._scipy_rv.pdf(self._bkd.to_numpy(samples))
        )

    def __call__(self, samples: Array) -> Array:
        return self.pdf(samples)

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function (CDF).

        Parameters
        ----------
        samples : Array
            The points at which to evaluate the CDF.

        Returns
        -------
        cdf_vals : Array
            The evaluated CDF values.
        """
        return self._bkd.asarray(
            self._scipy_rv.cdf(self._bkd.to_numpy(samples))
        )

    def ppf(self, usamples: Array) -> Array:
        """
        Evaluate the percent point function (inverse CDF).

        Parameters
        ----------
        usamples : Array
            The points at which to evaluate the inverse CDF.

        Returns
        -------
        ppf_vals : Array
            The evaluated inverse CDF values.
        """
        return self._bkd.asarray(self._scipy_rv.ppf(usamples))

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        nsamples : int
            The number of samples to generate.

        Returns
        -------
        samples : Array
            Random samples from the distribution.
        """
        return self._bkd.asarray(self._scipy_rv.rvs(int(nsamples)))

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        mean : float
            The mean of the distribution.
        """
        return self._bkd.asarray([self._scipy_rv.mean()]).item()

    def median(self) -> float:
        """
        Compute the median of the distribution.

        Returns
        -------
        median : float
            The median of the distribution.
        """
        return self._bkd.asarray([self._scipy_rv.median()]).item()

    def var(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        var : float
            The variance of the distribution.
        """
        return self._bkd.asarray(self._scipy_rv.var()).item()

    def std(self) -> Array:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        std : float
            The standard deviation of the distribution.
        """
        return self._bkd.asarray(self._scipy_rv.std())

    def __eq__(self, other: object) -> bool:
        """
        Determine if two SciPy random variables are equivalent.

        Parameters
        ----------
        other : ScipyRandomVariable1D
            The other distribution to compare.

        Returns
        -------
        equal : bool
            True if the distributions are equivalent, False otherwise.
        """
        if not isinstance(other, ScipyRandomVariable1D):
            return False
        if self._name != other._name:
            return False
        if self._scales != other._scales:
            return False
        return self._shapes == other._shapes

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        repr : str
            A string representation of the object.
        """
        return "{0}(name={1})".format(self.__class__.__name__, self._name)

    def get_distribution_info(
        self,
    ) -> Tuple[str, Dict[str, Array], Dict[str, Any]]:
        """
        Get important information from a scipy.stats variable.

        Notes
        -----
        Shapes and scales can appear in either args or kwargs depending on how
        the user initializes the frozen object.

        Returns
        -------
        name : str
            The name of the distribution.
        scales : Dict[str, Array]
            The scale parameters of the distribution (e.g., loc and scale).
        shapes : Dict[str, Any]
            The shape parameters of the distribution.
        """
        name = self._scipy_rv.dist.name
        shape_names = self._scipy_rv.dist.shapes
        if shape_names is not None:
            shape_names = [name.strip() for name in shape_names.split(",")]
            shape_values = [
                self._scipy_rv.args[ii]
                for ii in range(
                    min(len(self._scipy_rv.args), len(shape_names))
                )
            ]
            shape_values += [
                self._scipy_rv.kwds[shape_names[ii]]
                for ii in range(len(self._scipy_rv.args), len(shape_names))
            ]
            shapes = dict(zip(shape_names, shape_values))
        else:
            shapes = dict()

        scale_values = [
            self._scipy_rv.args[ii]
            for ii in range(len(shapes), len(self._scipy_rv.args))
        ]
        scale_values += [
            self._scipy_rv.kwds[key]
            for key in self._scipy_rv.kwds
            if key not in shapes
        ]
        if len(scale_values) == 0:
            scale_values = [0, 1]
        elif len(scale_values) == 1 and len(self._scipy_rv.args) > len(shapes):
            scale_values += [1.0]
        elif len(scale_values) == 1 and "scale" not in self._scipy_rv.kwds:
            scale_values += [1.0]
        elif len(scale_values) == 1 and "loc" not in self._scipy_rv.kwds:
            scale_values = [0] + scale_values
        scale_names = ["loc", "scale"]
        scales = dict(
            zip(scale_names, [self._bkd.asarray([s]) for s in scale_values])
        )
        return name, scales, shapes

    @abstractmethod
    def is_bounded(self) -> bool:
        """
        Check if the distribution is bounded.

        Returns
        -------
        bounded : bool
            True if the distribution is bounded, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_marginal(self, marginal: FrozenScipyVariable) -> None:
        """
        Validate the marginal distribution.

        Parameters
        ----------
        marginal : rv_continuous
            The SciPy random variable to validate.

        Raises
        ------
        ValueError
            If the marginal distribution is invalid.
        """
        raise NotImplementedError


class ContinuousScipyRandomVariable1D(ScipyRandomVariable1D):
    """
    Continuous marginal distribution based on SciPy random variables.
    """

    def _validate_marginal(
        self, scipy_rv: FrozenScipyConinousVariable
    ) -> None:
        """
        Validate that the marginal is a continuous SciPy random variable.

        Parameters
        ----------
        marginal : rv_continuous
            The SciPy random variable to validate.

        Raises
        ------
        ValueError
            If the marginal is not a continuous SciPy random variable.
        """
        if scipy_rv.dist.name not in _continuous_distns._distn_names:
            raise ValueError("marginal is not a continuous SciPy variable")

    def is_bounded(self) -> bool:
        """
        Check if the variable is bounded and continuous.

        Returns
        -------
        is_bounded : bool
            True if the variable is bounded, False otherwise.
        """
        interval = self._scipy_rv.interval(1)
        return np.isfinite(interval[0]) and np.isfinite(interval[1])

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log of the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            The points at which to evaluate the log PDF.

        Returns
        -------
        logpdf_vals : Array
            The evaluated log PDF values.
        """
        return self._bkd.asarray(
            self._scipy_rv.logpdf(self._bkd.to_numpy(samples))
        )


class DiscreteScipyRandomVariable1D(ScipyRandomVariable1D):
    """
    Discrete marginal distribution based on SciPy random variables.
    """

    def _validate_marginal(self, scipy_rv: FrozenScipyDiscreteVariable):
        if not (scipy_rv.dist.name in _discrete_distns._distn_names):
            raise ValueError("marginal is not a discrete scipy variable")

    def pdf(self, samples: Array) -> Array:
        return self._scipy_rv.pmf(samples)

    def is_bounded(self) -> bool:
        interval = self._scipy_rv.interval(1)
        return np.isfinite(interval[0]) and np.isfinite(interval[1])

    def _probability_masses(self, alpha: float = None) -> Array:
        """
        Get the the locations and masses of a discrete random variable.

        Parameters
        ----------
        tol : float
            Fraction of total probability in (0, 1). Can be useful with
            extracting masses when numerical precision becomes a problem
        """
        if self._name == "hypergeom":
            M, n, N = [self._shapes[key] for key in ["M", "n", "N"]]
            xk = self._bkd.arange(
                max(0, N - M + n), min(n, N) + 1, dtype=float
            )
            pk = self._scipy_rv.pmf(xk)
            return xk, pk

        if self._name == "binom":
            n = self._shapes["n"]
            xk = self._bkd.arange(0, n + 1, dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk

        if (
            self._name == "nbinom"
            or self._name == "geom"
            or self._name == "logser"
            or self._name == "poisson"
            or self._name == "planck"
            or self._name == "zipf"
            or self._name == "dlaplace"
            or self._name == "skellam"
        ):
            lb, ub = self.truncated_range(alpha)
            xk = self._bkd.arange(int(lb), int(ub), dtype=float)
            #  pk = self._scipy_rv.pmf(xk)
            pk = self.pdf(xk)
            return xk, pk

        if self._name == "boltzmann":
            xk = self._bkd.arange(self._shapes["N"], dtype=float)
            pk = self._scipy_rv.pmf(xk)
            return xk, pk

        if self._name == "randint":
            xk = self._bkd.arange(
                self._shapes["low"], self._shapes["high"], dtype=float
            )
            pk = self._scipy_rv.pmf(xk)
            return xk, pk

        raise NotImplementedError(
            f"Variable {self._scipy_rv.dist.self._name} not supported"
        )

    def _shapes_equal(self, other: "DiscreteScipyRandomVariable1D") -> bool:
        if "xk" not in self._shapes:
            return super()._shapes_equal(other)
        # xk and pk shapes are list so != comparison will not work
        not_equiv = self._bkd.any(
            self._shapes["xk"] != other.shapes["xk"]
        ) or self._bkd.any(self._shapes["pk"] != other._shapes["pk"])
        return not not_equiv

    def plot(self, ax: Axes):
        """
        Plot the probability density function (PDF) of the marginal variable.

        Parameters
        ----------
        ax : matplotlib Axes
            The axes to plot on.
        """
        xk, pk = self._probability_masses()
        ax.plot(xk, pk, "o")
        for s, w in zip(xk, pk):
            ax.vlines(x=s, ymin=0, ymax=w)

    def probability_masses(self, alpha: float = None) -> Array:
        """
        Get the the locations and masses of a discrete random variable.

        Parameters
        ----------
        tol : float
            Fraction of total probability in (0, 1). Can be useful with
            extracting masses when numerical precision becomes a problem
        """
        xk, pk = self._probability_masses(alpha)
        return self._bkd.asarray(xk), self._bkd.asarray(pk)

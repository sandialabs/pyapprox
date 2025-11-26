from typing import Tuple, Dict, Any, Generic

import numpy as np
from scipy.stats import _continuous_distns

from pyapprox.typing.util.backend import Array, Backend, validate_backend


class ContinuousScipyRandomVariable1D(Generic[Array]):
    """
    Continuous marginal distribution based on SciPy random variables.

    Parameters
    ----------
    scipy_rv : rv_continuous
        The SciPy random variable representing the marginal distribution.
    bkd : Backend
        The backend to use for computations.
    """

    def __init__(
        self,
        scipy_rv: Any,  # Expected to be a ScipyContinuousFrozenProtocol
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._bkd = bkd
        self._validate_marginal(scipy_rv)
        self._scipy_rv = scipy_rv
        self._name, self._scales, self._shapes = self.get_distribution_info()

    def _validate_marginal(self, scipy_rv: Any) -> None:
        """
        Validate that the marginal is a continuous SciPy random variable.

        Parameters
        ----------
        scipy_rv : rv_continuous
            The SciPy random variable to validate.

        Raises
        ------
        ValueError
            If the marginal is not a continuous SciPy random variable.
        """
        if scipy_rv.dist.name not in _continuous_distns._distn_names:
            raise ValueError("marginal is not a continuous SciPy variable")

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
        return self._bkd.asarray(
            self._scipy_rv.ppf(self._bkd.to_numpy(usamples))
        )

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

    def mean(self) -> Array:
        """
        Compute the mean of the distribution.

        Returns
        -------
        mean : Array
            The mean of the distribution.
        """
        return self._bkd.asarray([self._scipy_rv.mean()])

    def median(self) -> Array:
        """
        Compute the median of the distribution.

        Returns
        -------
        median : Array
            The median of the distribution.
        """
        return self._bkd.asarray([self._scipy_rv.median()])

    def var(self) -> Array:
        """
        Compute the variance of the distribution.

        Returns
        -------
        var : Array
            The variance of the distribution.
        """
        return self._bkd.asarray(self._scipy_rv.var())

    def std(self) -> Array:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        std : Array
            The standard deviation of the distribution.
        """
        return self._bkd.asarray(self._scipy_rv.std())

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

    def is_bounded(self) -> bool:
        """
        Check if the variable is bounded and continuous.

        Returns
        -------
        is_bounded : bool
            True if the variable is bounded, False otherwise.
        """
        interval = self._scipy_rv.interval(1)
        return bool(np.isfinite(interval[0]) and np.isfinite(interval[1]))

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

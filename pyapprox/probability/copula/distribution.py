"""
Copula-based joint distribution.

Composes a copula with marginal distributions to form a joint distribution:
    f(x) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)
    log f(x) = log c(u) + sum_i log f_i(x_i)
where u_i = F_i(x_i) is the probability integral transform.
"""

from typing import Generic, List, Optional

import numpy as np

from pyapprox.interface.functions.plot.plot1d import Plotter1D
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
)
from pyapprox.probability.copula.protocols import CopulaProtocol
from pyapprox.probability.protocols import MarginalProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class CopulaDistribution(Generic[Array]):
    """
    Joint distribution from a copula and marginals.

    The joint density is:
        f(x) = c(F_1(x_1), ..., F_d(x_d)) * prod_i f_i(x_i)

    Parameters
    ----------
    copula : CopulaProtocol[Array]
        The copula modeling the dependence structure.
    marginals : List[MarginalProtocol[Array]]
        The marginal distributions.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If copula dimension doesn't match number of marginals.
    """

    def __init__(
        self,
        copula: CopulaProtocol[Array],
        marginals: List[MarginalProtocol[Array]],
        bkd: Backend[Array],
    ):
        if copula.nvars() != len(marginals):
            raise ValueError(
                f"Copula has {copula.nvars()} variables but "
                f"{len(marginals)} marginals were provided"
            )
        self._copula = copula
        self._marginals = marginals
        self._bkd = bkd
        self._nvars = len(marginals)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nparams(self) -> int:
        """Return the total number of parameters (copula + all marginals)."""
        total = self._copula.nparams()
        for m in self._marginals:
            if hasattr(m, "nparams"):
                total += m.nparams()
        return total

    def copula(self) -> CopulaProtocol[Array]:
        """Return the copula."""
        return self._copula

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Return the marginal distributions."""
        return self._marginals

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return concatenated hyperparameter list (copula + marginals).

        Returns
        -------
        HyperParameterList[Array]
            Combined hyperparameter list.
        """
        combined = self._copula.hyp_list()
        for m in self._marginals:
            if hasattr(m, "hyp_list"):
                combined = combined + m.hyp_list()
        return combined

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
        Evaluate the log joint density.

        log f(x) = log c(u_1, ..., u_d) + sum_i log f_i(x_i)
        where u_i = F_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log joint density values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        self._validate_input(samples)
        nsamples = samples.shape[1]

        # Apply PIT: u_i = F_i(x_i)
        u = self._bkd.zeros((self._nvars, nsamples))
        marginal_logpdf_sum = self._bkd.zeros((1, nsamples))

        for i, marginal in enumerate(self._marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            u[i] = marginal.cdf(row_2d)[0]
            marginal_logpdf_sum = marginal_logpdf_sum + marginal.logpdf(row_2d)

        # Copula log-density
        copula_logpdf = self._copula.logpdf(u)

        return copula_logpdf + marginal_logpdf_sum

    def sample(self, nsamples: int) -> Array:
        """
        Generate random samples from the joint distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        # Sample from copula
        u = self._copula.sample(nsamples)

        # Apply inverse PIT: x_i = F_i^{-1}(u_i)
        x = self._bkd.zeros((self._nvars, nsamples))
        for i, marginal in enumerate(self._marginals):
            u_row = self._bkd.reshape(u[i], (1, -1))
            x[i] = marginal.invcdf(u_row)[0]

        return x

    def nqoi(self) -> int:
        """Return the number of quantities of interest (always 1 for PDF)."""
        return 1

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
        """
        return self._bkd.exp(self.logpdf(samples))

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the joint probability density function.

        Alias for ``pdf()`` to satisfy ``FunctionProtocol``.

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
                return False
        return True

    def domain(self) -> Array:
        """
        Return the domain of the joint distribution.

        Returns
        -------
        Array
            Domain bounds. Shape: (nvars, 2).
        """
        domain_list = []
        for marginal in self._marginals:
            if hasattr(marginal, "is_bounded") and marginal.is_bounded():
                if hasattr(marginal, "interval"):
                    interval = marginal.interval(1.0)
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
        quad_factory: Optional[object] = None,
        variable_names: Optional[List[str]] = None,
    ) -> object:
        """
        Create a plotter for visualizing the joint PDF.

        For ``nvars <= 2``, returns a ``Plotter1D`` or
        ``Plotter2DRectangularDomain``.  For ``nvars > 2``, returns a
        ``PairPlotter``.

        Either *reducer* or *quad_factory* must be provided for
        ``nvars > 2``.  If *quad_factory* is given, a
        ``FunctionMarginalizer`` is built from it.  If *reducer* is
        given it is used directly (e.g. a ``CrossSectionReducer``).

        Parameters
        ----------
        plot_limits : Optional[Array]
            Plot limits.  For 1D: ``[xmin, xmax]``; for 2D:
            ``[xmin, xmax, ymin, ymax]``; for nvars > 2: shape
            ``(nvars, 2)`` with ``[lb, ub]`` per variable.
            Required if the distribution is unbounded.
        reducer : Optional[DimensionReducerProtocol[Array]]
            Dimension reducer for pair-plot panels.  Takes priority
            over *quad_factory*.
        quad_factory : Optional[QuadratureFactoryProtocol[Array]]
            Quadrature factory for marginalizing the joint PDF.
            Ignored if *reducer* is provided.
        variable_names : Optional[List[str]]
            Axis labels for the pair plot.

        Returns
        -------
        Union[Plotter1D, Plotter2DRectangularDomain, PairPlotter]
            A plotter object.

        Raises
        ------
        ValueError
            If ``nvars > 2`` and neither *reducer* nor *quad_factory*
            is provided, or if the distribution is unbounded and
            *plot_limits* is not given.
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
            FunctionMarginalizer,
        )
        from pyapprox.interface.functions.plot.pair_plot import (
            PairPlotter,
        )

        # nvars > 2
        if plot_limits is None:
            plot_limits = self.domain()
        if reducer is not None:
            return PairPlotter(reducer, plot_limits, self._bkd, variable_names)
        if quad_factory is not None:
            marginalizer = FunctionMarginalizer(self, quad_factory, self._bkd)
            return PairPlotter(marginalizer, plot_limits, self._bkd, variable_names)
        raise ValueError("For nvars > 2, provide either 'reducer' or 'quad_factory'")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CopulaDistribution(nvars={self._nvars}, copula={self._copula!r})"

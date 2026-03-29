"""
Visualization utilities for multivariate Gaussian distributions.

Provides functions for plotting 2D covariance ellipses from Gaussian
distribution objects.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from pyapprox.probability.gaussian.dense import DenseCholeskyMultivariateGaussian


def plot_gaussian_2d_contour(
    gaussian: DenseCholeskyMultivariateGaussian,
    ax: Axes,
    n_std: float = 2.0,
    **kwargs: object,
) -> Ellipse:
    """
    Plot a 2D covariance ellipse for a multivariate Gaussian.

    The ellipse shows the ``n_std``-sigma contour of the Gaussian
    distribution. Requires ``nvars() == 2``.

    Parameters
    ----------
    gaussian
        Any object with ``mean()`` returning shape ``(2, 1)`` and
        ``covariance()`` returning shape ``(2, 2)``. Typical types:
        ``DenseCholeskyMultivariateGaussian``,
        ``DiagonalMultivariateGaussian``.
    ax : matplotlib.axes.Axes
        Axes on which to draw the ellipse.
    n_std : float, optional
        Number of standard deviations for the contour radius. Default 2.0.
    **kwargs
        Passed to ``matplotlib.patches.Ellipse`` (e.g. ``facecolor``,
        ``edgecolor``, ``lw``, ``alpha``, ``ls``, ``label``, ``zorder``).

    Returns
    -------
    matplotlib.patches.Ellipse
        The ellipse patch added to ``ax``.

    Raises
    ------
    ValueError
        If the Gaussian is not 2-dimensional.

    Examples
    --------
    >>> import numpy as np, matplotlib.pyplot as plt
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.gaussian import (
    ...     DenseCholeskyMultivariateGaussian,
    ... )
    >>> from pyapprox.probability.gaussian.plot import (
    ...     plot_gaussian_2d_contour,
    ... )
    >>> bkd = NumpyBkd()
    >>> g = DenseCholeskyMultivariateGaussian(
    ...     np.array([[1.0], [2.0]]),
    ...     np.array([[4.0, 1.0], [1.0, 9.0]]),
    ...     bkd,
    ... )
    >>> fig, ax = plt.subplots()
    >>> ell = plot_gaussian_2d_contour(g, ax, n_std=2,
    ...     facecolor="none", edgecolor="blue", lw=2)
    """
    bkd = gaussian.bkd()
    mean = bkd.to_numpy(gaussian.mean()).ravel()
    cov = bkd.to_numpy(gaussian.covariance())

    if mean.shape[0] != 2 or cov.shape != (2, 2):
        raise ValueError(
            f"plot_gaussian_2d_contour requires a 2D Gaussian, "
            f"got mean shape {mean.shape} and covariance shape {cov.shape}"
        )

    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width = 2 * n_std * np.sqrt(eigvals[1])
    height = 2 * n_std * np.sqrt(eigvals[0])

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell

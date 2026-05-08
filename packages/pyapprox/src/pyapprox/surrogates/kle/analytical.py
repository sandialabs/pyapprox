"""Analytical Karhunen-Loève Expansion for the 1D exponential kernel.

Used for test validation. The exponential kernel K(x,y) = sigma^2 * exp(-|x-y|/l)
has known eigenvalues and eigenfunctions on [0, L].

References
----------
https://doi.org/10.1016/j.jcp.2003.09.015
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import brenth

NDArrayFloat = npt.NDArray[np.floating[Any]]


def _compute_roots_of_characteristic_equation(
    corr_len: float,
    nvars: int,
    dom_len: float,
    maxw: Optional[float] = None,
) -> NDArrayFloat:
    """Compute roots of the characteristic equation for exponential kernel.

    Parameters
    ----------
    corr_len : float
        Correlation length l of the covariance kernel.
    nvars : int
        Number of roots to compute.
    dom_len : float
        Domain length L.
    maxw : float or None
        Maximum frequency range for root search.

    Returns
    -------
    omega : ndarray, shape (nvars,)
        Roots of the characteristic equation.
    """

    def func(w: float) -> np.floating[Any]:
        result: np.floating[Any] = (corr_len**2 * w**2 - 1.0) * np.sin(
            w * dom_len
        ) - 2 * corr_len * w * np.cos(w * dom_len)
        return result

    omega = np.empty(nvars, float)
    dw = 1e-2
    tol = 1e-5
    if maxw is None:
        maxw = nvars * 5
    w = np.linspace(dw, maxw, int(maxw // dw))
    fw = func(w)
    fw_sign = np.sign(fw)
    signchange = ((np.roll(fw_sign, -1) - fw_sign) != 0).astype(int)
    idx = np.where(signchange)[0]
    wI = w[idx]
    if idx.shape[0] < nvars + 1:
        raise RuntimeError(f"Not enough roots found. Increase maxw (currently {maxw}).")

    prev_root = 0
    for ii in range(nvars):
        root = brenth(func, wI[ii], wI[ii + 1], maxiter=1000, xtol=tol)
        assert root > 0 and abs(root - prev_root) > tol * 100
        omega[ii] = root
        prev_root = root
    return omega


def _exponential_kle_eigenvalues(
    sigma2: float, corr_len: float, omega: NDArrayFloat
) -> NDArrayFloat:
    """Compute analytical eigenvalues for exponential kernel KLE.

    Parameters
    ----------
    sigma2 : float
        Variance of the random field.
    corr_len : float
        Correlation length.
    omega : ndarray, shape (nvars,)
        Roots of the characteristic equation.

    Returns
    -------
    ndarray, shape (nvars,)
        Eigenvalues.
    """
    result: NDArrayFloat = 2 * corr_len * sigma2 / (1.0 + (omega * corr_len) ** 2)
    return result


def _exponential_kle_basis(
    x: NDArrayFloat,
    corr_len: float,
    sigma2: float,
    dom_len: float,
    omega: NDArrayFloat,
) -> NDArrayFloat:
    """Compute analytical eigenfunctions for exponential kernel KLE.

    Parameters
    ----------
    x : ndarray, shape (npts,)
        Spatial coordinates on [0, dom_len].
    corr_len : float
        Correlation length.
    sigma2 : float
        Variance.
    dom_len : float
        Domain length.
    omega : ndarray, shape (nvars,)
        Roots of the characteristic equation.

    Returns
    -------
    ndarray, shape (npts, nvars)
        Basis function values at each spatial location.
    """
    nvars = omega.shape[0]
    npts = x.shape[0]
    basis_vals = np.empty((npts, nvars), float)
    for jj in range(nvars):
        bn = 1 / ((corr_len**2 * omega[jj] ** 2 + 1) * dom_len / 2.0 + corr_len)
        bn = np.sqrt(bn)
        an = corr_len * omega[jj] * bn
        basis_vals[:, jj] = an * np.cos(omega[jj] * x) + bn * np.sin(omega[jj] * x)
    return basis_vals


class AnalyticalExponentialKLE1D:
    """Analytical Karhunen-Loève Expansion for 1D exponential kernel.

    Solves the Fredholm integral equation analytically for the kernel
    K(x,y) = sigma^2 * exp(-|x-y|/corr_len) on [0, dom_len].

    This is a numpy-only class used for test validation.

    Parameters
    ----------
    corr_len : float
        Correlation length of the exponential kernel.
    sigma2 : float
        Variance of the random field.
    dom_len : float
        Length of the domain.
    nterms : int
        Number of KLE terms to compute.
    maxw : float or None
        Maximum frequency range for root search.
    """

    def __init__(
        self,
        corr_len: float,
        sigma2: float,
        dom_len: float,
        nterms: int,
        maxw: Optional[float] = None,
    ) -> None:
        self._corr_len = corr_len
        self._sigma2 = sigma2
        self._dom_len = dom_len
        self._nterms = nterms

        self._omega = _compute_roots_of_characteristic_equation(
            corr_len, nterms, dom_len, maxw=maxw
        )
        self._eig_vals = _exponential_kle_eigenvalues(sigma2, corr_len, self._omega)
        self._basis_vals: Optional[NDArrayFloat] = None

    def eigenvalues(self) -> NDArrayFloat:
        """Return analytical eigenvalues, shape (nterms,)."""
        return self._eig_vals

    def basis_values(self, mesh_1d: NDArrayFloat) -> NDArrayFloat:
        """Compute basis function values at mesh points.

        Parameters
        ----------
        mesh_1d : ndarray, shape (npts,)
            1D spatial coordinates on [0, dom_len].

        Returns
        -------
        ndarray, shape (npts, nterms)
            Basis function values.
        """
        self._basis_vals = _exponential_kle_basis(
            mesh_1d, self._corr_len, self._sigma2, self._dom_len, self._omega
        )
        return self._basis_vals

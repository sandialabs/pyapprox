"""Exact PDF computation for univariate Polynomial Chaos Expansions.

Given a 1D PCE Y = g(xi) = sum_k c_k psi_k(xi) where xi has known density,
computes the exact PDF of Y via companion matrix root-finding and the
multi-branch change-of-variables formula.
"""

import math
from typing import Callable, Generic, Tuple

import numpy as np
from numpy.polynomial import polynomial as nppoly
from numpy.polynomial.legendre import leggauss

from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.univariate.globalpoly.monomial_conversion import (
    convert_orthonormal_to_monomials_1d,
)
from pyapprox.surrogates.affine.univariate.transformed import (
    TransformedBasis1D,
)
from pyapprox.util.backends.protocols import Array, Backend


class UnivariatePCEDensity(Generic[Array]):
    """Exact PDF of a scalar QoI defined by a 1D PCE.

    Uses companion matrix root-finding and the change-of-variables formula
    to compute the exact density of Y = g(xi) = sum_k c_k psi_k(xi).

    Parameters
    ----------
    pce : PolynomialChaosExpansion[Array]
        A 1D PCE with nvars=1 and nqoi=1.
    marginal
        The marginal distribution of the input variable xi.
        Must implement ``pdf(samples)`` returning shape ``(1, nsamples)``
        and the samples must be in the physical domain.

    Raises
    ------
    ValueError
        If the PCE is not univariate (nvars != 1) or not scalar-valued
        (nqoi != 1).
    """

    def __init__(
        self,
        pce: PolynomialChaosExpansion[Array],
        marginal,
    ) -> None:
        if pce.nvars() != 1:
            raise ValueError(
                f"UnivariatePCEDensity requires nvars=1, got {pce.nvars()}"
            )
        if pce.nqoi() != 1:
            raise ValueError(f"UnivariatePCEDensity requires nqoi=1, got {pce.nqoi()}")

        self._pce = pce
        self._bkd = pce.bkd()
        self._marginal = marginal

        # Access the 1D basis wrapper (TransformedBasis1D or NativeBasis1D).
        # Note: _bases_1d is private with no public accessor; fragile but
        # necessary since OrthonormalPolynomialBasis does not expose it.
        basis = pce.get_basis()
        basis_1d = basis._bases_1d[0]

        # Get the domain transform (physical <-> canonical)
        if isinstance(basis_1d, TransformedBasis1D):
            self._transform = basis_1d.transform()
            poly = basis_1d.polynomial()
        else:
            # NativeBasis1D: no transform needed
            self._transform = None
            poly = basis_1d.polynomial()

        self._jacobian_factor = (
            self._transform.jacobian_factor() if self._transform is not None else 1.0
        )

        # Determine canonical domain bounds for root filtering.
        # For bounded distributions (e.g., Uniform), roots outside the
        # canonical support must be excluded since f_xi = 0 there.
        if hasattr(marginal, "is_bounded") and marginal.is_bounded():
            # Map physical bounds to canonical domain
            lb_phys = marginal.lower()
            ub_phys = marginal.upper()
            if self._transform is not None:
                lb_can = float(
                    self._bkd.to_numpy(
                        self._transform.map_to_canonical(self._bkd.asarray([[lb_phys]]))
                    ).ravel()[0]
                )
                ub_can = float(
                    self._bkd.to_numpy(
                        self._transform.map_to_canonical(self._bkd.asarray([[ub_phys]]))
                    ).ravel()[0]
                )
                self._can_lb = min(lb_can, ub_can)
                self._can_ub = max(lb_can, ub_can)
            else:
                self._can_lb = lb_phys
                self._can_ub = ub_phys
        else:
            self._can_lb = -np.inf
            self._can_ub = np.inf

        # Get recursion coefficients and build monomial representation.
        # Use only the first pce.nterms() recursion coefficients, since
        # the polynomial may have been expanded (e.g., by quadrature calls)
        # beyond what the PCE actually uses.
        pce_nterms = pce.nterms()
        rcoefs_full = poly.recursion_coefficients()
        rcoefs = rcoefs_full[:pce_nterms, :]
        mono_matrix = convert_orthonormal_to_monomials_1d(rcoefs, self._bkd)

        # Compute g(xi) monomial coefficients: ascending powers
        # coefs shape: (pce_nterms,), mono_matrix shape: (pce_nterms, pce_nterms)
        coefs = pce.get_coefficients()[:, 0]  # (pce_nterms,)
        mono_coefs = self._bkd.dot(coefs, mono_matrix)  # (pce_nterms,)

        # Convert to numpy for root-finding operations
        self._mono_np = self._bkd.to_numpy(mono_coefs).copy()

        # Compute g'(xi) monomial coefficients by explicit differentiation:
        # if mono_coefs = [a_0, a_1, ..., a_P] then
        # deriv_mono = [a_1, 2*a_2, 3*a_3, ..., P*a_P]
        nterms = len(self._mono_np)
        if nterms > 1:
            js = np.arange(1, nterms)
            self._deriv_mono_np = self._mono_np[1:] * js
        else:
            self._deriv_mono_np = np.array([0.0])

        self._nterms = nterms

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def find_real_roots(self, y: float) -> np.ndarray:
        """Find all real roots of g(xi) = y in the canonical domain.

        Parameters
        ----------
        y : float
            The target value.

        Returns
        -------
        np.ndarray
            Verified real roots in canonical domain. Shape: (nroots,).
        """
        # Shift polynomial: g(xi) - y
        p = self._mono_np.copy()
        p[0] -= y

        # Handle constant polynomial (degree 0)
        if len(p) <= 1:
            return np.array([])

        # Strip trailing near-zero coefficients for numerical stability
        # but keep at least 2 coefficients
        while len(p) > 2 and abs(p[-1]) < 1e-30:
            p = p[:-1]

        # Find roots via companion matrix eigenvalues
        roots = nppoly.polyroots(p)

        if len(roots) == 0:
            return np.array([])

        # Filter to real roots
        tol_imag = 1e-10
        real_mask = np.abs(roots.imag) < tol_imag
        real_roots = roots[real_mask].real

        if len(real_roots) == 0:
            return np.array([])

        # Filter to roots within canonical domain support
        in_support = (real_roots >= self._can_lb) & (real_roots <= self._can_ub)
        real_roots = real_roots[in_support]

        if len(real_roots) == 0:
            return np.array([])

        # Verify each root by evaluating in monomial form (canonical domain)
        tol_verify = 1e-8
        g_vals = nppoly.polyval(real_roots, self._mono_np)
        verified_mask = np.abs(g_vals - y) < tol_verify
        return real_roots[verified_mask]

    def find_critical_points(self) -> np.ndarray:
        """Find roots of g'(xi) = 0 (critical points in canonical domain).

        Returns
        -------
        np.ndarray
            Real critical points. Shape: (ncrit,).
        """
        if len(self._deriv_mono_np) <= 1:
            return np.array([])

        p = self._deriv_mono_np.copy()

        # Strip trailing near-zero coefficients
        while len(p) > 2 and abs(p[-1]) < 1e-30:
            p = p[:-1]

        roots = nppoly.polyroots(p)

        if len(roots) == 0:
            return np.array([])

        tol_imag = 1e-10
        real_mask = np.abs(roots.imag) < tol_imag
        real_roots = roots[real_mask].real

        # Verify
        tol_verify = 1e-8
        deriv_vals = nppoly.polyval(real_roots, self._deriv_mono_np)
        verified_mask = np.abs(deriv_vals) < tol_verify
        return real_roots[verified_mask]

    def _canonical_density(self, xi_canonical: np.ndarray) -> np.ndarray:
        """Evaluate the input density in the canonical domain.

        f_can(xi) = f_phys(T^{-1}(xi)) / |jacobian_factor|

        Parameters
        ----------
        xi_canonical : np.ndarray
            Points in canonical domain. Shape: (npoints,).

        Returns
        -------
        np.ndarray
            Density values. Shape: (npoints,).
        """
        # Map to physical domain for marginal evaluation
        xi_2d = self._bkd.reshape(self._bkd.asarray(xi_canonical), (1, -1))
        if self._transform is not None:
            x_phys = self._transform.map_from_canonical(xi_2d)
        else:
            x_phys = xi_2d

        # Evaluate marginal PDF in physical domain: shape (1, npoints)
        pdf_phys = self._marginal.pdf(x_phys)

        # Convert to canonical density
        pdf_can = self._bkd.to_numpy(pdf_phys[0]) / abs(self._jacobian_factor)
        return pdf_can

    def pdf(self, y_values: Array) -> Array:
        """Evaluate the PDF of Y at query points.

        Uses the multi-branch change-of-variables formula:
            f_Y(y) = sum_i f_xi(xi_i) / |g'(xi_i)|
        where xi_i are roots of g(xi) = y.

        Parameters
        ----------
        y_values : Array
            Query points. Shape: (1, npoints) following typing conventions.

        Returns
        -------
        Array
            PDF values. Shape: (1, npoints).
        """
        y_np = self._bkd.to_numpy(y_values)
        if y_np.ndim == 2:
            y_flat = y_np[0]
        else:
            y_flat = y_np

        npoints = len(y_flat)
        pdf_vals = np.zeros(npoints)
        tol_deriv = 1e-14

        for ii in range(npoints):
            y = float(y_flat[ii])
            roots = self.find_real_roots(y)

            if len(roots) == 0:
                continue

            # Evaluate g'(xi) at roots
            g_prime = nppoly.polyval(roots, self._deriv_mono_np)

            # Evaluate canonical density at roots
            f_xi = self._canonical_density(roots)

            # Sum contributions, skipping near-critical roots
            for jj in range(len(roots)):
                abs_gp = abs(g_prime[jj])
                if abs_gp > tol_deriv:
                    pdf_vals[ii] += f_xi[jj] / abs_gp

        return self._bkd.reshape(self._bkd.asarray(pdf_vals), (1, npoints))

    def moments_exact(self, max_order: int) -> Array:
        """Compute exact moments E[Y^m] via Gaussian quadrature in xi-space.

        The quadrature weights from the orthonormal polynomial basis already
        include the probability measure, so:
            E[Y^m] = sum_i w_i * g(xi_i)^m

        Parameters
        ----------
        max_order : int
            Maximum moment order. Computes E[Y^m] for m = 1, ..., max_order.

        Returns
        -------
        Array
            Moment values. Shape: (max_order,).
        """
        basis = self._pce.get_basis()
        nterms = self._pce.nterms()

        moments = []
        for m in range(1, max_order + 1):
            # Need enough quadrature points for exactness
            # Integrand is g(xi)^m which is degree m*P
            npts = max(math.ceil((m * (nterms - 1) + 1) / 2) + 1, nterms)
            quad_pts, quad_wts = basis.univariate_quadrature(0, npts)

            # Evaluate PCE at quadrature points: shape (1, npts)
            g_vals = self._pce(quad_pts)  # (1, npts)

            # Compute weighted sum: E[g^m]
            g_pow = g_vals[0] ** m
            quad_wts_flat = self._bkd.flatten(quad_wts)
            moment_m = self._bkd.sum(g_pow * quad_wts_flat)
            moments.append(moment_m)

        return self._bkd.stack(moments)

    def get_range(self) -> Tuple[float, float]:
        """Estimate the effective range of Y for plotting/evaluation.

        For bounded distributions, evaluates g at quadrature nodes within
        the canonical support. For unbounded distributions, evaluates g
        on the interval corresponding to a high-probability region of the
        input density (using the marginal's interval method).

        Returns
        -------
        Tuple[float, float]
            (y_min, y_max) range estimate.
        """
        npts = 500

        if np.isfinite(self._can_lb) and np.isfinite(self._can_ub):
            # Bounded: evaluate on a uniform grid in canonical domain
            xi_np = np.linspace(self._can_lb, self._can_ub, npts)
        else:
            # Unbounded: use high-probability region
            # Get 99.99% interval from the marginal
            if hasattr(self._marginal, "interval"):
                interval = self._marginal.interval(0.9999)
                interval_np = self._bkd.to_numpy(interval)[0]
                lb_phys, ub_phys = float(interval_np[0]), float(interval_np[1])
            else:
                lb_phys, ub_phys = -6.0, 6.0

            # Map to canonical domain
            if self._transform is not None:
                lb_can = float(
                    self._bkd.to_numpy(
                        self._transform.map_to_canonical(self._bkd.asarray([[lb_phys]]))
                    ).ravel()[0]
                )
                ub_can = float(
                    self._bkd.to_numpy(
                        self._transform.map_to_canonical(self._bkd.asarray([[ub_phys]]))
                    ).ravel()[0]
                )
            else:
                lb_can, ub_can = lb_phys, ub_phys

            xi_np = np.linspace(min(lb_can, ub_can), max(lb_can, ub_can), npts)

        # Evaluate g in monomial form (faster than full PCE evaluation)
        g_np = np.polynomial.polynomial.polyval(xi_np, self._mono_np)
        g_min = float(np.min(g_np))
        g_max = float(np.max(g_np))

        margin = 0.1 * (g_max - g_min)
        if margin < 1e-10:
            margin = 1.0
        return (g_min - margin, g_max + margin)


def composite_gauss_legendre(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_intervals: int = 200,
    n_points: int = 5,
) -> float:
    """Integrate f over [a, b] using composite Gauss-Legendre quadrature.

    Divides [a, b] into n_intervals subintervals and applies n_points-point
    Gauss-Legendre quadrature on each.

    Parameters
    ----------
    f : callable
        Function to integrate. Takes and returns numpy arrays.
    a : float
        Left endpoint.
    b : float
        Right endpoint.
    n_intervals : int
        Number of subintervals. Default: 200.
    n_points : int
        Number of Gauss-Legendre points per subinterval. Default: 5.

    Returns
    -------
    float
        Approximate integral value.
    """
    # Reference Gauss-Legendre nodes/weights on [-1, 1]
    ref_nodes, ref_weights = leggauss(n_points)

    edges = np.linspace(a, b, n_intervals + 1)
    total = 0.0

    for kk in range(n_intervals):
        left = edges[kk]
        right = edges[kk + 1]
        half_width = (right - left) / 2.0
        midpoint = (right + left) / 2.0

        # Map reference nodes to [left, right]
        nodes = midpoint + half_width * ref_nodes
        total += half_width * np.sum(ref_weights * f(nodes))

    return float(total)


__all__ = [
    "UnivariatePCEDensity",
    "composite_gauss_legendre",
]

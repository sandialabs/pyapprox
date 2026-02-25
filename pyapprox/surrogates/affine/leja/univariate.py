"""Univariate Leja sequence generation.

This module provides optimization-based univariate Leja sequence generation.
Leja sequences are nested sequences of points optimal for polynomial
interpolation.

Key classes:
- LejaObjective: One-point objective function for Leja point optimization
- TwoPointLejaObjective: Two-point objective optimizing pairs simultaneously
- LejaSequence1D: Univariate Leja sequence with internal caching
- ScipyTrustConstrMinimizer: Default optimizer factory using scipy trust-constr

The default optimizer uses ScipyTrustConstrOptimizer from
pyapprox.optimization.minimize. Users can provide custom optimizer
factories that accept (objective, bounds) and return an optimizer with
a minimize() method.

Example with custom optimizer settings:

    from functools import partial
    from pyapprox.optimization.minimize.scipy.trust_constr import (
        ScipyTrustConstrOptimizer,
    )
    # Create a factory with custom settings
    optimizer = partial(ScipyTrustConstrOptimizer, gtol=1e-8, maxiter=500)
    leja = LejaSequence1D(bkd, basis, weighting, bounds, optimizer=optimizer)
"""

from typing import Callable, Dict, Generic, Optional, Tuple, Type, Union, Any
import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.protocols import Basis1DProtocol

from .protocols import (
    LejaWeightingProtocol,
    LejaWeightingWithJacobianProtocol,
)


class ScipyTrustConstrMinimizer:
    """Default optimizer factory for Leja sequences.

    Creates ScipyTrustConstrOptimizer instances with the provided settings.
    This is the default optimizer used by LejaSequence1D.

    Parameters
    ----------
    gtol : float, optional
        Gradient tolerance for termination. Default uses scipy's default.
    maxiter : int, optional
        Maximum number of iterations. Default uses scipy's default.
    verbosity : int, optional
        Verbosity level (0=silent). Default is 0.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create optimizer factory with custom settings
    >>> optimizer = ScipyTrustConstrMinimizer(gtol=1e-8, maxiter=500)
    >>> # leja = LejaSequence1D(bkd, basis, weighting, bounds, optimizer=optimizer)
    """
    # todo: delete this class and use scipy wrapper in typing.optimization.minimize as default, but leja should accept anything that meets the minimizer protocol, so we can change optimizers if the user desires
    def __init__(
        self,
        gtol: Optional[float] = None,
        maxiter: Optional[int] = None,
        verbosity: int = 0,
    ):
        self._gtol = gtol
        self._maxiter = maxiter
        self._verbosity = verbosity

    def __call__(self, objective, bounds):
        """Create optimizer instance for given objective and bounds."""
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        return ScipyTrustConstrOptimizer(
            objective=objective,
            bounds=bounds,
            gtol=self._gtol,
            maxiter=self._maxiter,
            verbosity=self._verbosity,
        )

    def __repr__(self) -> str:
        return (
            f"ScipyTrustConstrMinimizer(gtol={self._gtol}, "
            f"maxiter={self._maxiter})"
        )


class LejaObjective(Generic[Array]):
    """Objective function for Leja point optimization.

    The objective function finds the point that maximizes the weighted
    residual of the polynomial approximation. This is achieved by finding
    minima of the negative weighted squared residual.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : Basis1DProtocol[Array]
        Univariate polynomial basis.
    weighting : LejaWeightingProtocol[Array]
        Weighting strategy.
    bounds : Tuple[float, float]
        Domain bounds (lower, upper).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Assuming basis and weighting are defined
    >>> # objective = LejaObjective(bkd, basis, weighting, (-1, 1))
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: Basis1DProtocol[Array],
        weighting: LejaWeightingProtocol[Array],
        bounds: Tuple[float, float],
    ):
        self._bkd = bkd
        self._basis = basis
        self._weighting = weighting
        self._bounds = bounds
        self._sequence: Optional[Array] = None
        self._coefficients: Optional[Array] = None
        self._basis_mat: Optional[Array] = None
        self._basis_vec: Optional[Array] = None
        self._weights: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of optimization variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest (required by FunctionProtocol)."""
        return 1

    def nsamples(self) -> int:
        """Return current number of points in sequence."""
        if self._sequence is None:
            return 0
        return self._sequence.shape[1]

    def sequence(self) -> Array:
        """Return copy of current sequence."""
        return self._bkd.copy(self._sequence)

    def get_bounds(self) -> Tuple[float, float]:
        """Return domain bounds."""
        return self._bounds

    def set_sequence(self, sequence: Array) -> None:
        """Set the current Leja sequence.

        Parameters
        ----------
        sequence : Array
            Current sequence points. Shape: (1, npoints)
        """
        if sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError("sequence must be a 2D row vector with shape (1, n)")
        self._sequence = sequence
        self._update_cached_data()

    def _update_cached_data(self) -> None:
        """Update cached basis matrix and coefficients."""
        nterms = self.nsamples() + 1
        self._basis.set_nterms(nterms)
        basis_vals = self._basis(self._sequence)
        self._basis_mat = basis_vals[:, :-1]
        self._basis_vec = basis_vals[:, -1:]

        # Compute weights at sequence points
        self._weights = self._weighting(self._sequence, self._basis_mat)

        # Compute interpolation coefficients using weighted least squares
        sqrt_weights = self._bkd.sqrt(self._weights)
        self._coefficients = self._bkd.lstsq(
            sqrt_weights * self._basis_mat,
            sqrt_weights * self._basis_vec,
        )

    def __call__(self, samples: Array) -> Array:
        """Evaluate objective at sample points.

        The objective is the negative weighted squared residual:
        -w(x) * (p_{n+1}(x) - sum_i c_i p_i(x))^2

        Parameters
        ----------
        samples : Array
            Points to evaluate. Shape: (1, nsamples)

        Returns
        -------
        Array
            Objective values. Shape: (1, nsamples)
        """
        basis_vals = self._basis(samples)
        basis_mat = basis_vals[:, :-1]
        new_basis = basis_vals[:, -1:]
        weights = self._weighting(samples, basis_mat)

        pvals = basis_mat @ self._coefficients
        residual = new_basis - pvals
        vals = -weights * self._bkd.sum(residual ** 2, axis=1)[:, None]
        return vals.T

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of objective.

        Parameters
        ----------
        sample : Array
            Point to evaluate. Shape: (1, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, 1)
        """
        # Get basis values and derivatives separately
        # Note: derivatives(order=1) returns only derivatives, not values
        basis_vals = self._basis(sample)
        basis_jac = self._basis.derivatives(sample, order=1)

        bvals = basis_vals[:, -1:]
        pvals = basis_vals[:, :-1] @ self._coefficients
        bderivs = basis_jac[:, -1:]
        pderivs = basis_jac[:, :-1] @ self._coefficients

        residual = bvals - pvals
        residual_jac = bderivs - pderivs

        # Get weights and weight jacobians
        weight = self._weighting(sample, basis_vals[:, :-1])

        # Compute weight jacobian if available
        if hasattr(self._weighting, "jacobian"):
            weight_jac = self._weighting.jacobian(
                sample, basis_vals[:, :-1], basis_jac[:, :-1]
            )
        else:
            weight_jac = self._bkd.zeros((1, 1))

        # Overflow protection
        if float(self._bkd.max(self._bkd.abs(residual))) > np.sqrt(
            np.finfo(float).max / 100
        ):
            return self._bkd.full((1, 1), np.inf)

        # d/dx (-w * r^2) = -w' * r^2 - 2 * w * r * r'
        jac = self._bkd.sum(
            residual ** 2 * weight_jac + 2 * weight * residual * residual_jac,
            axis=1,
        )
        return -jac[None, :]

    def initial_iterates_and_bounds(self) -> Tuple[Array, list]:
        """Generate initial guesses and their bounds for optimization.

        Returns
        -------
        Tuple[Array, list]
            (iterates, bounds_list) where iterates has shape (1, n_iterates)
            and bounds_list contains per-iterate bounds arrays.
        """
        eps = 1e-6
        lb, ub = self._bounds

        # Sort sequence points (convert to numpy for sorting)
        seq_np = self._bkd.to_numpy(self._sequence)
        sorted_seq = np.sort(seq_np.flatten())

        # Build interval list
        intervals = sorted_seq.tolist()

        # Add bounds if sequence doesn't reach them
        if np.isfinite(lb) and float(self._bkd.min(self._sequence)) > lb + eps:
            intervals = [lb] + intervals
        if np.isfinite(ub) and float(self._bkd.max(self._sequence)) < ub - eps:
            intervals = intervals + [ub]

        # Handle infinite bounds
        if not np.isfinite(lb):
            min_val = float(self._bkd.min(self._sequence))
            intervals = [min(1.1 * min_val, min_val - 1.0)] + intervals
        if not np.isfinite(ub):
            max_val = float(self._bkd.max(self._sequence))
            intervals = intervals + [max(1.1 * max_val, max_val + 1.0)]

        # Generate iterates at midpoints
        iterates = []
        bounds_list = []
        for i in range(len(intervals) - 1):
            mid = (intervals[i] + intervals[i + 1]) / 2
            iterates.append(mid)
            bound = self._bkd.asarray([[intervals[i], intervals[i + 1]]])
            bounds_list.append(bound)

        iterates_arr = self._bkd.asarray([iterates])
        return iterates_arr, bounds_list

    def __repr__(self) -> str:
        return (
            f"LejaObjective(nsamples={self.nsamples()}, "
            f"bounds={self._bounds})"
        )


class TwoPointLejaObjective(LejaObjective[Array]):
    """Two-point objective function for Leja sequence optimization.

    Optimizes pairs of points simultaneously by maximizing the squared
    determinant of the 2x2 weighted residual matrix. This produces
    better-conditioned sequences than adding one point at a time.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : Basis1DProtocol[Array]
        Univariate polynomial basis.
    weighting : LejaWeightingProtocol[Array]
        Weighting strategy.
    bounds : Tuple[float, float]
        Domain bounds (lower, upper).
    """

    def nvars(self) -> int:
        """Return number of optimization variables (2 for two-point)."""
        return 2

    def _update_cached_data(self) -> None:
        """Update cached basis matrix and coefficients for two new terms."""
        nterms = self.nsamples() + 2
        self._basis.set_nterms(nterms)
        basis_vals = self._basis(self._sequence)
        self._basis_mat = basis_vals[:, :-2]
        self._basis_vec = basis_vals[:, -2:]

        # Compute weights at sequence points
        self._weights = self._weighting(self._sequence, self._basis_mat)

        # Compute interpolation coefficients using weighted least squares
        sqrt_weights = self._bkd.sqrt(self._weights)
        self._coefficients = self._bkd.lstsq(
            sqrt_weights * self._basis_mat,
            sqrt_weights * self._basis_vec,
        )

    def __call__(self, samples: Array) -> Array:
        """Evaluate two-point objective at sample pairs.

        The objective is the negative squared determinant of the 2x2
        weighted residual matrix.

        Parameters
        ----------
        samples : Array
            Pairs of points. Shape: (2, nsamples).
            Row 0 = first point of each pair, row 1 = second point.

        Returns
        -------
        Array
            Objective values. Shape: (1, nsamples)
        """
        if samples.shape[0] != 2:
            raise ValueError("samples must have shape (2, nsamples)")
        nsamples = samples.shape[1]

        # Flatten to (1, 2*nsamples) for basis evaluation
        flat_samples = self._bkd.reshape(samples, (1, nsamples * 2))
        basis_vals = self._basis(flat_samples)
        basis_mat = basis_vals[:, :-2]
        new_basis = basis_vals[:, -2:]

        sqrt_weights = self._bkd.sqrt(
            self._weighting(flat_samples, basis_mat)
        )
        pvals = basis_mat @ self._coefficients
        residuals = sqrt_weights * (new_basis - pvals)

        # Compute 2x2 determinant for each pair
        det = (
            residuals[:nsamples, 0] * residuals[nsamples:, 1]
            - residuals[:nsamples, 1] * residuals[nsamples:, 0]
        )
        return -(det[None, :] ** 2)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of two-point objective.

        Parameters
        ----------
        sample : Array
            Single pair of points. Shape: (2, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, 2) — (nqoi, nvars)
        """
        # Flatten to (1, 2)
        flat_sample = self._bkd.reshape(sample, (1, 2))

        # Get basis values and derivatives
        basis_vals = self._basis(flat_sample)
        basis_jac = self._basis.derivatives(flat_sample, order=1)

        bvals = basis_vals[:, -2:]
        pvals = basis_vals[:, :-2] @ self._coefficients
        bderivs = basis_jac[:, -2:]
        pderivs = basis_jac[:, :-2] @ self._coefficients

        sqrt_weights = self._bkd.sqrt(
            self._weighting(flat_sample, basis_vals[:, :-2])
        )

        # Compute weight jacobian for sqrt_weights
        if hasattr(self._weighting, "jacobian"):
            weights_jac = self._weighting.jacobian(
                flat_sample, basis_vals[:, :-2], basis_jac[:, :-2]
            )
        else:
            weights_jac = self._bkd.zeros((2, 1))

        sqrt_weights_jac = weights_jac / (2 * sqrt_weights[:, 0:1])

        # Weighted residuals and their jacobians
        residuals = sqrt_weights * (bvals - pvals)
        residuals_jac = sqrt_weights * (bderivs - pderivs) + (
            sqrt_weights_jac * (bvals - pvals)
        )

        # Determinant and its jacobian
        # det = r[0,0]*r[1,1] - r[0,1]*r[1,0]
        determinant = (
            residuals[:1, 0] * residuals[1:, 1]
            - residuals[:1, 1] * residuals[1:, 0]
        )
        determinant_jac = self._bkd.stack(
            (
                residuals_jac[:1, 0] * residuals[1:2, 1]
                - residuals_jac[:1, 1] * residuals[1:2, 0],
                residuals[:1, 0] * residuals_jac[1:2, 1]
                - residuals[:1, 1] * residuals_jac[1:2, 0],
            ),
            axis=1,
        )
        # d/dx (-(det)^2) = -2 * det * d(det)/dx
        # Shape: (1, 2) = (nqoi, nvars)
        jac = -2 * determinant * determinant_jac
        return jac

    def initial_iterates_and_bounds(self) -> Tuple[Array, list]:
        """Generate initial guesses as pairs of 1D interval midpoints.

        Returns
        -------
        Tuple[Array, list]
            (iterates, bounds_list) where iterates has shape (2, n_pairs)
            and bounds_list contains per-pair bounds arrays of shape (2, 2).
        """
        iterates_1d, bounds_1d = super().initial_iterates_and_bounds()
        iterates = []
        bounds = []
        for ii in range(iterates_1d.shape[1]):
            for jj in range(ii + 1, iterates_1d.shape[1]):
                iterates.append(
                    self._bkd.vstack(
                        (iterates_1d[:, ii:ii+1], iterates_1d[:, jj:jj+1])
                    )
                )
                bounds.append(
                    self._bkd.vstack((bounds_1d[ii], bounds_1d[jj]))
                )
        return self._bkd.hstack(iterates), bounds

    def __repr__(self) -> str:
        return (
            f"TwoPointLejaObjective(nsamples={self.nsamples()}, "
            f"bounds={self._bounds})"
        )


class LejaSequence1D(Generic[Array]):
    """Univariate Leja sequence with internal caching.

    Leja sequences are nested sequences of points optimal for polynomial
    interpolation. Points are added by solving an optimization problem to
    find the best next point(s). Supports both one-point and two-point
    objectives via the ``objective_class`` parameter.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : Basis1DProtocol[Array]
        Univariate polynomial basis.
    weighting : LejaWeightingProtocol[Array]
        Weighting strategy (e.g., ChristoffelWeighting, PDFWeighting).
    bounds : Tuple[float, float]
        Domain bounds (lower, upper).
    initial_points : Array, optional
        Initial sequence points. Shape: (1, n). If None, uses domain midpoint.
    optimizer : callable, optional
        Optimizer factory that accepts (objective, bounds) and returns an
        optimizer with a minimize() method. Defaults to ScipyTrustConstrMinimizer().
    objective_class : Type[LejaObjective], optional
        Objective class to use. Defaults to LejaObjective (one-point).
        Use TwoPointLejaObjective for two-point optimization.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate import (
    ...     JacobiPolynomial1D,
    ... )
    >>> from pyapprox.surrogates.affine.leja.weighting import (
    ...     ChristoffelWeighting,
    ... )
    >>> bkd = NumpyBkd()
    >>> basis = JacobiPolynomial1D(bkd, alpha=0.0, beta=0.0)
    >>> weighting = ChristoffelWeighting(bkd)
    >>> leja = LejaSequence1D(bkd, basis, weighting, bounds=(-1.0, 1.0))
    >>> samples, weights = leja.quadrature_rule(10)

    Custom optimizer settings:

    >>> from pyapprox.surrogates.affine.leja import ScipyTrustConstrMinimizer
    >>> optimizer = ScipyTrustConstrMinimizer(gtol=1e-8, maxiter=500)
    >>> leja = LejaSequence1D(bkd, basis, weighting, bounds, optimizer=optimizer)

    Two-point Leja sequence:

    >>> from pyapprox.surrogates.affine.leja.univariate import (
    ...     TwoPointLejaObjective,
    ... )
    >>> leja2 = LejaSequence1D(
    ...     bkd, basis, weighting, bounds,
    ...     objective_class=TwoPointLejaObjective,
    ... )
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: Basis1DProtocol[Array],
        weighting: LejaWeightingProtocol[Array],
        bounds: Tuple[float, float],
        initial_points: Optional[Array] = None,
        optimizer: Optional[Callable] = None,
        objective_class: Optional[Type[LejaObjective]] = None,
    ):
        self._bkd = bkd
        self._basis = basis
        self._weighting = weighting
        self._bounds = bounds

        # Set up optimizer factory (default to ScipyTrustConstrMinimizer)
        if optimizer is None:
            optimizer = ScipyTrustConstrMinimizer()
        self._optimizer_factory = optimizer

        # Initialize objective
        if objective_class is None:
            objective_class = LejaObjective
        self._objective: LejaObjective[Array] = objective_class(
            bkd, basis, weighting, bounds
        )

        # Cache for quadrature weights: {npoints: weights_array}
        self._cached_weights: Dict[int, Array] = {}

        # Set initial sequence
        if initial_points is None:
            # Use domain midpoint as initial point
            mid = (bounds[0] + bounds[1]) / 2
            initial_points = bkd.asarray([[mid]])

        self._objective.set_sequence(initial_points)

        # Compute and cache initial weights
        self._cache_weights_for_current_sequence()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def npoints(self) -> int:
        """Return current number of points in sequence."""
        return self._objective.nsamples()

    def _step(self) -> None:
        """Add new point(s) to the sequence.

        For one-point objectives, adds 1 point. For two-point objectives,
        adds 2 points per step.
        """
        iterates, bounds_list = self._objective.initial_iterates_and_bounds()

        # Try optimization from each initial guess
        results = []
        fun_vals = []
        for j in range(iterates.shape[1]):
            # Create optimizer instance with current objective and bounds
            optimizer = self._optimizer_factory(self._objective, bounds_list[j])
            result = optimizer.minimize(iterates[:, j : j + 1])
            results.append(result)
            fun_vals.append(result.fun())

        # Select best result
        fun_arr = self._bkd.asarray(fun_vals)
        best_idx = int(self._bkd.argmin(fun_arr))
        best_point = results[best_idx].optima()

        # Reshape to (1, nopt_vars) for hstack with sequence
        nopt_vars = self._objective.nvars()
        chosen = self._bkd.reshape(best_point, (1, nopt_vars))
        new_sequence = self._bkd.hstack([self._objective.sequence(), chosen])
        self._objective.set_sequence(new_sequence)

        # Cache weights for the new sequence length
        self._cache_weights_for_current_sequence()

    def _cache_weights_for_current_sequence(self) -> None:
        """Compute and cache quadrature weights for current sequence length."""
        npoints = self.npoints()
        sequence = self._objective.sequence()
        weights = self._compute_quadrature_weights(sequence)
        self._cached_weights[npoints] = weights

    def extend(self, n_new_points: int) -> None:
        """Extend sequence by adding new points.

        Parameters
        ----------
        n_new_points : int
            Number of new points to add. Must be divisible by the
            objective's nvars() (1 for one-point, 2 for two-point).

        Raises
        ------
        ValueError
            If n_new_points is not divisible by objective.nvars().
        """
        nopt_vars = self._objective.nvars()
        if n_new_points % nopt_vars != 0:
            raise ValueError(
                f"n_new_points ({n_new_points}) must be divisible by "
                f"objective nvars ({nopt_vars})"
            )
        n_steps = n_new_points // nopt_vars
        for _ in range(n_steps):
            self._step()

    def quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Get Leja sequence with specified number of points.

        Extends sequence if needed and returns both sample locations
        and quadrature weights. Weights are cached for each sequence length
        to avoid recomputation.

        Parameters
        ----------
        npoints : int
            Number of points in sequence.

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (1, npoints) and (npoints, 1)
        """
        if npoints <= 0:
            raise ValueError("npoints must be positive")

        # Extend if needed (this also caches weights for each new length)
        current = self.npoints()
        if npoints > current:
            self.extend(npoints - current)

        # Get samples
        samples = self._objective.sequence()[:, :npoints]

        # Return cached weights
        weights = self._cached_weights[npoints]

        return samples, weights

    def _compute_quadrature_weights(self, sequence: Array) -> Array:
        """Compute quadrature weights for the sequence.

        Parameters
        ----------
        sequence : Array
            Sample locations. Shape: (1, npoints)

        Returns
        -------
        Array
            Quadrature weights. Shape: (npoints, 1)
        """
        npoints = sequence.shape[1]

        # Save current nterms (shared basis with objective)
        saved_nterms = self._basis.nterms()

        self._basis.set_nterms(npoints)
        basis_mat = self._basis(sequence)

        # Get weights for preconditioning
        sqrt_weights = self._bkd.sqrt(self._weighting(sequence, basis_mat))

        # Solve for quadrature weights
        # weights @ basis_mat = [1, 0, 0, ...] (integrate first basis = 1)
        precond_mat = sqrt_weights * basis_mat
        basis_mat_inv = self._bkd.inv(precond_mat)

        # First row gives quadrature weights (adjusted for preconditioning)
        quad_weights = (basis_mat_inv[0, :] * sqrt_weights[:, 0])[:, None]

        # Restore nterms for objective
        self._basis.set_nterms(saved_nterms)

        return quad_weights

    def clear_cache(self) -> None:
        """Clear cached sequence and weights (reset to initial point)."""
        mid = (self._bounds[0] + self._bounds[1]) / 2
        initial = self._bkd.asarray([[mid]])
        self._objective.set_sequence(initial)
        self._cached_weights.clear()
        self._cache_weights_for_current_sequence()

    def __repr__(self) -> str:
        return (
            f"LejaSequence1D(npoints={self.npoints()}, "
            f"bounds={self._bounds})"
        )

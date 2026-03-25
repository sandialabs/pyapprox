"""Multi-start acquisition function optimization."""

from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pyapprox.optimization.bayesian.protocols import (
    AcquisitionContext,
    AcquisitionFunctionProtocol,
    BODomainProtocol,
    SurrogateProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class CandidateGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for generating initial candidates for multi-start optimization.

    Implementations generate candidate points within a bounded domain for
    use as starting points in local optimization.
    """

    def generate(
        self,
        n_candidates: int,
        bounds: Array,
        bkd: Backend[Array],
    ) -> Array:
        """Generate candidate points within bounds.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to generate.
        bounds : Array
            Variable bounds, shape (nvars, 2). Each row is [lower, upper].
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        Array
            Candidate points, shape (nvars, n_candidates).
        """
        ...


class UniformRandomCandidateGenerator(Generic[Array]):
    """Generate candidates via uniform random sampling.

    Uses numpy's uniform random generator. Simple but can leave
    gaps in high dimensions.
    """

    def generate(
        self,
        n_candidates: int,
        bounds: Array,
        bkd: Backend[Array],
    ) -> Array:
        """Generate uniform random candidates within bounds."""
        np_bounds = bkd.to_numpy(bounds)
        lb = np_bounds[:, 0]
        ub = np_bounds[:, 1]
        nvars = len(lb)
        raw_np = np.random.uniform(
            lb[:, None], ub[:, None], size=(nvars, n_candidates)
        )
        return bkd.array(raw_np)


class SobolCandidateGenerator(Generic[Array]):
    """Generate candidates via Sobol quasi-random sequences.

    Provides better space-filling coverage than uniform random sampling,
    especially in higher dimensions. Uses Owen scrambling for improved
    uniformity.

    Parameters
    ----------
    seed : Optional[int]
        Random seed for scrambling reproducibility. Default None.
    scramble : bool
        If True, use Owen scrambling. Default True.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        scramble: bool = True,
    ) -> None:
        self._seed = seed
        self._scramble = scramble

    def generate(
        self,
        n_candidates: int,
        bounds: Array,
        bkd: Backend[Array],
    ) -> Array:
        """Generate Sobol sequence candidates scaled to bounds."""
        from scipy.stats import qmc

        np_bounds = bkd.to_numpy(bounds)
        nvars = np_bounds.shape[0]
        engine = qmc.Sobol(d=nvars, scramble=self._scramble, seed=self._seed)
        # Round up to next power of 2 for Sobol balance properties
        n_pow2 = 1
        while n_pow2 < n_candidates:
            n_pow2 *= 2
        unit_samples = engine.random(n=n_pow2)
        # Trim back to requested count
        unit_samples = unit_samples[:n_candidates]
        # Scale to bounds
        lb = np_bounds[:, 0]
        ub = np_bounds[:, 1]
        scaled = qmc.scale(unit_samples, lb, ub)
        # Transpose to (nvars, n_candidates)
        return bkd.array(scaled.T)


class _AcquisitionObjective(Generic[Array]):
    """Wraps acquisition function as ObjectiveProtocol for scipy optimizer.

    Negates the acquisition value since the optimizer minimizes,
    but acquisition functions should be maximized.

    If the acquisition function has a ``jacobian`` method, this wrapper
    exposes a ``jacobian`` method too (negated), enabling analytical
    gradients in the scipy optimizer.
    """

    def __init__(
        self,
        acquisition: AcquisitionFunctionProtocol[Array],
        ctx: AcquisitionContext[Array],
        domain: BODomainProtocol[Array],
    ) -> None:
        self._acquisition = acquisition
        self._ctx = ctx
        self._domain = domain

        # Dynamically bind jacobian if acquisition supports it
        if hasattr(acquisition, "jacobian"):
            self.jacobian = self._jacobian

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._ctx.bkd

    def nvars(self) -> int:
        """Return number of variables."""
        return self._domain.nvars()

    def nqoi(self) -> int:
        """Return 1 (scalar objective)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate negated acquisition at samples.

        Parameters
        ----------
        samples : Array
            Input points, shape (nvars, n).

        Returns
        -------
        Array
            Negated acquisition values, shape (1, n).
        """
        acq_vals = self._acquisition.evaluate(samples, self._ctx)  # (n,)
        bkd = self._ctx.bkd
        return -bkd.reshape(acq_vals, (1, acq_vals.shape[0]))

    def _jacobian(self, sample: Array) -> Array:
        """Compute negated Jacobian of acquisition at a single point.

        Parameters
        ----------
        sample : Array
            Single input, shape (nvars, 1).

        Returns
        -------
        Array
            Negated Jacobian, shape (1, nvars).
        """
        acq_jac = self._acquisition.jacobian(sample, self._ctx)  # (1, nvars)
        return -acq_jac


class _SurrogateMeanObjective(Generic[Array]):
    """Wraps surrogate predict() as ObjectiveProtocol for polishing.

    When ``minimize`` is True, the objective returns the surrogate mean
    directly (the scipy optimizer minimizes it). When ``minimize`` is
    False, it negates the mean so the optimizer effectively maximizes.

    If the surrogate has a ``jacobian`` method, this wrapper exposes
    a ``jacobian`` method too, enabling analytical gradients.
    """

    def __init__(
        self,
        surrogate: SurrogateProtocol[Array],
        domain: BODomainProtocol[Array],
        bkd: Backend[Array],
        minimize: bool,
    ) -> None:
        self._surrogate = surrogate
        self._domain = domain
        self._bkd = bkd
        self._sign = 1.0 if minimize else -1.0

        if hasattr(surrogate, "jacobian"):
            self.jacobian = self._jacobian

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of variables."""
        return self._domain.nvars()

    def nqoi(self) -> int:
        """Return 1 (scalar objective)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate surrogate mean at samples.

        Parameters
        ----------
        samples : Array
            Input points, shape (nvars, n).

        Returns
        -------
        Array
            Surrogate mean values, shape (1, n).
        """
        pred = self._surrogate.predict(samples)
        return self._sign * pred[0:1, :]

    def _jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of surrogate mean at a single point.

        Parameters
        ----------
        sample : Array
            Single input, shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian, shape (1, nvars).
        """
        jac = self._surrogate.jacobian(sample)  # (nqoi, nvars)
        return self._sign * jac[0:1, :]


class AcquisitionOptimizer(Generic[Array]):
    """Multi-start optimizer for acquisition function maximization.

    Generates candidates via a pluggable candidate generator, selects
    top-scoring ones as starting points, then runs local optimization
    from each.

    Parameters
    ----------
    optimizer : BindableOptimizerProtocol[Array]
        Local optimizer template (will be copied per restart).
    bkd : Backend[Array]
        Computational backend.
    n_restarts : int
        Number of local optimization restarts. Default 20.
    n_raw_candidates : int
        Number of candidates for initial screening. Default 512.
    candidate_generator : Optional[CandidateGeneratorProtocol[Array]]
        Generator for initial candidate points. Default
        SobolCandidateGenerator (quasi-random, better space coverage
        than uniform random).
    """

    def __init__(
        self,
        optimizer: object,
        bkd: Backend[Array],
        n_restarts: int = 20,
        n_raw_candidates: int = 512,
        candidate_generator: Optional[CandidateGeneratorProtocol[Array]] = None,
    ) -> None:
        self._optimizer = optimizer
        self._bkd = bkd
        self._n_restarts = n_restarts
        self._n_raw_candidates = n_raw_candidates
        self._candidate_generator: CandidateGeneratorProtocol[Array] = (
            candidate_generator
            if candidate_generator is not None
            else SobolCandidateGenerator()
        )

    def generate_candidates(
        self,
        n_candidates: int,
        bounds: Array,
        bkd: Backend[Array],
    ) -> Array:
        """Generate candidate points using the configured generator.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to generate.
        bounds : Array
            Variable bounds, shape (nvars, 2).
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        Array
            Candidate points, shape (nvars, n_candidates).
        """
        return self._candidate_generator.generate(n_candidates, bounds, bkd)

    def maximize(
        self,
        acquisition: AcquisitionFunctionProtocol[Array],
        ctx: AcquisitionContext[Array],
        domain: BODomainProtocol[Array],
    ) -> Array:
        """Find the point that maximizes the acquisition function.

        Parameters
        ----------
        acquisition : AcquisitionFunctionProtocol[Array]
            Acquisition function to maximize.
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.
        domain : BODomainProtocol[Array]
            Search domain.

        Returns
        -------
        Array
            Best point found, shape (nvars, 1).
        """
        best_x, _ = self.maximize_with_value(acquisition, ctx, domain)
        return best_x

    def maximize_with_value(
        self,
        acquisition: AcquisitionFunctionProtocol[Array],
        ctx: AcquisitionContext[Array],
        domain: BODomainProtocol[Array],
    ) -> Tuple[Array, float]:
        """Find the point that maximizes the acquisition function.

        Parameters
        ----------
        acquisition : AcquisitionFunctionProtocol[Array]
            Acquisition function to maximize.
        ctx : AcquisitionContext[Array]
            Context with fitted surrogate and best value.
        domain : BODomainProtocol[Array]
            Search domain.

        Returns
        -------
        Tuple[Array, float]
            (best_x, best_val) where best_x has shape (nvars, 1) and
            best_val is the acquisition value at best_x.
        """
        bkd = self._bkd
        nvars = domain.nvars()
        bounds = domain.bounds()

        # Generate candidates for initial screening
        raw_X = self._candidate_generator.generate(
            self._n_raw_candidates, bounds, bkd
        )

        # Evaluate acquisition at all candidates
        acq_vals = acquisition.evaluate(raw_X, ctx)  # (n_raw_candidates,)
        acq_vals_np = bkd.to_numpy(acq_vals)

        # Select top n_restarts candidates
        n_restarts = min(self._n_restarts, self._n_raw_candidates)
        top_indices = np.argsort(acq_vals_np)[::-1][:n_restarts]

        best_x: Optional[Array] = None
        best_val = float("-inf")

        for idx in top_indices:
            init_guess = bkd.reshape(raw_X[:, int(idx)], (nvars, 1))

            # Create objective that negates acquisition (optimizer minimizes)
            obj = _AcquisitionObjective(acquisition, ctx, domain)

            # Copy optimizer and bind
            opt = self._optimizer.copy()
            opt.bind(obj, bounds)

            try:
                result = opt.minimize(init_guess)
                opt_x = result.optima()  # (nvars, 1)
                # Evaluate acquisition at optimum
                opt_acq = acquisition.evaluate(opt_x, ctx)
                opt_val = bkd.to_float(opt_acq[0])

                if opt_val > best_val:
                    best_val = opt_val
                    best_x = opt_x
            except Exception:
                # Skip failed restarts
                continue

        if best_x is None:
            # Fall back to best raw candidate
            best_raw_idx = int(top_indices[0])
            best_x = bkd.reshape(raw_X[:, best_raw_idx], (nvars, 1))
            best_val = float(acq_vals_np[best_raw_idx])

        return best_x, best_val

    def optimize_surrogate(
        self,
        surrogate: SurrogateProtocol[Array],
        domain: BODomainProtocol[Array],
        bkd: Backend[Array],
        minimize: bool,
    ) -> Array:
        """Optimize the surrogate mean to find the polished best point.

        Uses the same multi-start pattern as acquisition maximization
        but optimizes the surrogate mean directly.

        Parameters
        ----------
        surrogate : SurrogateProtocol[Array]
            Fitted surrogate model.
        domain : BODomainProtocol[Array]
            Search domain.
        bkd : Backend[Array]
            Computational backend.
        minimize : bool
            If True, minimize the surrogate mean.

        Returns
        -------
        Array
            Best point found, shape (nvars, 1).
        """
        nvars = domain.nvars()
        bounds = domain.bounds()

        # Generate candidates for initial screening
        raw_X = self._candidate_generator.generate(
            self._n_raw_candidates, bounds, bkd
        )

        # Evaluate surrogate mean at all candidates
        pred = surrogate.predict(raw_X)  # (nqoi, n_raw_candidates)
        pred_np = bkd.to_numpy(pred[0])  # first QoI

        # Select top n_restarts candidates (best = lowest for min)
        n_restarts = min(self._n_restarts, self._n_raw_candidates)
        if minimize:
            top_indices = np.argsort(pred_np)[:n_restarts]
        else:
            top_indices = np.argsort(pred_np)[::-1][:n_restarts]

        obj = _SurrogateMeanObjective(surrogate, domain, bkd, minimize)

        best_x: Optional[Array] = None
        best_val = float("inf") if minimize else float("-inf")

        for idx in top_indices:
            init_guess = bkd.reshape(raw_X[:, int(idx)], (nvars, 1))

            opt = self._optimizer.copy()
            opt.bind(obj, bounds)

            try:
                result = opt.minimize(init_guess)
                opt_x = result.optima()  # (nvars, 1)
                opt_pred = surrogate.predict(opt_x)
                opt_val = bkd.to_float(opt_pred[0][0])

                improved = (opt_val < best_val) if minimize else (opt_val > best_val)
                if improved:
                    best_val = opt_val
                    best_x = opt_x
            except Exception:
                continue

        if best_x is None:
            best_raw_idx = int(top_indices[0])
            best_x = bkd.reshape(raw_X[:, best_raw_idx], (nvars, 1))

        return best_x

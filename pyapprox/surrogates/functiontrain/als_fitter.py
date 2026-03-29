"""Alternating Least Squares fitter for FunctionTrain.

ALS exploits the tensor train structure to solve local least squares problems
for each core while holding other cores fixed. This is much more efficient
than full gradient descent for FunctionTrain surrogates.
"""

from typing import Generic, List

from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.util.backends.protocols import Array, Backend


class ALSFitterResult(Generic[Array]):
    """Result of ALS fitting.

    Parameters
    ----------
    surrogate : FunctionTrain
        The fitted FunctionTrain.
    n_sweeps : int
        Number of forward/backward sweeps performed.
    residual_history : List[float]
        History of residual norms at each sweep.
    converged : bool
        Whether the algorithm converged to tolerance.
    """

    def __init__(
        self,
        surrogate: FunctionTrain[Array],
        n_sweeps: int,
        residual_history: List[float],
        converged: bool,
    ):
        self._surrogate = surrogate
        self._n_sweeps = n_sweeps
        self._residual_history = residual_history
        self._converged = converged

    def surrogate(self) -> FunctionTrain[Array]:
        """Return the fitted FunctionTrain."""
        return self._surrogate

    def params(self) -> Array:
        """Return the fitted parameters."""
        return self._surrogate._flatten_params()

    def n_sweeps(self) -> int:
        """Return the number of sweeps performed."""
        return self._n_sweeps

    def residual_history(self) -> List[float]:
        """Return the residual history."""
        return self._residual_history

    def converged(self) -> bool:
        """Return whether the algorithm converged."""
        return self._converged


class ALSFitter(Generic[Array]):
    """Alternating Least Squares fitter for FunctionTrain.

    ALS solves for one core at a time while holding all other cores fixed.
    For each core, this reduces to a standard linear least squares problem:

        min_c ||J_k @ c - y||^2

    where J_k is the Jacobian of the output with respect to core k's
    parameters, computed analytically using the tensor train structure.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_sweeps : int
        Maximum number of forward sweeps through all cores. Default: 10.
    tol : float
        Convergence tolerance on the normalized residual. Default: 1e-6.
    verbosity : int
        Verbosity level (0=silent, 1=summary, 2=per-sweep). Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_sweeps: int = 10,
        tol: float = 1e-6,
        verbosity: int = 0,
    ):
        self._bkd = bkd
        self._max_sweeps = max_sweeps
        self._tol = tol
        self._verbosity = verbosity

    def fit(
        self,
        surrogate: FunctionTrain[Array],
        samples: Array,
        values: Array,
    ) -> ALSFitterResult[Array]:
        """Fit FunctionTrain to data using Alternating Least Squares.

        Parameters
        ----------
        surrogate : FunctionTrain
            The FunctionTrain to fit. Must be a FunctionTrain instance.
        samples : Array
            Training samples. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples) or (nsamples,)

        Returns
        -------
        ALSFitterResult
            Result containing fitted surrogate and diagnostics.

        Raises
        ------
        TypeError
            If surrogate is not a FunctionTrain.
        ValueError
            If dimensions don't match.
        """
        # Validate input type
        if not isinstance(surrogate, FunctionTrain):
            raise TypeError(
                f"ALSFitter only works with FunctionTrain, "
                f"got {type(surrogate).__name__}"
            )

        # Normalize values shape
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        # Validate dimensions
        nvars = surrogate.nvars()
        nqoi = surrogate.nqoi()
        nsamples = samples.shape[1]

        if samples.shape[0] != nvars:
            raise ValueError(
                f"samples has {samples.shape[0]} variables, surrogate expects {nvars}"
            )
        if values.shape[0] != nqoi:
            raise ValueError(f"values has {values.shape[0]} QoIs, surrogate has {nqoi}")
        if values.shape[1] != nsamples:
            raise ValueError(
                f"values has {values.shape[1]} samples, samples has {nsamples}"
            )

        # Work on a copy of the surrogate
        current_ft = surrogate

        residual_history: List[float] = []
        converged = False

        # Compute initial residual
        pred = current_ft(samples)
        residual = self._compute_residual(pred, values)
        residual_history.append(residual)

        if self._verbosity > 0:
            print(f"ALSFitter: Initial residual = {residual:.6e}")

        for sweep in range(self._max_sweeps):
            # Forward sweep through all cores
            for core_id in range(nvars):
                current_ft = self._solve_core(current_ft, samples, values, core_id)

            # Compute residual after sweep
            pred = current_ft(samples)
            residual = self._compute_residual(pred, values)
            residual_history.append(residual)

            if self._verbosity > 1:
                print(f"  Sweep {sweep + 1}: residual = {residual:.6e}")

            # Check convergence
            if residual < self._tol:
                converged = True
                if self._verbosity > 0:
                    print(
                        f"ALSFitter: Converged after {sweep + 1} sweeps "
                        f"(residual = {residual:.6e})"
                    )
                break

        if not converged and self._verbosity > 0:
            print(
                f"ALSFitter: Max sweeps ({self._max_sweeps}) reached "
                f"(residual = {residual:.6e})"
            )

        return ALSFitterResult(
            surrogate=current_ft,
            n_sweeps=sweep + 1,
            residual_history=residual_history,
            converged=converged,
        )

    def _compute_residual(self, pred: Array, values: Array) -> float:
        """Compute normalized RMSE residual."""
        diff = pred - values
        mse = self._bkd.to_float(self._bkd.mean(diff * diff))
        return float(mse**0.5)

    def _solve_core(
        self,
        ft: FunctionTrain[Array],
        samples: Array,
        values: Array,
        core_id: int,
    ) -> FunctionTrain[Array]:
        """Solve least squares for a single core.

        The Jacobian includes columns for ALL basis functions (including
        constants). We solve the full lstsq problem, then extract only
        the coefficients corresponding to trainable parameters.

        Parameters
        ----------
        ft : FunctionTrain
            Current FunctionTrain state.
        samples : Array
            Training samples. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples)
        core_id : int
            Which core to solve for.

        Returns
        -------
        FunctionTrain
            New FunctionTrain with updated core.
        """
        # Get Jacobian w.r.t. ALL basis functions in this core
        # Returns List[Array], one per QoI, each shape (nsamples, total_nterms)
        jacs = ft._core_jacobian(samples, core_id)

        nqoi = ft.nqoi()
        core = ft.cores()[core_id]

        # Get indices of trainable columns
        trainable_indices = core.get_trainable_indices()

        if len(trainable_indices) == 0:
            # No trainable params in this core
            return ft

        # Solve least squares for each QoI
        coefs_list = []
        for qq in range(nqoi):
            # jacs[qq] shape: (nsamples, total_nterms)
            # values[qq, :] shape: (nsamples,)
            target = values[qq : qq + 1, :].T  # (nsamples, 1)
            full_coef = self._bkd.lstsq(jacs[qq], target)  # (total_nterms, 1)

            # Extract only trainable coefficients
            trainable_coef = full_coef[trainable_indices, :]  # (nparams, 1)
            coefs_list.append(trainable_coef)

        # Combine coefficients: shape (nparams, nqoi)
        new_core_params = self._bkd.hstack(coefs_list)
        new_core_params_flat = self._bkd.flatten(new_core_params)

        # Create new core with updated parameters
        new_core = core.with_params(new_core_params_flat)

        # Create new FunctionTrain with updated core
        new_cores = list(ft.cores())
        new_cores[core_id] = new_core

        return ft.with_cores(new_cores)

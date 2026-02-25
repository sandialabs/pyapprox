"""
Lotka-Volterra ODE residual with full HVP support.

Implements the competitive Lotka-Volterra model:
    dx_i/dt = r_i * x_i * (1 - sum_j a_ij * x_j)

This provides a nonlinear benchmark with non-trivial second derivatives
for testing adjoint gradient and HVP computations.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class LotkaVolterraResidual(Generic[Array]):
    """
    Lotka-Volterra ODE residual with HVP support.

    Implements the competitive Lotka-Volterra equations:

        f_i(y, p) = r_i * y_i * (1 - sum_j a_ij * y_j)

    where:
        - y_i: population of species i
        - r_i: intrinsic growth rate of species i
        - a_ij: competition coefficient (effect of species j on species i)

    Parameters are organized as:
        p = [r_0, r_1, ..., r_{n-1}, a_00, a_01, ..., a_{n-1,n-1}]

    Parameters
    ----------
    nspecies : int
        Number of species (states).
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        nspecies: int,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._bkd = bkd
        self._nspecies = nspecies
        self._nparams = nspecies + nspecies * nspecies  # r + A
        self._time = 0.0
        self._param = None
        self._growth_rates = None  # r
        self._alpha = None  # A (competition matrix)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._nparams

    def set_time(self, time: float) -> None:
        """Set the current time."""
        self._time = time

    def set_param(self, param: Array) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters [r_0, ..., r_{n-1}, a_00, a_01, ..., a_{n-1,n-1}].
            Shape: (nparams,) or (nparams, 1)
        """
        if param.ndim == 2:
            param = param.flatten()
        if param.shape[0] != self._nparams:
            raise ValueError(
                f"Expected {self._nparams} parameters, got {param.shape[0]}"
            )
        self._param = param
        n = self._nspecies
        self._growth_rates = param[:n]
        self._alpha = self._bkd.reshape(param[n:], (n, n))

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        f_i(y) = r_i * y_i * (1 - (A @ y)_i)

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)

        Returns
        -------
        Array
            f(y). Shape: (nspecies,)
        """
        return self._growth_rates * state * (1.0 - self._alpha @ state)

    def jacobian(self, state: Array) -> Array:
        """
        Compute the state Jacobian df/dy.

        J[i,k] = r_i * delta_ik * (1 - (A@y)_i) - r_i * y_i * a_ik

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nspecies, nspecies)
        """
        r = self._growth_rates
        A = self._alpha
        Ay = A @ state

        # Diagonal: r * (1 - A@y)
        # Off-diagonal contribution: -r * y * A
        diag_part = self._bkd.diag(r * (1.0 - Ay))
        off_diag = -self._bkd.diag(r * state) @ A

        return diag_part + off_diag

    def mass_matrix(self, nstates: int) -> Array:
        """Return the identity mass matrix."""
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector (identity, returns vec)."""
        return vec

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute the parameter Jacobian df/dp.

        df_i/dr_j = delta_ij * y_i * (1 - (A@y)_i)
        df_i/da_jk = -delta_ij * r_i * y_i * y_k

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nspecies, nparams)
        """
        n = self._nspecies
        r = self._growth_rates
        A = self._alpha
        Ay = A @ state

        # df/dr: diagonal matrix with y * (1 - A@y)
        jac_r = self._bkd.diag(state * (1.0 - Ay))

        # df/dA: for species i, only a_i* matters
        # df_i/da_ik = -r_i * y_i * y_k
        # jac_A[i, i*n + k] = -r_i * y_i * y_k
        # This is a block diagonal structure in the flattened view
        ry = r * state  # r_i * y_i
        # Build as outer product then reshape
        # outer(ry, state) gives ry[i] * state[k] for block at row i
        outer_ry_state = ry[:, None] * state[None, :]  # (n, n)
        # Each row i of jac_A should have entries at columns i*n:(i+1)*n
        # We can build this using einsum or explicit indexing
        jac_A = self._bkd.zeros((n, n * n))
        # Use advanced indexing: row i, columns i*n to (i+1)*n
        for ii in range(n):
            # Create a mask and add
            row = self._bkd.zeros((n * n,))
            row = self._bkd.concatenate(
                [
                    self._bkd.zeros((ii * n,)),
                    -outer_ry_state[ii, :],
                    self._bkd.zeros(((n - ii - 1) * n,)),
                ]
            )
            jac_A = (
                jac_A
                + self._bkd.reshape(self._bkd.eye(n)[ii, :], (n, 1)) * row[None, :]
            )

        return self._bkd.hstack([jac_r, jac_A])

    def initial_param_jacobian(self) -> Array:
        """Initial condition is fixed, so this is zero."""
        return self._bkd.zeros((self._nspecies, self._nparams))

    # =========================================================================
    # HVP Methods for second-order adjoints
    # =========================================================================

    def state_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        """
        Compute lambda^T (d^2f/dy^2) w.

        For f_i = r_i * y_i * (1 - sum_j a_ij * y_j):

        d^2f_i/(dy_k dy_l) = -r_i * (delta_ik * a_il + delta_il * a_ik)

        Contracting with w then lambda:
        result_k = -sum_i lambda_i * r_i * (delta_ik * (A@w)_i + a_ik * w_i)
                 = -lambda_k * r_k * (A@w)_k - (A^T @ (lambda * r * w))_k

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)
        adj_state : Array
            Adjoint state lambda. Shape: (nspecies,)
        wvec : Array
            Direction vector w. Shape: (nspecies,)

        Returns
        -------
        Array
            HVP result. Shape: (nspecies,)
        """
        r = self._growth_rates
        A = self._alpha
        lr = adj_state * r  # lambda * r
        Aw = A @ wvec

        return -(lr * Aw) - (A.T @ (lr * wvec))

    def state_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        """
        Compute lambda^T (d^2f/dy dp) v.

        For the r parameters (v_r = vvec[:n]):
        d^2f_i/(dy_l dr_j) = delta_ij * [delta_il * (1 - (A@y)_i) - y_i * a_il]

        result_l^(r) = lambda_l * (1 - (A@y)_l) * v_r_l
                     - (A^T @ (lambda * y * v_r))_l

        For the A parameters (v_A = vvec[n:] reshaped to (n,n)):
        d^2f_i/(dy_l da_jk) = -delta_ij * r_i * (delta_il * y_k + y_i * delta_lk)

        result_l^(A) = -(lambda * r)_l * (v_A @ y)_l
                     - (v_A^T @ (lambda * r * y))_l

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)
        adj_state : Array
            Adjoint state lambda. Shape: (nspecies,)
        vvec : Array
            Parameter direction v. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nspecies,)
        """
        n = self._nspecies
        r = self._growth_rates
        A = self._alpha
        y = state
        lam = adj_state

        if vvec.ndim == 2:
            vvec = vvec.flatten()

        v_r = vvec[:n]
        v_A = self._bkd.reshape(vvec[n:], (n, n))

        Ay = A @ y

        # Contribution from r parameters
        result_r = lam * (1.0 - Ay) * v_r - A.T @ (lam * y * v_r)

        # Contribution from A parameters
        lr = lam * r
        lry = lr * y
        result_A = -(lr * (v_A @ y)) - (v_A.T @ lry)

        return result_r + result_A

    def param_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        """
        Compute lambda^T (d^2f/dp dy) w.

        For r_j:
        result_r_j = lambda_j * [(1 - (A@y)_j) * w_j - y_j * (A@w)_j]

        For a_jk:
        result_A_jk = -lambda_j * r_j * (y_k * w_j + y_j * w_k)

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)
        adj_state : Array
            Adjoint state lambda. Shape: (nspecies,)
        wvec : Array
            State direction w. Shape: (nspecies,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        r = self._growth_rates
        A = self._alpha
        y = state
        lam = adj_state

        Ay = A @ y
        Aw = A @ wvec

        # Result for r parameters
        result_r = lam * ((1.0 - Ay) * wvec - y * Aw)

        # Result for A parameters
        # result_A_jk = -lambda_j * r_j * (y_k * w_j + y_j * w_k)
        lrw = lam * r * wvec  # lambda_j * r_j * w_j
        lry = lam * r * y  # lambda_j * r_j * y_j

        # outer product via broadcasting: a[:, None] * b[None, :]
        # outer(lrw, y) gives lrw[j] * y[k] = lambda_j * r_j * w_j * y_k
        # outer(lry, w) gives lry[j] * w[k] = lambda_j * r_j * y_j * w_k
        result_A = -(lrw[:, None] * y[None, :] + lry[:, None] * wvec[None, :])
        result_A = self._bkd.flatten(result_A)

        return self._bkd.concatenate([result_r, result_A])

    def param_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        """
        Compute lambda^T (d^2f/dp^2) v.

        Most second derivatives are zero because f is linear/bilinear in params.
        Only r-A cross terms are non-zero:

        d^2f_i/(dr_j da_mn) = -delta_ij * delta_im * y_i * y_n

        result_r_j = -lambda_j * y_j * (v_A @ y)_j
        result_A_mn = -lambda_m * y_m * y_n * v_r_m

        Parameters
        ----------
        state : Array
            State y. Shape: (nspecies,)
        adj_state : Array
            Adjoint state lambda. Shape: (nspecies,)
        vvec : Array
            Parameter direction v. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        n = self._nspecies
        y = state
        lam = adj_state

        if vvec.ndim == 2:
            vvec = vvec.flatten()

        v_r = vvec[:n]
        v_A = self._bkd.reshape(vvec[n:], (n, n))

        ly = lam * y

        # result_r_j = -lambda_j * y_j * (v_A[j,:] @ y)
        # (v_A @ y) gives v_A[j,:] @ y for each j
        result_r = -ly * (v_A @ y)

        # result_A_mn = -lambda_m * y_m * y_n * v_r_m
        # = -outer(lambda * y * v_r, y)
        lyv = ly * v_r
        result_A = -(lyv[:, None] * y[None, :])
        result_A = self._bkd.flatten(result_A)

        return self._bkd.concatenate([result_r, result_A])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nspecies={self._nspecies}, "
            f"nparams={self._nparams})"
        )

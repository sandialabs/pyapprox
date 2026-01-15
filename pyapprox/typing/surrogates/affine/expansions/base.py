"""Base class for basis expansions.

A basis expansion represents a function as a linear combination of basis
functions: f(x) = Σ_i c_i φ_i(x).
"""

from typing import Generic, Optional, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    BasisProtocol,
    BasisHasJacobianProtocol,
    BasisHasHessianProtocol,
    LinearSystemSolverProtocol,
)


class BasisExpansion(Generic[Array]):
    """Base class for basis expansions.

    A basis expansion approximates functions as linear combinations of basis
    functions: f(x) ≈ Σ_i c_i φ_i(x).

    Parameters
    ----------
    basis : BasisProtocol[Array]
        Basis functions for the expansion.
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest. Default: 1.
    solver : LinearSystemSolverProtocol[Array], optional
        Solver for fitting. If None, uses least squares.
    """

    def __init__(
        self,
        basis: BasisProtocol[Array],
        bkd: Backend[Array],
        nqoi: int = 1,
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._nqoi = nqoi
        self._solver = solver

        # Initialize coefficients to zeros
        self._coef: Optional[Array] = None
        self._initialize_coefficients()

        # Dynamically bind derivative methods based on basis capabilities
        self._setup_derivative_methods()

    def _initialize_coefficients(self) -> None:
        """Initialize coefficients to zeros."""
        self._coef = self._bkd.zeros((self.nterms(), self._nqoi))

    def _setup_derivative_methods(self) -> None:
        """Dynamically bind derivative methods based on basis capabilities.

        This method conditionally binds jacobian_batch, hessian_batch, and
        single-sample methods (jacobian, hessian, hvp, whvp) only if the
        underlying basis supports them. This allows gradient-based optimizers
        to check for method availability via hasattr().
        """
        # Batch methods
        if isinstance(self._basis, BasisHasJacobianProtocol):
            self.jacobian_batch = self._jacobian_batch  # type: ignore[method-assign]
            # Single-sample methods
            self.jacobian = self._jacobian  # type: ignore[method-assign]

        # Hessian methods only available for nqoi=1
        if isinstance(self._basis, BasisHasHessianProtocol) and self._nqoi == 1:
            self.hessian_batch = self._hessian_batch  # type: ignore[method-assign]
            self.hessian = self._hessian  # type: ignore[method-assign]
            self.hvp = self._hvp  # type: ignore[method-assign]
            self.whvp = self._whvp  # type: ignore[method-assign]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._basis.nvars()

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._basis.nterms()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def get_coefficients(self) -> Array:
        """Return coefficients. Shape: (nterms, nqoi)."""
        return self._coef

    def set_coefficients(self, coef: Array) -> None:
        """Set coefficients.

        Parameters
        ----------
        coef : Array
            Coefficients. Shape: (nterms, nqoi) or (nterms,) for nqoi=1.
        """
        if coef.ndim == 1:
            coef = self._bkd.reshape(coef, (-1, 1))
        if coef.shape != (self.nterms(), self._nqoi):
            raise ValueError(
                f"Expected shape ({self.nterms()}, {self._nqoi}), "
                f"got {coef.shape}"
            )
        self._coef = coef

    def get_basis(self) -> BasisProtocol[Array]:
        """Return the basis object."""
        return self._basis

    def __call__(self, samples: Array) -> Array:
        """Evaluate expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        # basis(samples): (nsamples, nterms)
        # coef: (nterms, nqoi)
        # result: (nsamples, nqoi) -> transpose to (nqoi, nsamples)
        return (self._basis(samples) @ self._coef).T

    def _jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nvars)
        """
        # basis.jacobian_batch(samples): (nsamples, nterms, nvars)
        # coef: (nterms, nqoi)
        # result: (nsamples, nqoi, nvars)
        basis_jac = self._basis.jacobian_batch(samples)  # type: ignore[union-attr]
        return self._bkd.einsum("ijk,jl->ilk", basis_jac, self._coef)

    def _hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessians. Shape: (nsamples, nvars, nvars)

        Raises
        ------
        ValueError
            If nqoi != 1 (Hessian only supported for scalar-valued functions).
        """
        if self._nqoi != 1:
            raise ValueError(
                f"Hessian only supported for nqoi=1, got nqoi={self._nqoi}"
            )
        # basis.hessian_batch(samples): (nsamples, nterms, nvars, nvars)
        # coef: (nterms, 1)
        # result: (nsamples, 1, nvars, nvars) -> squeeze to (nsamples, nvars, nvars)
        basis_hess = self._basis.hessian_batch(samples)  # type: ignore[union-attr]
        result = self._bkd.einsum("ijkl,jm->imkl", basis_hess, self._coef)
        return result[:, 0, :, :]

    # Single-sample derivative methods

    def _jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        jac_batch = self._jacobian_batch(sample)  # (1, nqoi, nvars)
        return jac_batch[0, :, :]  # (nqoi, nvars)

    def _jvp(self, sample: Array, vec: Array) -> Array:
        """Compute Jacobian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            JVP result. Shape: (nqoi, 1)
        """
        jac = self._jacobian(sample)  # (nqoi, nvars)
        return jac @ vec  # (nqoi, 1)

    def _hessian(self, sample: Array) -> Array:
        """Compute Hessian at a single sample (nqoi=1 only).

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian. Shape: (nvars, nvars)
        """
        hess_batch = self._hessian_batch(sample)  # (1, nvars, nvars)
        return hess_batch[0, :, :]  # (nvars, nvars)

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample (nqoi=1 only).

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nvars, 1)
        """
        hess = self._hessian(sample)  # (nvars, nvars)
        return hess @ vec  # (nvars, 1)

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product at a single sample.

        For nqoi > 1, computes the HVP of the weighted sum of outputs.

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)
        weights : Array
            QoI weights. Shape: (nqoi, 1) or (1, nqoi)

        Returns
        -------
        Array
            Weighted HVP result. Shape: (nvars, 1)
        """
        # For basis expansions, we need the weighted Hessian
        # H_weighted = Σ_q w_q H_q
        # Since hessian_batch requires nqoi=1, we compute this differently:
        # For expansion f(x) = Σ_i c_i φ_i(x), the weighted function is
        # g(x) = Σ_q w_q f_q(x) = Σ_i (Σ_q w_q c_{i,q}) φ_i(x)
        # So the Hessian is: Σ_i (Σ_q w_q c_{i,q}) ∇²φ_i(x)

        # Flatten weights to (nqoi,)
        w = self._bkd.flatten(weights)  # (nqoi,)
        # Compute weighted coefficients: (nterms,)
        weighted_coef = self._coef @ w  # (nterms, nqoi) @ (nqoi,) = (nterms,)
        weighted_coef = self._bkd.reshape(weighted_coef, (-1, 1))  # (nterms, 1)

        # Compute basis Hessians
        basis_hess = self._basis.hessian_batch(sample)  # type: ignore[union-attr]
        # (1, nterms, nvars, nvars)

        # Weighted Hessian: einsum over terms
        result = self._bkd.einsum(
            "ijkl,jm->imkl", basis_hess, weighted_coef
        )  # (1, 1, nvars, nvars)
        hess = result[0, 0, :, :]  # (nvars, nvars)

        return hess @ vec  # (nvars, 1)

    def fit(
        self,
        samples: Array,
        values: Array,
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        """Fit expansion to data.

        Parameters
        ----------
        samples : Array
            Training sample points. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples) or (nsamples,) for nqoi=1.
        solver : LinearSystemSolverProtocol[Array], optional
            Solver to use. If None, uses solver from initialization.
        """
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        if values.shape[0] != self._nqoi:
            raise ValueError(
                f"Expected {self._nqoi} QoIs, got {values.shape[0]}"
            )

        active_solver = solver if solver is not None else self._solver
        if active_solver is None:
            # Use default least squares
            from pyapprox.typing.surrogates.affine.expansions.solvers import (
                LeastSquaresSolver,
            )
            active_solver = LeastSquaresSolver(self._bkd)

        # Evaluate basis at training samples
        basis_matrix = self._basis(samples)  # (nsamples, nterms)

        # Solve for coefficients: solver expects (nsamples, nqoi)
        coef = active_solver.solve(basis_matrix, values.T)
        self.set_coefficients(coef)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nvars={self.nvars()}, "
            f"nterms={self.nterms()}, nqoi={self.nqoi()})"
        )

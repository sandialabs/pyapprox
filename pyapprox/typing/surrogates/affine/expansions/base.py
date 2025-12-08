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

        # Check derivative support
        self._jacobian_supported = isinstance(basis, BasisHasJacobianProtocol)
        self._hessian_supported = isinstance(basis, BasisHasHessianProtocol)

    def _initialize_coefficients(self) -> None:
        """Initialize coefficients to zeros."""
        self._coef = self._bkd.zeros((self.nterms(), self._nqoi))

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

    def jacobian_supported(self) -> bool:
        """Return whether Jacobian computation is supported."""
        return self._jacobian_supported

    def hessian_supported(self) -> bool:
        """Return whether Hessian computation is supported."""
        return self._hessian_supported

    def __call__(self, samples: Array) -> Array:
        """Evaluate expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nsamples, nqoi)
        """
        # basis(samples): (nsamples, nterms)
        # coef: (nterms, nqoi)
        # result: (nsamples, nqoi)
        return self._bkd.dot(self._basis(samples), self._coef)

    def jacobians(self, samples: Array) -> Array:
        """Compute Jacobians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nvars)
        """
        if not self._jacobian_supported:
            raise NotImplementedError("Basis does not support Jacobians")

        # basis.jacobians(samples): (nsamples, nterms, nvars)
        # coef: (nterms, nqoi)
        # result: (nsamples, nqoi, nvars)
        basis_jac = self._basis.jacobians(samples)
        return self._bkd.einsum("ijk,jl->ilk", basis_jac, self._coef)

    def hessians(self, samples: Array) -> Array:
        """Compute Hessians of expansion at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Hessians. Shape: (nsamples, nqoi, nvars, nvars)
        """
        if not self._hessian_supported:
            raise NotImplementedError("Basis does not support Hessians")

        # basis.hessians(samples): (nsamples, nterms, nvars, nvars)
        # coef: (nterms, nqoi)
        # result: (nsamples, nqoi, nvars, nvars)
        basis_hess = self._basis.hessians(samples)
        return self._bkd.einsum("ijkl,jm->imkl", basis_hess, self._coef)

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
            Training values. Shape: (nsamples, nqoi) or (nsamples,).
        solver : LinearSystemSolverProtocol[Array], optional
            Solver to use. If None, uses solver from initialization.
        """
        if values.ndim == 1:
            values = self._bkd.reshape(values, (-1, 1))

        if values.shape[1] != self._nqoi:
            raise ValueError(
                f"Expected {self._nqoi} QoIs, got {values.shape[1]}"
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

        # Solve for coefficients
        coef = active_solver.solve(basis_matrix, values)
        self.set_coefficients(coef)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nvars={self.nvars()}, "
            f"nterms={self.nterms()}, nqoi={self.nqoi()})"
        )

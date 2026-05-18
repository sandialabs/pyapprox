"""Fixed-Poisson variable-Hamiltonian surrogate.

RHS = L @ grad_x H_eta(x, t, mu)

L is a user-supplied skew-symmetric matrix (fixed). H_eta is a scalar
BasisExpansion(nqoi=1) whose coefficients eta are learned. Canonical
Hamiltonian systems are the special case L = J = [[0, I], [-I, 0]].
"""

from typing import Generic

from pyapprox.surrogates.affine.expansions.base import BasisExpansion
from pyapprox.surrogates.affine.protocols import BasisHasJacobianProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class FixedPoissonVariableHamiltonianSurrogate(Generic[Array]):
    """Surrogate for dot{x} = L @ grad_x H_eta(x, t, mu).

    Wraps a scalar BasisExpansion H_eta and a fixed skew-symmetric Poisson
    matrix L. Satisfies LearnedFunctionProtocol.

    Parameters
    ----------
    hamiltonian : BasisExpansion[Array]
        Scalar Hamiltonian expansion with nqoi=1. Its basis must support
        jacobian_batch and hessian_batch.
    poisson_matrix : Array
        Skew-symmetric structure matrix. Shape: (n_dynamic, n_dynamic).
    has_time_input : bool
        Whether the Hamiltonian's input includes a time row.
    n_params : int
        Number of system-parameter input rows (for parametric Hamiltonians).
    """

    def __init__(
        self,
        hamiltonian: BasisExpansion[Array],
        poisson_matrix: Array,
        has_time_input: bool = False,
        n_params: int = 0,
    ) -> None:
        if hamiltonian.nqoi() != 1:
            raise ValueError(
                f"hamiltonian must have nqoi=1, got {hamiltonian.nqoi()}"
            )
        if poisson_matrix.ndim != 2:
            raise ValueError(
                f"poisson_matrix must be 2D, got ndim={poisson_matrix.ndim}"
            )
        if poisson_matrix.shape[0] != poisson_matrix.shape[1]:
            raise ValueError(
                f"poisson_matrix must be square, got shape "
                f"{poisson_matrix.shape}"
            )
        bkd = hamiltonian.bkd()
        skew_check = bkd.max(bkd.abs(poisson_matrix + bkd.transpose(poisson_matrix)))
        if float(skew_check) > 1e-12:
            raise ValueError(
                "poisson_matrix must be skew-symmetric (L + L^T = 0), "
                f"max |L + L^T| = {float(skew_check)}"
            )
        if not hasattr(hamiltonian, "jacobian_batch"):
            raise ValueError(
                "hamiltonian's basis must support jacobian_batch"
            )
        if not hasattr(hamiltonian, "hessian_batch"):
            raise ValueError(
                "hamiltonian's basis must support hessian_batch "
                "(required for jacobian_batch of the surrogate)"
            )
        n_dynamic = poisson_matrix.shape[0]
        expected_nvars = n_dynamic + int(has_time_input) + n_params
        if hamiltonian.nvars() != expected_nvars:
            raise ValueError(
                f"hamiltonian.nvars()={hamiltonian.nvars()} != "
                f"n_dynamic({n_dynamic}) + has_time_input({int(has_time_input)})"
                f" + n_params({n_params}) = {expected_nvars}"
            )
        self._H = hamiltonian
        self._L = poisson_matrix
        self._n_dynamic = n_dynamic
        self._has_time_input = has_time_input
        self._n_params = n_params
        self._bkd = bkd
        basis = hamiltonian.get_basis()
        assert isinstance(basis, BasisHasJacobianProtocol)
        self._jac_basis: BasisHasJacobianProtocol[Array] = basis

    @staticmethod
    def canonical(
        hamiltonian: "BasisExpansion[Array]",
        has_time_input: bool = False,
        n_params: int = 0,
    ) -> "FixedPoissonVariableHamiltonianSurrogate[Array]":
        """Build a canonical Hamiltonian surrogate with J = [[0, I], [-I, 0]].

        Parameters
        ----------
        hamiltonian : BasisExpansion[Array]
            Scalar Hamiltonian with nqoi=1. Its nvars must equal
            n_dynamic + int(has_time_input) + n_params where n_dynamic
            is even.
        has_time_input : bool
            Whether input includes a time row.
        n_params : int
            Number of system-parameter input rows.
        """
        bkd = hamiltonian.bkd()
        n_dynamic = hamiltonian.nvars() - int(has_time_input) - n_params
        if n_dynamic % 2 != 0:
            raise ValueError(
                f"Canonical J requires even n_dynamic, got {n_dynamic}"
            )
        n = n_dynamic // 2
        eye_n = bkd.eye(n)
        zero_n = bkd.zeros((n, n))
        J = bkd.vstack([
            bkd.hstack([zero_n, eye_n]),
            bkd.hstack([-eye_n, zero_n]),
        ])
        return FixedPoissonVariableHamiltonianSurrogate(
            hamiltonian, J, has_time_input, n_params
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._H.nvars()

    def nqoi(self) -> int:
        return self._n_dynamic

    def n_dynamic(self) -> int:
        return self._n_dynamic

    def poisson_matrix(self) -> Array:
        return self._L

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._H.hyp_list()

    def basis_jacobian_batch(self, samples: Array) -> Array:
        """Per-basis-function gradients at samples.

        Returns
        -------
        Array
            Shape: (nsamples, nterms, nvars).
        """
        return self._jac_basis.jacobian_batch(samples)

    def __call__(self, samples: Array) -> Array:
        """Evaluate L @ grad_x H_eta at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (n_dynamic, nsamples)
        """
        grad_H = self._H.jacobian_batch(samples)[:, 0, : self._n_dynamic]
        rhs = grad_H @ self._bkd.transpose(self._L)
        return self._bkd.transpose(rhs)

    def jacobian_batch(self, samples: Array) -> Array:
        """d(L @ grad_x H)/d(input) at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, n_dynamic, nvars)
        """
        hess = self._H.hessian_batch(samples)
        hess_state_rows = hess[:, : self._n_dynamic, :]
        return self._bkd.einsum(
            "rs,isk->irk", self._L, hess_state_rows
        )

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """d(L @ grad_x H)/d(eta) at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, n_dynamic, nactive_params)
        """
        basis_jac = self._jac_basis.jacobian_batch(samples)
        basis_jac_state = basis_jac[:, :, : self._n_dynamic]
        result = self._bkd.einsum(
            "rs,ias->ira", self._L, basis_jac_state
        )
        active_indices = self._H.hyp_list().get_active_indices()
        return result[:, :, active_indices]

    def with_params(
        self, params: Array
    ) -> "FixedPoissonVariableHamiltonianSurrogate[Array]":
        """Return a copy with new H_eta coefficients.

        Parameters
        ----------
        params : Array
            New coefficients. Shape: (nterms, nqoi) or (nterms,).
        """
        new_H = self._H.with_params(params)
        return FixedPoissonVariableHamiltonianSurrogate(
            new_H, self._L, self._has_time_input, self._n_params
        )

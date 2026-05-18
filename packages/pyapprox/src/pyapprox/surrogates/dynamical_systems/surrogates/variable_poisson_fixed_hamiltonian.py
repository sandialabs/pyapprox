"""Variable-Poisson fixed-Hamiltonian surrogate.

RHS = L_phi @ grad H(x)

grad H is user-supplied (known). L_phi is a learnable constant
skew-symmetric matrix parametrized by its strictly upper-triangular
entries.

User-supplied callables (grad_hamiltonian, grad_hamiltonian_jacobian)
must use backend-agnostic operations if intended for trajectory matching
with autograd through the computation graph.
"""

from typing import Callable, Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameter, HyperParameterList


class VariablePoissonFixedHamiltonianSurrogate(Generic[Array]):
    """Surrogate for dot{x} = L_phi @ grad H(x).

    The Hamiltonian H is known (user-supplied callables for grad H and
    its Jacobian). The Poisson operator L_phi is a learnable constant
    skew-symmetric matrix whose strictly upper-triangular entries are
    the parameters.

    Parameters
    ----------
    grad_hamiltonian : Callable[[Array], Array]
        Known gradient of H. Input: (n_dynamic + n_aux, nsamples).
        Output: (n_dynamic, nsamples) — state-direction gradient only.
    grad_hamiltonian_jacobian : Callable[[Array], Array]
        Jacobian of grad_hamiltonian (Hessian of H in state variables).
        Input: (n_dynamic + n_aux, nsamples).
        Output: (nsamples, n_dynamic, n_dynamic).
    n_dynamic : int
        Dynamic state dimension. No parity restriction.
    bkd : Backend[Array]
    n_aux : int
        Number of auxiliary input rows beyond dynamic state (time, params).
    """

    def __init__(
        self,
        grad_hamiltonian: Callable[[Array], Array],
        grad_hamiltonian_jacobian: Callable[[Array], Array],
        n_dynamic: int,
        bkd: Backend[Array],
        n_aux: int = 0,
    ) -> None:
        if n_dynamic < 2:
            raise ValueError(f"n_dynamic must be >= 2, got {n_dynamic}")

        self._grad_H = grad_hamiltonian
        self._grad_H_jac = grad_hamiltonian_jacobian
        self._n_dynamic = n_dynamic
        self._n_aux = n_aux
        self._bkd = bkd
        self._n_skew = n_dynamic * (n_dynamic - 1) // 2

        self._upper_indices: List[Tuple[int, int]] = [
            (p, q)
            for p in range(n_dynamic)
            for q in range(p + 1, n_dynamic)
        ]

        self._hyp_list = HyperParameterList(
            [
                HyperParameter(
                    name="skew_entries",
                    nparams=self._n_skew,
                    values=bkd.zeros((self._n_skew,)),
                    bounds=(-1e10, 1e10),
                    bkd=bkd,
                )
            ],
            bkd,
        )

    def _build_L(self) -> Array:
        """Assemble skew-symmetric L from current parameter values."""
        eta = self._hyp_list.get_values()
        L = self._bkd.zeros((self._n_dynamic, self._n_dynamic))
        for k, (p, q) in enumerate(self._upper_indices):
            L = self._bkd.index_update(L, (p, q), eta[k])
            L = self._bkd.index_update(L, (q, p), -eta[k])
        return L

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._n_dynamic + self._n_aux

    def nqoi(self) -> int:
        return self._n_dynamic

    def n_dynamic(self) -> int:
        return self._n_dynamic

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def __call__(self, samples: Array) -> Array:
        """Evaluate L_phi @ grad H(samples).

        Parameters
        ----------
        samples : Array
            Shape: (n_dynamic + n_aux, nsamples)

        Returns
        -------
        Array
            Shape: (n_dynamic, nsamples)
        """
        L = self._build_L()
        gH = self._grad_H(samples)
        return L @ gH

    def jacobian_batch(self, samples: Array) -> Array:
        """d(L @ grad H)/d(input) = L @ Hessian_H at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (n_dynamic + n_aux, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, n_dynamic, n_dynamic + n_aux)
        """
        L = self._build_L()
        hess_H = self._grad_H_jac(samples)
        if (
            hess_H.shape[1] == self._n_dynamic
            and hess_H.shape[2] == self._n_dynamic
        ):
            result = self._bkd.einsum("ij,njk->nik", L, hess_H)
            if self._n_aux > 0:
                nsamples = samples.shape[1]
                zeros_aux = self._bkd.zeros(
                    (nsamples, self._n_dynamic, self._n_aux)
                )
                result = self._bkd.concatenate([result, zeros_aux], axis=2)
            return result
        return self._bkd.einsum("ij,njk->nik", L, hess_H)

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """d(L @ grad H)/d(eta) at each sample.

        For skew entry k=(p,q) with p < q:
            d(RHS_r)/d(eta_k) = delta_{r,p} * gH_q - delta_{r,q} * gH_p

        Uses the precomputed basis-matrix representation: each skew entry k
        corresponds to the antisymmetric unit matrix e_p e_q^T - e_q e_p^T.

        Parameters
        ----------
        samples : Array
            Shape: (n_dynamic + n_aux, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, n_dynamic, n_skew)
        """
        gH = self._grad_H(samples)
        nsamples = samples.shape[1]
        n = self._n_dynamic
        eye = self._bkd.eye(n)
        columns = []
        for _k, (p, q) in enumerate(self._upper_indices):
            e_p = self._bkd.reshape(eye[p, :], (1, n))
            e_q = self._bkd.reshape(eye[q, :], (1, n))
            gH_q = self._bkd.reshape(gH[q, :], (nsamples, 1))
            gH_p = self._bkd.reshape(gH[p, :], (nsamples, 1))
            col = gH_q * e_p - gH_p * e_q
            columns.append(col)
        return self._bkd.stack(columns, axis=2)

    def with_params(
        self, params: Array
    ) -> "VariablePoissonFixedHamiltonianSurrogate[Array]":
        """Return a copy with new skew-matrix entries.

        Parameters
        ----------
        params : Array
            New skew entries. Shape: (n_skew,).
        """
        new = VariablePoissonFixedHamiltonianSurrogate(
            grad_hamiltonian=self._grad_H,
            grad_hamiltonian_jacobian=self._grad_H_jac,
            n_dynamic=self._n_dynamic,
            bkd=self._bkd,
            n_aux=self._n_aux,
        )
        new._hyp_list.set_active_values(params)
        return new

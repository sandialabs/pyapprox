"""Closed-form derivative matching fitter for fixed-Poisson Hamiltonian surrogates.

Builds the J/L-composed design matrix from the surrogate's basis Jacobian
and Poisson matrix, then delegates the linear solve to a
LinearSystemSolverProtocol.

If the basis includes a constant term (all-zero multi-index), use
RidgeSolver — constants in H don't contribute to dynamics, making the
design matrix rank-deficient.
"""

from typing import Generic, Optional

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import LinearSystemSolverProtocol
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.surrogates.fixed_poisson_variable_hamiltonian import (  # noqa: E501
    FixedPoissonVariableHamiltonianSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend


class FixedPoissonVariableHamiltonianDerivativeMatchingFitter(
    Generic[Array]
):
    """Closed-form derivative matching for fixed-Poisson Hamiltonian surrogates.

    Parameters
    ----------
    bkd : Backend[Array]
    solver : LinearSystemSolverProtocol[Array], optional
        Defaults to LeastSquaresSolver.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._solver = solver if solver is not None else LeastSquaresSolver(bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        surrogate: FixedPoissonVariableHamiltonianSurrogate[Array],
        dataset: SnapshotDataset[Array],
    ) -> DirectSolverResult[
        Array, FixedPoissonVariableHamiltonianSurrogate[Array]
    ]:
        """Fit H_eta coefficients via least squares.

        Parameters
        ----------
        surrogate : FixedPoissonVariableHamiltonianSurrogate[Array]
            Unfitted surrogate with the desired basis and Poisson matrix.
        dataset : SnapshotDataset[Array]
            states shape: (nvars, nsamples),
            derivatives shape: (n_dynamic, nsamples).
        """
        if dataset.nstates_input() != surrogate.nvars():
            raise ValueError(
                f"dataset.nstates_input()={dataset.nstates_input()} != "
                f"surrogate.nvars()={surrogate.nvars()}"
            )
        if dataset.nstates_output() != surrogate.nqoi():
            raise ValueError(
                f"dataset.nstates_output()={dataset.nstates_output()} != "
                f"surrogate.nqoi()={surrogate.nqoi()}"
            )

        n_dynamic = surrogate.n_dynamic()
        L = surrogate.poisson_matrix()
        basis_jac = surrogate.basis_jacobian_batch(dataset.states())
        basis_jac_state = basis_jac[:, :, :n_dynamic]
        Phi_3d = self._bkd.einsum("rs,ias->ira", L, basis_jac_state)
        nsamples = dataset.nsamples()
        nterms = basis_jac.shape[1]
        Phi = self._bkd.reshape(Phi_3d, (nsamples * n_dynamic, nterms))

        targets_2d = self._bkd.transpose(dataset.derivatives())
        targets = self._bkd.reshape(targets_2d, (nsamples * n_dynamic, 1))

        coefs = self._solver.solve(Phi, targets)

        fitted = surrogate.with_params(coefs)

        return DirectSolverResult(
            surrogate=fitted,
            params=coefs,
        )

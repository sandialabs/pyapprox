"""Closed-form derivative matching fitter for variable-Poisson Hamiltonian surrogates.

The RHS L @ grad H is linear in the skew entries of L (with H known),
so fitting reduces to a standard linear regression.
"""

from typing import Generic, Optional

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import LinearSystemSolverProtocol
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend


class VariablePoissonFixedHamiltonianDerivativeMatchingFitter(
    Generic[Array]
):
    """Closed-form derivative matching for variable-Poisson Hamiltonian surrogates.

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
        surrogate: VariablePoissonFixedHamiltonianSurrogate[Array],
        dataset: SnapshotDataset[Array],
    ) -> DirectSolverResult[
        Array, VariablePoissonFixedHamiltonianSurrogate[Array]
    ]:
        """Fit skew-matrix entries via least squares.

        Parameters
        ----------
        surrogate : VariablePoissonFixedHamiltonianSurrogate[Array]
            Surrogate with known grad H and initial L entries.
        dataset : SnapshotDataset[Array]
            states shape: (n_dynamic + n_aux, nsamples),
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
        nsamples = dataset.nsamples()

        jac_wrt_params = surrogate.jacobian_wrt_params(dataset.states())
        n_skew = jac_wrt_params.shape[2]
        Phi = self._bkd.reshape(
            jac_wrt_params, (nsamples * n_dynamic, n_skew)
        )

        targets_2d = self._bkd.transpose(dataset.derivatives())
        targets = self._bkd.reshape(targets_2d, (nsamples * n_dynamic, 1))

        coefs = self._solver.solve(Phi, targets)

        fitted = surrogate.with_params(coefs[:, 0])

        return DirectSolverResult(
            surrogate=fitted,
            params=coefs,
        )

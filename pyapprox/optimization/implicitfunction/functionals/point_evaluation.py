"""Point evaluation functional for collocation-based PDE solutions.

Evaluates Q(u) = u(x*) at an arbitrary point x* using Lagrange interpolation
through the collocation nodes. This is a linear functional with constant
state Jacobian.
"""

from typing import Generic

from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.decomposition.interface.interpolation import (
    lagrange_interpolation_matrix,
)
from pyapprox.util.backends.protocols import Array, Backend


class PointEvaluationFunctional(Generic[Array]):
    """Evaluate the PDE solution at a single physical point.

    Computes Q(u) = u(x*) via Lagrange interpolation through the
    collocation nodes. The interpolation row vector is precomputed
    and serves as the constant state Jacobian.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    eval_point : float
        Evaluation point x* in physical coordinates.
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: ChebyshevBasis1D[Array],
        eval_point: float,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._nparams = nparams

        ref_nodes = basis.nodes()  # shape (npts,)
        self._nstates = ref_nodes.shape[0]

        # Map eval_point to reference coordinates
        transform = basis.mesh().transform()
        if transform is not None:
            ref_pt = transform.map_to_reference(
                bkd.asarray([[eval_point]])
            )  # shape (1, 1)
            target = ref_pt[0, :]  # shape (1,)
        else:
            target = bkd.asarray([eval_point])  # shape (1,)

        # Build interpolation row: L_j(x*) for j = 0..npts-1
        self._interp_row = lagrange_interpolation_matrix(
            ref_nodes, target, bkd
        )  # shape (1, npts)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate Q(u) = u(x*).

        Parameters
        ----------
        state : Array, shape (nstates, 1)
            PDE state vector.
        param : Array, shape (nparams, 1)
            Parameters (unused).

        Returns
        -------
        Array, shape (1, 1)
            Solution value at x*.
        """
        return self._interp_row @ state

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/du = interpolation row vector (constant).

        Returns
        -------
        Array, shape (1, nstates)
        """
        return self._interp_row

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0 (no parameter dependence).

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams}, "
            f"bkd={type(self._bkd).__name__})"
        )

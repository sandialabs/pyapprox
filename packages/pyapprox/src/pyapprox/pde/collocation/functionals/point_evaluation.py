"""Point evaluation functional for collocation-based PDE solutions.

Evaluates Q(u) = u(x*) at an arbitrary point x* using Lagrange interpolation
through the collocation nodes. This is a linear functional with constant
state Jacobian.
"""
# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Generic, Literal, Sequence

from pyapprox.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.surrogates.affine.univariate.lagrange import (
    univariate_lagrange_polynomial,
)
from pyapprox.util.backends.protocols import Array, Backend


def point_interpolation_weights(
    basis: ChebyshevBasis1D[Array], points: Sequence[float], bkd: Backend[Array]
) -> Array:
    r"""The weights :math:`W` whose column :math:`k` interpolates a nodal
    state at :math:`x_k`: :math:`w_k^\top u = u(x_k)`, the Lagrange
    interpolant through the collocation nodes.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    points : sequence of float
        The points :math:`x_k`, in physical coordinates, each in the mesh's
        closed domain.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        :math:`W`, shape ``(npts, npoints)``.

    Raises
    ------
    ValueError
        If ``points`` is empty or a point lies outside the domain, where
        the interpolant would extrapolate.
    """
    points = list(points)
    if not points:
        raise ValueError("an interpolation needs at least one point")
    physical = basis.mesh().points()[0]
    lower = bkd.to_float(bkd.min(physical))
    upper = bkd.to_float(bkd.max(physical))
    slack = 1e-12 * (upper - lower)
    outside = [x for x in points if x < lower - slack or x > upper + slack]
    if outside:
        raise ValueError(
            f"points {outside} lie outside the domain [{lower}, {upper}]"
        )
    targets = bkd.asarray([points])  # shape (1, npoints)
    transform = basis.mesh().transform()
    if transform is not None:
        targets = transform.map_to_reference(targets)
    rows = univariate_lagrange_polynomial(basis.nodes(), targets[0], bkd)
    return rows.T


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

        self._nstates = basis.nodes().shape[0]
        # Interpolation row: L_j(x*) for j = 0..npts-1, shape (1, npts)
        self._interp_row = point_interpolation_weights(
            basis, [eval_point], bkd
        ).T

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

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return SteadyFunctionalDerivatives.first_order(
            self.state_jacobian, self.param_jacobian
        )

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

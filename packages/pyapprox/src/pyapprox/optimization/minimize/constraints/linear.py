from typing import Generic

from scipy.optimize import LinearConstraint as ScipyLinearConstraint

from pyapprox.util.backends.protocols import Array, Backend


class PyApproxLinearConstraint(Generic[Array]):
    """
    Generic array version of SciPy's LinearConstraint.

    This class supports different array backends (e.g., NumPy, PyTorch) and
    ensures
    compatibility with SciPy optimizers.

    Parameters
    ----------
    A : Array
        Coefficient matrix for the linear constraint.
    lb : Array
        Lower bounds for the constraint.
    ub : Array
        Upper bounds for the constraint.
    backend : Backend[Array]
        Backend used for computations (e.g., NumPy, PyTorch).
    """

    def __init__(self, A: Array, lb: Array, ub: Array, bkd: Backend[Array]):
        self._bkd = bkd
        self._A = A
        self._lb = lb
        self._ub = ub

    def to_scipy(self) -> ScipyLinearConstraint:
        """
        Convert the generic linear constraint to a SciPy-compatible
        LinearConstraint.

        Returns
        -------
        ScipyLinearConstraint
            SciPy-compatible LinearConstraint object.
        """
        return ScipyLinearConstraint(
            self._bkd.to_numpy(self._A),
            self._bkd.to_numpy(self._lb),
            self._bkd.to_numpy(self._ub),
        )

    def A(self) -> Array:
        """
        Get the coefficient matrix.

        Returns
        -------
        Array
            Coefficient matrix for the linear constraint.
        """
        return self._A

    def lb(self) -> Array:
        """
        Get the lower bounds.

        Returns
        -------
        Array
            Lower bounds for the constraint.
        """
        return self._lb

    def ub(self) -> Array:
        """
        Get the upper bounds.

        Returns
        -------
        Array
            Upper bounds for the constraint.
        """
        return self._ub

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        return self._bkd

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the PyApproxLinearConstraint
        object.

        Returns
        -------
        str
            String representation of the object.
        """
        return "{0}(nrows={1}, ncols={2})".format(
            self.__class__.__name__, self._A.shape[0], self._A.shape[1]
        )

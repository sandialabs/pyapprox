"""Standard (non-normalized) Chebyshev polynomials of the first kind.

Evaluates T_n(x) = cos(n*arccos(x)) via the three-term recurrence:
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)

These are the standard Chebyshev polynomials used in SUPN (Morrow et al. 2025),
not the orthonormal variants used in PCE. This class satisfies Basis1DProtocol
and Basis1DHasJacobianProtocol for use with MultiIndexBasis.
"""

from typing import Generic, List

from pyapprox.util.backends.protocols import Array, Backend


class StandardChebyshev1D(Generic[Array]):
    """Standard Chebyshev polynomials of the first kind on [-1, 1].

    Satisfies Basis1DProtocol and Basis1DHasJacobianProtocol so it can be
    used directly with MultiIndexBasis for multivariate tensor products.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nterms = 0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms (polynomial degrees 0..nterms-1)."""
        self._nterms = nterms

    def nterms(self) -> int:
        """Return the current number of terms."""
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        """Evaluate T_0(x), ..., T_{nterms-1}(x).

        Uses list-based construction then stack to preserve autograd graph.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Chebyshev values. Shape: (nsamples, nterms)
        """
        bkd = self._bkd
        x = samples[0]  # (nsamples,)
        nterms = self._nterms

        if nterms == 0:
            return bkd.zeros((x.shape[0], 0))

        cols: List[Array] = [bkd.ones(x.shape)]  # T_0 = 1
        if nterms >= 2:
            cols.append(x)  # T_1 = x
        for n in range(2, nterms):
            cols.append(2.0 * x * cols[n - 1] - cols[n - 2])

        return bkd.stack(cols, axis=1)  # (nsamples, nterms)

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate dT_n/dx for n = 0, ..., nterms-1.

        Uses the identity: dT_n/dx = n*U_{n-1}(x), where U is the
        Chebyshev polynomial of the second kind, with recurrence:
            U_0(x) = 1,  U_1(x) = 2x,  U_{n+1}(x) = 2x*U_n - U_{n-1}

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError(
                f"samples must have shape (1, nsamples), got {samples.shape}"
            )
        bkd = self._bkd
        x = samples[0]  # (nsamples,)
        nterms = self._nterms

        if nterms == 0:
            return bkd.zeros((x.shape[0], 0))

        cols: List[Array] = [bkd.zeros(x.shape)]  # dT_0/dx = 0
        if nterms >= 2:
            cols.append(bkd.ones(x.shape))  # dT_1/dx = 1

        if nterms >= 3:
            # Build U_{n-1} via recurrence for dT_n/dx = n * U_{n-1}
            u_prev = bkd.ones(x.shape)  # U_0
            u_curr = 2.0 * x  # U_1
            cols.append(2.0 * u_curr)  # dT_2/dx = 2 * U_1

            for n in range(3, nterms):
                u_next = 2.0 * x * u_curr - u_prev
                u_prev = u_curr
                u_curr = u_next
                cols.append(float(n) * u_curr)

        return bkd.stack(cols, axis=1)  # (nsamples, nterms)

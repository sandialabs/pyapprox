"""Shallow Universal Polynomial Network (SUPN) surrogate.

Implements the SUPN from Morrow et al. (2025, arXiv:2511.21414):

    f_{N,Lambda}(x) = sum_{n=1}^{N} c_n * tanh(sum_{m in Lambda} a_{n,m} * T_m(x))

where T_m(x) = prod_d T_{m_d}(x_d) are multivariate Chebyshev polynomials
defined by a multi-index lower set Lambda, and c_n, a_{n,m} are trainable
parameters.
"""

from typing import Generic, List, Optional, Self

import numpy as np

from pyapprox.surrogates.affine.basis.multiindex import MultiIndexBasis
from pyapprox.surrogates.affine.indices.utils import compute_hyperbolic_indices
from pyapprox.surrogates.affine.protocols.basis1d import Basis1DProtocol
from pyapprox.surrogates.supn.chebyshev import StandardChebyshev1D
from pyapprox.util.backends.protocols import Array, Backend


class SUPN(Generic[Array]):
    """Shallow Universal Polynomial Network surrogate.

    A single-hidden-layer network with tanh activation applied to a
    multivariate Chebyshev polynomial lift:

        f(x) = C @ tanh(A @ B(x).T)

    where B(x) is the basis matrix of Chebyshev tensor products T_m(x)
    for m in the multi-index set Lambda.

    Parameters
    ----------
    basis : MultiIndexBasis[Array]
        Multivariate basis providing T_m(x) evaluation. Indices define
        the multi-index set Lambda.
    width : int
        Number of neurons N (outer sum terms).
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest. Default: 1.
    outer_coefs : Array, optional
        Outer coefficients C. Shape: (nqoi, width). If None, uses
        Kaiming uniform initialization.
    inner_coefs : Array, optional
        Inner coefficients A. Shape: (width, nterms). If None, uses
        Kaiming uniform initialization.
    """

    def __init__(
        self,
        basis: MultiIndexBasis[Array],
        width: int,
        bkd: Backend[Array],
        nqoi: int = 1,
        outer_coefs: Optional[Array] = None,
        inner_coefs: Optional[Array] = None,
    ) -> None:
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if nqoi < 1:
            raise ValueError(f"nqoi must be >= 1, got {nqoi}")
        if basis.nterms() == 0:
            raise ValueError("basis must have indices set (nterms > 0)")

        self._basis = basis
        self._width = width
        self._bkd = bkd
        self._nqoi = nqoi

        nterms = basis.nterms()

        if outer_coefs is not None:
            if outer_coefs.shape != (nqoi, width):
                raise ValueError(
                    f"outer_coefs shape must be ({nqoi}, {width}), "
                    f"got {outer_coefs.shape}"
                )
            self._outer_coefs = outer_coefs
        else:
            self._outer_coefs = self._kaiming_init((nqoi, width), nterms)

        if inner_coefs is not None:
            if inner_coefs.shape != (width, nterms):
                raise ValueError(
                    f"inner_coefs shape must be ({width}, {nterms}), "
                    f"got {inner_coefs.shape}"
                )
            self._inner_coefs = inner_coefs
        else:
            self._inner_coefs = self._kaiming_init((width, nterms), nterms)

    def _kaiming_init(self, shape: tuple[int, ...], fan_in: int) -> Array:
        """Kaiming uniform initialization for tanh activation."""
        bound = np.sqrt(6.0 / fan_in)
        return self._bkd.asarray(
            np.random.uniform(-bound, bound, shape)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._basis.nvars()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def width(self) -> int:
        """Return the network width N."""
        return self._width

    def nterms(self) -> int:
        """Return the number of basis terms |Lambda|."""
        return self._basis.nterms()

    def nparams(self) -> int:
        """Return the total number of trainable parameters.

        Layout: [outer_coefs (nqoi*N), inner_coefs (N*nterms)]
        """
        return self._nqoi * self._width + self._width * self._basis.nterms()

    def basis(self) -> MultiIndexBasis[Array]:
        """Return the multivariate basis."""
        return self._basis

    def outer_coefs(self) -> Array:
        """Return the outer coefficients C. Shape: (nqoi, width)."""
        return self._outer_coefs

    def inner_coefs(self) -> Array:
        """Return the inner coefficients A. Shape: (width, nterms)."""
        return self._inner_coefs

    def __call__(self, samples: Array) -> Array:
        """Evaluate the SUPN at sample points.

        f(x) = C @ tanh(A @ B(x).T)

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            SUPN values. Shape: (nqoi, nsamples)
        """
        B = self._basis(samples)  # (nsamples, nterms)
        Z = self._bkd.dot(self._inner_coefs, B.T)  # (N, nsamples)
        H = self._bkd.tanh(Z)  # (N, nsamples)
        return self._bkd.dot(self._outer_coefs, H)  # (nqoi, nsamples)

    def eval_from_basis_matrix(self, basis_matrix: Array) -> Array:
        """Evaluate using a pre-computed basis matrix.

        Useful when the basis matrix is cached (e.g., during training).

        Parameters
        ----------
        basis_matrix : Array
            Pre-computed basis matrix B. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            SUPN values. Shape: (nqoi, nsamples)
        """
        Z = self._bkd.dot(self._inner_coefs, basis_matrix.T)
        H = self._bkd.tanh(Z)
        return self._bkd.dot(self._outer_coefs, H)

    def jacobian_batch(self, samples: Array) -> Array:
        """Jacobian w.r.t. inputs for a batch of samples.

        df_q/dx_d = sum_n C[q,n] * sech^2(Z[n,k]) * sum_j A[n,j] * dB[k,j,d]

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nqoi, nvars)
        """
        bkd = self._bkd
        B = self._basis(samples)  # (K, M)
        dB = self._basis.jacobian_batch(samples)  # (K, M, D)
        Z = bkd.dot(self._inner_coefs, B.T)  # (N, K)
        S = 1.0 - bkd.tanh(Z) ** 2  # sech^2, (N, K)

        # G_all[n,k,d] = sum_j A[n,j] * dB[k,j,d]
        G_all = bkd.einsum("nj,kjd->nkd", self._inner_coefs, dB)  # (N, K, D)
        # J[k,q,d] = sum_n C[q,n] * S[n,k] * G_all[n,k,d]
        return bkd.einsum(
            "qn,nk,nkd->kqd", self._outer_coefs, S, G_all
        )  # (K, Q, D)

    def jacobian(self, sample: Array) -> Array:
        """Jacobian w.r.t. inputs for a single sample.

        Parameters
        ----------
        sample : Array
            Single sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        return self.jacobian_batch(sample)[0]  # (nqoi, nvars)

    def jacobian_wrt_params_batch(self, samples: Array) -> Array:
        """Jacobian of output w.r.t. all trainable parameters.

        Parameter layout: [C.flatten() (nqoi*N), A.flatten() (N*M)]

        For QoI q:
            df_q/dc_{q',n} = delta_{qq'} * H[n,k]
            df_q/da_{n,j}  = C[q,n] * S[n,k] * B[k,j]

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nqoi, nparams)
        """
        bkd = self._bkd
        B = self._basis(samples)  # (K, M)
        Z = bkd.dot(self._inner_coefs, B.T)  # (N, K)
        H = bkd.tanh(Z)  # (N, K)
        S = 1.0 - H ** 2  # sech^2, (N, K)

        K = B.shape[0]
        N = self._width
        M = self.nterms()
        Q = self._nqoi
        P_outer = Q * N
        P_inner = N * M
        P = P_outer + P_inner

        # Allocate output columns
        jac_cols: List[Array] = []

        # --- Outer coefs: df_q/dc_{q',n} = delta_{qq'} * H[n,k] ---
        # For each QoI q, the block is (K, N) = H.T, placed at cols q*N..(q+1)*N
        # Full shape contribution: (K, Q, Q*N) with block-diagonal structure
        for q in range(Q):
            # (K, Q) for each n: all zeros except row q = H[n,:]
            for n in range(N):
                col = bkd.zeros((K, Q))
                # col[:, q] = H[n, :] but we need autograd-safe construction
                col_parts: List[Array] = []
                for qq in range(Q):
                    if qq == q:
                        col_parts.append(H[n, :])  # (K,)
                    else:
                        col_parts.append(bkd.zeros((K,)))
                jac_cols.append(bkd.stack(col_parts, axis=1))  # (K, Q)

        # --- Inner coefs: df_q/da_{n,j} = C[q,n] * S[n,k] * B[k,j] ---
        # For each (n, j): (K, Q) column where entry [k, q] = C[q,n]*S[n,k]*B[k,j]
        for n in range(N):
            # C[:,n] * S[n,:] gives (Q,) * (K,) -> need outer product
            # weighted[k,q] = C[q,n] * S[n,k]
            c_col = self._outer_coefs[:, n]  # (Q,)
            s_row = S[n, :]  # (K,)
            weighted = s_row[:, None] * c_col[None, :]  # (K, Q)
            for j in range(M):
                jac_cols.append(weighted * B[:, j : j + 1])  # (K, Q)

        # Stack: list of (K, Q) -> (K, Q, P)
        return bkd.stack(jac_cols, axis=2)  # (K, Q, P)

    def jacobian_wrt_params(self, sample: Array) -> Array:
        """Jacobian w.r.t. parameters for a single sample.

        Parameters
        ----------
        sample : Array
            Single sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nparams)
        """
        return self.jacobian_wrt_params_batch(sample)[0]

    def with_params(self, params: Array) -> Self:
        """Return a NEW SUPN with the given parameters.

        Parameters
        ----------
        params : Array
            Flattened parameters. Shape: (nparams,) or (nparams, 1)

        Returns
        -------
        Self
            New SUPN with parameters set.
        """
        bkd = self._bkd
        params_flat = bkd.flatten(params)
        if params_flat.shape[0] != self.nparams():
            raise ValueError(
                f"Expected {self.nparams()} params, got {params_flat.shape[0]}"
            )

        N = self._width
        Q = self._nqoi
        M = self.nterms()
        P_outer = Q * N

        outer_coefs = bkd.reshape(params_flat[:P_outer], (Q, N))
        inner_coefs = bkd.reshape(params_flat[P_outer:], (N, M))

        return self.__class__(
            basis=self._basis,
            width=self._width,
            bkd=self._bkd,
            nqoi=self._nqoi,
            outer_coefs=outer_coefs,
            inner_coefs=inner_coefs,
        )

    def _flatten_params(self) -> Array:
        """Flatten all parameters to a single vector.

        Layout: [outer_coefs.flatten(), inner_coefs.flatten()]

        Returns
        -------
        Array
            Flattened parameters. Shape: (nparams,)
        """
        return self._bkd.hstack([
            self._bkd.flatten(self._outer_coefs),
            self._bkd.flatten(self._inner_coefs),
        ])


def create_supn(
    nvars: int,
    width: int,
    max_level: int,
    bkd: Backend[Array],
    pnorm: float = 1.0,
    nqoi: int = 1,
    indices: Optional[Array] = None,
) -> SUPN[Array]:
    """Create a SUPN with standard Chebyshev basis on [-1,1]^D.

    Parameters
    ----------
    nvars : int
        Number of input variables D.
    width : int
        Network width N (number of neurons).
    max_level : int
        Maximum polynomial level for multi-index generation.
    bkd : Backend[Array]
        Computational backend.
    pnorm : float
        p-norm for multi-index set generation. 1.0 = total degree,
        < 1.0 = hyperbolic cross. Default: 1.0.
    nqoi : int
        Number of quantities of interest. Default: 1.
    indices : Array, optional
        Custom multi-index set. Shape: (nvars, nterms). If provided,
        max_level and pnorm are ignored.

    Returns
    -------
    SUPN[Array]
        Initialized SUPN surrogate.
    """
    bases_1d: List[Basis1DProtocol[Array]] = [
        StandardChebyshev1D(bkd) for _ in range(nvars)
    ]
    if indices is None:
        indices = compute_hyperbolic_indices(nvars, max_level, pnorm, bkd)
    basis = MultiIndexBasis(bases_1d, bkd, indices)
    return SUPN(basis, width, bkd, nqoi=nqoi)

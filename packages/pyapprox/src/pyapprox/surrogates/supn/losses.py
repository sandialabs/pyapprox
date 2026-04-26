"""Loss functions for SUPN fitting.

Provides MSE loss with analytical gradient and Hessian-vector product (HVP)
for use with trust-region Newton-CG optimization, matching the training
procedure recommended in Morrow et al. (2025, Section 4).
"""

from typing import Generic, Optional

from pyapprox.surrogates.supn.supn import SUPN
from pyapprox.util.backends.protocols import Array, Backend


class SUPNMSELoss(Generic[Array]):
    """Mean squared error loss for SUPN fitting.

    L(params) = (1/2K) ||f_params(X) - Y||_F^2

    Provides analytical jacobian and HVP for use with
    ScipyTrustConstrOptimizer (trust-region Newton-CG).

    The basis matrix B = basis(train_samples) is cached at construction
    since it is independent of trainable parameters. Intermediate quantities
    (H, S, residual) are cached between __call__, jacobian, and hvp calls
    since scipy calls all three with the same params.

    Parameters
    ----------
    surrogate : SUPN[Array]
        SUPN surrogate to fit.
    train_samples : Array
        Training samples. Shape: (nvars, nsamples)
    train_values : Array
        Target values. Shape: (nqoi, nsamples)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        surrogate: SUPN[Array],
        train_samples: Array,
        train_values: Array,
        bkd: Backend[Array],
    ) -> None:
        self._surrogate = surrogate
        self._train_values = train_values  # (Q, K)
        self._bkd = bkd
        self._nsamples = train_samples.shape[1]
        self._nqoi = surrogate.nqoi()
        self._width = surrogate.width()
        self._nterms = surrogate.nterms()

        # Cache basis matrix (independent of trainable params)
        self._B = surrogate.basis()(train_samples)  # (K, M)

        # Param cache: scipy calls fun(x) then jac(x) then hessp(x,v)
        self._cached_params: Optional[Array] = None
        self._cached_H: Optional[Array] = None  # tanh(Z), (N, K)
        self._cached_S: Optional[Array] = None  # sech^2(Z), (N, K)
        self._cached_residual: Optional[Array] = None  # F - Y, (Q, K)
        self._cached_C: Optional[Array] = None  # outer coefs, (Q, N)
        self._cached_c_flat: Optional[Array] = None  # for nqoi=1, (N,)

    def _ensure_cached(self, params_flat: Array) -> None:
        """Recompute cached quantities if params changed."""
        requires_grad = getattr(params_flat, "requires_grad", False)
        if not requires_grad and (
            self._cached_params is not None
            and self._bkd.allclose(
                params_flat, self._cached_params, rtol=0.0, atol=0.0
            )
        ):
            return

        bkd = self._bkd
        if not requires_grad:
            self._cached_params = bkd.copy(params_flat)
        else:
            self._cached_params = None

        supn = self._surrogate.with_params(params_flat)
        self._cached_C = supn.outer_coefs()  # (Q, N)

        Z = bkd.dot(supn.inner_coefs(), self._B.T)  # (N, K)
        self._cached_H = bkd.tanh(Z)  # (N, K)
        self._cached_S = 1.0 - self._cached_H ** 2  # (N, K)

        pred = bkd.dot(self._cached_C, self._cached_H)  # (Q, K)
        self._cached_residual = pred - self._train_values  # (Q, K)

        if self._nqoi == 1:
            self._cached_c_flat = self._cached_C[0, :]  # (N,)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of parameters to optimize."""
        return self._surrogate.nparams()

    def nqoi(self) -> int:
        """Loss is scalar."""
        return 1

    def __call__(self, params: Array) -> Array:
        """Compute MSE loss.

        Parameters
        ----------
        params : Array
            SUPN parameters. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Loss value. Shape: (1, 1)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(params)
        self._ensure_cached(params_flat)

        mse = (
            0.5 * bkd.sum(self._cached_residual ** 2) / self._nsamples
        )
        return bkd.reshape(mse, (1, 1))

    def jacobian(self, params: Array) -> Array:
        """Compute gradient of MSE loss w.r.t. params.

        For nqoi=1:
            dL/dc_n = (1/K) sum_k r_k * H[n,k]
            dL/da_{n,j} = (1/K) sum_k r_k * c_n * S[n,k] * B[k,j]

        For multi-QoI:
            dL/dc_{q,n} = (1/K) sum_k R[q,k] * H[n,k]
            dL/da_{n,j} = (1/K) sum_k sum_q R[q,k] * C[q,n] * S[n,k] * B[k,j]

        Parameters
        ----------
        params : Array
            SUPN parameters. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Gradient. Shape: (1, P)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(params)
        self._ensure_cached(params_flat)

        H = self._cached_H  # (N, K)
        S = self._cached_S  # (N, K)
        R = self._cached_residual  # (Q, K)
        C = self._cached_C  # (Q, N)
        B = self._B  # (K, M)
        K = self._nsamples

        if self._nqoi == 1:
            r = R[0, :]  # (K,)
            c = self._cached_c_flat  # (N,)

            # grad_c[n] = (1/K) * r @ H[n,:].T
            grad_c = bkd.dot(H, r) / K  # (N,)

            # grad_A[n,j] = (1/K) * c[n] * (r * S[n,:]) @ B[:,j]
            # weighted[n,k] = c[n] * r[k] * S[n,k]
            weighted = c[:, None] * (r[None, :] * S)  # (N, K)
            grad_A = bkd.dot(weighted, B) / K  # (N, M)

            grad = bkd.hstack([grad_c, bkd.flatten(grad_A)])
        else:
            # grad_C[q,n] = (1/K) * R[q,:] @ H[n,:].T
            grad_C = bkd.dot(R, H.T) / K  # (Q, N)

            # grad_A[n,j] = (1/K) * sum_q C[q,n] * (R[q,:] * S[n,:]) @ B[:,j]
            # weighted[n,k] = sum_q C[q,n] * R[q,k] * S[n,k]
            weighted = bkd.einsum("qn,qk,nk->nk", C, R, S)  # (N, K)
            grad_A = bkd.dot(weighted, B) / K  # (N, M)

            grad = bkd.hstack([
                bkd.flatten(grad_C), bkd.flatten(grad_A)
            ])

        return bkd.reshape(grad, (1, -1))

    def hvp(self, params: Array, direction: Array) -> Array:
        """Exact Hessian-vector product of MSE loss.

        H*v = (1/K) * [J.T @ (J @ v)  +  second_order_term]

        where J is the (K, P) surrogate jacobian w.r.t. params.

        The second-order term accounts for curvature of tanh through
        the residual-weighted second derivatives of f w.r.t. params.

        Parameters
        ----------
        params : Array
            SUPN parameters. Shape: (P, 1) or (P,)
        direction : Array
            Direction vector. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (P, 1)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(params)
        self._ensure_cached(params_flat)

        v = bkd.flatten(direction)  # (P,)
        N = self._width
        M = self._nterms
        Q = self._nqoi
        K = self._nsamples
        H = self._cached_H  # (N, K)
        S = self._cached_S  # (N, K)
        B = self._B  # (K, M)

        P_outer = Q * N

        if Q == 1:
            return self._hvp_single_qoi(v, N, M, K, P_outer, H, S, B)
        else:
            return self._hvp_multi_qoi(v, N, M, K, Q, P_outer, H, S, B)

    def _hvp_single_qoi(
        self,
        v: Array,
        N: int,
        M: int,
        K: int,
        P_outer: int,
        H: Array,
        S: Array,
        B: Array,
    ) -> Array:
        """HVP for nqoi=1. See plan for derivation.

        Gauss-Newton term: (1/K) J.T @ (J @ v)
        Second-order term: (1/K) sum_k r_k * d^2f/(dtheta dp) * v
        """
        bkd = self._bkd
        r = self._cached_residual[0, :]  # (K,)
        c = self._cached_c_flat  # (N,)

        # Unpack direction
        v_c = v[:N]  # (N,)
        v_A = bkd.reshape(v[P_outer:], (N, M))  # (N, M)

        # W[n,k] = sum_j v_A[n,j] * B[k,j] = (v_A @ B.T)[n,k]
        W = bkd.dot(v_A, B.T)  # (N, K)

        # === Gauss-Newton term: J.T @ (J @ v) ===
        # J @ v = v_c @ H + c @ (S * W)
        # v_c @ H: (N,) @ (N, K) but we need sum_n v_c[n]*H[n,k]
        Jv = bkd.dot(v_c, H) + bkd.dot(c, S * W)  # (K,)

        # J.T @ Jv:
        #   (J.T @ Jv)_{c_n} = sum_k Jv[k] * H[n,k]
        gn_c = bkd.dot(H, Jv)  # (N,)

        #   (J.T @ Jv)_{a_{n,j}} = sum_k Jv[k] * c[n] * S[n,k] * B[k,j]
        gn_A = bkd.dot(c[:, None] * S * Jv[None, :], B)  # (N, M)

        # === Second-order term ===
        # T2_{c_m} = sum_k r_k * S[m,k] * W[m,k]
        rS = r[None, :] * S  # (N, K)
        T2_c = bkd.sum(rS * W, axis=1)  # (N,)

        # T2_{a_{m,l}} = sum_k r_k * S[m,k] * B[k,l] * U[m,k]
        # where U[m,k] = v_c[m] - 2*c[m]*H[m,k]*W[m,k]
        U = v_c[:, None] - 2.0 * c[:, None] * H * W  # (N, K)
        T2_A = bkd.dot(rS * U, B)  # (N, M)

        # Combine
        hvp_c = (gn_c + T2_c) / K
        hvp_A = (gn_A + T2_A) / K

        result = bkd.hstack([hvp_c, bkd.flatten(hvp_A)])
        return bkd.reshape(result, (-1, 1))

    def _hvp_multi_qoi(
        self,
        v: Array,
        N: int,
        M: int,
        K: int,
        Q: int,
        P_outer: int,
        H: Array,
        S: Array,
        B: Array,
    ) -> Array:
        """HVP for multi-QoI. Same structure as single-QoI but with
        outer coefficients indexed by QoI.

        Parameter layout: [C.flatten() (Q*N), A.flatten() (N*M)]
        """
        bkd = self._bkd
        R = self._cached_residual  # (Q, K)
        C = self._cached_C  # (Q, N)

        # Unpack direction
        v_C = bkd.reshape(v[:P_outer], (Q, N))  # (Q, N)
        v_A = bkd.reshape(v[P_outer:], (N, M))  # (N, M)

        W = bkd.dot(v_A, B.T)  # (N, K)

        # === Gauss-Newton term ===
        # J @ v for each QoI q:
        # Jv[q,k] = sum_n v_C[q,n]*H[n,k] + sum_n C[q,n]*S[n,k]*W[n,k]
        Jv = bkd.dot(v_C, H) + bkd.dot(C, S * W)  # (Q, K)

        # J.T @ Jv for outer coefs:
        # gn_C[q,n] = sum_k Jv[q,k] * H[n,k]
        gn_C = bkd.dot(Jv, H.T)  # (Q, N)

        # J.T @ Jv for inner coefs:
        # gn_A[n,j] = sum_q sum_k Jv[q,k] * C[q,n] * S[n,k] * B[k,j]
        weighted_Jv = bkd.einsum("qk,qn,nk->nk", Jv, C, S)  # (N, K)
        gn_A = bkd.dot(weighted_Jv, B)  # (N, M)

        # === Second-order term ===
        # T2_C[q,n] = sum_k R[q,k] * S[n,k] * W[n,k]
        # = sum_k (R[q,k] * S[n,k] * W[n,k])
        T2_C = bkd.einsum("qk,nk,nk->qn", R, S, W)  # (Q, N)

        # T2_A[n,j] = sum_q sum_k R[q,k] * S[n,k] * B[k,j] * U_q[n,k]
        # U_q[n,k] = v_C[q,n] - 2*C[q,n]*H[n,k]*W[n,k]
        # Summed over q:
        # T2_A[n,j] = sum_k S[n,k] * B[k,j] * sum_q R[q,k] * U_q[n,k]
        # sum_q R[q,k]*U_q[n,k] = sum_q R[q,k]*v_C[q,n]
        #                        - 2*H[n,k]*W[n,k]*sum_q R[q,k]*C[q,n]
        Rv = bkd.einsum("qk,qn->nk", R, v_C)  # (N, K)
        RC = bkd.einsum("qk,qn->nk", R, C)  # (N, K)
        sum_RU = Rv - 2.0 * H * W * RC  # (N, K)
        T2_A = bkd.dot(S * sum_RU, B)  # (N, M)

        # Combine
        hvp_C = (gn_C + T2_C) / K
        hvp_A = (gn_A + T2_A) / K

        result = bkd.hstack([bkd.flatten(hvp_C), bkd.flatten(hvp_A)])
        return bkd.reshape(result, (-1, 1))

    def __repr__(self) -> str:
        return (
            f"SUPNMSELoss(nvars={self.nvars()}, "
            f"nsamples={self._nsamples}, nqoi={self._nqoi})"
        )

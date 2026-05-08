"""Loss functions for SUPN fitting.

Provides MSE loss with analytical gradient and Hessian-vector product (HVP)
for use with trust-region Newton-CG optimization, matching the training
procedure recommended in Morrow et al. (2025, Section 4).
"""

from typing import Generic

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
        # with the same parameter values. The wrapper chain creates new
        # array objects each call, so we cache by data pointer + first/last
        # element for O(1) staleness check.
        self._cached_sentinel: float = float("nan")
        self._cached_H: Array = bkd.zeros((0,))
        self._cached_S: Array = bkd.zeros((0,))
        self._cached_residual: Array = bkd.zeros((0,))
        self._cached_C: Array = bkd.zeros((0,))
        self._cached_c_flat: Array = bkd.zeros((0,))

        # Eagerly populate the cache from the surrogate's initial params.
        self._ensure_cached(surrogate._flatten_params())

    def _ensure_cached(self, params_flat: Array) -> None:
        """Recompute cached quantities if params changed.

        The trust-region CG inner solver calls hessp(x, v) many times
        with the same parameter values but different directions.  The
        wrapper chain (trust_constr → numpy wrapper → loss) creates new
        array objects on each call, so ``id()`` is useless.  Instead we
        compare a cheap sentinel: first_element + last_element.  A
        sentinel match means the params almost certainly haven't changed
        (same Newton iterate); a mismatch triggers recomputation.
        """
        requires_grad = getattr(params_flat, "requires_grad", False)
        if not requires_grad:
            sentinel = float(params_flat[0]) + float(params_flat[-1])
            if sentinel == self._cached_sentinel:
                return
            self._cached_sentinel = sentinel
        else:
            self._cached_sentinel = float("nan")

        bkd = self._bkd

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

    def __call__(self, samples: Array) -> Array:
        """Compute MSE loss.

        Parameters
        ----------
        samples : Array
            SUPN parameters. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Loss value. Shape: (1, 1)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(samples)
        self._ensure_cached(params_flat)

        mse = (
            0.5 * bkd.sum(self._cached_residual ** 2) / self._nsamples
        )
        return bkd.reshape(mse, (1, 1))

    def jacobian(self, sample: Array) -> Array:
        """Compute gradient of MSE loss w.r.t. params.

        For nqoi=1:
            dL/dc_n = (1/K) sum_k r_k * H[n,k]
            dL/da_{n,j} = (1/K) sum_k r_k * c_n * S[n,k] * B[k,j]

        For multi-QoI:
            dL/dc_{q,n} = (1/K) sum_k R[q,k] * H[n,k]
            dL/da_{n,j} = (1/K) sum_k sum_q R[q,k] * C[q,n] * S[n,k] * B[k,j]

        Parameters
        ----------
        sample : Array
            SUPN parameters. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Gradient. Shape: (1, P)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(sample)
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

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Exact Hessian-vector product of MSE loss.

        H*v = (1/K) * [J.T @ (J @ v)  +  second_order_term]

        where J is the (K, P) surrogate jacobian w.r.t. params.

        The second-order term accounts for curvature of tanh through
        the residual-weighted second derivatives of f w.r.t. params.

        Parameters
        ----------
        sample : Array
            SUPN parameters. Shape: (P, 1) or (P,)
        vec : Array
            Direction vector. Shape: (P, 1) or (P,)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (P, 1)
        """
        bkd = self._bkd
        params_flat = bkd.flatten(sample)
        self._ensure_cached(params_flat)

        v = bkd.flatten(vec)  # (P,)
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
        """HVP for nqoi=1.

        Gauss-Newton term: (1/K) J.T @ (J @ v)
        Second-order term: (1/K) sum_k r_k * d^2f/(dtheta dp) * v

        Fused to 3 dot calls (down from 6) by combining the two
        (N,K)@(K,M) matmuls and the two (N,K)@(K,) products.
        """
        bkd = self._bkd
        r = self._cached_residual[0, :]  # (K,)
        c = self._cached_c_flat  # (N,)

        # Unpack direction
        v_c = v[:N]  # (N,)
        v_A = bkd.reshape(v[P_outer:], (N, M))  # (N, M)

        # --- dot 1: W = v_A @ B.T  (N,M)@(M,K) → (N,K) ---
        W = bkd.dot(v_A, B.T)

        # Shared intermediates (elementwise, cheap)
        SW = S * W  # (N, K)
        rS = r[None, :] * S  # (N, K)

        # --- Jv (uses einsum to avoid 2 separate dot calls) ---
        # Jv[k] = sum_n v_c[n]*H[n,k] + sum_n c[n]*SW[n,k]
        Jv = bkd.einsum("n,nk->k", v_c, H) + bkd.einsum(
            "n,nk->k", c, SW
        )  # (K,)

        # --- hvp_c: fuse gn_c + T2_c into one product ---
        # gn_c[n] = sum_k Jv[k]*H[n,k]
        # T2_c[n] = sum_k r[k]*S[n,k]*W[n,k]
        # Combined: hvp_c[n] = (1/K) * H[n,:] @ Jv + (1/K) * sum(rS*W)
        # = (1/K) * sum_k (Jv[k]*H[n,k] + r[k]*S[n,k]*W[n,k])
        hvp_c_K = bkd.einsum("nk,k->n", H, Jv) + bkd.sum(
            rS * W, axis=1
        )  # (N,)

        # --- dot 2+3 fused: hvp_A from one (N,K)@(K,M) matmul ---
        # gn_A[n,j] = sum_k c[n]*S[n,k]*Jv[k]*B[k,j]
        # T2_A[n,j] = sum_k r[k]*S[n,k]*U[n,k]*B[k,j]
        #   where U[n,k] = v_c[n] - 2*c[n]*H[n,k]*W[n,k]
        # Combined[n,k] = c[n]*S[n,k]*Jv[k] + r[k]*S[n,k]*U[n,k]
        #               = S[n,k] * (c[n]*Jv[k] + r[k]*(v_c[n] - 2*c[n]*H[n,k]*W[n,k]))
        combined = S * (
            c[:, None] * Jv[None, :]
            + r[None, :] * (v_c[:, None] - 2.0 * c[:, None] * H * W)
        )  # (N, K)
        hvp_A_K = bkd.dot(combined, B)  # (N, M) --- single matmul

        # Scale and pack
        result = bkd.hstack([hvp_c_K / K, bkd.flatten(hvp_A_K) / K])
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

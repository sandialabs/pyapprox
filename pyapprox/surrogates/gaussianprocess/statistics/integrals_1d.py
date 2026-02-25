"""
1D kernel integral computations for GP statistics.

This module provides functions to compute the 1D kernel integrals that
are the building blocks for multidimensional GP statistics with separable
(product) kernels.

For a product kernel C(x, z) = prod_k C_k(x_k, z_k), the multidimensional
integrals factor as products of 1D integrals computed by these functions.
"""

from typing import Callable

from pyapprox.util.backends.protocols import Array, Backend


def compute_tau_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D tau vector for a single dimension.

    tau_k,i = integral C_k(z_k, x_k^(i)) rho_k(z_k) dz_k
            = sum_j w_j C_k(quad_j, x_k^(i))

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        tau vector for dimension k, shape (N,).
    """
    # K_quad_train shape: (nquad, N)
    K_quad_train = kernel_1d(quad_samples, train_samples_1d)

    # Weighted sum over quadrature points: sum_j w_j K(quad_j, train_i)
    # quad_weights shape: (nquad,), K_quad_train shape: (nquad, N)
    # Result shape: (N,)
    tau = quad_weights @ K_quad_train

    return tau


def compute_P_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D P matrix for a single dimension.

    P_k,ij = integral C_k(z_k, x_k^(i)) C_k(z_k, x_k^(j)) rho_k(z_k) dz_k
           = sum_m w_m C_k(quad_m, x_k^(i)) C_k(quad_m, x_k^(j))

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        P matrix for dimension k, shape (N, N).
    """
    # K_quad_train shape: (nquad, N)
    K_quad_train = kernel_1d(quad_samples, train_samples_1d)

    # Weight the kernel values: sqrt(w_m) * K(quad_m, train_i)
    # This allows us to compute P = K^T W K as (sqrt(W) K)^T (sqrt(W) K)
    sqrt_weights = bkd.sqrt(quad_weights)  # (nquad,)
    weighted_K = sqrt_weights[:, None] * K_quad_train  # (nquad, N)

    # P_ij = sum_m w_m K(quad_m, train_i) K(quad_m, train_j)
    #      = (weighted_K)^T @ (weighted_K)
    P = weighted_K.T @ weighted_K  # (N, N)

    return P


def compute_u_1d(
    quad_samples: Array,
    quad_weights: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D u scalar for a single dimension.

    u_k = integral integral C_k(z_k, w_k) rho_k(z_k) rho_k(w_k) dz_k dw_k
        = sum_i sum_j w_i w_j C_k(quad_i, quad_j)

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        u scalar for dimension k (0-dimensional array or scalar).
    """
    # K_quad_quad shape: (nquad, nquad)
    K_quad_quad = kernel_1d(quad_samples, quad_samples)

    # u = sum_i sum_j w_i w_j K(quad_i, quad_j)
    #   = w^T K w
    u = quad_weights @ K_quad_quad @ quad_weights

    return u


def compute_nu_1d(
    quad_samples: Array,
    quad_weights: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D nu scalar for Var[gamma] computation.

    nu_k = integral integral C_k(z_k, w_k)^2 rho_k(z_k) rho_k(w_k) dz_k dw_k
         = sum_i sum_j w_i w_j C_k(quad_i, quad_j)^2

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        nu scalar for dimension k (0-dimensional array or scalar).
    """
    # K_quad_quad shape: (nquad, nquad)
    K_quad_quad = kernel_1d(quad_samples, quad_samples)

    # nu = sum_i sum_j w_i w_j K(quad_i, quad_j)^2
    #    = w^T (K * K) w  where * is element-wise
    K_squared = K_quad_quad * K_quad_quad
    nu = quad_weights @ K_squared @ quad_weights

    return nu


def compute_lambda_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D lambda vector for Var[gamma] computation.

    lambda_k,i = integral C_k(z_k, z_k) C_k(z_k, x_k^(i)) rho_k(z_k) dz_k
               = sum_j w_j C_k(quad_j, quad_j) C_k(quad_j, x_k^(i))

    Note: This involves the kernel diagonal C_k(z_k, z_k) multiplied by
    the kernel evaluated between the quadrature point and training point.

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        lambda vector for dimension k, shape (N,).
    """
    # Get kernel diagonal at quadrature points: C_k(quad_j, quad_j)
    # K_quad_quad shape: (nquad, nquad), we need diagonal
    K_quad_quad = kernel_1d(quad_samples, quad_samples)
    K_diag = bkd.diag(K_quad_quad)  # (nquad,)

    # K_quad_train shape: (nquad, N)
    K_quad_train = kernel_1d(quad_samples, train_samples_1d)

    # lambda_i = sum_j w_j K(quad_j, quad_j) K(quad_j, train_i)
    # weighted_diag shape: (nquad,)
    weighted_diag = quad_weights * K_diag

    # Result: weighted_diag @ K_quad_train -> (N,)
    lambda_vec = weighted_diag @ K_quad_train

    return lambda_vec


def compute_Pi_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D Pi matrix for Var[gamma] computation.

    Pi_k,ij = integral integral C_k(x_k, x_k^(i)) C_k(x_k, z_k) C_k(z_k, x_k^(j))
              rho_k(x_k) rho_k(z_k) dx_k dz_k
            = sum_m sum_n w_m w_n C_k(quad_m, train_i) C_k(quad_m, quad_n) C_k(quad_n,
            train_j)

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        Pi matrix for dimension k, shape (N, N).
    """
    # K_quad_train shape: (nquad, N)
    K_quad_train = kernel_1d(quad_samples, train_samples_1d)

    # K_quad_quad shape: (nquad, nquad)
    K_quad_quad = kernel_1d(quad_samples, quad_samples)

    # Pi_ij = sum_m sum_n w_m w_n K(quad_m, train_i) K(quad_m, quad_n) K(quad_n,
    # train_j)
    #
    # Let W = diag(w), then:
    # Pi = K_quad_train^T @ W @ K_quad_quad @ W @ K_quad_train
    # = (W^{1/2} K_quad_train)^T @ (W^{1/2} K_quad_quad W^{1/2}) @ (W^{1/2}
    # K_quad_train)

    sqrt_weights = bkd.sqrt(quad_weights)  # (nquad,)

    # Weight the kernel matrices
    weighted_K_train = sqrt_weights[:, None] * K_quad_train  # (nquad, N)
    weighted_K_quad = (
        sqrt_weights[:, None] * K_quad_quad * sqrt_weights[None, :]
    )  # (nquad, nquad)

    # Pi = weighted_K_train^T @ weighted_K_quad @ weighted_K_train
    Pi = weighted_K_train.T @ weighted_K_quad @ weighted_K_train  # (N, N)

    return Pi


def compute_xi1_1d(
    quad_samples: Array,
    quad_weights: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D xi1 scalar for Var[gamma] computation.

    xi1_k = integral integral integral C_k(w_k, x_k) C_k(w_k, z_k)
            rho_k(w_k) rho_k(x_k) rho_k(z_k) dw_k dx_k dz_k
          = sum_i sum_j sum_m w_i w_j w_m C_k(quad_i, quad_j) C_k(quad_i, quad_m)

    This can be rewritten as:
    xi1_k = sum_i w_i [sum_j w_j C_k(quad_i, quad_j)]^2
          = sum_i w_i tau_i^2

    where tau_i = sum_j w_j C_k(quad_i, quad_j) is the tau for quadrature
    points (not training points).

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        xi1 scalar for dimension k (0-dimensional array or scalar).
    """
    # K_quad_quad shape: (nquad, nquad)
    K_quad_quad = kernel_1d(quad_samples, quad_samples)

    # tau_quad_i = sum_j w_j K(quad_i, quad_j) = K @ w
    tau_quad = K_quad_quad @ quad_weights  # (nquad,)

    # xi1 = sum_i w_i tau_quad_i^2 = w^T (tau_quad * tau_quad)
    xi1 = quad_weights @ (tau_quad * tau_quad)

    return xi1


def compute_conditional_P_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D conditional P matrix for sensitivity analysis.

    P̃_{k,ij} = ∫∫ C_k(x, x^(i)) C_k(z, x^(j)) ρ(x) ρ(z) dx dz
             = τ_{k,i} · τ_{k,j}  (outer product)

    Mathematical derivation:
    - The double integral factors because x and z are INDEPENDENT
    - Unlike standard P where the SAME point appears in both kernels
    - Result is a rank-1 matrix: P̃ = τ τᵀ

    This is used for dimensions that are INTEGRATED OUT (not conditioned on)
    in the conditional variance computation for sensitivity analysis.

    Comparison with standard P:
    - Standard P_k: ∫ C(x, x^(i)) C(x, x^(j)) ρ(x) dx (single integration point)
    - Conditional P̃_k: ∫∫ C(x, x^(i)) C(z, x^(j)) ρ(x)ρ(z) dx dz = τ_i · τ_j

    The key difference is that in P̃, the two kernel evaluations use INDEPENDENT
    integration points (x and z), so the integral factors. In standard P, the
    SAME point x appears in both kernels, so it doesn't factor.

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        Conditional P matrix for dimension k, shape (N, N).
        This is a rank-1 matrix equal to τ_k τ_k^T.
    """
    # Compute tau for this dimension
    tau = compute_tau_1d(quad_samples, quad_weights, train_samples_1d, kernel_1d, bkd)
    # P̃ = τ τᵀ (rank-1 outer product)
    return bkd.outer(tau, tau)


def compute_Gamma_1d(
    quad_samples: Array,
    quad_weights: Array,
    train_samples_1d: Array,
    kernel_1d: Callable[[Array, Array], Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute the 1D Gamma integral for variance of variance.

    Gamma_k,i = integral integral C_k(x_k^(i), z_k) C_k(z_k, v_k)
                rho_k(z_k) rho_k(v_k) dz_k dv_k

    This is a double integral where one argument is the training point
    and the other two are integration variables.

    Parameters
    ----------
    quad_samples : Array
        Quadrature points, shape (1, nquad).
    quad_weights : Array
        Quadrature weights, shape (nquad,).
    train_samples_1d : Array
        Training points for dimension k, shape (1, N).
    kernel_1d : Callable[[Array, Array], Array]
        1D kernel function: kernel(x1, x2) -> (n1, n2).
    bkd : Backend[Array]
        Backend for numerical operations.

    Returns
    -------
    Array
        Gamma values for each training point, shape (N,).
    """
    # K_train_quad[i, j] = k(x_i, z_j), shape (N, nquad)
    K_train_quad = kernel_1d(train_samples_1d, quad_samples)

    # K_quad_quad[j, l] = k(z_j, v_l), shape (nquad, nquad)
    K_quad_quad = kernel_1d(quad_samples, quad_samples)

    # Gamma_i = sum_j sum_l w_j w_l k(x_i, z_j) k(z_j, v_l)
    #         = sum_j w_j k(x_i, z_j) [sum_l w_l k(z_j, v_l)]
    #         = sum_j w_j k(x_i, z_j) (K_quad_quad @ w)_j
    #         = K_train_quad @ diag(w) @ K_quad_quad @ w
    inner = K_quad_quad @ quad_weights  # shape (nquad,)
    Gamma = K_train_quad @ (quad_weights * inner)  # shape (N,)

    return Gamma

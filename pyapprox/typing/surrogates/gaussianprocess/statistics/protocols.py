"""
Protocol definitions for GP statistics computations.

This module defines protocols for computing integrals needed for GP statistics
with separable (product) kernels. For a product kernel:

    C(x, z) = prod_k C_k(x_k, z_k)

the multidimensional integrals factor into products of 1D integrals, which
are computed by the KernelIntegralCalculatorProtocol.
"""

from typing import Protocol, runtime_checkable, Generic
from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class KernelIntegralCalculatorProtocol(Protocol, Generic[Array]):
    """
    Protocol for computing kernel integrals needed for GP statistics.

    This protocol defines the interface for computing the 1D kernel integrals
    that, when combined via products, give the full multidimensional integrals
    needed for E[f], Var[f], and sensitivity indices.

    For a product kernel C(x, z) = prod_k C_k(x_k, z_k), the required integrals
    factor as products of 1D integrals. This protocol computes those 1D factors.

    Required Methods for E[f] and Var[f]
    ------------------------------------
    tau : Array
        Shape (N,). The vector tau_i = integral C(x, x^(i)) rho(x) dx
        where x^(i) are training points and rho is the input density.
        For separable kernels: tau = prod_k tau_k

    P : Array
        Shape (N, N). The matrix P_ij = integral C(x, x^(i)) C(x, x^(j)) rho(x) dx
        For separable kernels: P = prod_k P_k

    u : Array
        Scalar. The double integral u = integral integral C(x, z) rho(x) rho(z) dx dz
        For separable kernels: u = prod_k u_k

    Extended Methods for Var[gamma] (variance of posterior variance)
    ----------------------------------------------------------------
    nu : Array
        Scalar. nu = integral integral C(x, z)^2 rho(x) rho(z) dx dz
        For separable kernels: nu = prod_k nu_k

    lambda_vec : Array
        Shape (N,). lambda_i = integral C(z, z) C(z, x^(i)) rho(z) dz
        For separable kernels: lambda = prod_k lambda_k

    Pi : Array
        Shape (N, N). Pi_ij = integral integral C(x, x^(i)) C(x, z) C(z, x^(j)) rho(x) rho(z) dx dz
        For separable kernels: Pi = prod_k Pi_k

    xi1 : Array
        Scalar. xi1 = integral integral integral C(w, x) C(w, z) rho(w) rho(x) rho(z) dw dx dz
        For separable kernels: xi1 = prod_k xi1_k

    Sensitivity Methods
    -------------------
    conditional_P : Array
        P matrix computed with variables in 'index' fixed.
        For sensitivity index computations.

    conditional_u : Array
        u scalar computed with variables in 'index' fixed.
        For sensitivity index computations.

    Notes
    -----
    The implementation should use numerical quadrature to approximate these
    integrals. The accuracy depends on the quadrature rule and the smoothness
    of the kernel.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical operations.

        Returns
        -------
        Backend[Array]
            The backend instance (e.g., NumpyBkd or TorchBkd).
        """
        ...

    def tau_C(self) -> Array:
        """
        Compute the tau vector using the correlation kernel C (unit variance).

        tau_C_i = integral C(x, x^(i)) rho(x) dx

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        ...

    def tau_K(self) -> Array:
        """
        Compute the tau vector using the full kernel K = s² C.

        tau_K = s² tau_C

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        ...

    def P(self) -> Array:
        """
        Compute the P matrix for Var[f] computation.

        P_ij = integral C(x, x^(i)) C(x, x^(j)) rho(x) dx

        This is the integral of the product of two kernel evaluations,
        weighted by the input density.

        For separable kernels, the full P is: P = prod_k P_k
        where P_k is the 1D factor for dimension k.

        Returns
        -------
        Array
            Shape (N, N) where N is the number of training points.
            This matrix is symmetric positive semi-definite.
        """
        ...

    def u(self) -> Array:
        """
        Compute the u scalar for prior variance.

        u = integral integral C(x, z) rho(x) rho(z) dx dz

        This is the double integral of the kernel weighted by the
        input density in both arguments.

        For separable kernels, the full u is: u = prod_k u_k
        where u_k is the 1D factor for dimension k.

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        ...

    def nu(self) -> Array:
        """
        Compute the nu scalar for Var[gamma] computation.

        nu = integral integral C(x, z)^2 rho(x) rho(z) dx dz

        This is the double integral of the squared kernel weighted by
        the input density.

        For separable kernels, the full nu is: nu = prod_k nu_k

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        ...

    def lambda_vec(self) -> Array:
        """
        Compute the lambda vector for Var[gamma] computation.

        lambda_i = integral C(z, z) C(z, x^(i)) rho(z) dz

        This involves the kernel diagonal C(z, z) multiplied by C(z, x^(i)).

        For separable kernels, the full lambda is: lambda = prod_k lambda_k

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        ...

    def Pi(self) -> Array:
        """
        Compute the Pi matrix for Var[gamma] computation.

        Pi_ij = integral integral C(x, x^(i)) C(x, z) C(z, x^(j)) rho(x) rho(z) dx dz

        This is a triple product of kernel evaluations.

        For separable kernels, the full Pi is: Pi = prod_k Pi_k

        Returns
        -------
        Array
            Shape (N, N) where N is the number of training points.
        """
        ...

    def xi1(self) -> Array:
        """
        Compute the xi1 scalar for Var[gamma] computation.

        xi1 = integral integral integral C(w, x) C(w, z) rho(w) rho(x) rho(z) dw dx dz

        This is a triple integral involving the kernel evaluated at two pairs.

        For separable kernels, the full xi1 is: xi1 = prod_k xi1_k

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        ...

    def Gamma(self) -> Array:
        """
        Compute the Gamma vector for Var[gamma] computation.

        Gamma_i = integral integral C(x^(i), z) C(z, v) rho(z) rho(v) dz dv

        This is a double integral where one argument is a training point and
        the other two are integration variables.

        For separable kernels, the full Gamma is: Gamma = prod_k Gamma_k

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        ...

    def conditional_P(self, index: Array) -> Array:
        """
        Compute P matrix with variables in 'index' fixed.

        This is used for computing sensitivity indices. The integral is
        computed only over variables NOT in the index set, with the
        indexed variables treated as fixed.

        Parameters
        ----------
        index : Array
            Indices of variables to treat as fixed (complement set).
            Shape (n_fixed,) with integer values in [0, nvars).

        Returns
        -------
        Array
            Shape (N, N) where N is the number of training points.
        """
        ...

    def conditional_u(self, index: Array) -> Array:
        """
        Compute u scalar with variables in 'index' fixed.

        This is used for computing sensitivity indices. The integral is
        computed only over variables NOT in the index set.

        Parameters
        ----------
        index : Array
            Indices of variables to treat as fixed (complement set).
            Shape (n_fixed,) with integer values in [0, nvars).

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        ...

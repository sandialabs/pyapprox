"""
KL divergence functions for copulas.

Provides:
- gaussian_copula_kl_divergence: KL between two Gaussian copulas
"""

from typing import TYPE_CHECKING

from pyapprox.typing.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.typing.probability.copula.gaussian import GaussianCopula


def gaussian_copula_kl_divergence(
    p: "GaussianCopula[Array]",
    q: "GaussianCopula[Array]",
) -> Array:
    """
    Compute KL(p || q) for two Gaussian copulas.

    Reduces to the KL divergence between two zero-mean multivariate
    normals with the corresponding correlation matrices:

        KL(c_p || c_q) = 0.5 * (tr(Sigma_q^{-1} Sigma_p) - d
                                  + log|Sigma_q| - log|Sigma_p|)

    Parameters
    ----------
    p : GaussianCopula
        First Gaussian copula (the "true" distribution).
    q : GaussianCopula
        Second Gaussian copula (the approximating distribution).

    Returns
    -------
    Array
        KL divergence value (scalar Array, preserves autograd).
    """
    return p.kl_divergence(q)

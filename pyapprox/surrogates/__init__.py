from pyapprox.surrogates.orthopoly.quadrature import (
    get_gauss_quadrature_rule_from_marginal
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    tensor_product_barycentric_lagrange_interpolation
)

__all__ = ["get_gauss_quadrature_rule_from_marginal",
           "tensor_product_barycentric_lagrange_interpolation"]

from pyapprox.surrogates.orthopoly.quadrature import (
    get_gauss_quadrature_rule_from_marginal
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    tensor_product_barycentric_lagrange_interpolation,
)
from pyapprox.surrogates.interp.tensorprod import (
    tensor_product_piecewise_polynomial_interpolation,
    get_tensor_product_piecewise_polynomial_quadrature_rule
)
from pyapprox.surrogates.approximate import adaptive_approximate, approximate


__all__ = ["get_gauss_quadrature_rule_from_marginal",
           "tensor_product_barycentric_lagrange_interpolation",
           "tensor_product_piecewise_polynomial_interpolation",
           "get_tensor_product_piecewise_polynomial_quadrature_rule",
           "approximate", "adaptive_approximate"]

import numpy as np
from warnings import warn

from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence, hermite_recurrence, krawtchouk_recurrence,
    hahn_recurrence, charlier_recurrence
)
from pyapprox.surrogates.orthopoly.numeric_orthonormal_recursions import (
    predictor_corrector, get_function_independent_vars_recursion_coefficients,
    get_product_independent_vars_recursion_coefficients, lanczos
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d
)
from pyapprox.variables.marginals import (
    get_distribution_info, is_continuous_variable,
    transform_scale_parameters, is_bounded_continuous_variable,
    is_bounded_discrete_variable, get_probability_masses, get_pdf
)


# There is a one-to-one correspondence in these two lists
askey_poly_names = ["legendre", "hermite", "jacobi", "charlier",
                    "krawtchouk", "hahn"][:-2]
askey_variable_names = ["uniform", "norm", "beta", "poisson",
                        "binom", "hypergeom"][:-2]
# The Krawtchouk and Hahn polynomials are not defined
# on the canonical domain [-1,1]. Must use numeric recursion
# to generate polynomials on [-1,1] for consistency, so remove from Askey list


def get_askey_recursion_coefficients(poly_name, opts, num_coefs):
    if poly_name not in askey_poly_names:
        raise ValueError(f"poly_name {poly_name} not in {askey_poly_names}")

    if poly_name == "legendre":
        return jacobi_recurrence(num_coefs, alpha=0, beta=0, probability=True)

    if poly_name == "jacobi":
        return jacobi_recurrence(
            num_coefs, alpha=opts["alpha_poly"], beta=opts["beta_poly"],
            probability=True)

    if poly_name == "hermite":
        return hermite_recurrence(num_coefs, rho=0., probability=True)

    if poly_name == "krawtchouk":
        msg = "Although bounded the Krawtchouk polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        num_coefs = min(num_coefs, opts["n"])
        return krawtchouk_recurrence(num_coefs, opts["n"], opts["p"])

    if poly_name == "hahn":
        msg = "Although bounded the Hahn polynomials are not defined "
        msg += "on the canonical domain [-1,1]. Must use numeric recursion "
        msg += "to generate polynomials on [-1,1] for consistency"
        warn(msg, UserWarning)
        num_coefs = min(num_coefs, opts["N"])
        return hahn_recurrence(
            num_coefs, opts["N"], opts["alpha_poly"], opts["beta_poly"])

    if poly_name == "charlier":
        return charlier_recurrence(num_coefs, opts["mu"])


def get_askey_recursion_coefficients_from_variable(var, num_coefs):
    var_name, scales, shapes = get_distribution_info(var)

    if var_name not in askey_variable_names:
        msg = f"Variable name {var_name} not in {askey_variable_names}"
        raise ValueError(msg)

    # Askey polynomials associated with continuous variables
    if var_name == "uniform":
        poly_name, opts = "legendre", {}
    elif var_name == "beta":
        poly_name = "jacobi"
        opts = {"alpha_poly": shapes["b"]-1, "beta_poly": shapes["a"]-1}
    elif var_name == "norm":
        poly_name, opts = "hermite", {}
        opts

    # Askey polynomials associated with discrete variables
    elif var_name == "binom":
        poly_name, opts = "krawtchouk", shapes
    elif var_name == "hypergeom":
        # note xk = np.arange(max(0, N-M+n), min(n, N)+1, dtype=float)
        poly_name = "hahn"
        M, n, N = [shapes[key] for key in ["M", "n", "N"]]
        opts = {"alpha_poly": -(n+1), "beta_poly": -M-1+n, "N": N}
    elif var_name == "poisson":
        poly_name, opts = "charlier", shapes

    return get_askey_recursion_coefficients(poly_name, opts, num_coefs)


def get_numerically_generated_recursion_coefficients_from_samples(
        xk, pk, num_coefs, orthonormality_tol, truncated_probability_tol=0):

    if num_coefs > xk.shape[0]:
        msg = "Number of coefs requested is larger than number of "
        msg += "probability masses"
        raise ValueError(msg)
    recursion_coeffs = lanczos(xk, pk, num_coefs, truncated_probability_tol)

    p = evaluate_orthonormal_polynomial_1d(
        np.asarray(xk, dtype=float), num_coefs-1, recursion_coeffs)
    error = np.absolute((p.T*pk).dot(p)-np.eye(num_coefs)).max()
    if error > orthonormality_tol:
        msg = "basis created is ill conditioned. "
        msg += f"Max error: {error}. Max terms: {xk.shape[0]}, "
        msg += f"Terms requested: {num_coefs}"
        raise ValueError(msg)
    return recursion_coeffs


def predictor_corrector_known_pdf(nterms, lb, ub, pdf, opts={}):
    if "quad_options" not in opts:
        tol = opts.get("orthonormality_tol", 1e-8)
        quad_options = {'epsrel': tol, 'epsabs': tol}
    else:
        quad_options = opts["quad_options"]

    return predictor_corrector(nterms, pdf, lb, ub, quad_options)


def get_recursion_coefficients_from_variable(var, num_coefs, opts):
    """
    Generate polynomial recursion coefficients by inspecting a random variable.
    """
    var_name, _, shapes = get_distribution_info(var)
    if var_name == "continuous_monomial":
        return None

    loc, scale = transform_scale_parameters(var)

    if var_name == "rv_function_indpndt_vars":
        shapes["loc"] = loc
        shapes["scale"] = scale
        return get_function_independent_vars_recursion_coefficients(
            shapes, num_coefs)

    if var_name == "rv_product_indpndt_vars":
        shapes["loc"] = loc
        shapes["scale"] = scale
        return get_product_independent_vars_recursion_coefficients(
            shapes, num_coefs)

    if (var_name in askey_variable_names and
            opts.get("numeric", False) is False):
        return get_askey_recursion_coefficients_from_variable(var, num_coefs)

    orthonormality_tol = opts.get("orthonormality_tol", 1e-8)
    truncated_probability_tol = opts.get("truncated_probability_tol", 0)

    if (not is_continuous_variable(var) or
            var.dist.name == "continuous_rv_sample"):
        if hasattr(shapes, "xk"):
            xk, pk = shapes["xk"], shapes["pk"]
        else:
            xk, pk = get_probability_masses(
                var, truncated_probability_tol)
        xk = (xk-loc)/scale

        return get_numerically_generated_recursion_coefficients_from_samples(
            xk, pk, num_coefs, orthonormality_tol, truncated_probability_tol)

    # integration performed in canonical domain so need to map back to
    # domain of pdf
    lb, ub = var.interval(1)

    # Get version var.pdf without error checking which runs much faster
    pdf = get_pdf(var)

    def canonical_pdf(x):
        # print(x, lb, ub, x*scale+loc)
        # print(var.pdf(x*scale+loc)*scale)
        # assert np.all(x*scale+loc >= lb) and np.all(x*scale+loc <= ub)
        return pdf(x*scale+loc)*scale
        # return var.pdf(x*scale+loc)*scale

    if (is_bounded_continuous_variable(var) or
            is_bounded_discrete_variable(var)):
        can_lb, can_ub = -1, 1
    elif is_continuous_variable(var):
        can_lb = (lb-loc)/scale
        can_ub = (ub-loc)/scale

    return predictor_corrector_known_pdf(
        num_coefs, can_lb, can_ub, canonical_pdf, opts)

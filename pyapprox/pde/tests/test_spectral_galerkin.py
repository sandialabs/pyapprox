import unittest
import numpy as np
from scipy import stats

from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D)
from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import AffineTransform
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation)
from pyapprox.surrogates.interp.indexing import tensor_product_indices

# imports for unittests
import sympy as sp
import matplotlib.pyplot as plt
from functools import partial


class SpectralGalerkinDomain():
    def __init__(self, domain_bounds, degrees):
        self._poly = None  # spectral basis
        self._basis_vals_at_can_quad = None  # basis vals at quadrature points
        self._basis_derivs_at_can_quad = None # basis derivs at quadrature points
        self._nbasis = None  # the number of basis terms
        self._quad_samples = None  # the quad samples in the user space
        self._canonical_quad_samples = None # canonical quad samples
        self._canonical_quad_weights = None # canonical quad samples
        self._nquad_samples = None
        self._bndry_canonical_quad_samples = None # quad rules on the boundary
        self._bndry_canonical_quad_weights = None
        self._bndry_quad_samples = None

        self._degrees = degrees  # degree of basis
        self._domain_bounds = domain_bounds
        self._nphys_vars = len(domain_bounds)//2

        self._define_quadrature_rule()
        self._define_basis()
        self._define_boundary()

    def _define_quadrature_rule(self):
        self._quad_rules_1d = [
            gauss_jacobi_pts_wts_1D(degree+2, 0, 0)
            for degree in self._degrees]
        self._canonical_quad_samples = (
            cartesian_product([rule[0] for rule in self._quad_rules_1d]))
        self._canonical_quad_weights = (
            outer_product(
                [rule[1] for rule in self._quad_rules_1d]))[:, None]
        self._nquad_samples = self._canonical_quad_samples.shape[1]


    def _define_boundary(self):
        if self._nphys_vars == 1:
            self._bndry_canonical_quad_samples = [
                np.array([[-1.0]]), np.array([[1.0]])]
            self._bndry_canonical_quad_weights = [
                np.array([[1.0]]) for ii in range(2)]
        else:
            # boundaries ordered left, right, bottom, top
            self._bndry_canonical_quad_samples = [
                np.vstack((
                    np.full((1, self._quad_rules_1d[1][0].shape[0]), -1),
                    self._quad_rules_1d[1][0][None, :])),
                np.vstack((
                    np.full((1, self._quad_rules_1d[1][0].shape[0]), 1),
                    self._quad_rules_1d[1][0][None, :])),
                np.vstack((
                    self._quad_rules_1d[0][0][None, :],
                    np.full((1, self._quad_rules_1d[0][0].shape[0]), -1))),
                np.vstack((
                    self._quad_rules_1d[0][0][None, :],
                    np.full((1, self._quad_rules_1d[0][0].shape[0]), 1)))]
            self._bndry_canonical_quad_weights = [
                self._quad_rules_1d[1][1], self._quad_rules_1d[1][1],
                self._quad_rules_1d[1][0], self._quad_rules_1d[0][1]]

        self._bndry_quad_samples = [
            self._poly.var_trans.map_from_canonical(s)
            for s in self._bndry_canonical_quad_samples]

    def _define_basis(self):
        marginals = [stats.uniform(
            self._domain_bounds[2*ii],
            self._domain_bounds[2*ii+1]-self._domain_bounds[2*ii])
                     for ii in range(self._nphys_vars)]
        var_trans = AffineTransform(IndependentMarginalsVariable(marginals))
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        self._poly = PolynomialChaosExpansion()
        self._poly.configure(poly_opts)
        self._poly.set_indices(tensor_product_indices(self._degrees))
        self._nbasis = self._poly.indices.shape[1]
        self._quad_samples = self._poly.var_trans.map_from_canonical(
            self._canonical_quad_samples)
        vals = self._poly.canonical_basis_matrix(
            self._canonical_quad_samples, {"deriv_order": 1})
        self._basis_vals_at_can_quad = vals[:self._nquad_samples, :]
        # derivs may be wrong
        self._basis_derivs_at_can_quad = vals[self._nquad_samples:, :].reshape(
            (self._nphys_vars, self._nquad_samples, self._nbasis))


class SpectralGalerkinSolver():
    def __init__(self, domain):
        self._diffusivity_fun = None
        self._velocity_fun = None
        self._forcing_fun = None
        self._domain = domain

    def initialize(self, diffusivity_fun, velocity_fun, forcing_fun,
                   bndry_conds):
        self._diffusivity_fun = diffusivity_fun
        self._velocity_fun = velocity_fun
        self._forcing_fun = forcing_fun
        self._bndry_conds = bndry_conds

    def _form_matrix(self):
        Amat = np.zeros(
            (self._domain._nbasis, self._domain._nbasis))
        basis_derivs = self._domain._basis_derivs_at_can_quad
        diff_vals = self._diffusivity_fun(self._domain._quad_samples)
        for dd in range(self._domain._nphys_vars):
            kdiff_vals = (diff_vals*basis_derivs[dd])
            for ii in range(self._domain._nbasis):
                Amat[ii, :] += (
                    (basis_derivs[dd][:, ii:ii+1]*kdiff_vals).T.dot(
                        self._domain._canonical_quad_weights))[:, 0]
                # TODO add velocity contribution
        return Amat

    def _form_rhs(self):
        rhs = (self._forcing_fun(self._domain._quad_samples) *
               self._domain._basis_vals_at_can_quad).T.dot(
                   self._domain._canonical_quad_weights)
        return rhs

    def _apply_boundary_conditions_to_matrix(self, matrix):
        return matrix

    def _apply_boundary_conditions_to_rhs(self, rhs):
        rhs_bc = rhs.copy()
        basis_vals = self._domain._basis_derivs_at_can_quad
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "D":
                continue
            # Neumann
            bndry_vals = bndry_cond[0](
                self._domain._bndry_quad_samples[ii])
            assert bndry_vals.ndim == 2 and bndry_vals.shape[1] == 1
            rhs_bc += (bndry_vals*basis_vals).dot(
                self._domain._bndry_canonical_quad_weights[ii])
        return rhs_bc

    def solve(self, sample):
        Amat = self._form_matrix()
        rhs = self._form_rhs()
        print(Amat)
        print(rhs)
        Amat_bc = self._apply_boundary_conditions_to_matrix(Amat)
        rhs_bc = self._apply_boundary_conditions_to_rhs(rhs)
        sol = np.linalg.solve(Amat_bc, rhs_bc)
        return sol

class BiLaplacian(SpectralGalerkinSolver):
    def __init__(self, domain):
        self._diffusivity_fun = None
        self._mass_fun = None
        self._forcing_fun = None
        self._domain = domain

    def initialize(self, diffusivity_fun, mass_fun, forcing_fun, bndry_conds):
        self._diffusivity_fun = diffusivity_fun
        self._mass_fun = mass_fun
        self._forcing_fun = forcing_fun
        self._bndry_conds = bndry_conds



def evaluate_sp_lambda(sp_lambda, xx, *args):
    # sp_lambda returns a single function output
    sp_args = tuple([x for x in xx])+args
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


def evaluate_list_of_sp_lambda(sp_lambdas, xx, *args):
    # sp_lambda returns list of values from multiple functions
    vals = [evaluate_sp_lambda(sp_lambda, xx) for sp_lambda in sp_lambdas]
    return np.array(vals).T


def setup_steady_advection_diffusion_manufactured_solution(
        sol_string, diff_string, vel_strings):
    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(symbs, sol_expr, "numpy")
    sol_fun = partial(evaluate_sp_lambda, sol_lambda)

    diff_expr = sp.sympify(diff_string)
    diff_lambda = sp.lambdify(symbs, diff_expr, "numpy")
    diff_fun = partial(evaluate_sp_lambda, diff_lambda)
    diffusion_expr = sum([(diff_expr*sol_expr.diff(symb, 1)).diff(symb, 1)
                          for symb in symbs])

    vel_exprs = [sp.sympify(vel_string) for vel_string in vel_strings]
    vel_lambdas = [
        sp.lambdify(symbs, vel_expr, "numpy") for vel_expr in vel_exprs]
    vel_fun = partial(evaluate_list_of_sp_lambda, vel_lambdas)
    advection_expr = sum(
        [vel_expr*sol_expr.diff(symb, 1)
         for vel_expr, symb in zip(vel_exprs, symbs)])

    forc_expr = -(diffusion_expr-advection_expr)
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(evaluate_sp_lambda, forc_lambda)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)

    return sol_fun, diff_fun, vel_fun, forc_fun


def setup_steady_bilaplacian_manufactured_solution(
        sol_string, diff_string, mass_string, nphys_vars):
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(symbs, sol_expr, "numpy")
    sol_fun = partial(evaluate_sp_lambda, sol_lambda)

    diff_expr = sp.sympify(diff_string)
    diff_lambda = sp.lambdify(symbs, diff_expr, "numpy")
    diff_fun = partial(evaluate_sp_lambda, diff_lambda)
    diffusion_expr = sum([(diff_expr*sol_expr.diff(symb, 1)).diff(symb, 1)
                          for symb in symbs])

    mass_expr = sp.sympify(mass_string)
    mass_lambda = sp.lambdify(symbs, mass_expr, "numpy")
    mass_fun = partial(evaluate_sp_lambda, mass_lambda)
    second_expr = mass_expr*sol_expr

    forc_expr = -diffusion_expr+second_expr
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(evaluate_sp_lambda, forc_lambda)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)

    return sol_fun, diff_fun, mass_fun, forc_fun


class TestSpectralGalerkin(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_advection_diffusion(self):
        np.set_printoptions(linewidth=400, threshold=5000, precision=2)
        domain_bounds, orders = [0, 1], [2]
        sol_string, diff_string, vel_strings = "x**2", "1", ["0"]

        domain_bounds, orders = [0, 1, 0, 1], [2, 2]
        sol_string, diff_string, vel_strings = "x**2*y**2", "1", ["0", "0"]

        sol_fun, diff_fun, vel_fun, forc_fun = (
            setup_steady_advection_diffusion_manufactured_solution(
                sol_string, diff_string, vel_strings))

        nphys_vars = len(orders)
        bdnry_conds = [[lambda xx: sol_fun(xx), "D"]
                       for dd in range(2**nphys_vars)]

        domain = SpectralGalerkinDomain(domain_bounds, orders)
        model = SpectralGalerkinSolver(domain)
        model.initialize(diff_fun, vel_fun, forc_fun, bdnry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)

    def test_bilaplacian(self):
        np.set_printoptions(linewidth=400, threshold=5000, precision=2)
        domain_bounds, orders = [0, 1], [2]
        sol_string, diff_string, mass_string = "x**2", "1", "1"

        # domain_bounds, orders = [0, 1, 0, 1], [2, 2]
        # sol_string, diff_string, mass_string = "x**2*y**2", "1", "1"

        sol_fun, diff_fun, mass_fun, forc_fun = (
            setup_steady_bilaplacian_manufactured_solution(
                sol_string, diff_string, mass_string, len(orders)))

        nphys_vars = len(orders)
        bdnry_conds = [[lambda xx: np.zeros((xx.shape[1], 1)), "N"]
                       for dd in range(2**nphys_vars)]

        domain = SpectralGalerkinDomain(domain_bounds, orders)
        model = BiLaplacian(domain)
        model.initialize(diff_fun, mass_fun, forc_fun, bdnry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)



if __name__ == "__main__":
    stokes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralGalerkin)
    unittest.TextTestRunner(verbosity=2).run(stokes_test_suite)

import unittest
from scipy import stats
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D)
from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import AffineTransform
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation)
from pyapprox.surrogates.interp.indexing import tensor_product_indices
from pyapprox.util.visualization import get_meshgrid_function_data

# imports for unittests
import sympy as sp


class AbstractSpectralGalerkinDomain(ABC):
    def __init__(self, degrees):
        self._degrees = degrees
        self._nphys_vars = len(self._degrees)

        self._quad_samples, self._quad_weights = self._define_quadrature_rule()
        self._nquad_samples = self._quad_samples.shape[1]

        self._poly, self._nbasis = self._define_basis()

        self._basis_vals_at_quad, self._basis_derivs_at_quad = (
            self._evaluate_basis_at_quadrature_samples())

        (self._bndry_quad_samples, self._bndry_quad_weights,
         self._bndry_normals) = self._define_boundary()

        self._basis_vals_at_bndry_quad = (
            self._evaluate_basis_at_bndry_quadrature_samples())

    def _evaluate_basis_at_samples(self, samples):
        vals = self._poly.basis_matrix(samples, {"deriv_order": 1})
        basis_vals = vals[:samples.shape[1], :]
        basis_derivs = vals[samples.shape[1]:, :].reshape(
            (self._nphys_vars, samples.shape[1], self._nbasis))
        return basis_vals, basis_derivs

    def _evaluate_basis_at_quadrature_samples(self):
        return self._evaluate_basis_at_samples(self._quad_samples)

    def _evaluate_basis_at_bndry_quadrature_samples(self):
        basis_vals_at_bndry_quad = [
            None for ii in range(len(self._bndry_quad_samples))]
        for ii in range(len(self._bndry_quad_samples)):
            vals = self._poly.basis_matrix(
                self._bndry_quad_samples[ii], {"deriv_order": 0})
            nbndry_quad = self._bndry_quad_samples[ii].shape[1]
            basis_vals_at_bndry_quad[ii] = vals[:nbndry_quad, :]
        return basis_vals_at_bndry_quad

    def interpolate(self, basis_coefs, samples):
        self._poly.set_coefficients(basis_coefs)
        return self._poly(samples)

    def _define_quadrature_rule(self):
        raise NotImplementedError()

    def _define_basis(self):
        raise NotImplementedError()

    def _define_boundary(self):
        raise NotImplementedError()


class CartesianProductSpectralGalerkinDomain(AbstractSpectralGalerkinDomain):
    def __init__(self, domain_bounds, degrees):
        marginals = [stats.uniform(
            domain_bounds[2*ii], domain_bounds[2*ii+1]-domain_bounds[2*ii])
                     for ii in range(len(domain_bounds)//2)]
        self._var_trans = AffineTransform(
            IndependentMarginalsVariable(marginals))
        self._bounds = domain_bounds

        super().__init__(degrees)

    def _define_quadrature_rule(self):
        canonical_quad_rules_1d = [
            gauss_jacobi_pts_wts_1D(degree+3, 0, 0)
            for degree in self._degrees]
        canonical_quad_samples = (
            cartesian_product([rule[0] for rule in canonical_quad_rules_1d]))
        quad_samples = self._var_trans.map_from_canonical(
            canonical_quad_samples)
        quad_weights = (
            outer_product(
                [rule[1] for rule in canonical_quad_rules_1d]))[:, None]
        return quad_samples, quad_weights

    def _boundary_normal(self, val, xx):
        return np.tile(val, (xx.shape[1], 1)).T

    def _define_boundary(self):
        if self._nphys_vars == 1:
            canonical_quad_samples = [
                np.array([[-1.0]]), np.array([[1.0]])]
            bndry_quad_weights = [
                np.array([[1.0]]) for ii in range(2)]
            bndry_normals = [
                partial(self._boundary_normal, [-1**(ii+1)])
                for ii in range(2)]
        else:
            canonical_quad_rules_1d = [
                 gauss_jacobi_pts_wts_1D(degree+2, 0, 0)
                 for degree in self._degrees]
            nquad_rules_1d = [len(q[0]) for q in canonical_quad_rules_1d]
            # boundaries ordered left, right, bottom, top
            canonical_quad_samples = [
                np.vstack((
                    np.full((1, nquad_rules_1d[1]), -1),
                    canonical_quad_rules_1d[1][0][None, :])),
                np.vstack((
                    np.full((1, nquad_rules_1d[1]), 1),
                    canonical_quad_rules_1d[1][0][None, :])),
                np.vstack((
                    canonical_quad_rules_1d[0][0][None, :],
                    np.full((1, nquad_rules_1d[0]), -1))),
                np.vstack((
                    canonical_quad_rules_1d[0][0][None, :],
                    np.full((1, nquad_rules_1d[0]), 1)))]
            bndry_quad_weights = [
                canonical_quad_rules_1d[1][1][:, None],
                canonical_quad_rules_1d[1][1][:, None],
                canonical_quad_rules_1d[1][0][:, None],
                canonical_quad_rules_1d[0][1][:, None]]
            bndry_normals = [
                partial(self._boundary_normal, val)
                for val in [[-1, 0], [1, 0], [0, -1], [1, 0]]]
        bndry_quad_samples = [
            self._poly.var_trans.map_from_canonical(s)
            for s in canonical_quad_samples]
        return bndry_quad_samples, bndry_quad_weights, bndry_normals

    def _define_basis(self):
        poly_opts = define_poly_options_from_variable_transformation(
            self._var_trans)
        _poly = PolynomialChaosExpansion()
        _poly.configure(poly_opts)
        _poly.set_indices(tensor_product_indices(self._degrees))
        _nbasis = _poly.indices.shape[1]
        return _poly, _nbasis

    def plot(self, fun, nplot_pts_1d, ax=None, ncontour_levels=20,
             **plt_kwargs):
        if ax is None:
            ax = plt
        if self._nphys_vars == 1:
            xx = np.linspace(
                self._bounds[0], self._bounds[1], nplot_pts_1d)[None, :]
            values = fun(xx)
            return ax.plot(xx[0, :], values, **plt_kwargs)

        X, Y, Z = get_meshgrid_function_data(fun, self._bounds, nplot_pts_1d)
        return ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels),
            **plt_kwargs)

    def plot_poly(self, coef, nplot_pts_1d, ax=None, **plt_kwargs):
        fun = partial(self.interpolate, coef)
        self.plot(fun, nplot_pts_1d, ax, **plt_kwargs)


class AbstractSpectralGalerkinSolver(ABC):
    def __init__(self, domain, dirichlet_penalty=10):
        self.domain = domain
        self._dirichlet_penalty = dirichlet_penalty
        self._bndry_conds = None

    def _form_rhs(self):
        rhs = (self._forcing_fun(self.domain._quad_samples) *
               self.domain._basis_vals_at_quad).T.dot(
                   self.domain._quad_weights)
        return rhs

    def _apply_boundary_conditions_to_matrix(self, matrix):
        return matrix

    def _apply_boundary_conditions_to_rhs(self, rhs):
        rhs_bc = rhs.copy()
        for ii, bndry_cond in enumerate(self._bndry_conds):
            bndry_vals = bndry_cond[0](
                self.domain._bndry_quad_samples[ii])
            basis_vals = self.domain._basis_vals_at_bndry_quad[ii]
            if bndry_cond[1] == "D":
                bndry_integral = self._dirichlet_penalty*(
                    (bndry_vals*basis_vals).T.dot(self.domain._quad_weights))
                bndry_integral += self._bndry_adjustment(ii, bndry_vals)
                # continue
            # Neumann
            assert bndry_vals.ndim == 2 and bndry_vals.shape[1] == 1
            bndry_integral = (bndry_vals*basis_vals).T.dot(
                    self.domain._bndry_quad_weights[ii])
            rhs_bc += bndry_integral
        return rhs_bc

    def solve(self, sample):
        Amat = self._form_matrix()
        Amat += self._matrix_adjustment()
        rhs = self._form_rhs()
        # print(Amat)
        # print(rhs)
        Amat_bc = self._apply_boundary_conditions_to_matrix(Amat)
        rhs_bc = self._apply_boundary_conditions_to_rhs(rhs)
        sol = np.linalg.solve(Amat_bc, rhs_bc)
        return sol

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def _form_matrix(self):
        raise NotImplementedError()

    @abstractmethod
    def _bndry_adjustment(self, bndry_id, bndry_vals):
        raise NotImplementedError()


class SpectralGalerkinAdvectionDiffusionSolver(AbstractSpectralGalerkinSolver):

    def __init__(self, domain):
        super().__init__(domain)
        self._diffusivity_fun = None
        self._velocity_fun = None
        self._forcing_fun = None

    def initialize(self, diffusivity_fun, velocity_fun, forcing_fun,
                   bndry_conds):
        self._diffusivity_fun = diffusivity_fun
        self._velocity_fun = velocity_fun
        self._forcing_fun = forcing_fun
        self._bndry_conds = bndry_conds

    def _form_matrix(self):
        Amat = np.zeros(
            (self.domain._nbasis, self.domain._nbasis))
        basis_derivs = self.domain._basis_derivs_at_quad
        diff_vals = self._diffusivity_fun(self.domain._quad_samples)
        for dd in range(self.domain._nphys_vars):
            kderiv_vals = (diff_vals*basis_derivs[dd])
            for ii in range(self.domain._nbasis):
                Amat[ii, :] += (
                    (basis_derivs[dd][:, ii:ii+1]*kderiv_vals).T.dot(
                        self.domain._quad_weights))[:, 0]
                # TODO add velocity contribution
        return Amat


class SpectralGalerkinBiLaplacianSolver(AbstractSpectralGalerkinSolver):
    def __init__(self, domain):
        self._diffusivity_fun = None
        self._mass_fun = None
        self._forcing_fun = None
        self.domain = domain

    def initialize(self, diffusivity_fun, mass_fun, forcing_fun, bndry_conds):
        self._diffusivity_fun = diffusivity_fun
        self._mass_fun = mass_fun
        self._forcing_fun = forcing_fun
        self._bndry_conds = bndry_conds

    def _form_matrix(self):
        Amat = np.zeros(
            (self.domain._nbasis, self.domain._nbasis))
        basis_derivs = self.domain._basis_derivs_at_quad
        basis_vals = self.domain._basis_vals_at_quad
        diff_vals = self._diffusivity_fun(self.domain._quad_samples)
        mass_vals = self._mass_fun(self.domain._quad_samples)
        for dd in range(self.domain._nphys_vars):
            kderiv_vals = (diff_vals*basis_derivs[dd])
            mbasis_vals = (mass_vals*basis_vals)
            for ii in range(self.domain._nbasis):
                Amat[ii, :] += (
                    (basis_derivs[dd][:, ii:ii+1]*kderiv_vals).T.dot(
                        self.domain._quad_weights))[:, 0]
        for ii in range(self.domain._nbasis):
            Amat[ii, :] += (
                (basis_vals[:, ii:ii+1]*mbasis_vals).T.dot(
                    self.domain._quad_weights))[:, 0]
        return Amat

    def _bndry_adjustment(self, bndry_id, bndry_vals):
        bndry_quad_samples = self.domain._bndry_quad_samples[bndry_id]
        basis_derivs = self.domain._evaluate_basis_at_samples(
            bndry_quad_samples)[1]
        diff_vals = self._diffusivity_fun(bndry_quad_samples)
        normals = self.domain._bndry_normals[bndry_id](bndry_quad_samples)
        adjust = np.empty((self.domain._nbasis, 1))
        for ii in range(self.domain._nbasis):
            adjust[ii] = -np.dot((bndry_vals*diff_vals*(
                np.sum(normals*basis_derivs[:, :, ii], axis=0)[:, None])).T,
                            self.domain._bndry_quad_weights[bndry_id])
        return adjust

    def _matrix_adjustment(self):
        adjust = np.zeros(
            (self.domain._nbasis, self.domain._nbasis))
        for ii, bndry_cond in enumerate(self._bndry_conds):
            #TODO do not compute bdnry vals again make this code used
            # values computed when forming rhs
            if bndry_cond[1] == "D":
                bndry_vals = bndry_cond[0](
                    self.domain._bndry_quad_samples[ii])
                adjust += self._matrix_adjustment_single_bndry(ii, bndry_vals)
        return adjust

    def _matrix_adjustment_single_bndry(self, bndry_id, bndry_vals):
        bndry_quad_samples = self.domain._bndry_quad_samples[bndry_id]
        adjust = np.empty(
            (self.domain._nbasis, self.domain._nbasis))
        basis_vals, basis_derivs = self.domain._evaluate_basis_at_samples(
            bndry_quad_samples)
        diff_vals = self._diffusivity_fun(bndry_quad_samples)
        kbasis_vals = basis_vals*diff_vals
        normals = self.domain._bndry_normals[bndry_id](bndry_quad_samples)
        for ii in range(self.domain._nbasis):
            adjust[ii, :] = -((kbasis_vals*np.sum(
                normals*basis_derivs[:, :, ii], axis=0)[:, None]).T.dot(
                    self.domain._bndry_quad_weights[bndry_id]))[:, 0]
            adjust[ii, :] += ((basis_vals[:, ii:ii+1]*basis_vals).T.dot(
                self.domain._quad_weights))[:, 0]
        return adjust


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
    return np.hstack(vals)


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

    flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    flux_funs = partial(evaluate_list_of_sp_lambda, flux_lambdas)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)
    print("flux", flux_exprs)

    return sol_fun, diff_fun, mass_fun, forc_fun, flux_funs


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
        bndry_conds = [[lambda xx: sol_fun(xx), "D"]
                       for dd in range(2**nphys_vars)]

        domain = CartesianProductSpectralGalerkinDomain(domain_bounds, orders)
        model = SpectralGalerkinAdvectionDiffusionSolver(domain)
        model.initialize(diff_fun, vel_fun, forc_fun, bndry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)

    def check_bilaplacian(self, domain_bounds, orders, sol_string, diff_string,
                          mass_string):

        sol_fun, diff_fun, mass_fun, forc_fun, flux_funs = (
            setup_steady_bilaplacian_manufactured_solution(
                sol_string, diff_string, mass_string, len(orders)))
        # Solution designed to have zero flux at boundaries.
        # Thus we are isolating testing of assembling bilinear form and rhs

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(nphys_vars):
            bndry_conds += [[lambda xx: -0*flux_funs(xx)[:, dd:dd+1], "N"],
                            [lambda xx: 0*flux_funs(xx)[:, dd:dd+1], "N"]]

        domain = CartesianProductSpectralGalerkinDomain(domain_bounds, orders)
        model = SpectralGalerkinBiLaplacianSolver(domain)
        model.initialize(diff_fun, mass_fun, forc_fun, bndry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)

        xx = cartesian_product(
            [np.linspace(domain_bounds[2*ii], domain_bounds[2*ii+1], 20)
             for ii in range(model.domain._nphys_vars)])
        sol_vals = model.domain.interpolate(sol, xx)
        exact_sol_vals = sol_fun(xx)
        # print(sol_vals)
        # # print(exact_sol_vals)
        # fig, axs = plt.subplots(1, model.domain._nphys_vars)
        # axs = np.atleast_1d(axs)
        # model.domain.plot(sol_fun, 50, ax=axs[0])
        # model.domain.plot_poly(sol, 50, ax=axs[-1], ls='--')
        # plt.show()
        print(sol_vals-exact_sol_vals)
        assert np.allclose(sol_vals, exact_sol_vals)

    def test_bilaplacian(self):
        domain_bounds, orders = [0, 1], [3]
        sol_string, diff_string, mass_string = "5/3*x**2-10/9*x**3", "1", "1"
        self.check_bilaplacian(domain_bounds, orders, sol_string, diff_string,
                               mass_string)

        # domain_bounds, orders = [0, 1], [3]
        # sol_string, diff_string, mass_string = "x**3", "1", "1"
        # self.check_bilaplacian(domain_bounds, orders, sol_string, diff_string,
        #                        mass_string)
        
        domain_bounds, orders = [0, 1, 0, 1], [3, 3]
        sol_string = "(5/3*x**2-10/9*x**3)*(5/3*y**2-10/9*y**3)"
        diff_string, mass_string = "1", "1"
        self.check_bilaplacian(domain_bounds, orders, sol_string, diff_string,
                               mass_string)
        



if __name__ == "__main__":
    stokes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralGalerkin)
    unittest.TextTestRunner(verbosity=2).run(stokes_test_suite)

from abc import ABC, abstractmethod
from functools import partial
from scipy import stats
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
            gauss_jacobi_pts_wts_1D(degree+1+degree % 2, 0, 0)
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
                partial(self._boundary_normal, [-1]),
                partial(self._boundary_normal, [1])]
        else:
            canonical_quad_rules_1d = [
                gauss_jacobi_pts_wts_1D(degree+1+degree % 2, 0, 0)
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
                canonical_quad_rules_1d[0][1][:, None],
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
        return self._matrix_adjustment() + matrix

    def _apply_boundary_conditions_to_rhs(self, rhs):
        rhs_bc = rhs.copy()
        for ii, bndry_cond in enumerate(self._bndry_conds):
            bndry_vals = bndry_cond[0](
                self.domain._bndry_quad_samples[ii])
            assert bndry_vals.ndim == 2 and bndry_vals.shape[1] == 1
            if bndry_cond[1] == "D":
                rhs_bc += self._bndry_adjustment(ii, bndry_vals)
            else:
                # Neumann
                basis_vals = self.domain._basis_vals_at_bndry_quad[ii]
                rhs_bc += (bndry_vals*basis_vals).T.dot(
                    self.domain._bndry_quad_weights[ii])
        return rhs_bc

    def solve(self):
        Amat = self._form_matrix()
        rhs = self._form_rhs()
        # Amat_bc should be symmetic when velocity and thus advection term
        # is zero
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


class SpectralGalerkinLinearDiffusionReactionSolver(
        AbstractSpectralGalerkinSolver):
    def __init__(self, domain, dirichlet_penalty=10):
        super().__init__(domain, dirichlet_penalty)
        self._diffusivity_fun = None
        self._mass_fun = None
        self._forcing_fun = None

    def initialize(self, diffusivity_fun, mass_fun, forcing_fun, bndry_conds):
        self._diffusivity_fun = diffusivity_fun
        self._mass_fun = mass_fun
        self._forcing_fun = forcing_fun
        self._bndry_conds = bndry_conds

    def _form_matrix(self):
        basis_derivs = self.domain._basis_derivs_at_quad
        basis_vals = self.domain._basis_vals_at_quad
        diff_vals = self._diffusivity_fun(self.domain._quad_samples)
        mass_vals = self._mass_fun(self.domain._quad_samples)
        weights = self.domain._quad_weights
        Amat = np.sum(
            (diff_vals*weights)[:, :, None]*np.einsum(
                "ijk,kjm->mij", basis_derivs.T, basis_derivs).T, axis=0)
        Amat += (weights*basis_vals).T.dot(mass_vals*basis_vals)
        return Amat

    def _bndry_adjustment(self, bndry_id, bndry_vals):
        bndry_quad_samples = self.domain._bndry_quad_samples[bndry_id]
        basis_vals = self.domain._basis_vals_at_bndry_quad[bndry_id]
        basis_derivs = self.domain._evaluate_basis_at_samples(
            bndry_quad_samples)[1]
        diff_vals = self._diffusivity_fun(bndry_quad_samples)
        normals = self.domain._bndry_normals[bndry_id](bndry_quad_samples)
        weights = self.domain._bndry_quad_weights[bndry_id]
        adjust = self._dirichlet_penalty*(
            (bndry_vals*basis_vals).T.dot(weights))
        weights = self.domain._bndry_quad_weights[bndry_id]
        adjust -= (np.einsum("jk,jkm->km", normals, basis_derivs)).T.dot(
            (weights*diff_vals*bndry_vals))
        return adjust

    def _matrix_adjustment(self):
        adjust = np.zeros(
            (self.domain._nbasis, self.domain._nbasis))
        for ii, bndry_cond in enumerate(self._bndry_conds):
            # TODO do not compute bdnry vals again make this code used
            # values computed when forming rhs
            if bndry_cond[1] == "D":
                adjust += self._matrix_adjustment_single_bndry(ii)
        return adjust

    def _matrix_adjustment_single_bndry(self, bndry_id):
        bndry_quad_samples = self.domain._bndry_quad_samples[bndry_id]
        basis_vals, basis_derivs = self.domain._evaluate_basis_at_samples(
            bndry_quad_samples)
        diff_vals = self._diffusivity_fun(bndry_quad_samples)
        weights = self.domain._bndry_quad_weights[bndry_id]
        kwbasis_vals = basis_vals*diff_vals*weights
        normals = self.domain._bndry_normals[bndry_id](bndry_quad_samples)
        deriv_normals = np.sum(normals[:, :, None]*basis_derivs, axis=0)
        adjust = -(deriv_normals.T.dot(kwbasis_vals))
        adjust -= kwbasis_vals.T.dot(deriv_normals)
        adjust += self._dirichlet_penalty*(weights*basis_vals).T.dot(
            basis_vals)
        return adjust


class SpectralGalerkinAdvectionDiffusionSolver(
        SpectralGalerkinLinearDiffusionReactionSolver):

    def __init__(self, domain, dirichlet_penalty=10):
        super().__init__(domain, dirichlet_penalty)
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
        basis_vals = self.domain._basis_vals_at_quad
        basis_derivs = self.domain._basis_derivs_at_quad
        diff_vals = self._diffusivity_fun(self.domain._quad_samples)
        vel_vals = self._velocity_fun(self.domain._quad_samples)
        weights = self.domain._quad_weights
        Amat = np.sum(
            (diff_vals*weights)[:, :, None]*np.einsum(
                "ijk,kjm->mij", basis_derivs.T, basis_derivs).T, axis=0)
        Amat += (weights*basis_vals).T.dot(np.einsum(
            "jk,jkm->km", vel_vals.T, basis_derivs))
        return Amat


class SpectralGalerkinStokes():
    def initialize(self, bndry_conds):
        self._bndry_conds = bndry_conds

    def _form_matrix(self):
        basis_derivs = self.domain._basis_derivs_at_quad
        weights = self.domain._quad_weights
        Amat = np.sum(
            (weights)[:, :, None]*np.einsum(
                "ijk,kjm->mij", basis_derivs.T, basis_derivs).T, axis=0)
        return Amat
        

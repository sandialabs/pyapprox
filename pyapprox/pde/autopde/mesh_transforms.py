import numpy as np
from functools import partial
from abc import ABC, abstractmethod
import sympy as sp

from pyapprox.variables.transforms import _map_hypercube_samples
from pyapprox.pde.autopde.sympy_utils import (
    _evaluate_list_of_sp_lambda, _evaluate_sp_lambda)


class OrthogonalCoordinateTransform2D(ABC):
    @abstractmethod
    def map_from_orthogonal(self, orth_samples):
        raise NotImplementedError()

    @abstractmethod
    def map_to_orthogonal(self, samples):
        raise NotImplementedError()

    @abstractmethod
    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        raise NotImplementedError()

    @abstractmethod
    def scale_factor(self, basis_id, orth_samples):
        raise NotImplementedError()

    def _normalized_curvelinear_basis(self, orth_samples):
        basis = self.curvelinear_basis(orth_samples)
        for ii in range(2):
            scale = self.scale_factor(ii, orth_samples)
            basis[:, :, ii] = scale*basis[:, :, ii]
        return basis

    @staticmethod
    def _normal_sign(bndry_id):
        if bndry_id % 2 == 0:
            return -1.0
        return 1.0

    @staticmethod
    def _normal(map_to_orthogonal, normalized_curvelinear_basis,
                bndry_id, sign, samples):
        orth_samples = map_to_orthogonal(samples)
        basis_id = int(bndry_id > 1)
        normals = sign*normalized_curvelinear_basis(
            orth_samples)[..., basis_id]
        return normals

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        return self._normal(
            self.map_to_orthogonal, self._normalized_curvelinear_basis,
            bndry_id, sign, samples)

    @staticmethod
    def scale_orthogonal_gradients(basis, orth_grads):
        return np.einsum("ijk,ik->ij", basis, orth_grads)

    def modify_quadrature_weights(self, orth_samples, orth_weights):
        basis = self.curvelinear_basis(orth_samples)
        dets = np.linalg.det(basis)
        return orth_weights/np.abs(dets)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


def _sample_ranges(samples):
    return np.vstack([samples.min(axis=1)[None, :],
                     samples.max(axis=1)[None, :]]).T


class CompositionTransform():
    def __init__(self, transforms):
        self._transforms = transforms

        self._normal_sign = OrthogonalCoordinateTransform2D._normal_sign

    def map_from_orthogonal(self, orth_samples):
        samples = self._transforms[0].map_from_orthogonal(orth_samples)
        for transform in self._transforms[1:]:
            samples = transform.map_from_orthogonal(samples)
        return samples

    def map_to_orthogonal(self, samples):
        orth_samples = samples
        for transform in self._transforms[::-1]:
            orth_samples = transform.map_to_orthogonal(orth_samples)
        return orth_samples

    def _basis_product(self, basis1, basis2):
        basis = np.einsum("ijk,ikl->ijl", basis1, basis2)
        return basis

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        basis = self._transforms[0].curvelinear_basis(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples)
            new_basis = transform.curvelinear_basis(orth_samples)
            basis = self._basis_product(new_basis, basis)
        return basis

    def _normalized_curvelinear_basis(self, orth_samples):
        basis = self._transforms[0]._normalized_curvelinear_basis(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples)
            new_basis = transform._normalized_curvelinear_basis(orth_samples)
            basis = self._basis_product(new_basis, basis)
        return basis

    @staticmethod
    def scale_orthogonal_gradients(basis, orth_grads):
        return OrthogonalCoordinateTransform2D.scale_orthogonal_gradients(
            basis, orth_grads)

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        normals = OrthogonalCoordinateTransform2D._normal(
            self.map_to_orthogonal,
            self._normalized_curvelinear_basis,
            bndry_id, sign, samples)
        return normals

    def modify_quadrature_weights(self, orth_samples, orth_weights):
        weights = self._transforms[0].modify_quadrature_weights(
            orth_samples, orth_weights)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples)
            weights = self._transforms[ii+1].modify_quadrature_weights(
                orth_samples, weights)
        return weights

    def __repr__(self):
        return self.__class__.__name__+"[{0}]".format(", ".join(
            map("{}".format, self._transforms)))


class ScaleAndTranslationTransform(OrthogonalCoordinateTransform2D):
    def __init__(self, orthog_ranges, ranges):
        self._orthog_ranges = np.asarray(orthog_ranges)
        self._ranges = np.asarray(ranges)
        # if following not satisfied it will mess up how normals are computed
        if np.any(orthog_ranges[1::2] <= orthog_ranges[::2]):
            msg = f"orthog_ranges {orthog_ranges} must be increasing"
            raise ValueError(msg)
        if np.any(ranges[1::2] <= ranges[::2]):
            msg = f"ranges {ranges} must be increasing"
            raise ValueError(msg)

    def map_from_orthogonal(self, orth_samples):
        return _map_hypercube_samples(
            orth_samples, self._orthog_ranges, self._ranges)

    def map_to_orthogonal(self, samples):
        return _map_hypercube_samples(
            samples, self._ranges, self._orthog_ranges)

    def scale_factor(self, basis_id, orth_samples):
        nsamples = orth_samples.shape[1]
        if basis_id == 0:
            return np.full(
                (nsamples, 1),
                np.diff(self._ranges[:2])/np.diff(self._orthog_ranges[:2]))
        return np.full(
            (nsamples, 1),
            np.diff(self._ranges[2:])/np.diff(self._orthog_ranges[2:]))

    def _curvelinear_basis_2d(self, orth_samples):
        r, theta = orth_samples
        zeros = np.zeros((r.shape[0], 1))
        a11 = np.full(
            zeros.shape,
            np.diff(self._ranges[:2])/np.diff(self._orthog_ranges[:2]))
        a22 = np.full(
            zeros.shape,
            np.diff(self._ranges[2:])/np.diff(self._orthog_ranges[2:]))
        basis = np.dstack(
            [np.hstack([1/a11, zeros])[..., None],
             np.hstack([zeros, 1/a22])[..., None]])
        return basis

    def _curvelinear_basis_1d(self, orth_samples):
        r = orth_samples[0]
        zeros = np.zeros((r.shape[0], 1))
        a11 = np.full(
            zeros.shape,
            np.diff(self._ranges[:2])/np.diff(self._orthog_ranges[:2]))
        basis = (1/a11)[..., None]
        return basis

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        if orth_samples.shape[0] == 2:
            return self._curvelinear_basis_2d(orth_samples)
        return self._curvelinear_basis_1d(orth_samples)

    def _orthog_ranges_repr(self):
        return "[{0}]".format(
            ", ".join(map("{0:.3g}".format, self._orthog_ranges)),
        )

    def _ranges_repr(self):
        return "[{0}]".format(
            ", ".join(map("{0:.3g}".format, self._ranges)),
        )

    def __repr__(self):
        return "{0}(orthog_ranges={1}, ranges={2})".format(
            self.__class__.__name__, self._orthog_ranges_repr(),
            self._ranges_repr())


class PolarTransform(OrthogonalCoordinateTransform2D):
    def map_from_orthogonal(self, orth_samples):
        if orth_samples[1].max() > 2*np.pi or orth_samples[1].min() < 0:
            raise ValueError("theta must be in [0, 2*pi]")
        if orth_samples[0].min() < 0:
            print(orth_samples[0])
            raise ValueError("r must be in [0, np.inf]")
        r, theta = orth_samples
        samples = np.vstack(
            [r*np.cos(theta)[None, :], r*np.sin(theta)[None, :]])
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        r = np.sqrt(x**2+y**2)[None, :]
        orth_samples = np.vstack(
            [r, np.arccos(x[None, :]/r)])
        II = np.where(y < 0)[0]
        orth_samples[1, II] = -orth_samples[1, II]+2*np.pi
        return orth_samples

    def scale_factor(self, basis_id, orth_samples):
        nsamples = orth_samples.shape[1]
        if basis_id == 0:
            return np.ones((nsamples, 1))
        r = orth_samples[0]
        return r[:, None]

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        r, theta = orth_samples
        r, theta = r[:, None], theta[:, None]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # basis is [b1, b2]
        # form b1, b2 using hstack then form basis using dstack
        basis = np.dstack(
            [np.hstack([cos_t, sin_t])[..., None],  # first basis
             np.hstack([-sin_t/r, cos_t/r])[..., None]])  # second basis
        return basis


class EllipticalTransform(OrthogonalCoordinateTransform2D):
    def __init__(self, a):
        self._a = a

    def map_from_orthogonal(self, orth_samples):
        r, theta = orth_samples
        samples = np.vstack(
            [self._a*(np.cosh(r)*np.cos(theta))[None, :],
             self._a*(np.sinh(r)*np.sin(theta))[None, :]])
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        II = np.where(y < 0)[0]
        orth_samples = np.vstack(
            [np.real(np.arccosh(x/self._a+1.j*y/self._a))[None, :],
             np.imag(np.arccosh(x/self._a+1.j*y/self._a))[None, :]])
        orth_samples[1, II] = orth_samples[1, II]+2*np.pi
        return orth_samples

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        r, theta = orth_samples
        denom = (self._a*np.sinh(r)**2+np.sin(theta)**2)[:, None]
        cosh_r = np.cosh(r)[:, None]
        sinh_r = np.sinh(r)[:, None]
        cos_t = np.cos(theta)[:, None]
        sin_t = np.sin(theta)[:, None]
        basis = np.dstack(
            [np.hstack([sinh_r*cos_t/denom, cosh_r*sin_t/denom])[..., None],
             np.hstack([-cosh_r*sin_t/denom, sinh_r*cos_t/denom])[..., None]])
        return basis

    def scale_factor(self, basis_id, orth_samples):
        r, theta = orth_samples
        return self._a*np.sqrt(np.sinh(r)**2+np.sin(theta)**2)[:, None]


from pyapprox.pde.autopde.manufactured_solutions import (
    _evaluate_list_of_sp_lambda, _evaluate_sp_lambda)
class SympyTransform(OrthogonalCoordinateTransform2D):
    @staticmethod
    def _lambdify_map(strings, symbs):
        exprs = [sp.sympify(string) for string in strings]
        lambdas = [sp.lambdify(symbs, expr, "numpy") for expr in exprs]
        return partial(_evaluate_list_of_sp_lambda, lambdas)

    @staticmethod
    def _jacobian_expr(strings, symbs):
        exprs = [sp.sympify(string) for string in strings]
        jacobian_exprs = [
            [expr.diff(symb) for expr in exprs]
            for symb in symbs]
        return sp.Matrix(jacobian_exprs)

    def __init__(self, map_from_orthogonal_strings, map_to_orthogonal_strings):
        self._symbs = sp.symbols(['_x_', '_y_'])
        self._orth_symbs = sp.symbols(['_r_', '_t_'])
        assert ("_r_" in map_from_orthogonal_strings[0] and
                "_t_" in map_from_orthogonal_strings[1])
        assert ("_x" in map_to_orthogonal_strings[0] and
                "_y_" in map_to_orthogonal_strings[1])
        self._map_from_orthogonal_trans = self._lambdify_map(
              map_from_orthogonal_strings, self._orth_symbs)
        self._map_to_orthogonal_trans = self._lambdify_map(
              map_to_orthogonal_strings, self._symbs)

        self._from_orth_jacobian_expr = self._jacobian_expr(
            map_from_orthogonal_strings, self._orth_symbs)
        self._to_orth_jacobian_expr = sp.simplify(
            self._from_orth_jacobian_expr.inv())
        self._bases = [
            partial(_evaluate_list_of_sp_lambda,
                    [sp.lambdify(self._orth_symbs, expr, "numpy")
                     for expr in self._to_orth_jacobian_expr[:, ii]])
            for ii in range(2)]
        self._scale_factor_expr = [
            1/sp.simplify(sp.sqrt(self._to_orth_jacobian_expr[:, ii].dot(
                self._to_orth_jacobian_expr[:, ii]))) for ii in range(2)]
        self._scale_factors = [
            partial(_evaluate_sp_lambda,
                    sp.lambdify(self._orth_symbs, expr, "numpy"))
            for expr in self._scale_factor_expr]

    def map_from_orthogonal(self, orth_samples):
        return self._map_from_orthogonal_trans(orth_samples).T

    def map_to_orthogonal(self, samples):
        return self._map_to_orthogonal_trans(samples).T

    def scale_factor(self, basis_id, orth_samples):
        return self._scale_factors[basis_id](orth_samples)

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        basis = np.dstack(
            [basis(orth_samples)[..., None] for basis in self._bases])
        return basis

    def __repr__(self):
        return "{0}, {1}, {2}".format(
            self.__class__.__name__, self._from_orth_jacobian_expr,
            self._to_orth_jacobian_expr)

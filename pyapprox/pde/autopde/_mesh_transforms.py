from abc import ABC, abstractmethod
from functools import partial

import sympy as sp
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.manufactured_solutions import (
    _evaluate_list_of_sp_lambda, _evaluate_sp_lambda)


class OrthogonalCoordinateTransform(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def map_from_orthogonal(self, orth_samples):
        raise NotImplementedError()

    @abstractmethod
    def map_to_orthogonal(self, samples):
        raise NotImplementedError()

    @abstractmethod
    def unit_curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        raise NotImplementedError()

    def scale_factors(self, orth_samples):
        raise NotImplementedError()

    @abstractmethod
    def determinants(self, orth_samples):
        raise NotImplementedError()

    # def scale_factor(self, basis_id, orth_samples):
    #     raise NotImplementedError()

    # def _normalized_curvelinear_basis(self, orth_samples):
    #     basis = self.curvelinear_basis(orth_samples)
    #     for ii in range(self.nphys_vars()):
    #         scale = self.scale_factor(ii, orth_samples)
    #         basis[:, :, ii] = scale*basis[:, :, ii]
    #     return basis

    @staticmethod
    def _normal_sign(bndry_id):
        if bndry_id % 2 == 0:
            return -1.0
        return 1.0

    @staticmethod
    def _normal(map_to_orthogonal, unit_curvelinear_basis,
                bndry_id, sign, samples):
        orth_samples = map_to_orthogonal(samples)
        normals = sign*unit_curvelinear_basis(
            orth_samples)[..., bndry_id//2]
        return normals

    @abstractmethod
    def nphys_vars():
        raise NotImplementedError

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        return self._normal(
            self.map_to_orthogonal, self.unit_curvelinear_basis,
            bndry_id, sign, samples)

    # def scale_orthogonal_gradients(self, basis, orth_grads):
    #     return self._bkd.einsum("ijk,ik->ij", basis, orth_grads)

    def curvelinear_basis(self, orth_samples):
        unit_basis = self.unit_curvelinear_basis(orth_samples)
        scale_factors = self.scale_factors(orth_samples)
        basis = unit_basis/scale_factors[:, None, :]
        return basis

    def scale_orthogonal_gradients(self, orth_samples, orth_grads):
        basis = self.curvelinear_basis(orth_samples)
        return self._bkd.einsum("ijk,ik->ij", basis, orth_grads)

    def modify_quadrature_weights(self, orth_samples, orth_weights):
        if orth_weights.ndim != 2:
            raise ValueError("oth weights must be 2d column vector")
        basis = self.curvelinear_basis(orth_samples)
        dets = self._bkd.det(basis)
        #print(dets, "D1")
        dets = self.determinants(orth_samples)
        #print(dets, "D2")
        return (orth_weights[:, 0]/self._bkd.abs(dets))[:, None]

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class ScaleAndTranslationTransformMixIn:
    def __init__(self, orthog_ranges, ranges, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        # if following not satisfied it will mess up how normals are computed
        if self._bkd.any(orthog_ranges[1::2] <= orthog_ranges[::2]):
            msg = f"orthog_ranges {orthog_ranges} must be increasing"
            raise ValueError(msg)
        if self._bkd.any(ranges[1::2] <= ranges[::2]):
            msg = f"ranges {ranges} must be increasing"
            raise ValueError(msg)
        self._orthog_ranges = self._bkd.asarray(orthog_ranges)
        self._ranges = self._bkd.asarray(ranges)

    def _map_hypercube_samples(
            self, current_samples, current_ranges, new_ranges
    ):
        # no error checking or notion of active_vars
        clbs, cubs = current_ranges[0::2], current_ranges[1::2]
        nlbs, nubs = new_ranges[0::2], new_ranges[1::2]
        return ((current_samples.T-clbs)/(cubs-clbs)*(nubs-nlbs)+nlbs).T

    def map_from_orthogonal(self, orth_samples):
        return self._map_hypercube_samples(
            orth_samples, self._orthog_ranges, self._ranges)

    def map_to_orthogonal(self, samples):
        return self._map_hypercube_samples(
            samples, self._ranges, self._orthog_ranges)

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

    def curvelinear_basis(self, orth_samples):
        zeros = self._bkd.zeros((orth_samples.shape[1], 1))
        bases = []
        for ii in range(self.nphys_vars()):
            row = []
            for jj in range(self.nphys_vars()):
                if ii != jj:
                    row.append(zeros)
                else:
                    row.append(
                        1/self._bkd.full(
                            zeros.shape,
                            self._bkd.diff(
                                self._ranges[2*ii:2*ii+2]) /
                            self._bkd.diff(self._orthog_ranges[2*ii:2*ii+2])
                        )
                    )
            bases.append(self._bkd.hstack(row)[..., None])
        basis = self._bkd.dstack(bases)
        return basis

    def unit_curvelinear_basis(self, orth_samples):
        nsamples = orth_samples.shape[1]
        # print(self._normalized_curvelinear_basis(orth_samples)[0])
        # print(self._bkd.reshape(
        #     self._bkd.repeat(
        #         self._bkd.eye(self.nphys_vars()), (nsamples, 1)
        #     ),
        #     (nsamples, self.nphys_vars(), self.nphys_vars())
        # )[0])
        # assert False
        return self._bkd.reshape(
            self._bkd.repeat(
                self._bkd.eye(self.nphys_vars()), (nsamples, 1)
            ),
            (nsamples, self.nphys_vars(), self.nphys_vars())
        )

    def scale_factors(self, orth_samples):
        nsamples = orth_samples.shape[1]
        lbs, ubs = self._ranges[0::2], self._ranges[1::2]
        olbs, oubs = self._orthog_ranges[0::2], self._orthog_ranges[1::2]
        diffs = ubs-lbs
        odiffs = oubs-olbs
        ratios = diffs/odiffs
        return self._bkd._stack(
            [self._bkd.full((nsamples,), ratio) for ratio in ratios],
            axis=1
        )

    # def scale_factor(self, basis_id, orth_samples):
    #     nsamples = orth_samples.shape[1]
    #     lb, ub = 2*basis_id, 2*basis_id+2
    #     return self._bkd.full(
    #         (nsamples, 1),
    #         self._bkd.diff(
    #             self._ranges[lb:ub])/self._bkd.diff(self._orthog_ranges[lb:ub])
    #     )

    def determinants(self, orth_samples):
        nsamples = orth_samples.shape[1]
        lbs, ubs = self._ranges[0::2], self._ranges[1::2]
        olbs, oubs = self._orthog_ranges[0::2], self._orthog_ranges[1::2]
        det = 1/(self._bkd.prod(ubs-lbs)/self._bkd.prod(oubs-olbs))
        return self._bkd.full((nsamples), det)


# scale and translation are the only things you can do in 1D
class ScaleAndTranslationTransform1D(
        ScaleAndTranslationTransformMixIn, OrthogonalCoordinateTransform
):
    def nphys_vars(self):
        return 1


class OrthogonalCoordinateTransform2D(OrthogonalCoordinateTransform):
    def nphys_vars(self):
        return 2


class ScaleAndTranslationTransform2D(
        ScaleAndTranslationTransformMixIn, OrthogonalCoordinateTransform2D
):
    pass


class OrthogonalCoordinateTransform3D(OrthogonalCoordinateTransform):
    def nphys_vars(self):
        return 3


class ScaleAndTranslationTransform3D(
        ScaleAndTranslationTransformMixIn, OrthogonalCoordinateTransform3D
):
    pass
    # def curvelinear_basis(self, orth_samples):
    #     r, theta = orth_samples
    #     zeros = self._bkd.zeros((r.shape[0], 1))
    #     a11 = self._bkd.full(
    #         zeros.shape,
    #         self._bkd.diff(
    #             self._ranges[:2])/self._bkd.diff(self._orthog_ranges[:2])
    #     )
    #     a22 = self._bkd.full(
    #         zeros.shape,
    #         self._bkd.diff(
    #             self._ranges[2:4])/self._bkd.diff(self._orthog_ranges[2:4])
    #     )
    #     a33 = self._bkd.full(
    #         zeros.shape,
    #         self._bkd.diff(
    #             self._ranges[4:6])/self._bkd.diff(self._orthog_ranges[4:6])
    #     )
    #     basis = self._bkd.dstack(
    #         [self._bkd.hstack([1/a11, zeros, zeros])[..., None],
    #          self._bkd.hstack([zeros, 1/a22, zeros])[..., None],
    #          self._bkd.hstack([zeros, zeros, 1/a33])[..., None]
    #          ]
    #     )
    #     return basis

    # def scale_factor(self, basis_id, orth_samples):
    #     nsamples = orth_samples.shape[1]
    #     if basis_id == 0:
    #         lb, ub = 0, 2
    #     elif basis_id == 1:
    #         lb, ub = 2, 4
    #     else:  # basis_id == 2:
    #         lb, ub = 4, 6

    #     return self._bkd.full(
    #         (nsamples, 1),
    #         self._bkd.diff(
    #             self._ranges[lb:ub])/self._bkd.diff(self._orthog_ranges[lb:ub])
    #     )


class PolarTransform(OrthogonalCoordinateTransform2D):
    def map_from_orthogonal(self, orth_samples):
        if (orth_samples[1].max() > np.pi) or (orth_samples[1].min() < -np.pi):
            raise ValueError("theta must be in [-pi, pi]")
        if orth_samples[0].min() < 0:
            raise ValueError("r must be in [0, np.inf]")
        r, theta = orth_samples
        samples = self._bkd.stack(
            [r*self._bkd.cos(theta), r*self._bkd.sin(theta)], axis=0
        )
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        r = self._bkd.sqrt(x**2+y**2)
        azimuth = np.arctan2(y, x)
        orth_samples = self._bkd.stack([r, azimuth], axis=0)
        # II = self._bkd.where(y < 0)[0]
        # orth_samples[1, II] = -orth_samples[1, II]+2*np.pi
        return orth_samples

    def unit_curvelinear_basis(self, orth_samples):
        r, theta = orth_samples
        cos_t = self._bkd.cos(theta)
        sin_t = self._bkd.sin(theta)
        # basis is [b1, b2]
        # form b1, b2 using hstack then form basis using dstack
        unit_basis = self._bkd.dstack(
            [
                # first basis
                self._bkd.stack([cos_t, sin_t], axis=1)[..., None],
                # second basis
                self._bkd.stack([-sin_t, cos_t], axis=1)[..., None]
            ]
        )
        # unit_basis (nsamples, nvars, nbasis)
        # it just happens that nvars always equals nbasis
        return unit_basis

    def scale_factors(self, orth_samples):
        nsamples = orth_samples.shape[1]
        r = orth_samples[0]
        return self._bkd.stack((self._bkd.ones((nsamples,)), r), axis=1)

    # def scale_factor(self, basis_id, orth_samples):
    #     nsamples = orth_samples.shape[1]
    #     if basis_id == 0:
    #         return self._bkd.ones((nsamples, 1))
    #     r = orth_samples[0]
    #     return r[:, None]

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        r, theta = orth_samples
        r, theta = r[:, None], theta[:, None]
        cos_t = self._bkd.cos(theta)
        sin_t = self._bkd.sin(theta)
        # basis is [b1, b2]
        # form b1, b2 using hstack then form basis using dstack
        basis = self._bkd.dstack(
            [self._bkd.hstack([cos_t, sin_t])[..., None],  # first basis
             self._bkd.hstack([-sin_t/r, cos_t/r])[..., None]])  # second basis
        return basis

    def determinants(self, orth_samples):
        r = orth_samples[0]
        return 1/r


class EllipticalTransform(OrthogonalCoordinateTransform2D):
    def __init__(self, a, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._a = a

    def map_from_orthogonal(self, orth_samples):
        r, theta = orth_samples
        samples = self._bkd.vstack(
            [self._a*(self._bkd.cosh(r)*self._bkd.cos(theta))[None, :],
             self._a*(self._bkd.sinh(r)*self._bkd.sin(theta))[None, :]])
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        II = self._bkd.where(y < 0)[0]
        orth_samples = self._bkd.vstack(
            [
                self._bkd.real(
                    self._bkd.arccosh(x/self._a+1.j*y/self._a)
                )[None, :],
                self._bkd.imag(
                    self._bkd.arccosh(x/self._a+1.j*y/self._a)
                )[None, :]
            ]
        )
        orth_samples[1, II] = orth_samples[1, II]+2*np.pi
        return orth_samples

    def unit_curvelinear_basis(self, orth_samples):
        r, theta = orth_samples
        print(r)
        cosh_r = self._bkd.cosh(r)
        sinh_r = self._bkd.sinh(r)
        cos_t = self._bkd.cos(theta)
        sin_t = self._bkd.sin(theta)
        denom = self._a*self._bkd.sqrt(
            self._bkd.sinh(r)**2+self._bkd.sin(theta)**2
        )
        basis = self._bkd.dstack(
            [
                self._bkd.stack(
                    [sinh_r*cos_t/denom, cosh_r*sin_t/denom],
                    axis=1
                )[..., None],
                self._bkd.stack(
                    [-cosh_r*sin_t/denom, sinh_r*cos_t/denom],
                    axis=1
                )[..., None]])
        return basis

    def scale_factors(self, orth_samples):
        r, theta = orth_samples
        return np.stack(
            [
                self._a*self._bkd.sqrt(
                    self._bkd.sinh(r)**2+self._bkd.sin(theta)**2
                )
            ]*2,
            axis=1
        )

    # def scale_factor(self, basis_id, orth_samples):
    #     r, theta = orth_samples
    #     return self._a*self._bkd.sqrt(
    #         self._bkd.sinh(r)**2+self._bkd.sin(theta)**2
    #     )[:, None]

    def determinants(self, orth_samples):
        r, theta = orth_samples
        return 1/(
            self._a**2*(
                self._bkd.sinh(r)**2+self._bkd.sin(theta)**2
            )
        )[:, None]


class SympyTransform2D(OrthogonalCoordinateTransform2D):
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
        basis = self._bkd.dstack(
            [basis(orth_samples)[..., None] for basis in self._bases])
        return basis

    def __repr__(self):
        return "{0}, {1}, {2}".format(
            self.__class__.__name__, self._from_orth_jacobian_expr,
            self._to_orth_jacobian_expr)


class CompositionTransform(OrthogonalCoordinateTransform):
    def __init__(self, transforms):
        for transform in transforms:
            if transform.nphys_vars() != transforms[0].nphys_vars():
                raise ValueError("Transforms must have the same nphys_vars")
        self._bkd = transforms[0]._bkd
        self._transforms = transforms
        # self._normal_sign = OrthogonalCoordinateTransform2D._normal_sign
        self._normal_sign = transforms[0]._normal_sign

    def nphys_vars(self):
        return self._transforms[0].nphys_vars()

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
        basis = self._bkd.einsum("ijk,ikl->ijl", basis1, basis2)
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

    def unit_curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        basis = self._transforms[0].unit_curvelinear_basis(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples)
            new_basis = transform.unit_curvelinear_basis(orth_samples)
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

    # ef scale_orthogonal_gradients(self, basis, orth_grads):
    #    return OrthogonalCoordinateTransform2D.scale_orthogonal_gradients(
    #        basis, orth_grads)

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        normals = OrthogonalCoordinateTransform2D._normal(
            self.map_to_orthogonal,
            self.unit_curvelinear_basis,
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

    def determinants(self, orth_samples):
        dets = self._transforms[0].curvelinear_basis(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples)
            new_dets = transform.determinants(orth_samples)
            dets *= new_dets
        return dets

class SphericalTransform(OrthogonalCoordinateTransform3D):
    def map_from_orthogonal(self, orth_samples):
        if orth_samples[2].min() < -np.pi or orth_samples[2].max() > np.pi:
            raise ValueError("phi must be in [-pi, pi]")
        if orth_samples[1].max() > np.pi or orth_samples[1].min() <= 0:
            raise ValueError("theta must be in [0, pi]")
        if orth_samples[0].min() < 0:
            raise ValueError("r must be in [0, np.inf]")
        # r, theta, phi = orth_samples
        r, elevation, azimuth = orth_samples
        x = r * self._bkd.sin(elevation) * self._bkd.cos(azimuth)
        y = r * self._bkd.sin(elevation) * self._bkd.sin(azimuth)
        z = r * self._bkd.cos(elevation)
        samples = self._bkd.stack([x, y, z], axis=0)
        return samples

    def map_to_orthogonal(self, samples):
        x, y, z = samples
        r = self._bkd.sqrt(x**2+y**2+z**2)
        elevation = self._bkd.arccos(z/r)
        azimuth = np.arctan2(y, x)
        orth_samples = self._bkd.stack([r, elevation, azimuth], axis=0)
        return orth_samples

    def scale_factor(self, basis_id, orth_samples):
        nsamples = orth_samples.shape[1]
        r, elevation, azimuth = orth_samples
        #den = r**2*self._bkd.sin(elevation)
        #return den[:, None]
        if basis_id == 0:
            return self._bkd.ones((nsamples, 1))
        return r[:, None]

    def determinants(self, orth_samples):
        r, elevation, azimuth = orth_samples
        return r**2*self._bkd.sin(elevation)

    def curvelinear_basis(self, orth_samples):
        # this is A^{-1} in my notes
        r, elevation, azimuth = orth_samples
        r, elevation, azimuth = (
            r[:, None], elevation[:, None], azimuth[:, None])
        cos_az = self._bkd.cos(azimuth)
        sin_az = self._bkd.sin(azimuth)
        cos_el = self._bkd.cos(elevation)
        sin_el = self._bkd.sin(elevation)
        # print(1/(r**2*sin_el)[:, 0], 'd')
        # basis is [b1, b2]
        # form b1, b2 using hstack then form basis using dstack
        # den = (r**2*sin_el)
        # print(den.shape)
        # basis = self._bkd.dstack(
        #     [
        #         # first basis
        #         self._bkd.hstack(
        #             [cos_az*sin_el/den, sin_az*sin_el/den, cos_el/den]
        #         )[..., None],
        #         # second basis
        #         self._bkd.hstack(
        #            [-r*sin_az/den, r*cos_az*sin_el/den, 0*r]
        #         )[..., None],
        #         # third basis
        #         self._bkd.hstack(
        #             [r*cos_az*cos_el/den, r*sin_az*cos_el/den, -r*sin_el/den]
        #         )[..., None],
        #     ]
        # )
        basis = self._bkd.dstack(
            [
                # first basis
                self._bkd.hstack(
                    [cos_az*sin_el, sin_az*sin_el, cos_el]
                )[..., None],
                # second basis
                self._bkd.hstack(
                   [-sin_az*sin_el, cos_az*sin_el, 0*r]
                )[..., None],
                # third basis
                self._bkd.hstack(
                    [cos_az*cos_el, sin_az*cos_el, -sin_el]
                )[..., None],
            ]
        )
        return basis

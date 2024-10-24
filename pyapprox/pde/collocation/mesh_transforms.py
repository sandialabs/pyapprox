from abc import ABC, abstractmethod
from functools import partial

import sympy as sp
import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.manufactured_solutions import (
    _evaluate_list_of_sp_lambda,
    _evaluate_sp_lambda,
)


class OrthogonalCoordinateTransform(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def map_from_orthogonal(self, orth_samples: Array) -> Array:
        """
        Parameters
        ----------
        orth_samples : Array (nvars, nsamples)

        Return
        ------
        samples : Array (nvars, nsamples)
        """
        raise NotImplementedError()

    @abstractmethod
    def map_to_orthogonal(self, samples: Array) -> Array:
        """
        Parameters
        ----------
        samples : Array (nvars, nsamples)

        Return
        ------
        orth_vals : Array (nvars, nsamples)
        """
        raise NotImplementedError()

    @abstractmethod
    def unit_curvelinear_basis(self, orth_samples: Array) -> Array:
        """
        Parameters
        ----------
        orth_samples : Array (nvars, nsamples)

        Return
        ------
        basis_vals : Array (nsamples, nvars, nvars)
        """
        raise NotImplementedError()

    def scale_factors(self, orth_samples: Array) -> Array:
        """
        Parameters
        ----------
        orth_samples : Array (nvars, nsamples)

        Return
        ------
        det_vals : Array (nsamples, nvars)
        """
        raise NotImplementedError()

    def determinants(self, orth_samples: Array) -> Array:
        """
        Parameters
        ----------
        orth_samples : Array (nvars, nsamples)

        Return
        ------
        det_vals : Array (nsamples, )
        """
        return self._bkd.prod(self.scale_factors(orth_samples), axis=1)

    @staticmethod
    def _normal_sign(bndry_id):
        if bndry_id % 2 == 0:
            return -1.0
        return 1.0

    @staticmethod
    def _normal(
        map_to_orthogonal, unit_curvelinear_basis, bndry_id, sign, samples
    ):
        orth_samples = map_to_orthogonal(samples)
        normals = (
            sign * unit_curvelinear_basis(orth_samples)[..., bndry_id // 2]
        )
        return normals

    @abstractmethod
    def nphys_vars(self) -> int:
        """
        Return
        ------
        nvars: int
            The number of physical variables
        """
        raise NotImplementedError

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        return self._normal(
            self.map_to_orthogonal,
            self.unit_curvelinear_basis,
            bndry_id,
            sign,
            samples,
        )

    def gradient_factors(self, orth_samples: Array):
        unit_basis = self.unit_curvelinear_basis(orth_samples)
        scale_factors = self.scale_factors(orth_samples)
        basis = unit_basis / scale_factors[:, None, :]
        return basis

    def scale_orthogonal_gradients(self, orth_samples, orth_grads):
        basis = self.gradient_factors(orth_samples)
        return self._bkd.einsum("ijk,ik->ij", basis, orth_grads)

    def modify_quadrature_weights(self, orth_samples, orth_weights):
        if orth_weights.ndim != 2:
            raise ValueError("oth weights must be 2d column vector")
        dets = self.determinants(orth_samples)
        return (orth_weights[:, 0] * self._bkd.abs(dets))[:, None]

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class ScaleAndTranslationTransformMixIn:
    def __init__(self, orthog_ranges, ranges, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        # if following not satisfied it will mess up how normals are computed
        if self._bkd.any(
            self._bkd.array(orthog_ranges[1::2])
            <= self._bkd.array(orthog_ranges[::2])
        ):
            msg = f"orthog_ranges {orthog_ranges} must be increasing"
            raise ValueError(msg)
        if self._bkd.any(
            self._bkd.array(ranges[1::2]) <= self._bkd.array(ranges[::2])
        ):
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
        return (
            (current_samples.T - clbs) / (cubs - clbs) * (nubs - nlbs) + nlbs
        ).T

    def map_from_orthogonal(self, orth_samples: Array) -> Array:
        return self._map_hypercube_samples(
            orth_samples, self._orthog_ranges, self._ranges
        )

    def map_to_orthogonal(self, samples: Array) -> Array:
        return self._map_hypercube_samples(
            samples, self._ranges, self._orthog_ranges
        )

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
            self.__class__.__name__,
            self._orthog_ranges_repr(),
            self._ranges_repr(),
        )

    def unit_curvelinear_basis(self, orth_samples: Array) -> Array:
        nsamples = orth_samples.shape[1]
        return self._bkd.reshape(
            self._bkd.repeat(self._bkd.eye(self.nphys_vars()), (nsamples, 1)),
            (nsamples, self.nphys_vars(), self.nphys_vars()),
        )

    def scale_factors(self, orth_samples: Array) -> Array:
        nsamples = orth_samples.shape[1]
        lbs, ubs = self._ranges[0::2], self._ranges[1::2]
        olbs, oubs = self._orthog_ranges[0::2], self._orthog_ranges[1::2]
        diffs = ubs - lbs
        odiffs = oubs - olbs
        ratios = diffs / odiffs
        return self._bkd.stack(
            [self._bkd.full((nsamples,), ratio) for ratio in ratios], axis=1
        )

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


class FixedScaleAndTranslationTransform3D(ScaleAndTranslationTransform3D):
    # fix one of the dimensions to a single value
    def __init__(self, orthog_ranges, ranges, backend=NumpyLinAlgMixin):
        OrthogonalCoordinateTransform3D.__init__(self, backend)
        self._orthog_ranges = self._bkd.asarray(orthog_ranges)
        self._ranges = self._bkd.asarray(ranges)
        self._fixed_idx = self._bkd.where(
            self._bkd.abs(self._ranges[::2]-self._ranges[1::2]) == 0
        )[0]
        self._active_idx = self._bkd.where(
            self._bkd.abs(self._ranges[::2]-self._ranges[1::2]) != 0
        )[0]

    def _map_hypercube_samples(
        self, current_samples, current_ranges, new_ranges
    ):
        # no error checking or notion of active_vars
        clbs, cubs = current_ranges[0::2], current_ranges[1::2]
        nlbs, nubs = new_ranges[0::2], new_ranges[1::2]
        new_samples = self._bkd.empty((3, current_samples.shape[1]))
        new_samples[self._fixed_idx] = self._ranges[self._fixed_idx]
        idx = self._active_idx
        new_samples[idx] = (
            (current_samples[idx].T - clbs[idx]) / (
                cubs[idx] - clbs[idx]) * (nubs[idx] - nlbs[idx]) + nlbs[idx]
        ).T
        return new_samples

    def scale_factors(self, orth_samples: Array):
        nsamples = orth_samples.shape[1]
        lbs, ubs = self._ranges[0::2], self._ranges[1::2]
        olbs, oubs = self._orthog_ranges[0::2], self._orthog_ranges[1::2]
        diffs = ubs - lbs
        odiffs = oubs - olbs
        idx = self._active_idx
        ratios = self._bkd.empty(diffs.shape)
        ratios[idx] = diffs[idx] / odiffs[idx]
        # value does not matter as this class will only composed
        # with physics with grad that does not depend on fixed vars
        ratios[self._fixed_idx] = 1
        return self._bkd.stack(
            [self._bkd.full((nsamples,), ratio) for ratio in ratios], axis=1
        )


class PolarTransform(OrthogonalCoordinateTransform2D):
    def map_from_orthogonal(self, orth_samples: Array):
        if (orth_samples[1].max() > np.pi) or (orth_samples[1].min() < -np.pi):
            raise ValueError("theta must be in [-pi, pi]")
        if orth_samples[0].min() < 0:
            raise ValueError("r must be in [0, np.inf]")
        r, theta = orth_samples
        samples = self._bkd.stack(
            [r * self._bkd.cos(theta), r * self._bkd.sin(theta)], axis=0
        )
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        r = self._bkd.sqrt(x**2 + y**2)
        azimuth = self._bkd.arctan2(y, x)
        orth_samples = self._bkd.stack([r, azimuth], axis=0)
        return orth_samples

    def unit_curvelinear_basis(self, orth_samples: Array):
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
                self._bkd.stack([-sin_t, cos_t], axis=1)[..., None],
            ]
        )
        # unit_basis (nsamples, nvars, nbasis)
        # it just happens that nvars always equals nbasis
        return unit_basis

    def scale_factors(self, orth_samples: Array):
        nsamples = orth_samples.shape[1]
        r = orth_samples[0]
        return self._bkd.stack((self._bkd.ones((nsamples,)), r), axis=1)


class EllipticalTransform(OrthogonalCoordinateTransform2D):
    def __init__(self, a, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._a = a

    def map_from_orthogonal(self, orth_samples: Array):
        r, theta = orth_samples
        samples = self._bkd.vstack(
            [
                self._a * (self._bkd.cosh(r) * self._bkd.cos(theta))[None, :],
                self._a * (self._bkd.sinh(r) * self._bkd.sin(theta))[None, :],
            ]
        )
        return samples

    def map_to_orthogonal(self, samples):
        x, y = samples
        II = self._bkd.where(y < 0)[0]
        orth_samples = self._bkd.vstack(
            [
                self._bkd.real(
                    self._bkd.arccosh(x / self._a + 1.0j * y / self._a)
                )[None, :],
                self._bkd.imag(
                    self._bkd.arccosh(x / self._a + 1.0j * y / self._a)
                )[None, :],
            ]
        )
        orth_samples[1, II] = orth_samples[1, II] + 2 * np.pi
        return orth_samples

    def unit_curvelinear_basis(self, orth_samples: Array):
        r, theta = orth_samples
        cosh_r = self._bkd.cosh(r)
        sinh_r = self._bkd.sinh(r)
        cos_t = self._bkd.cos(theta)
        sin_t = self._bkd.sin(theta)
        denom = self._a * self._bkd.sqrt(
            self._bkd.sinh(r) ** 2 + self._bkd.sin(theta) ** 2
        )
        basis = self._bkd.dstack(
            [
                self._bkd.stack(
                    [sinh_r * cos_t / denom, cosh_r * sin_t / denom], axis=1
                )[..., None],
                self._bkd.stack(
                    [-cosh_r * sin_t / denom, sinh_r * cos_t / denom], axis=1
                )[..., None],
            ]
        )
        return basis

    def scale_factors(self, orth_samples: Array):
        r, theta = orth_samples
        return np.stack(
            [
                self._a
                * self._bkd.sqrt(
                    self._bkd.sinh(r) ** 2 + self._bkd.sin(theta) ** 2
                )
            ]
            * 2,
            axis=1,
        )


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
            [expr.diff(symb) for expr in exprs] for symb in symbs
        ]
        return sp.Matrix(jacobian_exprs)

    def __init__(
        self,
        map_from_orthogonal_strings,
        map_to_orthogonal_strings,
        backend=NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._symbs = sp.symbols(["_x_", "_y_"])
        self._orth_symbs = sp.symbols(["_r_", "_t_"])
        assert (
            "_r_" in map_from_orthogonal_strings[0]
            and "_t_" in map_from_orthogonal_strings[1]
        )
        assert (
            "_x" in map_to_orthogonal_strings[0]
            and "_y_" in map_to_orthogonal_strings[1]
        )
        self._map_from_orthogonal_trans = self._lambdify_map(
            map_from_orthogonal_strings, self._orth_symbs
        )
        self._map_to_orthogonal_trans = self._lambdify_map(
            map_to_orthogonal_strings, self._symbs
        )

        self._from_orth_jacobian_expr = self._jacobian_expr(
            map_from_orthogonal_strings, self._orth_symbs
        )
        self._to_orth_jacobian_expr = sp.simplify(
            self._from_orth_jacobian_expr.inv()
        )
        self._bases = [
            partial(
                _evaluate_list_of_sp_lambda,
                [
                    sp.lambdify(self._orth_symbs, expr, "numpy")
                    for expr in self._to_orth_jacobian_expr[:, ii]
                ],
            )
            for ii in range(2)
        ]
        self._scale_factor_expr = [
            1
            / sp.simplify(
                sp.sqrt(
                    self._to_orth_jacobian_expr[:, ii].dot(
                        self._to_orth_jacobian_expr[:, ii]
                    )
                )
            )
            for ii in range(2)
        ]
        self._scale_factors = [
            partial(
                _evaluate_sp_lambda,
                sp.lambdify(self._orth_symbs, expr, "numpy"),
            )
            for expr in self._scale_factor_expr
        ]

    def map_from_orthogonal(self, orth_samples: Array):
        return self._map_from_orthogonal_trans(orth_samples).T

    def map_to_orthogonal(self, samples):
        return self._map_to_orthogonal_trans(samples).T

    def scale_factors(self, orth_samples: Array):
        return self._bkd.hstack(
            [factor(orth_samples) for factor in self._scale_factors]
        )

    def unit_curvelinear_basis(self, orth_samples: Array):
        scale_factors = self.scale_factors(orth_samples)
        basis = self._bkd.dstack(
            [
                s[:, None, None] * (basis(orth_samples))[..., None]
                for basis, s in zip(self._bases, scale_factors.T)
            ]
        )
        return basis

    def __repr__(self):
        return "{0}, {1}, {2}".format(
            self.__class__.__name__,
            self._from_orth_jacobian_expr,
            self._to_orth_jacobian_expr,
        )


class CompositionTransform(OrthogonalCoordinateTransform):
    def __init__(self, transforms):
        for transform in transforms:
            if transform.nphys_vars() != transforms[0].nphys_vars():
                raise ValueError("Transforms must have the same nphys_vars")
        self._bkd = transforms[0]._bkd
        self._transforms = transforms
        self._normal_sign = transforms[0]._normal_sign
        self._orthog_ranges = transforms[0]._orthog_ranges

    def nphys_vars(self):
        return self._transforms[0].nphys_vars()

    def map_from_orthogonal(self, orth_samples: Array):
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

    def gradient_factors(self, orth_samples: Array):
        basis = self._transforms[0].gradient_factors(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples
            )
            new_basis = transform.gradient_factors(orth_samples)
            basis = self._basis_product(new_basis, basis)
        return basis

    def unit_curvelinear_basis(self, orth_samples: Array):
        basis = self._transforms[0].unit_curvelinear_basis(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples
            )
            new_basis = transform.unit_curvelinear_basis(orth_samples)
            basis = self._basis_product(new_basis, basis)
        return basis

    def normal(self, bndry_id, samples):
        sign = self._normal_sign(bndry_id)
        normals = OrthogonalCoordinateTransform2D._normal(
            self.map_to_orthogonal,
            self.unit_curvelinear_basis,
            bndry_id,
            sign,
            samples,
        )
        return normals

    def modify_quadrature_weights(self, orth_samples, orth_weights):
        weights = self._transforms[0].modify_quadrature_weights(
            orth_samples, orth_weights
        )
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples
            )
            weights = self._transforms[ii + 1].modify_quadrature_weights(
                orth_samples, weights
            )
        return weights

    def __repr__(self):
        return self.__class__.__name__ + "[{0}]".format(
            ", ".join(map("{}".format, self._transforms))
        )

    def scale_factors(self, orth_samples: Array) -> Array:
        factors = self._transforms[0].scale_factors(orth_samples)
        for ii, transform in enumerate(self._transforms[1:]):
            orth_samples = self._transforms[ii].map_from_orthogonal(
                orth_samples
            )
            factors *= transform.scale_factors(orth_samples)
        return factors


class SphericalTransform(OrthogonalCoordinateTransform3D):
    def map_from_orthogonal(self, orth_samples: Array):
        if orth_samples[0].min() < 0:
            raise ValueError("radius must be in [0, np.inf]")
        if orth_samples[1].min() < -np.pi or orth_samples[1].max() >= np.pi:
            raise ValueError("azimuth must be in [-pi, pi]")
        if orth_samples[2].max() > np.pi or orth_samples[2].min() < 0:
            raise ValueError("elevation must be in [0, pi]")
        # r, theta, phi = orth_samples
        r, azimuth, elevation = orth_samples
        x = r * self._bkd.sin(elevation) * self._bkd.cos(azimuth)
        y = r * self._bkd.sin(elevation) * self._bkd.sin(azimuth)
        z = r * self._bkd.cos(elevation)
        samples = self._bkd.stack([x, y, z], axis=0)
        return samples

    def map_to_orthogonal(self, samples):
        x, y, z = samples
        r = self._bkd.sqrt(x**2 + y**2 + z**2)
        elevation = self._bkd.arccos(z / r)
        azimuth = np.arctan2(y, x)
        orth_samples = self._bkd.stack([r, azimuth, elevation], axis=0)
        return orth_samples

    def scale_factors(self, orth_samples: Array):
        nsamples = orth_samples.shape[1]
        r, azimuth, elevation = orth_samples
        return self._bkd.stack(
            [
                self._bkd.ones((nsamples,)),
                r * self._bkd.sin(elevation),
                r,
            ],
            axis=1,
        )

    def unit_curvelinear_basis(self, orth_samples: Array):
        r, azimuth, elevation = orth_samples
        r, azimuth, elevation = (
            r[:, None],
            azimuth[:, None],
            elevation[:, None],
        )
        cos_az = self._bkd.cos(azimuth)
        sin_az = self._bkd.sin(azimuth)
        cos_el = self._bkd.cos(elevation)
        sin_el = self._bkd.sin(elevation)
        basis = self._bkd.dstack(
            [
                # first basis (radius)
                self._bkd.hstack([cos_az * sin_el, sin_az * sin_el, cos_el])[
                    ..., None
                ],
                # second basis (azimuth)
                self._bkd.hstack([-sin_az, cos_az, 0 * r])[..., None],
                # third basis (elevation)
                self._bkd.hstack([cos_az * cos_el, sin_az * cos_el, -sin_el])[
                    ..., None
                ],
            ]
        )
        return basis


class SurfaceTransform(OrthogonalCoordinateTransform):
    """
    Surface transform for maps on surfaces and integration. 
    Unlike other transforms this does not support gradients or normals
    """
    def __init__(
            self,
            trans: OrthogonalCoordinateTransform3D,
            inactive_idx: int,
            inactive_val: float,
    ):
        self._trans = trans
        self._bkd = trans._bkd
        self._inactive_idx = inactive_idx
        self._active_idx = self._bkd.delete(self._bkd.arange(3), inactive_idx)
        self._inactive_val = inactive_val

    def nphys_vars(self) -> int:
        return 2

    def _expand_orth(self, orth_samples_2d: Array) -> Array:
        orth_samples_3d = self._bkd.empty((3, orth_samples_2d.shape[1]))
        orth_samples_3d[self._inactive_idx] = self._bkd.full(
            (orth_samples_2d.shape[1],), -1,
        )
        orth_samples_3d[self._active_idx] = orth_samples_2d
        return orth_samples_3d

    def map_from_orthogonal(self, orth_samples: Array) -> Array:
        return self._trans.map_from_orthogonal(self._expand_orth(orth_samples))

    def map_to_orthogonal(self, samples: Array) -> Array:
        return self._trans.map_to_orthogonal(samples)[self._active_idx]

    def unit_curvelinear_basis(self, orth_samples: Array) -> Array:
        raise NotImplementedError

    def gradient_factors(self, orth_samples: Array):
        raise NotImplementedError

    def normal(self, bndry_id, samples):
        raise NotImplementedError

    def scale_factors(self, orth_samples: Array) -> Array:
        return self._trans.scale_factors(
            self._expand_orth(orth_samples)
        )[:, self._active_idx]

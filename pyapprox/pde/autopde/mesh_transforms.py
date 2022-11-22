import numpy as np
from functools import partial
from abc import ABC, abstractmethod


from pyapprox.variables.transforms import _map_hypercube_samples


def _map_hypercube_derivatives_scale(
        current_range, new_range, samples, pkg=np):
    current_len = current_range[1]-current_range[0]
    new_len = new_range[1]-new_range[0]
    map_derivs = pkg.full(
        (samples.shape[1], ), (new_len/current_len), dtype=pkg.double)
    return map_derivs


def vertical_transform_2D_mesh(xdomain_bounds, bed_fun, surface_fun,
                               canonical_samples):
    samples = np.empty_like(canonical_samples)
    xx, yy = canonical_samples[0], canonical_samples[1]
    samples[0] = (xx+1)/2*(
        xdomain_bounds[1]-xdomain_bounds[0])+xdomain_bounds[0]
    bed_vals = bed_fun(samples[0:1])[:, 0]
    samples[1] = (yy+1)/2*(surface_fun(samples[0:1])[:, 0]-bed_vals)+bed_vals
    return samples


def vertical_transform_2D_mesh_inv(xdomain_bounds, bed_fun, surface_fun,
                                   samples):
    canonical_samples = np.empty_like(samples)
    uu = samples[0]
    canonical_samples[0] = 2*(uu-xdomain_bounds[0])/(
        xdomain_bounds[1]-xdomain_bounds[0])-1
    bed_vals = bed_fun(samples[0:1])[:, 0]
    canonical_samples[1] = 2*(samples[1]-bed_vals)/(
        surface_fun(samples[0:1])[:, 0]-bed_vals)-1
    return canonical_samples


def vertical_transform_2D_mesh_inv_dxdu(xdomain_bounds, samples):
    return np.full(samples.shape[1], 2/(xdomain_bounds[1]-xdomain_bounds[0]))


def vertical_transform_2D_mesh_inv_dydu(
        bed_fun, surface_fun, bed_grad_u, surf_grad_u, samples):
    surf_vals = surface_fun(samples[:1])[:, 0]
    bed_vals = bed_fun(samples[:1])[:, 0]
    return 2*(bed_grad_u(samples[:1])[:, 0]*(samples[1]-surf_vals) +
              surf_grad_u(samples[:1])[:, 0]*(bed_vals-samples[1]))/(
                  surf_vals-bed_vals)**2


def vertical_transform_2D_mesh_inv_dxdv(samples):
    return np.zeros(samples.shape[1])


def vertical_transform_2D_mesh_inv_dydv(bed_fun, surface_fun, samples):
    surf_vals = surface_fun(samples[:1])[:, 0]
    bed_vals = bed_fun(samples[:1])[:, 0]
    return 2/(surf_vals-bed_vals)



def elliptical_inv_scale_factor(a, elp_samples):
    u, v = elp_samples
    return 1./(a*np.sqrt(np.sinh(u)**2+np.sin(v)**2))



from pyapprox.pde.autopde.solvers import Function
def get_ellipitical_transform_functions(a, ranges, hemisphere):
    """
    transform maps orthogonal coordinates to cartesian coordinates
    transform_inv maps cartesian coordinates to orthogonal coordinates
    """
    ranges = np.asarray(ranges)
    return (
        partial(from_elliptical, a, ranges, hemisphere),
        partial(to_elliptical, a, ranges, hemisphere),
        [[Function(partial(elliptical_inv_scale_factor, a), oned=True)]*2,
         [Function(partial(elliptical_inv_scale_factor, a), oned=True)]*2],
        [Function(partial(elliptical_cartesian_normal, a), oned=True)]*4)


class OrthogonalCoordinateTransform2D(ABC):
    @abstractmethod
    def map_from_orthogonal(self):
        raise NotImplementedError()

    @abstractmethod
    def map_to_orthogonal(self):
        raise NotImplementedError()

    @abstractmethod
    def scale_orthogonal_derivatives(self):
        raise NotImplementedError()


class PolarTransform(OrthogonalCoordinateTransform2D):
    def map_from_orthogonal(self, orth_samples):
        if orth_samples[1].max() > 2*np.pi:
            raise ValueError("theta must be in [0, 2*pi]")
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

    def scale_orthogonal_derivatives(self, orth_samples):
        r, theta = orth_samples
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        scales = 1/r*np.hstack(
            [np.hstack([(r*cos_t)[None, :], -sin_t[None, :]]),
             np.hstack([r*sin_t[None, :], cos_t[None, :]])])
        return scales

    def normal(self, bndry_id, samples):
        r, theta = self.map_to_orthogonal(samples)
        cos_t = np.cos(theta)[:, None]
        sin_t = np.sin(theta)[:, None]
        if bndry_id % 2 == 0:
            sign = -1.0
        else:
            sign = 1.0
        if bndry_id >= 2:
            return sign*np.hstack([cos_t, sin_t])
        return sign*np.hstack([-sin_t, cos_t])


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

    def scale_orthogonal_derivatives(self, orth_samples):
        r, theta = orth_samples
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        scales = 1/r*np.hstack(
            [np.hstack([(r*cos_t)[None, :], -sin_t[None, :]]),
             np.hstack([r*sin_t[None, :], cos_t[None, :]])])
        return scales

    def normal(self, bndry_id, samples):
        r, theta = self.map_to_orthogonal(samples)
        denom = self._a*np.sqrt(np.sinh(r)**2+np.sin(theta)**2)[:, None]
        cosh_r = np.cosh(r)[:, None]
        sinh_r = np.sinh(r)[:, None]
        cos_t = np.cos(theta)[:, None]
        sin_t = np.sin(theta)[:, None]
        if bndry_id % 2 == 0:
            sign = -1.0
        else:
            sign = 1.0
        if bndry_id >= 2:
            return sign*np.hstack([sinh_r*cos_t/denom, cosh_r*sin_t/denom])
        return sign*np.hstack([-cosh_r*sin_t/denom, sinh_r*cos_t/denom])



class ScaleAndTranslationTransform(OrthogonalCoordinateTransform2D):
    def __init__(self, orthog_ranges, ranges, transform):
        self._orthog_ranges = np.asarray(orthog_ranges)
        self._ranges = np.asarray(ranges)
        self._transform = transform

    def map_from_orthogonal(self, orth_samples):
        samples = _map_hypercube_samples(
            orth_samples, self._orthog_ranges, self._ranges)
        # print(samples.min(axis=1), samples.max(axis=1), 's1')
        return self._transform.map_from_orthogonal(samples)

    def map_to_orthogonal(self, samples):
        # print(samples.min(axis=1), samples.max(axis=1), 's')
        orth_samples = self._transform.map_to_orthogonal(samples)
        # print(orth_samples.min(axis=1), orth_samples.max(axis=1), 'o')
        return _map_hypercube_samples(
            orth_samples, self._ranges, self._orthog_ranges)

    def scale_orthogonal_derivatives(self, orth_samples):
        scales = self._transform.scale_orthogonal_derivatives(orth_samples)
        r, theta = orth_samples
        scales = np.hstack(
            [np.hstack([_map_hypercube_derivatives_scale(
                self._orthog_ranges[:2], self._ranges[:2])[None, :], 0*r]),
             np.hstack([0*theta, _map_hypercube_derivatives_scale(
                 self._orthog_ranges[2:], self._ranges[2:])[None, :]])])
        return scales

    def normal(self, bndry_id, samples):
        return self._transform.normal(bndry_id, samples)


class IdentityTransform(OrthogonalCoordinateTransform2D):
    def map_from_orthogonal(self, orth_samples):
        return orth_samples

    def map_to_orthogonal(self, samples):
        return samples

    def scale_orthogonal_derivatives(self, orth_samples):
        ones = np.ones((1, orth_samples.shape[1]))
        zeros = np.zeros((1, orth_samples.shape[1]))
        scales = np.hstack(
            [np.hstack([ones, zeros]),
             np.hstack([zeros, ones])])
        return scales

    def normal(self, bndry_id, samples):
        zeros = np.zeros((samples.shape[1], 1))
        if bndry_id % 2 == 0:
            sign = np.full((samples.shape[1], 1), -1.0)
        else:
            sign = 1.0
        if bndry_id >= 2:
            return sign*np.hstack([zeros, sign])
        return sign*np.vstack([sign, zeros])

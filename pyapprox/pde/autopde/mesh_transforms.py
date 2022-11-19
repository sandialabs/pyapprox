import numpy as np
from functools import partial
from pyapprox.variables.transforms import _map_hypercube_samples


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


def from_elliptical(a, ranges, hemisphere, elp_samples):
    hemisphere = "low"
    assert ranges[2] >= 0 and ranges[3] <= np.pi
    u, v = _map_hypercube_samples(
        elp_samples, np.array([-1, 1, -1, 1]), ranges)
    cart_samples = np.vstack(
        [a*(np.cosh(u)*np.cos(v))[None, :],
         a*(np.sinh(u)*np.sin(v))[None, :]])
    if hemisphere == "low":
        cart_samples[1] *= -1
    return cart_samples


def to_elliptical(a, ranges, hemisphere, cart_samples):
    x, y = cart_samples
    if hemisphere == "low":
        cart_samples[1] *= -1
    elp_samples = np.vstack(
        [np.real(np.arccosh(x/a+1.j*y/a))[None, :],
         np.imag(np.arccosh(x/a+1.j*y/a))[None, :]])
    return _map_hypercube_samples(
        elp_samples, ranges, np.array([-1, 1, -1, 1]))


def elliptical_inv_scale_factor(a, elp_samples):
    u, v = elp_samples
    return 1./(a*np.sqrt(np.sinh(u)**2+np.sin(v)**2))


def elliptical_cartesian_normal(cart_samples):
    """
    get normal, in cartesian coordinates,
    of a domain mapped from elliptical orthogonal coordinates
    """
    normal = np.vstack([
        ellipise_grad(a, cart_samples),
        np.full((1, cart_samples.shape[1]), -1)])
    unit_normal = normal/np.linalg.norm(normal, axis=0)
    return unit_normal


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

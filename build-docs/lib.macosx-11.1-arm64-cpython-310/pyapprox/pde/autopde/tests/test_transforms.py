import unittest
import torch
import numpy as np
from functools import partial

from pyapprox.pde.autopde.mesh_transforms import (
    ScaleAndTranslationTransform, PolarTransform,
    EllipticalTransform, CompositionTransform, SympyTransform)
from pyapprox.util.utilities import (
    cartesian_product, approx_fprime, outer_product)


class TestMeshTransforms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    @staticmethod
    def _check_gradients(transform, orth_samples, samples):
        def _fun(samples):
            return np.sum(samples**2, axis=0)[:, None]

        grad_fd = []
        for sample in samples.T:
            grad_fd.append(approx_fprime(sample[:, None], _fun))
        grad_fd = np.asarray(grad_fd)
        # print(grad_fd, "f_x FD")

        def _orth_fun(orth_samples):
            return _fun(transform.map_from_orthogonal(orth_samples))

        orth_grad_fd = []
        for orth_sample in orth_samples.T:
            orth_grad_fd.append(approx_fprime(orth_sample[:, None], _orth_fun))
        basis = transform.curvelinear_basis(orth_samples)
        orth_grad_fd = np.asarray(orth_grad_fd)
        grad = transform.scale_orthogonal_gradients(basis, orth_grad_fd)
        # print(grad, "fx")
        # print(grad_fd)
        # print(grad_fd-grad)
        tol = 1e-7
        assert np.allclose(grad_fd, grad, atol=tol, rtol=tol)

    def _check_integral(self, transform, exact_integral, fun, npts_1d=20):
        # assumes transform is from [-1, -1]*ndims
        from pyapprox.surrogates.orthopoly.quadrature import (
            gauss_jacobi_pts_wts_1D)
        quad_rules = [
            gauss_jacobi_pts_wts_1D(npts_1d, 0, 0)]*2
        orth_samples = cartesian_product([q[0] for q in quad_rules])
        # multiply 1D weights by 2 to get them to remove PDF of uniform
        # variable on [-1, 1]
        orth_weights = outer_product([q[1]*2 for q in quad_rules])
        weights = transform.modify_quadrature_weights(orth_samples, orth_weights)
        integral = fun(transform.map_from_orthogonal(orth_samples)).dot(weights)
        print(integral, exact_integral)
        assert np.allclose(integral, exact_integral)

    @staticmethod
    def _get_orthogonal_boundary_samples(npts):
        s = np.linspace(-1, 1, npts)[None, :]
        # boundaries ordered left, right, bottom, top
        # this ordering is assumed by tests
        orth_lines = [np.vstack((np.full(s.shape, -1), s)),
                      np.vstack((np.full(s.shape, 1), s)),
                      np.vstack((s, np.full(s.shape, -1))),
                      np.vstack((s, np.full(s.shape, 1)))]
        return orth_lines

    @staticmethod
    def _check_normals(transform, orth_lines, get_exact_normals, plot=False):
        for bndry_id in range(2, 4):
            # print(orth_lines[bndry_id])
            line = transform.map_from_orthogonal(orth_lines[bndry_id])
            normals = transform.normal(bndry_id, line)
            assert np.allclose(np.linalg.norm(normals, axis=1), 1)
            exact_normals = get_exact_normals(
                bndry_id, orth_lines[bndry_id], line)
            if plot:
                import matplotlib.pyplot as plt
                plt.plot(line[0], line[1])
                for ii in range(normals.shape[0]):
                    plt.plot([line[0, ii], line[0, ii]+normals[ii, 0]],
                             [line[1, ii], line[1, ii]+normals[ii, 1]])
                    plt.plot(
                        [line[0, ii], line[0, ii]+exact_normals[ii, 0]],
                        [line[1, ii], line[1, ii]+exact_normals[ii, 1]], '--')
                plt.show()
            II = np.where(np.all(np.isfinite(exact_normals), axis=1))[0]
            assert II.shape[0] > 0
            assert np.allclose(normals[II], exact_normals[II])

    def test_scale_translation(self):
        nsamples_1d = [3, 3]
        ranges = np.array([0.5, 1., 0., 3])
        transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], ranges)

        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        samples = transform.map_from_orthogonal(orth_samples)
        assert np.allclose(samples, cartesian_product(
            [np.linspace(*ranges[:2], nsamples_1d[0]),
             np.linspace(*ranges[2:], nsamples_1d[1])]))
        assert np.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)
        self._check_integral(transform, np.prod(ranges[1::2]-ranges[::2]),
                             lambda xx: np.ones(xx.shape[1]))

        def _rectangle_normals(bndry_id, orth_line, line):
            zeros = np.zeros_like(orth_line[0])[:, None]
            ones = np.ones_like(orth_line[0])[:, None]
            if bndry_id == 0:
                return np.hstack([-ones, zeros])
            if bndry_id == 1:
                return np.hstack([ones, zeros])
            if bndry_id == 2:
                return np.hstack([zeros, -ones])
            return np.hstack([zeros, ones])

        orth_lines = self._get_orthogonal_boundary_samples(31)
        self._check_normals(transform, orth_lines, _rectangle_normals)

    def test_polar_transform(self):
        nsamples_1d = [3, 3]
        rmin, rmax = 0.5, 1
        tmin, tmax = 3*np.pi/4, np.pi/4
        scale_transform = ScaleAndTranslationTransform(
            # keep counter clockwise direction
            # [-1, 1, -1, 1], [0.5, 1., np.pi/4, 3*np.pi/4])
            # reverse direction from counter clockwise to clockwise
            [-1, 1, -1, 1], [rmin, rmax, 3*np.pi/4, np.pi/4])
        polar_transform = PolarTransform()
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        polar_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = polar_transform.map_from_orthogonal(polar_orth_samples)
        assert np.allclose(
            polar_transform.map_to_orthogonal(samples), polar_orth_samples)
        self._check_gradients(polar_transform, polar_orth_samples, samples)
        
        def _circle_normals(bndry_id, orth_line, line):
            r, theta = orth_line
            y = np.sqrt(r**2-line[0]**2)
            dydx = -line[0]/y
            active_var = int(bndry_id > 1)
            exact_normals = np.ones((line.shape[1], 2))
            if bndry_id < 2:
                exact_normals[theta > np.pi, :] = -1
            else:
                exact_normals *= -1
                exact_normals[theta > np.pi, :] = 1
            exact_normals[:, active_var] = -dydx
            exact_normals /= np.linalg.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1)**((bndry_id+1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples(31)
        polar_orth_lines = [scale_transform.map_from_orthogonal(
                orth_lines[bndry_id]) for bndry_id in range(4)]
        self._check_normals(polar_transform, polar_orth_lines, _circle_normals)

        transform = CompositionTransform(
            [scale_transform, polar_transform])
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        samples = transform.map_from_orthogonal(orth_samples)
        assert np.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)
        # use abs on tmax-tmin because we are just calculating ratio of angle
        # to possible angle in cicle
        # exact_integral = (np.pi*rmax**2-np.pi*rmin**2)*(abs(tmax-tmin)/(2*np.pi))
        # fun = lambda xx: np.ones(xx.shape[1])
        exact_integral = (np.pi*rmax**2-np.pi*rmin**2)*(abs(tmax-tmin)/(2*np.pi))
        fun = lambda xx: np.prod(xx**2, axis=0)
        # Mathematica input
        # Integrate[x^2*y^2, {x, y} \[Element] Disk[{0, 0}, r]]
        exact_integral = (np.pi*rmax**6/24-np.pi*rmin**6/24)*(
            abs(tmax-tmin)/(2*np.pi))
        self._check_integral(transform, exact_integral, fun)

        orth_lines = self._get_orthogonal_boundary_samples(31)
        # _circle_normals assumes ortho_samples have been transformed from
        # [-1, 1, -1, 1] to [0, inf, 0, 2*np.pi]
        # so use lambda to apply this mapping
        self._check_normals(
            transform, orth_lines,
            lambda ii, oline, line: _circle_normals(
                ii, scale_transform.map_from_orthogonal(oline), line))

    def test_elliptical_transform(self):
        foci = 1
        nsamples_1d = [3, 3]
        scale_transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], [0.5, 1., 0.5*np.pi, 7*np.pi/4])
        elliptical_transform = EllipticalTransform(foci)
        transform = CompositionTransform(
            [scale_transform, elliptical_transform])
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        samples = transform.map_from_orthogonal(orth_samples)
        assert np.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)

        def _ellipse_normals(foci, bndry_id, orth_line, line):
            r, theta = orth_line
            width = np.sqrt(foci**2*np.cosh(r)**2)
            height = np.sqrt(width**2-foci**2)
            # y = np.sqrt((1-line[0]**2/width**2)*height**2)
            dydx = -height*line[0]/(
                width**2*np.sqrt(1-line[0]**2/width**2))
            active_var = int((bndry_id) > 1)
            exact_normals = np.ones((line.shape[1], 2))
            if bndry_id < 2:
                exact_normals[theta > np.pi, :] = -1
            else:
                exact_normals *= -1
                exact_normals[theta > np.pi, :] = 1
            exact_normals[:, active_var] = -dydx
            exact_normals /= np.linalg.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1)**((bndry_id+1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples(31)
        self._check_normals(
            transform, orth_lines,
            lambda ii, oline, line: _ellipse_normals(
                foci, ii, scale_transform.map_from_orthogonal(oline), line))

    def test_sympy_transform(self):
        nsamples_1d = [3, 3]
        scale_transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], [0.5, 1., np.pi/4, 3*np.pi/4])
        # Note this will only work in upper half plane
        # dut to non uniqueness of inverse map
        sympy_transform = SympyTransform(
            ["_r_*cos(_t_)", "_r_*sin(_t_)"],
            ["sqrt(_x_**2+_y_**2)", "atan2(_y_,_x_)"])
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        sympy_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = sympy_transform.map_from_orthogonal(sympy_orth_samples)
        assert np.allclose(
            sympy_transform.map_to_orthogonal(samples), sympy_orth_samples)
        self._check_gradients(sympy_transform, sympy_orth_samples, samples)

        def _circle_normals(bndry_id, orth_line, line):
            r, theta = orth_line
            y = np.sqrt(r**2-line[0]**2)
            dydx = -line[0]/y
            active_var = int(bndry_id > 1)
            exact_normals = np.ones((line.shape[1], 2))
            if bndry_id < 2:
                exact_normals[theta > np.pi, :] = -1
            else:
                exact_normals *= -1
                exact_normals[theta > np.pi, :] = 1
            exact_normals[:, active_var] = -dydx
            exact_normals /= np.linalg.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1)**((bndry_id+1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples(31)
        sympy_orth_lines = [scale_transform.map_from_orthogonal(
                orth_lines[bndry_id]) for bndry_id in range(4)]
        self._check_normals(sympy_transform, sympy_orth_lines, _circle_normals)

        nsamples_1d = [31, 2]
        s0, depth, L, alpha = 2, 1, 1, 1e-1
        surf_string, bed_string = (
            f"{s0}-{alpha}*_r_**2", f"{s0}-{alpha}*_r_**2-{depth}")
        # brackets are essential around bed string
        y_from_orth_string = f"({surf_string}-({bed_string}))*_t_+{bed_string}"
        y_to_orth_string = (
            f"(_y_-({bed_string}))/({surf_string}-({bed_string}))".replace(
                "_r_", "_x_"))
        scale_transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], [-L, L, 0., 1.])
        # Note this will only work in upper half plane
        # dut to non uniqueness of inverse map
        sympy_transform = SympyTransform(
            ["_r_", y_from_orth_string], ["_x_", y_to_orth_string])
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        sympy_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = sympy_transform.map_from_orthogonal(sympy_orth_samples)
        assert np.allclose(
            sympy_transform.map_to_orthogonal(samples), sympy_orth_samples)
        self._check_gradients(sympy_transform, sympy_orth_samples, samples)

        def _normals(bndry_id, orth_line, line):
            zeros = np.zeros_like(orth_line[0])[:, None]
            ones = np.ones_like(orth_line[0])[:, None]
            if bndry_id == 0:
                return np.hstack([-ones, zeros])
            if bndry_id == 1:
                return np.hstack([ones, zeros])
            r, theta = orth_line
            active_var = int(bndry_id < 2)
            exact_normals = np.ones((line.shape[1], 2))
            dydx = -2*alpha*r
            exact_normals[:, active_var] = -dydx
            exact_normals /= np.linalg.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1)**((bndry_id+1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples(31)
        sympy_orth_lines = [scale_transform.map_from_orthogonal(
                orth_lines[bndry_id]) for bndry_id in range(4)]
        self._check_normals(sympy_transform, sympy_orth_lines, _normals)

if __name__ == "__main__":
    mesh_transforms_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestMeshTransforms)
    unittest.TextTestRunner(verbosity=2).run(mesh_transforms_test_suite)

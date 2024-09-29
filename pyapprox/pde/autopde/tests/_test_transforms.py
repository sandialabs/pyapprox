import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde._mesh_transforms import (
    ScaleAndTranslationTransform2D,
    ScaleAndTranslationTransform3D,
    PolarTransform,
    EllipticalTransform,
    CompositionTransform,
    SympyTransform2D,
    SphericalTransform,
)
from pyapprox.util.utilities import approx_fprime
from pyapprox.surrogates.bases.orthopoly import GaussLegendreQuadratureRule
from pyapprox.surrogates.bases.basis import FixedTensorProductQuadratureRule


class TestMeshTransforms:
    def setUp(self):
        np.random.seed(1)

    def _check_gradients(self, transform, orth_samples, samples, forward=True):
        bkd = self.get_backend()

        def _fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]

        grad_fd = []
        for sample in samples.T:
            grad_fd.append(
                approx_fprime(sample[:, None], _fun, forward=forward)
            )
        grad_fd = bkd.asarray(grad_fd)
        # print(grad_fd, "f_x FD")

        def _orth_fun(orth_samples):
            return _fun(transform.map_from_orthogonal(orth_samples))

        orth_grad_fd = []
        for orth_sample in orth_samples.T:
            orth_grad_fd.append(
                approx_fprime(orth_sample[:, None], _orth_fun, forward=forward)
            )
        orth_grad_fd = bkd.asarray(orth_grad_fd)
        grad = transform.scale_orthogonal_gradients(orth_samples, orth_grad_fd)
        tol = 3e-7
        assert bkd.allclose(grad_fd, grad, atol=tol, rtol=tol)

    def _check_integral(self, transform, exact_integral, fun, npts_1d=20):
        bkd = self.get_backend()
        # assumes transform is from [-1, -1]*ndims
        quad_rule = FixedTensorProductQuadratureRule(
            transform.nphys_vars(),
            [GaussLegendreQuadratureRule([-1, 1])] * transform.nphys_vars(),
            [npts_1d] * transform.nphys_vars(),
        )
        orth_samples, orth_weights = quad_rule()
        if not isinstance(transform, CompositionTransform):
            assert bkd.allclose(
                transform.determinants(orth_samples),
                bkd.prod(transform.scale_factors(orth_samples), axis=1),
            )
        weights = transform.modify_quadrature_weights(
            orth_samples, orth_weights
        )
        integral = fun(transform.map_from_orthogonal(orth_samples)).dot(
            weights
        )
        # print(integral, exact_integral)
        assert bkd.allclose(integral, exact_integral)

    def _get_orthogonal_boundary_samples_2d(self, npts):
        bkd = self.get_backend()
        s = bkd.linspace(-1, 1, npts)[None, :]
        # boundaries ordered left, right, bottom, top
        # this ordering is assumed by tests
        orth_lines = [
            bkd.vstack((bkd.full(s.shape, -1), s)),
            bkd.vstack((bkd.full(s.shape, 1), s)),
            bkd.vstack((s, bkd.full(s.shape, -1))),
            bkd.vstack((s, bkd.full(s.shape, 1))),
        ]
        return orth_lines

    def _check_normals_2d(
        self, transform, orth_lines, get_exact_normals, plot=False
    ):
        bkd = self.get_backend()
        for bndry_id in range(transform.nphys_vars() * 2):
            line = transform.map_from_orthogonal(orth_lines[bndry_id])
            normals = transform.normal(bndry_id, line)
            assert bkd.allclose(bkd.norm(normals, axis=1), 1)
            exact_normals = get_exact_normals(
                bndry_id, orth_lines[bndry_id], line
            )
            if plot:
                plt.plot(line[0], line[1])
                for ii in range(normals.shape[0]):
                    plt.plot(
                        [line[0, ii], line[0, ii] + normals[ii, 0]],
                        [line[1, ii], line[1, ii] + normals[ii, 1]],
                    )
                    plt.plot(
                        [line[0, ii], line[0, ii] + exact_normals[ii, 0]],
                        [line[1, ii], line[1, ii] + exact_normals[ii, 1]],
                        "--",
                    )
                plt.show()
            II = bkd.where(bkd.all(bkd.isfinite(exact_normals), axis=1))[0]
            assert II.shape[0] > 0
            assert bkd.allclose(normals[II], exact_normals[II])

    def _get_orthogonal_boundary_samples_3d(self, npts):
        bkd = self.get_backend()
        # boundaries ordered left, right, front, back, bottom, top
        # this ordering is assumed by tests
        one = bkd.full((1,), 1.0)
        orth_surfaces = [
            bkd.cartesian_product(
                [-one, bkd.linspace(-1, 1, npts), bkd.linspace(-1, 1, npts)]
            ),
            bkd.cartesian_product(
                [one, bkd.linspace(-1, 1, npts), bkd.linspace(-1, 1, npts)]
            ),
            bkd.cartesian_product(
                [bkd.linspace(-1, 1, npts), -one, bkd.linspace(-1, 1, npts)]
            ),
            bkd.cartesian_product(
                [bkd.linspace(-1, 1, npts), one, bkd.linspace(-1, 1, npts)]
            ),
            bkd.cartesian_product(
                [bkd.linspace(-1, 1, npts), bkd.linspace(-1, 1, npts), -one]
            ),
            bkd.cartesian_product(
                [bkd.linspace(-1, 1, npts), bkd.linspace(-1, 1, npts), one]
            ),
        ]
        return orth_surfaces

    def _check_normals_3d(
        self, transform, orth_surfaces, get_exact_normals, plot=False
    ):
        bkd = self.get_backend()
        if plot:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        for bndry_id in range(transform.nphys_vars() * 2):
            # print(orth_lines[bndry_id])
            surface = transform.map_from_orthogonal(orth_surfaces[bndry_id])
            normals = transform.normal(bndry_id, surface)
            assert bkd.allclose(bkd.norm(normals, axis=1), 1)
            if get_exact_normals is not None:
                exact_normals = get_exact_normals(
                    bndry_id, orth_surfaces[bndry_id], surface
                )
                II = bkd.where(bkd.all(bkd.isfinite(exact_normals), axis=1))[0]
                assert II.shape[0] > 0
                assert bkd.allclose(normals[II], exact_normals[II])
            if plot:
                ax.plot(*surface, "o")
                for ii in range(normals.shape[0]):
                    ax.plot(
                        [surface[0, ii], surface[0, ii] + normals[ii, 0]],
                        [surface[1, ii], surface[1, ii] + normals[ii, 1]],
                        [surface[2, ii], surface[2, ii] + normals[ii, 2]],
                        "-k",
                    )
                    if get_exact_normals is None:
                        continue
                    ax.plot(
                        [
                            surface[0, ii],
                            surface[0, ii] + exact_normals[ii, 0],
                        ],
                        [
                            surface[1, ii],
                            surface[1, ii] + exact_normals[ii, 1],
                        ],
                        [
                            surface[2, ii],
                            surface[2, ii] + exact_normals[ii, 2],
                        ],
                        "r--",
                    )

    def test_scale_translation_2d(self):
        bkd = self.get_backend()
        nsamples_1d = [3, 3]
        ranges = bkd.array([0.5, 1.0, 0.0, 3])
        transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], ranges)

        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        samples = transform.map_from_orthogonal(orth_samples)
        assert bkd.allclose(
            samples,
            bkd.cartesian_product(
                [
                    bkd.linspace(*ranges[:2], nsamples_1d[0]),
                    bkd.linspace(*ranges[2:], nsamples_1d[1]),
                ]
            ),
        )
        assert bkd.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)
        self._check_integral(
            transform,
            bkd.prod(ranges[1::2] - ranges[::2]),
            lambda xx: bkd.ones(xx.shape[1]),
        )

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

        orth_lines = self._get_orthogonal_boundary_samples_2d(31)
        self._check_normals_2d(transform, orth_lines, _rectangle_normals)

    def test_scale_translation_3d(self):
        bkd = self.get_backend()
        nsamples_1d = [3, 3, 3]
        ranges = bkd.array([0.5, 1.0, 0.0, 3, -1, 0])
        transform = ScaleAndTranslationTransform3D(
            [-1, 1, -1, 1, -1, 1], ranges
        )

        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
                bkd.linspace(-1, 1, nsamples_1d[2]),
            ]
        )
        samples = transform.map_from_orthogonal(orth_samples)
        assert bkd.allclose(
            samples,
            bkd.cartesian_product(
                [
                    bkd.linspace(*ranges[:2], nsamples_1d[0]),
                    bkd.linspace(*ranges[2:4], nsamples_1d[1]),
                    bkd.linspace(*ranges[4:6], nsamples_1d[2]),
                ]
            ),
        )
        assert bkd.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)
        self._check_integral(
            transform,
            bkd.prod(ranges[1::2] - ranges[::2]),
            lambda xx: bkd.ones(xx.shape[1]),
        )

        def _cube_normals(bndry_id, orth_line, line):
            zeros = bkd.zeros(orth_line[0].shape)[:, None]
            ones = bkd.ones(orth_line[0].shape)[:, None]
            if bndry_id == 0:
                return bkd.hstack([-ones, zeros, zeros])
            if bndry_id == 1:
                return bkd.hstack([ones, zeros, zeros])
            if bndry_id == 2:
                return bkd.hstack([zeros, -ones, zeros])
            if bndry_id == 3:
                return bkd.hstack([zeros, ones, zeros])
            if bndry_id == 4:
                return bkd.hstack([zeros, zeros, -ones])
            if bndry_id == 5:
                return bkd.hstack([zeros, zeros, ones])

        orth_surfaces = self._get_orthogonal_boundary_samples_3d(5)
        self._check_normals_3d(transform, orth_surfaces, _cube_normals)

    def test_polar_transform(self):
        bkd = self.get_backend()
        nsamples_1d = [11, 11]
        rmin, rmax = 0.8, 1
        tmin, tmax = -3 * np.pi / 4, 3 * np.pi / 4
        # numericall conditioning of transformation will mean the following
        # will cause a failure because transformed points are outside bounds
        # by just over machine precision
        # tmin, tmax = -np.pi+0.1, np.pi
        scale_transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1],
            [rmin, rmax, tmin, tmax],
        )
        polar_transform = PolarTransform()
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        polar_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = polar_transform.map_from_orthogonal(polar_orth_samples)
        assert bkd.allclose(
            polar_transform.map_to_orthogonal(samples), polar_orth_samples
        )
        self._check_gradients(polar_transform, polar_orth_samples, samples)

        def _circle_normals(bndry_id, orth_line, line):
            radius, azimuth = orth_line
            sign = (-1) ** ((bndry_id % 2) + 1)
            if bndry_id < 2:
                return sign*bkd.stack(
                    [bkd.cos(azimuth), bkd.sin(azimuth)],
                    axis=1)
            return sign*bkd.stack(
                    [-bkd.sin(azimuth), bkd.cos(azimuth)],
                    axis=1)

        orth_lines = self._get_orthogonal_boundary_samples_2d(30)
        polar_orth_lines = [
            scale_transform.map_from_orthogonal(orth_lines[bndry_id])
            for bndry_id in range(4)
        ]
        self._check_normals_2d(
            polar_transform, polar_orth_lines, _circle_normals
        )

        transform = CompositionTransform([scale_transform, polar_transform])
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        samples = transform.map_from_orthogonal(orth_samples)
        assert bkd.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)
        # use abs on tmax-tmin because we are just calculating ratio of angle
        # to possible angle in cicle
        exact_integral = (np.pi * rmax**2 - np.pi * rmin**2) * (
            abs(tmax - tmin) / (2 * np.pi)
        )

        def fun(xx):
            return bkd.prod(xx**2, axis=0)

        # Mathematica input
        # Integrate[x^2*y^2, {x, y} \[Element] Disk[{0, 0}, r]]
        exact_integral = (np.pi * rmax**6 / 24 - np.pi * rmin**6 / 24) * (
            abs(tmax - tmin) / (2 * np.pi)
        )
        self._check_integral(transform, exact_integral, fun)

        orth_lines = self._get_orthogonal_boundary_samples_2d(31)
        # _circle_normals assumes ortho_samples have been transformed from
        # [-1, 1, -1, 1] to [0, inf, 0, 2*np.pi]
        # so use lambda to apply this mapping
        self._check_normals_2d(
            transform,
            orth_lines,
            lambda ii, oline, line: _circle_normals(
                ii, scale_transform.map_from_orthogonal(oline), line
            ),
        )

    def test_elliptical_transform(self):
        bkd = self.get_backend()
        foci = 1
        nsamples_1d = [3, 3]
        scale_transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], [0.5, 1.0, 0.5 * np.pi, 7 * np.pi / 4]
        )
        elliptical_transform = EllipticalTransform(foci)
        transform = CompositionTransform(
            [scale_transform, elliptical_transform]
        )
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]) * 0 + 1,
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        samples = transform.map_from_orthogonal(orth_samples)
        assert bkd.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples)

        def _ellipse_normals(foci, bndry_id, orth_line, line):
            r, theta = orth_line
            width = bkd.sqrt(foci**2 * bkd.cosh(r) ** 2)
            height = bkd.sqrt(width**2 - foci**2)
            dydx = (
                -height
                * line[0]
                / (width**2 * bkd.sqrt(1 - line[0] ** 2 / width**2))
            )
            active_var = int((bndry_id) > 1)
            exact_normals = bkd.ones((line.shape[1], 2))
            if bndry_id < 2:
                exact_normals[theta > np.pi, :] = -1
            else:
                exact_normals *= -1
                exact_normals[theta > np.pi, :] = 1
            exact_normals[:, active_var] = -dydx
            exact_normals /= bkd.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1) ** ((bndry_id + 1) % 2)
            return exact_normals

        # avoid odd numbers are args so ortholines are not evaluated at 0
        # in canonical space, this causes divide by zero in test
        orth_lines = self._get_orthogonal_boundary_samples_2d(30)
        self._check_normals_2d(
            transform,
            orth_lines,
            lambda ii, oline, line: _ellipse_normals(
                foci, ii, scale_transform.map_from_orthogonal(oline), line
            ),
        )

    def test_sympy_transform(self):
        bkd = self.get_backend()
        nsamples_1d = [3, 3]
        scale_transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], [0.5, 1.0, np.pi / 4, 3 * np.pi / 4]
        )
        # Note this will only work in upper half plane
        # dut to non uniqueness of inverse map
        sympy_transform = SympyTransform2D(
            ["_r_*cos(_t_)", "_r_*sin(_t_)"],
            ["sqrt(_x_**2+_y_**2)", "atan2(_y_,_x_)"],
        )
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        sympy_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = sympy_transform.map_from_orthogonal(sympy_orth_samples)
        assert bkd.allclose(
            sympy_transform.map_to_orthogonal(samples), sympy_orth_samples
        )
        self._check_gradients(sympy_transform, sympy_orth_samples, samples)

        def _circle_normals(bndry_id, orth_line, line):
            r, theta = orth_line
            y = bkd.sqrt(r**2 - line[0] ** 2)
            dydx = -line[0] / y
            active_var = int(bndry_id > 1)
            exact_normals = bkd.ones((line.shape[1], 2))
            if bndry_id < 2:
                exact_normals[theta > np.pi, :] = -1
            else:
                exact_normals *= -1
                exact_normals[theta > np.pi, :] = 1
            exact_normals[:, active_var] = -dydx
            exact_normals /= bkd.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1) ** ((bndry_id + 1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples_2d(31)
        sympy_orth_lines = [
            scale_transform.map_from_orthogonal(orth_lines[bndry_id])
            for bndry_id in range(4)
        ]
        self._check_normals_2d(
            sympy_transform, sympy_orth_lines, _circle_normals
        )

        nsamples_1d = [31, 2]
        s0, depth, L, alpha = 2, 1, 1, 1e-1
        surf_string, bed_string = (
            f"{s0}-{alpha}*_r_**2",
            f"{s0}-{alpha}*_r_**2-{depth}",
        )
        # brackets are essential around bed string
        y_from_orth_string = f"({surf_string}-({bed_string}))*_t_+{bed_string}"
        y_to_orth_string = (
            f"(_y_-({bed_string}))/({surf_string}-({bed_string}))".replace(
                "_r_", "_x_"
            )
        )
        scale_transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], [-L, L, 0.0, 1.0]
        )
        # Note this will only work in upper half plane
        # dut to non uniqueness of inverse map
        sympy_transform = SympyTransform2D(
            ["_r_", y_from_orth_string], ["_x_", y_to_orth_string]
        )
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
            ]
        )
        sympy_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = sympy_transform.map_from_orthogonal(sympy_orth_samples)
        assert bkd.allclose(
            sympy_transform.map_to_orthogonal(samples), sympy_orth_samples
        )
        self._check_gradients(sympy_transform, sympy_orth_samples, samples)

        def _normals(bndry_id, orth_line, line):
            zeros = bkd.zeros(orth_line[0].shape)[:, None]
            ones = bkd.ones(orth_line[0].shape)[:, None]
            if bndry_id == 0:
                return bkd.hstack([-ones, zeros])
            if bndry_id == 1:
                return bkd.hstack([ones, zeros])
            r, theta = orth_line
            active_var = int(bndry_id < 2)
            exact_normals = bkd.ones((line.shape[1], 2))
            dydx = -2 * alpha * r
            exact_normals[:, active_var] = -dydx
            exact_normals /= bkd.norm(exact_normals, axis=1)[:, None]
            exact_normals *= (-1) ** ((bndry_id + 1) % 2)
            return exact_normals

        orth_lines = self._get_orthogonal_boundary_samples_2d(31)
        sympy_orth_lines = [
            scale_transform.map_from_orthogonal(orth_lines[bndry_id])
            for bndry_id in range(4)
        ]
        self._check_normals_2d(sympy_transform, sympy_orth_lines, _normals)

    def test_spherical_transform(self):
        bkd = self.get_backend()
        nsamples_1d = [10, 10, 10]
        rmin, rmax = 0.5, 2
        # add 0.1 so finite difference chec
        amin, amax = -np.pi / 2, np.pi / 2
        emin, emax = np.pi / 2, np.pi
        scale_transform = ScaleAndTranslationTransform3D(
            # reverse direction from counter clockwise to clockwise
            [-1, 1, -1, 1, -1, 1],
            [rmin, rmax, amin, amax, emin, emax],
        )
        sph_transform = SphericalTransform()
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
                bkd.linspace(-1, 1, nsamples_1d[2]),
            ]
        )
        sph_orth_samples = scale_transform.map_from_orthogonal(orth_samples)
        samples = sph_transform.map_from_orthogonal(sph_orth_samples)
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(1, 1, 1, projection="3d")
        # ax.plot(*samples, "o")
        # plt.show()

        assert bkd.allclose(
            sph_transform.map_to_orthogonal(samples), sph_orth_samples
        )
        self._check_gradients(
           sph_transform, sph_orth_samples, samples, forward=False
        )

        orth_surfaces = self._get_orthogonal_boundary_samples_3d(10)
        sph_orth_surfaces = [
            scale_transform.map_from_orthogonal(orth_surfaces[bndry_id])
            for bndry_id in range(len(orth_surfaces))
        ]

        def _sphere_normals(bndry_id, orth_line, line):
            radius, azimuth, elevation = orth_line
            zeros = bkd.zeros(orth_line[0].shape)
            sign = (-1) ** ((bndry_id % 2) + 1)
            if bndry_id < 2:
                sign = -1 if bndry_id == 0 else 1
                return sign*bkd.stack(
                    [
                        bkd.cos(azimuth)*bkd.sin(elevation),
                        bkd.sin(azimuth)*bkd.sin(elevation),
                        bkd.cos(elevation)
                    ],
                    axis=1)
            if bndry_id < 4:
                return sign*bkd.stack(
                    [
                        -bkd.sin(azimuth),
                        bkd.cos(azimuth),
                        zeros
                    ],
                    axis=1
                )
            return sign*bkd.stack(
                    [
                        bkd.cos(azimuth)*bkd.cos(elevation),
                        bkd.sin(azimuth)*bkd.cos(elevation),
                        -bkd.sin(elevation)
                    ],
                    axis=1
                )

        self._check_normals_3d(
            sph_transform, sph_orth_surfaces, _sphere_normals
        )

        transform = CompositionTransform([scale_transform, sph_transform])
        orth_samples = bkd.cartesian_product(
            [
                bkd.linspace(-1, 1, nsamples_1d[0]),
                bkd.linspace(-1, 1, nsamples_1d[1]),
                bkd.linspace(-1, 1, nsamples_1d[2]),
            ]
        )
        samples = transform.map_from_orthogonal(orth_samples)
        assert bkd.allclose(transform.map_to_orthogonal(samples), orth_samples)
        self._check_gradients(transform, orth_samples, samples, forward=False)
        # assume tmin, tmax ,smin, smax defined to integrate quarter of sphere
        exact_integral = (
            4 / 3 * np.pi * rmax**3 - 4 / 3 * np.pi * rmin**3
        ) / 4
        self._check_integral(
            transform, exact_integral, lambda xx: bkd.ones(xx[1].shape)
        )


class TestNumpyMeshTransforms(TestMeshTransforms, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchMeshTransforms(TestBasis, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()

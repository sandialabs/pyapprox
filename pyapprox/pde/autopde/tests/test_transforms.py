import unittest
import torch
import numpy as np

from pyapprox.pde.autopde.mesh_transforms import (
    IdentityTransform, ScaleAndTranslationTransform, PolarTransform,
    EllipticalTransform)
from pyapprox.util.utilities import cartesian_product


class TestMeshTransforms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def test(self):
        nsamples_1d = [11, 11]
        transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], [0.5, 1., 0., np.pi], IdentityTransform())

        # orth_samples = cartesian_product(
        #     [np.linspace(-1, 1, nsamples_1d[0]),
        #      np.linspace(-1, 1, nsamples_1d[1])])
        # samples = transform.map_from_orthogonal(orth_samples)
        # assert np.allclose(samples, cartesian_product(
        #     [np.linspace(0.5, 1, nsamples_1d[0]),
        #      np.linspace(0, np.pi, nsamples_1d[1])]))
        # assert np.allclose(transform.map_to_orthogonal(samples), orth_samples)

        # transform = ScaleAndTranslationTransform(
        #     [-1, 1, -1, 1], [0.5, 1., 0.5*np.pi, 7*np.pi/4], PolarTransform())
        # orth_samples = cartesian_product(
        #     [np.linspace(-1, 1, nsamples_1d[0]),
        #      np.linspace(-1, 1, nsamples_1d[1])])
        # samples = transform.map_from_orthogonal(orth_samples)

        foci = 1
        transform = ScaleAndTranslationTransform(
            # [-1, 1, -1, 1], [0.5, 1., 0.5*np.pi, 7*np.pi/4],
            [-1, 1, -1, 1], [0.5, 1., 0.0*np.pi, np.pi],
            EllipticalTransform(foci))
        orth_samples = cartesian_product(
            [np.linspace(-1, 1, nsamples_1d[0]),
             np.linspace(-1, 1, nsamples_1d[1])])
        samples = transform.map_from_orthogonal(orth_samples)

        import matplotlib.pyplot as plt
        # plt.plot(samples[0], samples[1], 'o')
        # recovered_orth_samples = transform.map_to_orthogonal(samples)
        # plt.plot(recovered_orth_samples[0], recovered_orth_samples[1], 's')
        # plt.show()
        
        assert np.allclose(transform.map_to_orthogonal(samples), orth_samples)

        s = np.linspace(-1, 1, 31)[None, :]
        orth_lines = [np.vstack((s, np.full(s.shape, -1))),
                      np.vstack((s, np.full(s.shape, 1))),
                      np.vstack((np.full(s.shape, -1), s)),
                      np.vstack((np.full(s.shape, 1), s))]


        for bndry_id in range(3, 4):
            line = transform.map_from_orthogonal(orth_lines[bndry_id])
            plt.plot(line[0], line[1])
            r = orth_lines[bndry_id][0][0]
            width = np.sqrt(foci**2*r**2*np.cosh(r)**2)
            height = np.sqrt(width**2-foci**2)
            y = np.sqrt((1-line[0]**2/width**2)*height**2)
            plt.plot(line[0], y, '--')
            # plt.show()
            dydx = -height*line[0]/(width**2*np.sqrt(1-line[0]**2/width**2))
            exact_normals = np.array([-dydx, 1+0*dydx])
            exact_normals /= np.linalg.norm(exact_normals, axis=0)
            normals = transform.normal(bndry_id, line)
            # exact_normals is not defined this way on x axis
            II = np.where(np.isfinite(exact_normals))
            print(normals.T[II]-exact_normals[II])
            assert np.allclose(normals.T[II], exact_normals[II])
            print(exact_normals)
            print(normals.shape)
            for ii in range(normals.shape[0]):
                plt.plot([line[0, ii], line[0, ii]+normals[ii][0]],
                         [line[1, ii], line[1, ii]+normals[ii][1]])
        plt.show()


if __name__ == "__main__":
    mesh_transforms_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestMeshTransforms)
    unittest.TextTestRunner(verbosity=2).run(mesh_transforms_test_suite)

import unittest

import matplotlib.pyplot as plt
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    HyperbolicIndexGenerator,
    compute_hyperbolic_indices,
    sort_indices_lexiographically,
    IsotropicSGIndexGenerator,
    DoublePlusOneIndexGrowthRule,
)


class TestMultiIndex:
    def setUp(self):
        np.random.seed(1)

    def test_hyperbolic_index_generator(self):
        bkd = self.get_backend()
        nvars, pnorm, level = 2, 1, 2
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        print(gen)
        indices = bkd._la_asarray(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
        ).T
        assert bkd._la_allclose(gen.get_indices(), indices)
        assert bkd._la_allclose(
            compute_hyperbolic_indices(nvars, level, pnorm, bkd),
            indices
        )

        nvars, pnorm, level = 3, 1, 4
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        assert bkd._la_allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(
                compute_hyperbolic_indices(nvars, level, pnorm, bkd)
            ),
        )

        nvars, pnorm, level = 3, 0.4, 4
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        assert bkd._la_allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(
                compute_hyperbolic_indices(nvars, level, pnorm, bkd)
            ),
        )

    def test_isotropic_sg_index_generator(self):
        bkd = self.get_backend()
        nvars, level = 2, 2
        gen = IsotropicSGIndexGenerator(
            nvars, level, DoublePlusOneIndexGrowthRule()
        )
        indices = bkd._la_array(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0],
             [4, 0], [1, 1], [1, 2], [2, 1], [2, 2]]).T
        assert bkd._la_allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(indices)
        )
        # check plot runs
        gen.plot_indices(plt.subplots(1, 1)[1])

        nvars, level = 3, 2
        gen = IsotropicSGIndexGenerator(
            nvars, level, DoublePlusOneIndexGrowthRule()
        )
        # check plot runs
        gen.plot_indices(plt.figure().add_subplot(projection="3d"))


class TestNumpyMultiIndex(TestMultiIndex, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


# class TestTorchMultiIndex(TestMultiIndex, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)

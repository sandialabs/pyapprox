import unittest

import matplotlib.pyplot as plt
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    HyperbolicIndexGenerator,
    compute_hyperbolic_indices,
    sort_indices_lexiographically,
    IsotropicSGIndexGenerator,
    DoublePlusOneIndexGrowthRule,
    LinearGrowthRule,
)


class TestMultiIndex:
    def setUp(self):
        np.random.seed(1)

    def test_hyperbolic_index_generator(self):
        bkd = self.get_backend()
        nvars, pnorm, level = 2, 1, 2
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        indices = bkd.asarray(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]],
            dtype=int
        ).T
        assert bkd.allclose(gen.get_indices(), indices)
        assert bkd.allclose(
            compute_hyperbolic_indices(nvars, level, pnorm, bkd),
            indices
        )

        on_margin = bkd.array(
            [gen._index_on_margin(index) for index in indices.T], dtype=bool
        )
        assert bkd.allclose(
            on_margin,
            bkd.array([False, False, False, True, True, True], dtype=bool)
        )

        gen.step()
        assert gen.nindices() == 10
        assert gen.ncandidate_indices() == 0

        # Check that setting selected indices can be set and candidate indices
        # computed correctly. This function should not be used by a user
        # for HyperbolicIndexGenerator. TODO move test to use different
        # of adaptive IndexGenerator
        gen.set_selected_indices(indices[:, :-3])
        assert gen.nselected_indices() == 3
        assert gen.ncandidate_indices() == 3
        assert bkd.allclose(gen.get_indices(), indices)

        self.assertRaises(
            ValueError, gen.set_selected_indices, indices[:, -3:]
        )

        nvars, pnorm, level = 3, 1, 4
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        assert bkd.allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(
                compute_hyperbolic_indices(nvars, level, pnorm, bkd)
            ),
        )

        nvars, pnorm, level = 3, 0.4, 4
        gen = HyperbolicIndexGenerator(nvars, level, pnorm, backend=bkd)
        assert bkd.allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(
                compute_hyperbolic_indices(nvars, level, pnorm, bkd)
            ),
        )

        nvars, pnorm, level, max_1d_levels = 2, 1, 2, [1, 2]
        gen = HyperbolicIndexGenerator(
            nvars, level, pnorm, max_1d_levels, backend=bkd
        )
        assert bkd.allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(
                bkd.array(
                    [[0, 0], [1, 0], [0, 1], [0, 2], [1, 1]]
                ).T,
            )
        )

    def test_isotropic_sg_index_generator(self):
        bkd = self.get_backend()
        nvars, level = 2, 2
        gen = IsotropicSGIndexGenerator(
            nvars, level, DoublePlusOneIndexGrowthRule(), backend=bkd
        )
        indices = bkd.array(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0],
             [4, 0], [1, 1], [1, 2], [2, 1], [2, 2]], dtype=int).T
        assert bkd.allclose(
            sort_indices_lexiographically(gen.get_indices()),
            sort_indices_lexiographically(indices)
        )
        # check plot runs
        gen.plot_indices(plt.subplots(1, 1)[1])

        nvars, level = 3, 2
        gen = IsotropicSGIndexGenerator(
            nvars, level, DoublePlusOneIndexGrowthRule(), backend=bkd
        )
        # check plot runs
        gen.plot_indices(plt.figure().add_subplot(projection="3d"))

        gen.step()
        assert gen.nindices() == 69

        # check total degree index set is recoverred by
        # IsotropicSGIndexGenerator when linear growth rule is used
        gen1 = IsotropicSGIndexGenerator(
            nvars, level, LinearGrowthRule(1, 1), backend=bkd
        )
        gen2 = HyperbolicIndexGenerator(nvars, level, 1., backend=bkd)
        assert bkd.allclose(
            sort_indices_lexiographically(gen1.get_indices()),
            sort_indices_lexiographically(gen2.get_indices())
        )


class TestNumpyMultiIndex(TestMultiIndex, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchMultiIndex(TestMultiIndex, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

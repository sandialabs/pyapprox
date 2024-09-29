import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution
)
from pyapprox.pde.autopde._collocation import (
    ChebyshevCollocationBasis,
)
from pyapprox.pde.autopde._mesh_transforms import (
    ScaleAndTranslationTransform1D
)
from pyapprox.pde.autopde._mesh import (
    ChebyshevCollocationMesh1D,
)


class TestCollocation:
    def setUp(self):
        np.random.seed(1)

    def _check_steady_state_advection_diffusion_reaction(
            self, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, mesh,
            nl_diff_funs=[None, None]):

        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], False,
                nl_diff_funs[0]))

        basis = ChebyshevCollocationBasis(mesh)

    def test_advection_diffusion_reaction(self):
        bkd = self.get_backend()
        test_cases = [
            ["-(x-1)*x/2", "4", ["0"], [None, None], ["D", "D"],
             ChebyshevCollocationMesh1D(
                 [5], ScaleAndTranslationTransform1D([-1, 1], [0, 1], bkd)
             )
             ],
        ]
        for test_case in test_cases:
            self._check_steady_state_advection_diffusion_reaction(*test_case)
        

class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCollocation(TestBasis, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()

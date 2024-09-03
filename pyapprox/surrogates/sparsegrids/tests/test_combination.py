import unittest

import matplotlib.pyplot as plt
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    DoublePlusOneIndexGrowthRule,
    MaxLevelAdmissibilityCriteria,
    IterativeIndexGenerator,
    IsotropicSGIndexGenerator,
)
from pyapprox.surrogates.bases.univariate import (
    UnivariateLagrangeBasis,
    ClenshawCurtisQuadratureRule,
    DydadicEquidistantNodeGenerator,
)
from pyapprox.surrogates.bases.orthopoly import LegendrePolynomial1D
from pyapprox.surrogates.bases.basis import (
    TensorProductInterpolatingBasis, MultiIndexBasis
)
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.sparsegrids.combination import (
    IsotropicCombinationSparseGrid,
    AdaptiveCombinationSparseGrid,
    LevelRefinementCriteria,
    LocallyAdaptiveCombinationSparseGrid,
    LocalRefinementCriteria,
)
from pyapprox.surrogates.bases.univariate import (
    setup_univariate_piecewise_polynomial_basis
)


class TestCombination:
    def setUp(self):
        np.random.seed(1)

    def test_isotropic_sparse_grid(self):
        bkd = self.get_backend()
        nvars, level, nqoi = 2, 3, 2
        quad_rule = ClenshawCurtisQuadratureRule(
            store=True, backend=bkd
        )
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, 3)
            for dim_id in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        sg = IsotropicCombinationSparseGrid(
            nqoi, nvars, level, DoublePlusOneIndexGrowthRule(), basis,
            backend=bkd
        )
        self.assertRaises(RuntimeError, sg.set_basis, basis)

        # check sparse grid exactly interpolates a monomial
        # with the same multiindex
        train_samples = sg.train_samples()
        assert train_samples.shape[1] == sg._basis_gen.nindices()

        basis = MultiIndexBasis(
            [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_indices(sg._basis_gen.get_indices())
        fun = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        fun.set_coefficients(
            bkd.asarray(np.random.normal(0, 1, (fun.nterms(), nqoi)))
        )

        train_values = fun(train_samples)
        sg.fit(train_samples, train_values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 101)))
        sg_test_values = sg(test_samples)
        assert bkd.allclose(sg_test_values, fun(test_samples), atol=1e-15)

        assert bkd.allclose(sg.integrate(), fun.get_coefficients()[0])

        # test plot runs
        sg.plot_grid(plt.figure().gca())
        sg.plot_subspace_indices(plt.figure().gca())

    def test_adaptive_sparse_grid(self):
        bkd = self.get_backend()
        nvars, level, nqoi = 2, 3, 2
        quad_rule = ClenshawCurtisQuadratureRule(
            store=True, backend=bkd
        )
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, 3)
            for dim_id in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)

        # check sparse grid exactly interpolates a monomial
        # with the same multiindex
        growth_rule = DoublePlusOneIndexGrowthRule()
        fun_basis = MultiIndexBasis(
            [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        )
        fun_gen = IsotropicSGIndexGenerator(nvars, level, growth_rule)
        fun_basis.set_indices(fun_gen.get_indices())
        fun = PolynomialChaosExpansion(fun_basis, solver=None, nqoi=nqoi)
        fun.set_coefficients(
            bkd.asarray(np.random.normal(0, 1, (fun.nterms(), nqoi)))
        )

        sg = AdaptiveCombinationSparseGrid(nqoi)
        sg.set_basis(basis)
        subspace_gen = IterativeIndexGenerator(nvars, backend=bkd)
        subspace_gen.set_verbosity(0)
        # TODO add admissiblity function that sets max budget on sparse grid
        subspace_gen.set_admissibility_function(
            MaxLevelAdmissibilityCriteria(level, 1., bkd)
        )
        sg.set_subspace_generator(subspace_gen, growth_rule)
        sg.set_refinement_criteria(LevelRefinementCriteria())
        sg.set_initial_subspace_indices()
        sg.build(fun)

        assert sg.train_samples().shape[1] == fun.nterms()

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 101)))
        sg_test_values = sg(test_samples)
        assert bkd.allclose(sg_test_values, fun(test_samples), atol=1e-15)

        assert bkd.allclose(sg.integrate(), fun.get_coefficients()[0])

    def test_locally_adaptive_sparse_grid(self):
        bkd = self.get_backend()
        nvars, level, nqoi = 2, 3, 2
        bounds = [-1, 1]
        node_gen = DydadicEquidistantNodeGenerator()
        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                bt, bounds, backend=bkd, node_gen=node_gen
            )
            for bt in ["linear"]*nvars
        ]
        print(node_gen(3))
        basis = TensorProductInterpolatingBasis(bases_1d)

        class CustomLocalRefinementCriteria(LocalRefinementCriteria):
            def _priority(self, subspace_index):
                return 1, 1

        sg = LocallyAdaptiveCombinationSparseGrid(nqoi)
        sg.set_refinement_criteria(CustomLocalRefinementCriteria())
        sg.set_basis(basis)
        subspace_gen = IterativeIndexGenerator(nvars, backend=bkd)
        subspace_gen.set_verbosity(0)
        # TODO add admissiblity function that sets max budget on sparse grid
        subspace_gen.set_admissibility_function(
            MaxLevelAdmissibilityCriteria(level, 1., bkd)
        )
        sg.set_subspace_generator(subspace_gen)
        sg.set_initial_subspace_indices()

        def fun(samples):
            return bkd.sum(samples**2)

        unique_samples = sg.step_samples()
        print(sg.plot_grid(plt.figure().gca()))
        # plt.show()
        print(unique_samples)
        unique_values = fun(unique_samples)
        sg.step_values(unique_values)
        print(sg.plot_grid(plt.figure().gca()))
        sg.step_samples()
        print(sg._train_samples)
        # plt.show()
        # sg.build(fun)


class TestNumpyCombination(TestCombination, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCombination(TestCombination, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

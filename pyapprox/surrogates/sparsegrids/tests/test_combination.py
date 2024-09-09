import unittest
import copy

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    DoublePlusOneIndexGrowthRule,
    IterativeIndexGenerator,
    IsotropicSGIndexGenerator,
    sort_indices_lexiographically,
    argsort_indices_lexiographically,
)
from pyapprox.surrogates.bases.univariate import (
    UnivariateLagrangeBasis,
    ClenshawCurtisQuadratureRule,
    DydadicEquidistantNodeGenerator,
)
from pyapprox.surrogates.bases.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.bases.basis import (
    TensorProductInterpolatingBasis,
    MultiIndexBasis,
)
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.sparsegrids.combination import (
    IsotropicCombinationSparseGrid,
    AdaptiveCombinationSparseGrid,
    LevelRefinementCriteria,
    LocallyAdaptiveCombinationSparseGrid,
    LocalRefinementCriteria,
    LocalHierarchicalRefinementCriteria,
    SparseGridMaxLevelAdmissibilityCriteria,
    SparseGridMaxCostBasisAdmissibilityCriteria,
)
from pyapprox.surrogates.bases.univariate import (
    setup_univariate_piecewise_polynomial_basis,
)


class TestCombination:
    def setUp(self):
        np.random.seed(1)

    def test_isotropic_sparse_grid(self):
        bkd = self.get_backend()
        nvars, level, nqoi = 2, 3, 2
        quad_rule = ClenshawCurtisQuadratureRule(
            store=True, backend=bkd, bounds=[-1, 1]
        )
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        sg = IsotropicCombinationSparseGrid(
            nqoi,
            nvars,
            level,
            DoublePlusOneIndexGrowthRule(),
            basis,
            backend=bkd,
        )
        self.assertRaises(RuntimeError, sg.set_basis, basis)

        # check sparse grid exactly interpolates a monomial
        # with the same multiindex
        train_samples = sg.train_samples()
        assert train_samples.shape[1] == sg._basis_gen.nindices()

        marginal = stats.uniform(-1, 2)
        basis = MultiIndexBasis(
            [
                setup_univariate_orthogonal_polynomial_from_marginal(
                    marginal, backend=bkd
                )
                for ii in range(nvars)
            ]
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
            store=True, backend=bkd, bounds=[-1, 1]
        )
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)

        # check sparse grid exactly interpolates a monomial
        # with the same multiindex
        growth_rule = DoublePlusOneIndexGrowthRule()
        marginal = stats.uniform(-1, 2)
        fun_basis = MultiIndexBasis(
            [
                setup_univariate_orthogonal_polynomial_from_marginal(
                    marginal, backend=bkd
                )
                for ii in range(nvars)
            ]
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
        sg.set_subspace_generator(subspace_gen, growth_rule)
        sg.set_subspace_admissibility_criteria(
            SparseGridMaxLevelAdmissibilityCriteria(level, 1.0)
        )
        sg.set_refinement_criteria(LevelRefinementCriteria())
        sg.set_initial_subspace_indices()
        sg.build(fun)

        assert sg.train_samples().shape[1] == fun.nterms()

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 101)))
        sg_test_values = sg(test_samples)
        assert bkd.allclose(sg_test_values, fun(test_samples), atol=1e-15)

        assert bkd.allclose(sg.integrate(), fun.get_coefficients()[0])

    def _setup_locally_adaptive_sparse_grid(
            self, nvars, level, nqoi, max_cost
    ):
        bkd = self.get_backend()
        bounds = [-1, 1]
        node_gen = DydadicEquidistantNodeGenerator()
        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                bt, bounds, backend=bkd, node_gen=node_gen
            )
            for bt in ["linear"] * nvars
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        sg = LocallyAdaptiveCombinationSparseGrid(nqoi)

        class CustomLocalRefinementCriteria(LocalRefinementCriteria):
            def _priority(self, subspace_index):
                if subspace_index[1] > 0:
                    return 1., 1.
                return 1., -1.

        # criteria = CustomLocalRefinementCriteria()
        criteria = LocalHierarchicalRefinementCriteria()
        sg.set_refinement_criteria(criteria)
        sg.set_basis(basis)
        subspace_gen = IterativeIndexGenerator(nvars, backend=bkd)
        subspace_gen.set_verbosity(0)
        # TODO add admissiblity function that sets max budget on sparse grid
        sg.set_subspace_generator(subspace_gen)
        sg.set_subspace_admissibility_criteria(
            SparseGridMaxLevelAdmissibilityCriteria(level, 1.0)
        )
        sg.set_basis_admissibility_criteria(
            SparseGridMaxCostBasisAdmissibilityCriteria(max_cost)
        )
        sg.set_initial_subspace_indices()
        return sg

    def _check_isotropic_locally_adaptive_sparse_grid(
            self, nvars, level, nqoi
    ):
        """Test locally adaptive sparse grid recovers isotropic sparse grid"""
        bkd = self.get_backend()
        sg = self._setup_locally_adaptive_sparse_grid(nvars, level, nqoi, 100)

        def fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]*bkd.arange(
                1, nqoi+1
            )[None, :]

        iso_sg = IsotropicCombinationSparseGrid(
            nqoi,
            nvars,
            level,
            DoublePlusOneIndexGrowthRule(),
            copy.deepcopy(sg._basis),
            backend=bkd,
        )
        iso_train_samples = iso_sg.train_samples()
        train_values = fun(iso_train_samples)
        iso_sg.fit(iso_train_samples, train_values)

        sg.build(fun)

        # test plot
        if nvars <= 2:
            grid_ax = plt.figure().gca()
            if nvars == 1:
                plot_limits = [-1, 1]
                plot_ax = plt.figure().gca()
            else:
                plot_limits = [-1, 1, -1, 1]
                plot_ax = plt.figure().add_subplot(projection="3d")
            sg.plot_surface(plot_ax, plot_limits)
            iso_sg.plot_surface(plot_ax, plot_limits)

        elif nvars == 3:
            grid_ax = plt.figure().add_subplot(projection="3d")

        sg.plot_grid(grid_ax)
        # plt.show()
        assert bkd.allclose(
            sort_indices_lexiographically(sg.train_samples()),
            sort_indices_lexiographically(iso_sg.train_samples()),
        )
        assert bkd.allclose(
            sort_indices_lexiographically(sg.train_values().T),
            sort_indices_lexiographically(iso_sg.train_values().T),
        )
        assert bkd.allclose(
            sg._smolyak_coefs[
                argsort_indices_lexiographically(
                    sg._subspace_gen.get_indices()
                )
            ],
            iso_sg._smolyak_coefs[
                argsort_indices_lexiographically(
                    iso_sg._subspace_gen.get_indices()
                )
            ],
        )

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 101)))
        sg_test_values = sg(test_samples)
        assert bkd.allclose(sg_test_values, iso_sg(test_samples), atol=1e-15)

    def test_isotropic_locally_adaptive_sparse_grid(self):
        test_cases = [[1, 3, 1], [2, 3, 1], [3, 3, 2]]
        for test_case in test_cases:
            self._check_isotropic_locally_adaptive_sparse_grid(*test_case)

    def test_early_termination_locally_adaptive_sparse_grid(self):
        nvars, level, nqoi = 2, 3, 2
        bkd = self.get_backend()
        # set max_cost to be much larger than needed and temrinate
        # early by only completing a few steps, setting termination with
        # max_cost will not work given the implementation of the test below
        sg = self._setup_locally_adaptive_sparse_grid(nvars, level, nqoi, 100)

        def fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]*bkd.arange(
                1, nqoi+1
            )[None, :]

        for ii in range(4):
            sg.step(fun)

        idx = []
        for basis_index in sg._basis_gen._get_all_basis_indices().T:
            key = sg._basis_gen._hash_index(basis_index)
            if (
                key not in sg._basis_gen._sel_basis_indices_dict
                and key not in sg._basis_gen._cand_basis_indices_dict
            ):
                idx.append(sg._basis_gen._basis_indices_dict[key][0])
                print(basis_index)
        assert bkd.allclose(
            sg.train_values()[idx], bkd.zeros((len(idx), sg.nqoi()))
        )
        sg.plot_grid(plt.figure().gca())




class TestNumpyCombination(TestCombination, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCombination(TestCombination, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

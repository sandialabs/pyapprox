import unittest
from pyapprox.indexing import *
from functools import partial

class TestIndexing(unittest.TestCase):
    def test_compute_hyperbolic_indices(self):
        num_vars = 3; level = 3; p=1.0
        indices = compute_hyperbolic_indices(num_vars, level, p)
        assert indices.shape[1]==nchoosek(num_vars+level,num_vars)


        num_vars = 4; level = 3; p=0.5
        indices = compute_hyperbolic_indices(num_vars, level, p)
        assert np.all(pnorm(indices,p,axis=0)<=level)

    def test_set_difference_1d_array(self):
        num_vars = 2; level1 = 1; p=1.0
        indices1 = np.arange(0,10,2)
        indices2 = np.arange(10)
        indices = set_difference(indices1,indices2)

        true_indices = np.arange(1,10,2)
        assert np.allclose(indices,true_indices)


    def test_set_difference_2d_array(self):
        num_vars = 2; level1 = 1; p=1.0
        indices1 = compute_hyperbolic_indices(num_vars, level1, p)

        level2 = 2;
        indices2 = compute_hyperbolic_indices(num_vars, level2, p)

        indices = set_difference(indices1,indices2)

        true_indices = np.asarray([[2,0],[0,2],[1,1]]).T
        assert np.allclose(indices,true_indices)

    def test_argsort_indices_leixographically(self):
        num_vars=2; degree=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        sorted_idx = argsort_indices_leixographically(indices)

        sorted_indices = indices[:,sorted_idx]
        true_sorted_indices = np.array([[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]]).T
        assert np.allclose(sorted_indices, true_sorted_indices)

        num_vars=3; degree=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        sorted_idx = argsort_indices_leixographically(indices)

        sorted_indices = indices[:,sorted_idx]
        true_sorted_indices = np.array(
            [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0,2],
             [0,1,1],[0,2,0],[1,0,1],[1,1,0],[2,0,0]]).T
        assert np.allclose(sorted_indices, true_sorted_indices)

    def test_compute_downward_closed_indices(self):
        num_vars, degree = [2,5]
        downward_closed_indices = compute_downward_closed_indices(
            num_vars,partial(total_degree_admissibility_criteria,degree))
        total_degree_indices = compute_hyperbolic_indices(num_vars,degree,1)
        assert np.allclose(sort_indices_lexiographically(total_degree_indices),
                           sort_indices_lexiographically(downward_closed_indices))

        num_vars, degree = [5,5]
        downward_closed_indices = compute_downward_closed_indices(
            num_vars,partial(pnorm_admissibility_criteria,degree,0.4))
        pnorm_indices = compute_hyperbolic_indices(num_vars,degree,0.4)
        assert np.allclose(sort_indices_lexiographically(pnorm_indices),
                           sort_indices_lexiographically(downward_closed_indices))

        num_vars, degree = [2,5]
        anisotropic_weights = np.asarray([1,2])
        min_weight = np.asarray(anisotropic_weights).min()
        admissibility_criteria = partial(
            anisotropic_admissibility_criteria,anisotropic_weights,min_weight,
            degree)
        downward_closed_indices = compute_downward_closed_indices(
            num_vars,admissibility_criteria)
        anisotropic_indices = compute_anisotropic_indices(
            num_vars,degree,anisotropic_weights)
        assert np.allclose(sort_indices_lexiographically(anisotropic_indices),
                           sort_indices_lexiographically(downward_closed_indices))



if __name__=='__main__':
    indexing_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestIndexing)
    unittest.TextTestRunner(verbosity=2).run(indexing_test_suite)


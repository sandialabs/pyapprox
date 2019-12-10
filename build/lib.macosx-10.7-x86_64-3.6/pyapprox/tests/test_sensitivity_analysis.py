from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import unittest
from pyapprox.sensitivity_analysis import *
class TestSensitivityAnalysis(unittest.TestCase):
    def test_get_sobol_indices_from_pce(self):
        num_vars = 5; degree = 5
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        interaction_indices, interaction_values = \
            get_sobol_indices(
                coefficients,indices,max_order=num_vars)
        assert np.allclose(
            interaction_values.sum(axis=0), np.ones(2))

    def test_get_sobol_indices_from_pce_max_order(self):
        num_vars = 3; degree = 4; max_order=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        interaction_indices, interaction_values = \
            get_sobol_indices(coefficients,indices,max_order)

        assert len(interaction_indices)==6
        true_interaction_indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        for ii in range(len(interaction_indices)):
            assert np.allclose(
                true_interaction_indices[ii],interaction_indices[ii])
        
        true_variance = np.asarray(
            [indices.shape[1]-1,2**2*(indices.shape[1]-1)])

        # get the number of interactions involving variables 0 and 1
        # test problem is symmetric so number is the same for all variables
        num_pairwise_interactions = np.where(
            np.all(indices[0:2,:]>0,axis=0)&(indices[2,:]==0))[0].shape[0]
        I = np.where(np.all(indices[0:2,:]>0,axis=0))[0]
        
        true_interaction_values = np.vstack((
            np.tile(np.arange(1,3)[np.newaxis,:],
                    (num_vars,1))**2*degree/true_variance,
            np.tile(np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*num_pairwise_interactions/true_variance))

        assert np.allclose(true_interaction_values,interaction_values)

        #plot_interaction_values( interaction_values, interaction_indices)

    def test_get_main_and_total_effect_indices_from_pce(self):
        num_vars = 3; degree = num_vars; max_order=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        main_effects, total_effects = \
            get_main_and_total_effect_indices_from_pce(coefficients,indices)
        true_variance = np.asarray(
            [indices.shape[1]-1,2**2*(indices.shape[1]-1)])
        true_main_effects = np.tile(
            np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*degree/true_variance
        assert np.allclose(main_effects,true_main_effects)

        # get the number of interactions variable 0 is involved in
        # test problem is symmetric so number is the same for all variables
        num_interactions_per_variable = np.where(indices[0,:]>0)[0].shape[0]
        true_total_effects = np.tile(
            np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*num_interactions_per_variable/true_variance
        assert np.allclose(true_total_effects,total_effects)

        #plot_total_effects(total_effects)
        #plot_main_effects(main_effects)
        
    
if __name__== "__main__":    
    sensitivity_analysis_test_suite=unittest.TestLoader().loadTestsFromTestCase(
        TestSensitivityAnalysis)
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)



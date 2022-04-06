import numpy as np
import unittest
from functools import partial
from scipy import stats

from pyapprox.variables.marginals import variables_equivalent
from pyapprox.surrogates.coupled_systems import SystemNetwork
from pyapprox.surrogates.system_analysis import (
    DecoupledSystemSurrogate, TerminateTest
)
from pyapprox.surrogates.tests.test_coupled_systems import (
    get_3_recursive_polynomial_components, build_chain_graph
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import variance_refinement_indicator
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_in_polynomial_order, clenshaw_curtis_rule_growth
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)


def get_polynomial_models():
    # Define polynomial functions
    def f1(x):  # x = [x1]
        f = 1*x[0]**2 + 1
        return f[:, np.newaxis]

    def f2(x):  # x = [y1, x3]
        print(x)
        f = 2*x[0]**2 + 1.5*x[1]**2 + -1
        return f[:, np.newaxis]

    def f3(x):  # x = [x2, y2]
        f = -2*x[0]**2 + 1*x[1]**1 + 1
        return f[:, np.newaxis]

    return f1, f2, f3


class TestSystemAnalysis(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_feed_forward_system_of_polynomials_fixed_coupling_bounds(self):
        graph, variables, graph_data = get_3_recursive_polynomial_components()
        # overwrite functions, they have the same network structure so this
        # is fine
        graph_data['functions'] = get_polynomial_models()
        graph = build_chain_graph(3, graph_data)
        network = SystemNetwork(graph)
        approx = DecoupledSystemSurrogate(
            network, variables, estimate_coupling_ranges=False, verbose=2)

        univariate_quad_rule_info = [
            clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth]
        refinement_indicator = partial(
            variance_refinement_indicator, convex_param=0.5)
        options = [{'univariate_quad_rule_info': univariate_quad_rule_info,
                    'max_nsamples': 30, 'tol': 0, 'verbose': 0,
                    'refinement_indicator': refinement_indicator}]*3

        # check pre-specified bounds for coupling variables are stored
        # correctly
        output_bounds = [[0, 2], [0.5, 7]]
        coupling_variables = {
            0: [],
            1: [stats.uniform(output_bounds[0][0], np.diff(output_bounds[0]))],
            2: [stats.uniform(output_bounds[1][0], np.diff(output_bounds[1]))]}
        approx.set_coupling_variables(coupling_variables)

        true_node_vars = [
            [stats.uniform(0, 1)],
            [stats.uniform(output_bounds[0][0], np.diff(output_bounds[0])),
             stats.uniform(0, 1)],
            [stats.uniform(0, 1),
             stats.uniform(output_bounds[1][0], np.diff(output_bounds[1]))]]
        surr_graph = approx.surrogate_network.graph
        for node_id in surr_graph.nodes:
            node_vars = approx.get_node_variables(surr_graph.nodes[node_id])
            assert len(node_vars) == len(true_node_vars[node_id])
            for ii in range(len(node_vars)):
                assert variables_equivalent(
                    node_vars[ii], true_node_vars[node_id][ii])

        # test component surrogates and final integrated surrogate are
        # exact
        approx.initialize_component_surrogates(options)
        approx.build(terminate_test=TerminateTest(max_work=50))

        nvalidation_samples = 10
        validation_samples = generate_independent_random_samples(
            variables, nvalidation_samples)
        validation_values = network(
            validation_samples, component_ids=[0, 1, 2])

        validation_values_approx = []
        for node_id in surr_graph.nodes:
            component_surr = surr_graph.nodes[node_id]['functions']
            idx = graph_data["global_random_var_indices"][node_id]
            if node_id == 0:
                component_vals_approx = component_surr(
                    validation_samples[idx, :])
            else:
                jdx = graph_data["local_random_var_indices"][node_id][0]
                component_samples = np.empty((2, validation_samples.shape[1]))
                component_samples[jdx] = validation_samples[idx, :]
                component_samples[1-jdx] = validation_values_approx[-1].T
                component_vals_approx = component_surr(component_samples)
            validation_values_approx.append(component_vals_approx)
            assert np.allclose(
                validation_values[node_id], component_vals_approx)
        assert np.allclose(
            approx(validation_samples), validation_values[-1])

    def test_feed_forward_system_of_polynomials_estimate_coupling_bounds(self):
        graph, variables, graph_data = get_3_recursive_polynomial_components()
        # overwrite functions, they have the same network structure so this
        # is fine
        graph_data['functions'] = get_polynomial_models()
        graph = build_chain_graph(3, graph_data)
        network = SystemNetwork(graph)
        approx = DecoupledSystemSurrogate(
            network, variables, estimate_coupling_ranges=True, verbose=2,
            nrefinement_samples=1e4)

        univariate_quad_rule_info = [
            clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth]
        refinement_indicator = None
        options = [{'univariate_quad_rule_info': univariate_quad_rule_info,
                    'max_nsamples': 30, 'tol': 0, 'verbose': 0,
                    'refinement_indicator': refinement_indicator}]*3
        surr_graph = approx.surrogate_network.graph

        # under estimate coupling ranges
        output_bounds = [[0.5, 1.5], [2, 5]]
        coupling_variables = {
            0: [],
            1: [stats.uniform(output_bounds[0][0], np.diff(output_bounds[0]))],
            2: [stats.uniform(output_bounds[1][0], np.diff(output_bounds[1]))]}
        approx.set_coupling_variables(coupling_variables)
        approx.initialize_component_surrogates(options)
        approx.build(terminate_test=TerminateTest(max_work=30))

        nvalidation_samples = 10
        validation_samples = generate_independent_random_samples(
            variables, nvalidation_samples)
        validation_values = network(
            validation_samples, component_ids=[0, 1, 2])

        validation_values_approx = []
        for node_id in surr_graph.nodes:
            component_surr = surr_graph.nodes[node_id]['functions']
            idx = graph_data["global_random_var_indices"][node_id]
            if node_id == 0:
                component_vals_approx = component_surr(
                    validation_samples[idx, :])
            else:
                jdx = graph_data["local_random_var_indices"][node_id][0]
                component_samples = np.empty((2, validation_samples.shape[1]))
                component_samples[jdx] = validation_samples[idx, :]
                component_samples[1-jdx] = validation_values_approx[-1].T
                component_vals_approx = component_surr(component_samples)
            validation_values_approx.append(component_vals_approx)
            assert np.allclose(
                validation_values[node_id], component_vals_approx)
        assert np.allclose(
            approx(validation_samples), validation_values[-1])

    def test_three_component_adaptive_refinement(self):
        """
        Make sure that if upstream model is not refined first that it still
        can be.
        """
        graph, variables, graph_data = get_3_recursive_polynomial_components()
        network = SystemNetwork(graph)
        approx = DecoupledSystemSurrogate(
            network, variables, verbose=2, nrefinement_samples=1e4)

        # functions are monotonic on [0,1] so get bounds of component outputs
        # by evaluating the function at 1
        output_bounds = [np.inf for ii in range(3)]
        output_bounds[0] = network.graph.nodes[0]['functions'](
            np.array([[1.0]]))[0, 0]
        output_bounds[1] = network.graph.nodes[1]['functions'](
            np.array([[output_bounds[0], 1.0]]).T)[0, 0]
        output_bounds[2] = network.graph.nodes[1]['functions'](
            np.array([[1.0, output_bounds[1]]]).T)[0, 0]
        coupling_variables = {
            0: [],
            1: [stats.uniform(0, output_bounds[0])],
            2: [stats.uniform(0, output_bounds[1])]}
        approx.set_coupling_variables(coupling_variables)

        univariate_quad_rule_info = [
            clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth]
        refinement_indicator = None
        options = [{'univariate_quad_rule_info': univariate_quad_rule_info,
                    'max_nsamples': 30, 'tol': 0, 'verbose': 2,
                    'refinement_indicator': refinement_indicator}]*3

        approx.initialize_component_surrogates(options)

        class Callback():
            def __init__(self):
                self.best_components = []
                self.best_subspace_indices = []

            def __call__(self, approx):
                self.best_components.append(approx.best_component)
                self.best_subspace_indices.append(approx.best_subspace_index)

        terminate_test = TerminateTest(max_iters=4)
        callback = Callback()
        approx.build(callback=callback, terminate_test=terminate_test)

        assert np.allclose(callback.best_components, [2, 1, 1, 0])


if __name__ == "__main__":
    system_analysis_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestSystemAnalysis)
    unittest.TextTestRunner(verbosity=2).run(system_analysis_test_suite)

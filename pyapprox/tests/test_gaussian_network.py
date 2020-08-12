#from work.multi_fidelity.latent_variable_networks import *
from pyapprox.gaussian_network import *
from functools import partial
import unittest

class TestLVN(unittest.TestCase):
    def test_get_var_ids_to_eliminate(self):
        nmodels,nvars_per_model=4,2
        network_labels = ['M%d'%ii for ii in range(nmodels)]
        network_ids = np.arange(len(network_labels))
        query_labels_idx = [1]
        
        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids,network_labels,query_labels,evidence_ids=None)
        mask = np.zeros(len(network_labels),dtype=bool)
        mask[query_labels_idx]=True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate,true_var_ids_to_eliminate)

        nmodels,nvars_per_model=4,2
        network_labels = ['M%d'%ii for ii in range(nmodels)]
        network_ids = np.arange(len(network_labels))
        query_labels_idx = [1,3]
        
        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids,network_labels,query_labels,evidence_ids=None)
        mask = np.zeros(len(network_labels),dtype=bool)
        mask[query_labels_idx]=True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate,true_var_ids_to_eliminate)

        nmodels,nvars_per_model,ndata_per_model=4,2,2
        network_labels = ['M%d'%ii for ii in range(nmodels)]
        query_labels_idx = [1]
        evidence_labels = []
        for jj in range(ndata_per_model):
            evidence_labels += ['M%d_%d'%(ii,jj) for ii in range(nmodels)]
        network_labels += evidence_labels
        network_ids = np.arange(len(network_labels))
        evidence_ids = network_ids[nmodels:]

        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids,network_labels,query_labels,evidence_ids=evidence_ids)
        # variable labels must not contain any labels associated with evidence
        # thus the slicing in the following loop
        mask = np.zeros(len(network_labels),dtype=bool)
        mask[query_labels_idx]=True
        mask[nmodels:]=True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate,true_var_ids_to_eliminate)

    def test_conditional_probability_density_factorization(self):
        nvars1, nvars2 = [3,2]
        Amat = np.random.normal(0.,1.,(nvars2,nvars1))
        bvec = np.random.normal(0,1,(nvars2))
        mean1 = np.random.normal(0,1,nvars1)
        true_joint_mean = np.concatenate([mean1,Amat.dot(mean1)+bvec])
        temp1 = np.random.normal(0.,1.,(nvars1,nvars1))
        cov1 = temp1.T.dot(temp1)
        temp2g1 = np.random.normal(0.,1.,(nvars2,nvars2))
        cov2g1 = temp2g1.T.dot(temp2g1)

        joint_mean, joint_covar = \
            joint_density_from_linear_conditional_relationship(
                mean1,cov1,cov2g1,Amat,bvec)

        precision_matrix1,shift1,normalization1 = \
            convert_gaussian_to_canonical_form(mean1,cov1)
        
        var1_ids, nvars_per_var1 = [0],[nvars1]
        var2_ids, nvars_per_var2 = [1],[nvars2]
        
        precision_matrix2g1,shift2g1,normalization2g1,var2g1_ids, \
            nvars_per_var2g1 = \
                convert_conditional_probability_density_to_canonical_form(
                    Amat,bvec,cov2g1,var1_ids,nvars_per_var1,
                    var2_ids, nvars_per_var2)

        precision_matrix,shift,normalization,all_var_ids,nvars_per_all_vars = \
            multiply_gaussian_densities_in_compact_canonical_form(
                precision_matrix1,shift1,normalization1,var1_ids,nvars_per_var1,
                precision_matrix2g1,shift2g1,normalization2g1,var2g1_ids,
                nvars_per_var2g1)
        mean,covariance = convert_gaussian_from_canonical_form(
            precision_matrix,shift)

        assert np.allclose(joint_mean,mean)
        assert np.allclose(joint_covar,covariance)

    def test_recursive_graph_prior(self):
        nmodels = 3
        num_vars=1
        prior_covs = [1,2,3]
        cpd_scales  =[0.5,0.4]
        univariate_variables = [stats.uniform(-1,2)]*num_vars

        degrees=[0]*nmodels
        
        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        network = build_recursive_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        network.convert_to_compact_factors()
        labels = [l[1] for l in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean,prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix,factor_prior.shift)
        #print('prior_cov\n',prior_cov)

        a21,a32=cpd_scales
        v1,v2,v3=prior_covs
        true_prior_cov = np.array([
            [v1,a21*v1,a32*a21*v1],[a21*v1,v2,a32*v2],[a32*a21*v1,a32*v2,v3]])
        assert np.allclose(true_prior_cov,prior_cov)


        degrees=[1]*(nmodels-1)+[2]

        
        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        network = build_recursive_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        network.convert_to_compact_factors()
        labels = [l[1] for l in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean,prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix,factor_prior.shift)
        #print('prior_cov\n',prior_cov)

        a21,a32=cpd_scales
        v1,v2,v3=prior_covs
        assert degrees[0]==degrees[1]==degrees[2]-1
        I1 = np.eye(degrees[0]+1)
        I2 = np.eye(degrees[1]+1)
        I3 = np.eye(degrees[2])
        true_prior_cov = np.zeros((nparams.sum(),nparams.sum()))
        true_prior_cov[:nparams.sum()-(degrees[2]-degrees[1]),:nparams.sum()-(degrees[2]-degrees[1])]=np.vstack([np.hstack([v1*I1,a21*v1*I1,a32*a21*v1*I1]),np.hstack([a21*v1*I2,v2*I2,a32*v2*I2]),np.hstack([a32*a21*v1*I3,a32*v2*I3,v3*I3])])
        # the poly terms not found in the other models are not correlated with
        # any term from the lower models. But when graph is setup we still
        # subtract a32**2*v2 from the specified covariance. So although
        # all other terms will have variance given by v3 these remainder terms
        # will not. So add rows to true_prior_cov accordingly
        true_prior_cov[-(degrees[2]-degrees[1]):,-(degrees[2]-degrees[1]):]=\
            np.eye(degrees[2]-degrees[1])*(v3-a32**2*v2)
        #print(true_prior_cov)
        assert np.allclose(true_prior_cov,prior_cov)

    def test_peer_graph_prior(self):
        nmodels = 3
        num_vars=1
        prior_covs = [1,2,3]
        cpd_scales  =[0.5,0.4]
        univariate_variables = [stats.uniform(-1,2)]*num_vars

        degrees=[0]*nmodels

        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        network = build_peer_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        network.convert_to_compact_factors()
        labels = [l[1] for l in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean,prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix,factor_prior.shift)
        #print('prior_cov\n',prior_cov)

        a31,a32=cpd_scales
        v1,v2,v3=prior_covs
        true_prior_cov = np.array([
            [v1,0,a31*v1],[0,v2,a32*v2],[a31*v1,a32*v2,v3]])
        #print('true_prior_cov\n',true_prior_cov)
        assert np.allclose(true_prior_cov,prior_cov)

        degrees=[6,6,6]

        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        network = build_peer_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        network.convert_to_compact_factors()
        labels = [l[1] for l in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean,prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix,factor_prior.shift)

        
        I1 = np.eye(degrees[0]+1)
        I2 = np.eye(degrees[1]+1)
        I3 = np.eye(degrees[2]+1)
        rows = [np.hstack([v1*I1,0*I1,a31*v1*I1]),
                np.hstack([0*I2,v2*I2,a32*v2*I2]),
                np.hstack([a31*v1*I3,a32*v2*I3,v3*I3])]
        true_prior_cov=np.vstack(rows)
        #print('true_prior_cov\n',true_prior_cov)
        assert np.allclose(true_prior_cov,prior_cov)

    def test_one_model_inference(self):
        np.random.seed(1)
        nmodels=1
        num_vars=1
        degrees=[2]
        prior_covs = [1]
        cpd_scales = []

        nsamples = 5
        noise_std = np.array([0.1])
        samples_train = [np.random.uniform(-1,1,(1,nsamples))]

        univariate_variables = [stats.uniform(-1,2)]*num_vars
        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        basis_matrices = [b(s) for b,s in zip(
            basis_matrix_funcs,samples_train)]
        true_coef = [
            np.random.normal(0,np.sqrt(prior_covs[ii]),(nparams[ii],1))
            for ii in range(len(prior_covs))]
        noise = [noise_std[ii]*np.random.normal(
            0,noise_std[ii],(samples_train[ii].shape[1],1))
                 for ii in range(len(prior_covs))]
        data_train = [b.dot(c) + n for b,c,n in zip(
            basis_matrices,true_coef,noise)]

        # solve using classical inversion formula
        from pyapprox.bayesian_inference.laplace import \
            laplace_posterior_approximation_for_linear_models
        true_post = laplace_posterior_approximation_for_linear_models(
            basis_matrices[0],np.zeros((nparams[0],1)),
            np.eye(nparams[0])/prior_covs[0],
            np.eye(samples_train[0].shape[1])/noise_std**2,data_train[0])
        #print('True post covar\n',true_post[1])

        # solve using network
        network = build_peer_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        labels = [l[1] for l in network.graph.nodes.data('label')]
        network.add_data_to_network(samples_train,noise_std**2)
        #convert_to_compact_factors must be after add_data when doing inference
        network.convert_to_compact_factors()
        evidence, evidence_ids = network.assemble_evidence(data_train)

        factor_post = cond_prob_variable_elimination(
            network,labels,evidence_ids=evidence_ids,evidence=evidence)    
        gauss_post = convert_gaussian_from_canonical_form(
            factor_post.precision_matrix,factor_post.shift)
        #print('Graph post covar\n',gauss_post[1])
        
        assert np.allclose(gauss_post[1],true_post[1])
        
    def test_recursive_graph_inference(self):
        #add tests for multiplie models for computing entire covariance and
        #for just computing covariance of high-fidelity. The later will check
        #variable elimination, i.e. marginalization
        np.random.seed(1)
        nmodels=3
        num_vars=1
        degrees=[1]*(nmodels-1)+[2]
        prior_covs = [1]*nmodels
        cpd_scales  =[0.9]*(nmodels-1) # Amat = cpd_scale * identity

        nsamples = [3]*(nmodels-1)+[2]
        noise_std = np.array([0.1]*(nmodels-1)+[0.2])
        samples_train = [
            np.random.uniform(-1,1,(1,nsamples[ii])) for ii in range(nmodels)]

        univariate_variables = [stats.uniform(-1,2)]*num_vars
        polys, nparams = get_total_degree_polynomials(
            [univariate_variables]*nmodels,degrees)
        basis_matrix_funcs = [p.basis_matrix for p in polys]
        basis_matrices = [b(s) for b,s in zip(
            basis_matrix_funcs,samples_train)]
        true_coef = [
            np.random.normal(0,np.sqrt(prior_covs[ii]),(nparams[ii],1))
            for ii in range(len(prior_covs))]
        noise = [noise_std[ii]*np.random.normal(
            0,noise_std[ii],(samples_train[ii].shape[1],1))
                 for ii in range(len(prior_covs))]
        data_train = [b.dot(c) + n for b,c,n in zip(
            basis_matrices,true_coef,noise)]

        # solve using classical inversion formula
        from pyapprox.bayesian_inference.laplace import \
            laplace_posterior_approximation_for_linear_models
        import scipy
        basis_mat = scipy.linalg.block_diag(*basis_matrices)
        noise_cov_inv = scipy.linalg.block_diag(
            *[1/noise_std[ii]**2*np.eye(samples_train[ii].shape[1])
              for ii in range(nmodels)])
        data = np.vstack(data_train)
        network = build_recursive_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        network.convert_to_compact_factors()
        labels = [l[1] for l in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean,prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix,factor_prior.shift)
        true_post = laplace_posterior_approximation_for_linear_models(
            basis_mat,np.zeros((basis_mat.shape[1],1)),
            np.linalg.inv(prior_cov),noise_cov_inv,data)
        #print('True post covar\n',true_post[1])

        # solve using network
        network = build_recursive_polynomial_network(
            prior_covs,cpd_scales,basis_matrix_funcs,nparams)
        labels = [l[1] for l in network.graph.nodes.data('label')]
        network.add_data_to_network(samples_train,noise_std**2)
        #convert_to_compact_factors must be after add_data when doing inference
        network.convert_to_compact_factors()
        evidence, evidence_ids = network.assemble_evidence(data_train)
        factor_post = cond_prob_variable_elimination(
            network,labels,evidence_ids=evidence_ids,evidence=evidence)    
        gauss_post = convert_gaussian_from_canonical_form(
            factor_post.precision_matrix,factor_post.shift)
        #print('Graph post covar\n',gauss_post[1])
        
        assert np.allclose(gauss_post[1],true_post[1])
    
if __name__ == '__main__':
    LVN_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLVN)
    unittest.TextTestRunner(verbosity=2).run(LVN_test_suite)
    

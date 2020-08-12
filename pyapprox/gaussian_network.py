import numpy as np
import networkx as nx
import copy
from pyapprox.multivariate_gaussian import *

def get_var_ids_to_eliminate(network_ids,network_labels,query_labels,
                             evidence_ids=None):
    """
    Get the ids of variables in a network which are not being queried.
    This function will always exclude from elimiation any ids which are in 
    evidence_ids.

    Parameters
    ----------
    network_ids : list
        A list of integer variable ids of the nodes to eliminate from the 
        network.

    network_labels: list
       A list of strings containing the names of each node in the network

    query_labels : list
        A list of variable labels which must remain after elimination

    evidence_ids : list
        A list of data variable tuples (var_name,1,np.ndarray (1))

    Returns
    -------
    elimiate_ids : list
       A list of labels of variables to be eliminated. Labels associated with
       evidence (data) will not be eliminated
    """

    nvars_to_query = len(query_labels)
    query_ids = np.empty(nvars_to_query,dtype=float)
    for ii in range(nvars_to_query):
        found = False
        label = query_labels[ii]
        for jj in range(len(network_labels)):
            if label==network_labels[jj]:
                query_ids[ii]=network_ids[jj]
                found=True
                break
        if not found:
            raise Exception('query label %s was not found'%label)

    if evidence_ids is not None:
        assert np.intersect1d(query_ids,evidence_ids).shape[0]==0
        query_ids = np.concatenate((query_ids,evidence_ids))

    mask = np.isin(network_ids,query_ids)
    #eliminate_ids=[network_ids[ii] for ii in range(len(mask)) if not mask[ii]]
    eliminate_ids = np.asarray(network_ids)[~mask]
    return eliminate_ids

def get_nparams_of_nodes(vands):
    num_nodes = len(vands)
    nparams = np.asarray([vands[ii].shape[1] for ii in range(num_nodes)])
    return nparams

from scipy import stats
def basis_matrix(ranges,degree,samples):
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
        define_poly_options_from_variable_transformation
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation 
    from pyapprox.indexing import compute_hyperbolic_indices
    #from pyapprox.configure_plots import *
    samples = np.atleast_2d(samples)
    num_vars = samples.shape[0]; alpha_poly = 0; beta_poly = 0
    assert (num_vars==len(ranges)/2)
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        stats.uniform(),num_vars)
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
    poly.set_indices(indices)
    return poly.basis_matrix(samples)

def basis_matrix_cols(nvars,degree):
    from pyapprox.utilities import total_degree_space_dimension
    ncols = total_degree_space_dimension(nvars, degree)
    return ncols

def get_cpd_linear_matrix(graph,node_index):
    child_indices = list(graph.predecessors(node_index))
    if len(child_indices)==0:
        return None,None

    Amat_blocks = []
    node_nparams = graph.nodes[node_index]["nparams"]
    for child in child_indices:
        cpd_scale = graph.edges[child,node_index]['cpd_scale']
        child_nparams = graph.nodes[child]["nparams"]
        if np.isscalar(cpd_scale) or cpd_scale.ndim==1:
            Amat_blocks.append(
                cpd_scale*np.eye(node_nparams,child_nparams))
        else:
            assert cpd_scale.shape==(node_nparams,child_nparams)
            Amat_blocks.append(cpd_scale)
        
    Amat = np.hstack(Amat_blocks)
    return Amat, child_indices

def get_cpd_prior_covariance(graph,node_index):
    prior_scale = graph.nodes[node_index]["prior_scale"]
    nparams = graph.nodes[node_index]["nparams"]
    if np.isscalar(prior_scale) or prior_scale.ndim==1:
        prior_cov=np.eye(nparams)*prior_scale**2
    else:
        assert prior_scale.shape==(nparams,nparams)
        prior_cov=prior_scale
    return prior_cov

def get_gaussian_factor_in_canonical_form(Amat,bvec,cov2g1,
                                          var1_ids,nvars_per_var1,
                                          var2_ids, nvars_per_var2):
    """
    Todo consider massing inv(cov2g1) to function so can leverage structure
    in matrix and not to inversion inside convert_conditional function
    """
    if bvec is None:
        bvec = np.zeros(Amat.shape[0])
    precision_matrix,shift,normalization,var_ids,nvars_per_var = \
        convert_conditional_probability_density_to_canonical_form(
            Amat,bvec,cov2g1,var1_ids,nvars_per_var1,
            var2_ids, nvars_per_var2)
    return GaussianFactor(
        precision_matrix,shift,normalization,var_ids,nvars_per_var)


def build_recursive_polynomial_network(prior_covs,cpd_scales,basis_matrix_funcs,
                                       nparams,model_labels=None):
    """
    prior_scales : list
        List of diagonal matrices (represented by either a scalar for a constant
        diagonal or a vector)

    cpd_scales : list
        List of diagonal matrices (represented by either a scalar for a constant
        diagonal or a vector)
    
    """
    nmodels = len(nparams)
    assert len(basis_matrix_funcs)==nmodels
    assert len(prior_covs)==nmodels
    assert len(cpd_scales)==nmodels-1

    if model_labels is None:
        model_labels = ['M_%d'%ii for ii in range(nmodels)]

    graph = nx.DiGraph()
    ii=0
    graph.add_node(
        ii,label=model_labels[ii],prior_scale=np.sqrt(prior_covs[ii]),
        nparams=nparams[ii],
        basis_matrix_func=basis_matrix_funcs[ii])
    #todo pass in list of nparams
    for ii in range(1,nmodels):
        prior_scale = np.sqrt(
            max(1e-8,prior_covs[ii] - cpd_scales[ii-1]**2*prior_covs[ii-1]))
        graph.add_node(
            ii,label=model_labels[ii],prior_scale=prior_scale,
            nparams=nparams[ii],basis_matrix_func=basis_matrix_funcs[ii])

    graph.add_edges_from(
        [(ii,ii+1,{'cpd_scale':cpd_scales[ii]}) for ii in range(nmodels-1)])

    network = BayesianNetwork(graph)
    return network

class BayesianNetwork(object):
    def __init__(self, graph):
        self.graph=graph
        self.ndata=None
        
        self.construct_dataless_network()

    def construct_dataless_network(self):
        """
        """
        nnodes = len(self.graph.nodes)
        if len(self.graph.nodes)>1:
            assert (np.max(self.graph.nodes)==nnodes-1 and
                    np.min(self.graph.nodes)==0)
        
        self.Amats, self.cpd_prior_covs = [None]*nnodes,[None]*nnodes
        self.node_childs, self.node_nvars = [None]*nnodes,[None]*nnodes
        self.node_labels, self.node_ids = [None]*nnodes, list(np.arange(nnodes))
        for ii in self.graph.nodes:
            # Extract node information from graph
            nparams = self.graph.nodes[ii]['nparams']
            self.Amats[ii],self.node_childs[ii] = get_cpd_linear_matrix(
                self.graph,ii)
            self.cpd_prior_covs[ii] = get_cpd_prior_covariance(
                self.graph,ii)
            self.node_labels[ii] = self.graph.nodes[ii]['label']
            self.node_nvars[ii] = self.cpd_prior_covs[ii].shape[0]

            # check the validity of the graph
            if self.node_childs[ii] is not None:
                for child in self.node_childs[ii]:
                    assert child < ii

        self.node_var_ids=[[ii] for ii in range(nnodes)] # move to when creating dataless network


    def xadd_data_to_network(self,samples,noise_variances):
        """
        """
        nnodes = len(self.graph.nodes)
        assert len(samples)==nnodes
        assert len(noise_variances)==nnodes
        self.samples=samples
        #self.build_matrix_functions = build_matrix_functions
        self.ndata = [samples[ii].shape[1] for ii in range(nnodes)]
        
        # retain copy of old dataless graph
        dataless_graph = copy.deepcopy(self.graph)
        kk = len(self.graph.nodes)
        for ii in dataless_graph.nodes:
            vand = self.graph.nodes[ii]['basis_matrix_func'](samples[ii])
            assert vand.shape[1]==self.graph.nodes[ii]['nparams']
            for jj in range(self.ndata[ii]):
                self.node_ids.append(kk)
                label=self.graph.nodes[ii]['label']+'_%d'%jj
                self.graph.add_node(len(self.Amats),label=label)
                self.graph.add_edge(ii,kk)
                
                self.Amats.append(vand[jj:jj+1,:]),self.node_childs.append([ii])
                self.cpd_prior_covs.append(noise_variances[ii]*np.eye(1))
                self.node_labels.append(label)
                self.node_nvars.append(self.cpd_prior_covs[-1].shape[0])
                kk+=1
        self.evidence_ids = np.arange(len(dataless_graph.nodes),kk)

    def add_data_to_network(self,samples,noise_variances):
        """
        """
        nnodes = len(self.graph.nodes)
        assert len(samples)==nnodes
        assert len(noise_variances)==nnodes
        self.samples=samples
        #self.build_matrix_functions = build_matrix_functions
        self.ndata = [samples[ii].shape[1] for ii in range(nnodes)]
        
        # retain copy of old dataless graph
        dataless_graph = copy.deepcopy(self.graph)
        kk = len(self.graph.nodes)
        jj=kk
        self.evidence_ids = []
        for ii in dataless_graph.nodes:
            vand = self.graph.nodes[ii]['basis_matrix_func'](samples[ii])
            assert vand.shape[1]==self.graph.nodes[ii]['nparams']
            self.node_ids.append(kk)
            label=self.graph.nodes[ii]['label']+'_data'
            self.graph.add_node(len(self.Amats),label=label)
            self.graph.add_edge(ii,kk)
            self.Amats.append(vand),self.node_childs.append([ii])
            self.cpd_prior_covs.append(
                noise_variances[ii]*np.eye(vand.shape[0]))
            self.node_labels.append(label)
            self.node_nvars.append(self.cpd_prior_covs[-1].shape[0])
            self.node_var_ids.append(np.arange(jj,jj+vand.shape[0]))
            jj+=vand.shape[0]
            kk+=1
        self.evidence_ids = np.arange(len(dataless_graph.nodes),jj)

    def assemble_evidence(self,data):
        """
        Relies on order vandermondes are added in network.add_data_to_network
        """
        nnodes = len(data)
        nevidence = np.sum([d.shape[0] for d in data])
        evidence = np.empty((nevidence))
        kk = 0
        for ii in range(nnodes):
            assert (data[ii].ndim==1 or data[ii].shape[1]==1),(
                ii,data[ii].shape)
            for jj in range(data[ii].shape[0]):
                evidence[kk] = data[ii][jj]
                kk += 1
        return evidence, self.evidence_ids


    def assemble_evidence_deprecated(self,data):
        """
        Relies on order vandermondes are added in network.add_data_to_network
        """
        nnodes = len(data)
        nevidence = np.sum([d.shape[0] for d in data])
        evidence = np.empty((nevidence))
        evidence_ids = np.empty((nevidence),dtype=int)
        kk = 0
        for ii in range(nnodes):
            assert (data[ii].ndim==1 or data[ii].shape[1]==1),(
                ii,data[ii].shape)
            for jj in range(data[ii].shape[0]):
                evidence[kk] = data[ii][jj]
                evidence_ids[kk] = self.evidence_ids[ii][jj]
                kk += 1
        #return evidence, self.evidence_ids
        return evidence, evidence_ids

                
    def convert_to_compact_factors(self):
        """
        Compute the factors of the network
        """
        self.factors = []
        for ii in self.graph.nodes:
            if self.node_childs[ii] is not None:
                # Not a leaf node - at least one child (predecessor in networkx)
                # Generate a Gaussian conditional probability density
                if len(self.node_var_ids[ii])==self.node_nvars[ii]:
                    # node contains data
                    nvars_per_var2 = [[1] for kk in range(self.node_nvars[ii])]
                else:
                    # node does not contain data
                    nvars_per_var2 = [self.node_nvars[ii]]
                cpd = get_gaussian_factor_in_canonical_form(
                    self.Amats[ii],None,self.cpd_prior_covs[ii],
                    self.node_childs[ii],
                    [self.node_nvars[jj]for jj in self.node_childs[ii]],
                    #[ii],[self.node_nvars[ii]])
                    self.node_var_ids[ii],nvars_per_var2)
                self.factors.append(cpd)
            else:
                # Leaf nodes - no children (predecessors in networkx)
                # TODO: replace inverse by method that takes advantage of matrix
                # structure, e.g. diagonal, constant diagonal
                precision_matrix = np.linalg.inv(self.cpd_prior_covs[ii])
                shift = np.zeros(self.cpd_prior_covs[ii].shape[0]); mean=shift
                normalization=compute_gaussian_pdf_canonical_form_normalization(
                    mean,shift,precision_matrix)
                self.factors.append(GaussianFactor(
                    #precision_matrix,shift,normalization,[ii],
                    precision_matrix,shift,normalization,self.node_var_ids[ii],
                    [self.node_nvars[ii]]))
                
def sum_product_eliminate_variable(factors, var_id_to_eliminate):
    """
    Marginalize out a variable from a multivariate Gaussian defined by 
    the product of the gaussian variables in factors.

    Algorithm 9.1 in Koller

    Parameters
    ----------
    factors : list (num_factors)
        List of gaussian variables in CanonicalForm

    var_id_to_eliminate : integer
        The variable to eliminate from each of the factors

    Returns
    -------
    fpp_tau : list
        List of gaussian variables in CanonicalForm. The first entries are 
        all the factors that did not contain the variable_to_eliminate on 
        entry to this function. The last entry is the multivariate gaussian 
        which is based upon the product of all factors that did contain the
        elimination variable for which the elimination var is then marginalized 
        out
    """

    # Get list of factors which contain the variable to eliminate
    fp,fpp = [],[]
    for factor in factors:
        if var_id_to_eliminate in factor.var_ids:
            fp.append(factor)
        else:
            fpp.append(factor)
            
    if len(fp)==0:
        return fpp
    
    # Of the factors which contain the variable to eliminate marginalize
    # out that variable from a multivariate Gaussian of all factors
    # containing that variable
    
    # construct multivariate Gaussian distribution in canonical form
    psi = copy.deepcopy(fp[0])
    for jj in range(1, len(fp)):
        psi *= fp[jj]

    #marginalize out all data associated with var_to_eliminate
    tau = copy.deepcopy(psi)
    tau.marginalize([var_id_to_eliminate])
    #tau = tau.reorder(ordering)
        
    # Combine the marginalized factors and the factors which did
    # not originally contain the variable to eliminate
    return fpp+[tau]


def sum_product_variable_elimination(factors,var_ids_to_eliminate):
    """
    Marginalize out a list of variables from the multivariate Gaussian variable
    which is the product of all factors.
    """
    nvars_to_eliminate = len(var_ids_to_eliminate)

    fup = copy.deepcopy(factors)
    for var_id in var_ids_to_eliminate:
        #print("eliminating ", var_id)
        fup = sum_product_eliminate_variable(fup, var_id)
        #print("factors left K= ",
        #      [np.linalg.pinv(f.precision_matrix) for f in fup])

    assert len(fup) > 0, "no factors left after elimination"
    assert len(fup[0].var_ids) != 0, "factor k = {0}".format(
        factor_ret.precision_matrix)

    factor_ret = fup[0]
    #print('f[0].K\n',factor_ret.precision_matrix,factor_ret.var_ids)
    for jj in range(1, len(fup)):
        #print(f'f[{jj}].K\n',fup[jj].precision_matrix,fup[jj].var_ids)
        factor_ret *= fup[jj]
        #print(f'f_ret[{jj}].K\n',factor_ret.precision_matrix)
    return factor_ret

def cond_prob_variable_elimination(network, query_labels, evidence_ids=None,
                                   evidence=None):
    """
    Marginalize out variables not in query labels.
    """
    eliminate_ids = get_var_ids_to_eliminate(
        network.node_ids,network.node_labels,query_labels,evidence_ids)

    factors = copy.deepcopy(network.factors)

    # Condition each node on available data
    if evidence is not None:
        for factor in factors:
            factor.condition(evidence_ids,evidence)

    # Marginalize out all unrequested variables
    factor_ret = sum_product_variable_elimination(factors,eliminate_ids)

    return factor_ret

from functools import partial
def build_peer_polynomial_network(prior_covs,cpd_scales,basis_matrix_funcs,
                                  nparams,model_labels=None):
    """
    All list arguments must contain high-fidelity info in last entry
    """
    # construct graph
    graph = nx.DiGraph()
    nmodels = len(nparams)
    assert len(cpd_scales)==nmodels-1
    assert len(basis_matrix_funcs)==nmodels

    if model_labels is None:
        model_labels = ['M_%d'%ii for ii in range(nmodels)]
        
    for ii in range(nmodels-1):
        graph.add_node(
            ii,label=model_labels[ii],prior_scale=np.sqrt(prior_covs[ii]),
            nparams=nparams[ii],basis_matrix_func=basis_matrix_funcs[ii])

    ii = nmodels-1
    hf_cpd_cov=prior_covs[ii]-np.dot(np.asarray(cpd_scales)**2,prior_covs[:ii])
    hf_cpd_cov = max(1e-8, hf_cpd_cov)
    graph.add_node(ii, label=model_labels[ii], prior_scale=np.sqrt(hf_cpd_cov),
                   nparams=nparams[ii],basis_matrix_func=basis_matrix_funcs[ii])

    for ii in range(nmodels-1):
        graph.add_edge(ii, nmodels-1, cpd_scale=cpd_scales[ii])

    network = BayesianNetwork(graph)
    return network

def nonlinear_constraint_peer(covs,scales):
    """
    All list arguments must contain high-fidelity info in last entry
    """
    cpd_cov = [covs[-1]-np.dot(scales**2, covs[:-1])-1e-7]
    return cpd_cov # must be > 0 to ensure cpd_cov is positive

def nonlinear_constraint_recursive(covs,scales):
    """
    All list arguments must contain model info ordered lowest-highest fidelity
    """
    cpd_cov = [None]*len(scales)
    for dim in range(len(scales)):
        cpd_cov[dim] = covs[dim+1] - scales[dim]**2 * covs[dim] - 1e-8
    return cpd_cov # must be > 0 to ensure cpd_cov is positive

def infer(build_network, scales, samples_train, data_train, noise_std):
    
    network = build_network(scales)
    network.add_data_to_network(samples_train,noise_std**2)
    network.convert_to_compact_factors()
    evidence, evidence_ids = network.assemble_evidence(data_train)

    # high fidelity model is always last label of models. It will not
    # be last lable in graph. These will be data
    hf_label = network.graph.nodes[len(samples_train)-1]['label']
    #print(network.graph.nodes.data('label'))
    #print(hf_label)
    factor_post = cond_prob_variable_elimination(
        network,[hf_label],evidence_ids=evidence_ids,evidence=evidence)    
    gauss_post = convert_gaussian_from_canonical_form(
        factor_post.precision_matrix,factor_post.shift)

    factor_prior = cond_prob_variable_elimination(network, [hf_label], None)
    gauss_prior = convert_gaussian_from_canonical_form(
        factor_prior.precision_matrix,factor_prior.shift)

    return gauss_post, gauss_prior, network

def obj(vand_funcs,input_samples,data,noise_std,labels,held_inputs,
        held_hf_data,build_network,scales):

    g = build_network(scales)

    # build network
    vands = [vfunc(s) for vfunc, s in zip(vand_funcs, input_samples)]
    gn = graph_to_gaussian_network(g, vands=vands, obs_std=noise_std)

    # Node names predict
    node_names_evidence = [n[1] for n in gn.graph.nodes.data('label')]
    evidence = assemble_evidence(node_names_evidence, data)

    mse = estimate_mse(
        gn, evidence, labels[-1], vand_funcs[-1], held_inputs, held_hf_data)
    return mse

def estimate_mse_from_posterior(held_inputs,held_out_data,ignore_variance,
                                network):
    node_names_evidence = [n[1] for n in gn.graph.nodes.data('label')]
    evidence = assemble_evidence(node_names_evidence, data)
    model_name = [network.graph.nodes[-1]['label'][1]]
    factor = cond_prob_variable_elimination(network, model_name, evidence)
    post_mean, post_cov = convert_gaussian_from_canonical_form(
        factor.precision_matrix,factor.shift)
    
    vand = network.graph.nodes[-1]['basis_matrix_func'][1](held_inputs)
    ypred = np.dot(vand, post_mean)

    assert np.allclose(ypred.shape,held_out_data.shape)
    if vand.shape[1]==1:
        l2err = (ypred.mean()-held_out_data.mean())**2
        #print(l2err,'bias')
    else:
        l2err = np.linalg.norm(ypred - held_out_data)**2/ypred.shape[0]
        
    if ignore_variance:
        return l2err

    if vand.shape[1]>1:
        approx_post_variance = np.diag(
            vand.dot(post_cov.dot(vand.T)))
        l2err += approx_post_variance.sum()/ypred.shape[0]
    else:
        assert post_cov.shape==(1,1)
        l2err += post_cov[-1,-1]

    return l2err

def regression(noise_std, data_train, samples_train, build_network,
               fit_metric, init_scales, optimize=True):
    """
    Build a MFNets approximation.


    Parameters
    ----------
    samples_train : list
        List of the samples at which each evaluation source is evaluated.
        Each entry of the list is a np.ndarray (num_vars,num_samples)

    data_train : list
        List of the values of each information source at each sample in 
        ``samples_train``. Each entry of the list is a 
        np.ndarray (num_samples,1)

    noise_std : np.narray (nmodels)
        the standard deviation of the noise of the observations in
        ``data_train``

    build_network : callable
        A function used to build the network with signature 
       
        ``build_network(scales) - > BayesianNetwork

    where scales is a np.ndarray (nscales) matrix of CPD scales which will be 
    treated as hyper-parameters and optimized.

    init_scales : np.ndarray (nscales)
        The initial guess for the hyper-parameters. If optimize=False
        the final approximation will use this value

    fit_metric : callable
        A function that estimates the fit of the approximation to the data
        with signature
    
        ``fit_metric(network) -> float``

        where network is a BayesianNetwork.

    optimize : boolean
        True - optimize the hyper-parameters
        False - use the values in init_scales

    Returns
    -------
    result : dictionary
        Dictionary with the following attributes.

    scales : np.ndarray (nscales)
        The final cpd_scales found by the optimizer.

    network : BayesianNetwork
        The Bayesian network built using ``scales``

    gauss_post : tuple
        The tuple (mean, cov) where mean is a np.ndarray (nparams) and 
        cov is a np.ndarray (nparams,nparams) contain the posterior mean and 
        covariance of the latent variables of the network, respectively.

    gauss_prior : tuple
        The tuple (mean, cov) where mean is a np.ndarray (nparams) and 
        cov is a np.ndarray (nparams,nparams) contain the prior mean and 
        covariance of the latent variables of the network, respectively.
    """
    nmodels = len(samples_train)

    if init_scales is None:
        x0 = np.ones((nmodels-1))/(nmodels-1)
        res = sciopt.fmin_cobyla(obj, x0, cons=cons,
                                 maxfun=40, disp=disp)
        final_scales = res
    else:
        final_scales = init_scales.copy()

    gauss_post, gauss_prior, network = infer(
        build_network, final_scales, samples_train, data_train,
        np.asarray(noise_std))
    
    result = {"scales":final_scales,'network':network,
              "gauss_post":gauss_post, "gauss_prior":gauss_prior}
    return result

def get_regression_data(result):
    gauss_prior = ret['gauss_prior']
    gauss_post  = ret['gauss_post']
    gauss_mean = gauss_post.get_mean()
    vand = vand_funcs[-1](held_hf_samples)
    ypred = np.dot(vand, gauss_mean)
    mean = ypred.mean()
    return vand_funcs[-1],gauss_post,gauss_prior,mean

def get_heterogeneous_data(ndata,noise_std):
    f1 = lambda x: np.cos(3*np.pi*x[0,:]+0.1*x[1,:])[:,np.newaxis]
    f2 = lambda x: np.exp(-(x-.5)**2/0.5).T
    f3 = lambda x: (f2(x).T+np.cos(3*np.pi*x)).T
    XX = [np.random.uniform(0,1,(2,ndata[0]))]+[
        np.random.uniform(0,1,(1,n)) for n in ndata[1:]]

    funcs = [f1,f2,f3]
    data = [f(xx) + e*np.random.normal(0,1,(n,1))
            for f,xx,n,e in zip(funcs,XX,ndata,noise_std)]
    samples = [x for x in XX]

    validation_samples = np.linspace(0,1,10001)
    validation_data= funcs[-1](validation_samples)

    ranges = [[0,1,0,1],[0,1],[0,1]]

    for ii in range(3):
        assert data[ii].shape==(ndata[ii],1),(ii,data[ii].shape,ndata[ii])
    return samples, data, validation_samples, validation_data, ranges

from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
    define_poly_options_from_variable_transformation
from pyapprox.variable_transformations import \
    AffineRandomVariableTransformation
from pyapprox.indexing import compute_hyperbolic_indices
def get_total_degree_polynomials(univariate_variables,degrees):
    assert type(univariate_variables[0])==list
    assert len(univariate_variables)==len(degrees)
    polys, nparams = [], []
    for ii in range(len(degrees)):
        poly = PolynomialChaosExpansion()
        var_trans = AffineRandomVariableTransformation(univariate_variables[ii])
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        indices=compute_hyperbolic_indices(var_trans.num_vars(),degrees[ii],1.0)
        poly.set_indices(indices)
        polys.append(poly)
        nparams.append(indices.shape[1])
    return polys, np.array(nparams)

def plot_1d_lvn_approx(xx,nmodels,hf_vandermonde,gauss_post,gauss_prior,
                       axs,samples,data,labels,ranges,hf_data_mean=None,
                       colors=None):    
    if samples[-1].ndim!=1 and samples[-1].shape[0]>1:
        print('Cannot plot num_vars>1')
        return
    xx =np.linspace(0,1,101)
    xx = (ranges[1]-ranges[0])*xx+ranges[0]
    basis_matrix = hf_vandermonde(xx[np.newaxis,:])
    approx_post_covariance =basis_matrix.dot(gauss_post[1].dot(basis_matrix.T))
    assert (np.diag(approx_post_covariance).min()>0),np.diag(approx_post_covariance).min()
    approx_prior_covariance=basis_matrix.dot(gauss_prior[1].dot(basis_matrix.T))
    # print (approx_covariance.shape,gauss_post.get_covariance().shape)


    approx_post_mean  = np.dot(basis_matrix,gauss_post[0]).squeeze()
    approx_prior_mean = np.dot(basis_matrix,gauss_prior[0]).squeeze()
    if hf_data_mean is not None:
        approx_post_mean += hf_data_mean
        approx_prior_mean+= hf_data_mean
        
    approx_post_std = np.sqrt(np.diag(approx_post_covariance))
    axs.plot(xx,approx_post_mean,'--g',label='MFNet')
    axs.fill_between(xx,approx_post_mean-2*approx_post_std,
                     approx_post_mean+2*approx_post_std,color='green',alpha=0.5)

    approx_prior_std = np.sqrt(np.diag(approx_prior_covariance))
    axs.fill_between(xx,approx_prior_mean-2*approx_prior_std,
                     approx_prior_mean+2*approx_prior_std,color='black',
                     alpha=0.2)
    #print('max prior var', (approx_prior_mean+2*approx_prior_std).max())

    if labels is None:
        labels = [None]*len(samples)
    for ii in range(len(samples)):
        samples[ii] = np.atleast_2d(samples[ii])
        if colors is not None:
            axs.plot(samples[ii][0,:],data[ii],'o',label=labels[ii],
                     c=colors[ii])
        else:
            axs.plot(samples[ii][0,:],data[ii],'o',label=labels[ii])
            #axs.ylim([(approx_post_mean-2*approx_post_std).min(),
            #          (approx_post_mean+2*approx_post_std).max()])
            #axs.set_ylim([(-2*approx_prior_std).min(),(2*approx_prior_std).max()])
    axs.set_xlim([xx.min(),xx.max()])
    axs.legend()

import itertools
def plot_peer_network(nmodels,ax):
    # Best use with axis of (8,3) or (8,6)
    options={'node_size':2000,'width':3,'arrowsize':20,'ax':ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels-1)/64,(nmodels-1)/64,nmodels-1),[0]))
    coords += [[0,1/16]]
    pos = dict(zip(np.arange(nmodels,dtype=int),coords))
    
    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)
        
    for ii in range(nmodels-1):
        graph.add_edge(ii, nmodels-1)
    
    nx.draw(graph,pos,**options)
    labels_str=[r'$\theta_{%d}$'%(ii+1) for ii in range(nmodels)]
    labels=dict(zip(np.arange(nmodels,dtype=int),labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)

def plot_recursive_network(nmodels,ax):
    # Best use with axis of (8,3) or (8,6)
    options={'node_size':2000,'width':3,'arrowsize':20,'ax':ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels)/64,(nmodels)/64,nmodels),[0]))
    pos = dict(zip(np.arange(nmodels,dtype=int),coords))
    
    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)
        
    for ii in range(nmodels-1):
        graph.add_edge(ii, ii+1)
    
    nx.draw(graph,pos,**options)
    labels_str=[r'$\theta_{%d}$'%(ii+1) for ii in range(nmodels)]
    labels=dict(zip(np.arange(nmodels,dtype=int),labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)

    
def set_polynomial_ensemble_coef_from_flattened(polys,coefs):
    idx1,idx2=0,0
    for ii in range(len(polys)):
        idx2 += polys[ii].num_terms()
        polys[ii].set_coefficients(coefs[idx1:idx2])
        idx1=idx2
    return polys

max_num_samples=100
error_tol=1e-10

candidate_samples=-np.cos(
    np.random.uniform(0,np.pi,(var_trans.num_vars(),int(1e4))))
pce = AdaptiveLejaPCE(
    var_trans.num_vars(),candidate_samples,factorization_type='fast')

max_level=np.inf
max_level_1d=[max_level]*(pce.num_vars)

admissibility_function = partial(
    max_level_admissibility_function,max_level,max_level_1d,
    max_num_samples,error_tol)

growth_rule =  partial(constant_increment_growth_rule,2)
#growth_rule = clenshaw_curtis_rule_growth
pce.set_function(model,var_trans)
pce.set_refinement_functions(
    variance_pce_refinement_indicator,admissibility_function,
    growth_rule)
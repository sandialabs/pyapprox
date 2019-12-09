ntrain_samples = int(poly.indices.shape[1]*1.1)
train_samples = -np.cos(np.random.uniform(0,2*np.pi,(poly.num_vars(),ntrain_samples)))
train_samples = var_trans.map_from_canonical_space(train_samples)
train_values  = model(train_samples)
validation_samples = generate_independent_random_samples(
var_trans.variable,int(1e3))
validation_values = model(validation_samples)

errors = []
num_samples = []
def callback(pce):
    error = compute_l2_error(validation_samples,validation_values,pce)
    errors.append(error)
    num_samples.append(pce.samples.shape[1])
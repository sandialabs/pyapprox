univariate_variables = [uniform(),beta(3,3)]
variable = IndependentMultivariateRandomVariable(univariate_variables)

c = np.array([10,0.01])
model = GenzFunction(
    "oscillatory",variable.num_vars(),c=c,w=np.zeros_like(c))
# model.set_coefficients(4,'exponential-decay')
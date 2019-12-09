univariate_variables = [uniform(),beta(3,3)]
variable = IndependentMultivariateRandomVariable(univariate_variables)

c = np.random.uniform(0.,1.,variable.num_vars())
c*=4/c.sum()
w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
model = GenzFunction( "oscillatory",variable.num_vars(),c=c,w=w )
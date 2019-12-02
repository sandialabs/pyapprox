from pyapprox.examples.cantilever_beam import *
objective,constraints,constraint_functions,uq_samples,res,opt_history = \
    find_deterministic_beam_design()
plot_beam_design(
    beam_obj,constraints,constraint_functions,uq_samples,
    res.x,res,opt_history,'deterministic')
plt.show()
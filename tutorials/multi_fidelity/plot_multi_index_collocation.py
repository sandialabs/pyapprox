r"""
Multi-index Stochastic Collocation
==================================
This tutorial describes how to implement and deploy multi-index collocation to construct a surrogate of the output of a high-fidelity model using a set of lower-fidelity models of lower accuracy and cost.

Despite the improved efficiency of surrogate methods relative to MC sampling, building a surrogate can still be prohibitively expensive for high-fidelity simulation models. Fortunately, a selection of models of varying fidelity and computational cost are typically available for many applications. For example, aerospace models span fluid dynamics, structural and thermal response, control systems, etc. 

Leveraging an ensemble of models can facilitate significant reductions in the overall computational cost of UQ, by integrating the predictions of quantities of interest (QoI) from multiple sources.

.. math::

   \frac{\partial u}{\partial t}(x,t,\rv) + \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] = g(x,t)& &
   (x,t,\rv)\in D\times [0,1]\times\rvdom\\
   u(x,t,\rv)=0 & & (x,t,\rv)\in \partial D\times[0,1]\times\rvdom

with forcing :math:`g(x,t)=(1.5+\cos(2\pi t))\cos(x_1)`, and subject to the initial condition :math:`u(x,0,\rv)=0`. Following [NTWSIAMNA2008]_, we model the diffusivity :math:`k` as a random field represented by the
Karhunen-Loeve (like) expansion (KLE)

.. math::

   \log(k(x,\rv)-0.5)=1+\rv_1\left(\frac{\sqrt{\pi L}}{2}\right)^{1/2}+\sum_{k=2}^d \lambda_k\phi(x)\rv_k,

with

.. math::

  \lambda_k=\left(\sqrt{\pi L}\right)^{1/2}\exp\left(-\frac{(\lfloor\frac{k}{2}\rfloor\pi L)^2}{4}\right) k>1,  & &  \phi(x)=
    \begin{cases}
      \sin\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ even}\,,\\
      \cos\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ odd}\,.
    \end{cases}

where :math:`L_p=\max(1,2L_c)`, :math:`L=\frac{L_c}{L_p}` and :math:`L_c=0.5`.

We choose a random field which is effectively one-dimensional so that the error in the finite element solution is more sensitive to refinement of the mesh in the :math:`x_1`-direction than to refinement in the :math:`x_2`-direction.

The advection diffusion equation \eqref{eq:advection-diffusion} is solved using linear finite elements and implicit backward-Euler timestepping implemented using `Fenics <https://fenicsproject.org/>`_. In the following we will show how solving the PDE with varying numbers of finite elements and timesteps can reduce the cost of approximating the quantity of interest

.. math:: f(\rv)=\int_D u(\rv)\frac{1}{2\pi\sigma^2}\exp\left(-\frac{\lVert x-x^\star \rVert_2^2}{\sigma^2}\right)\,dx,

where :math:`x^\star=(0.3,0.5)` and :math:`\sigma=0.16`.
The advection diffusion model, defined here, can be inexpensively evaluated and therefore allows for exhaustive exploration of the behavior of our proposed multi-index collocation approach.

Lets first consider a simple 1D example. The following sets up the problem
"""
import pyapprox as pya
from scipy.stats import uniform
from pyapprox.examples.multi_index_advection_diffusion import *
from pyapprox.models.wrappers import MultiLevelWrapper

nmodels  = 3
num_vars = 1
max_eval_concurrency = 1
base_model = setup_model(num_vars,max_eval_concurrency)
multilevel_model=MultiLevelWrapper(
    base_model,base_model.base_model.num_config_vars,
    base_model.cost_function)
variable = pya.IndependentMultivariateRandomVariable(
    [uniform(-np.sqrt(3),2*np.sqrt(3))],[np.arange(num_vars)])
#%%
#Now lets us plot each model as a function of the random variable
lb,ub = variable.get_statistics('interval',alpha=1)[0]
nsamples = 10
random_samples = np.linspace(lb,ub,nsamples)[np.newaxis,:]
config_vars = np.arange(nmodels)[np.newaxis,:]
samples = pya.get_all_sample_combinations(random_samples,config_vars)
values = multilevel_model(samples)
values = np.reshape(values,(nsamples,nmodels))

import dolfin as dl
plt.figure(figsize=(nmodels*8,2*6))
config_samples = multilevel_model.map_to_multidimensional_index(config_vars)
for ii in range(nmodels):
    nx,ny,dt = base_model.base_model.get_degrees_of_freedom_and_timestep(
        config_samples[:,ii])
    mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1), nx, ny)
    plt.subplot(2,nmodels,ii+1)
    dl.plot(mesh)
    label=r'$f_%d$'%ii
    if ii==0:
        ax = plt.subplot(2,nmodels,nmodels+ii+1)
    else:
        plt.subplot(2,nmodels,nmodels+ii+1,sharey=ax)
    plt.plot(random_samples[0,:],values[:,ii],label=label)
    if ii>0:
        label=r'$f_%d-f_%d$'%(ii,ii-1)
        plt.plot(random_samples[0,:],values[:,ii]-values[:,ii-1],label=label)
    plt.legend()
plt.show()

#%%
# The first row shows the spatial mesh of each model and the second row depicts the model response and the discrepancy between two consecutive models. The difference between the model output decreases as the resolution of the mesh is increased. Thus as the cost of the model increases (with increasing resolution) we need less samples to resolve 

#%%
#References
#^^^^^^^^^^
#.. [TJWGSIAMUQ2015] `Teckentrup, A. and Jantsch, P. and Webster, C. and Gunzburger, M. A Multilevel Stochastic Collocation Method for Partial Differential Equations with Random Input Data. SIAM/ASA Journal on Uncertainty Quantification, 3(1), 1046-1074, 2015. <https://doi.org/10.1137/140969002>`_
#
#.. [HNTTCMAME2016] `Haji-Ali, A. and Nobile, F. and Tamellini, L. and Tempone, R. Multi-Index Stochastic Collocation for random PDEs. Computer Methods in Applied Mechanics and Engineering, 306, 95-122, 2016. <https://doi.org/10.1016/j.cma.2016.03.029>`_

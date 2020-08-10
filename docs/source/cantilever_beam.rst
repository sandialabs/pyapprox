Cantilever Beam
===============

.. figure:: ./figures/cantilever-beam.*
   :align: center

   Conceptual model of the cantilever-beam
   
.. table:: Uncertainties
   :align: center
	    
   =============== ========= =======================
   Uncertainty     Symbol    Prior
   =============== ========= =======================
   Yield stress    :math:`R` :math:`N(40000,2000)`
   Young's modulus :math:`E` :math:`N(2.9e7,1.45e6)`
   Horizontal load :math:`X` :math:`N(500,100)`
   Vertical Load   :math:`Y` :math:`N(1000,100)`
   =============== ========= =======================

First we must specify the distribution of the random variables

..
  .. literalinclude:: ../../pyapprox/examples/cantilever_beam.py
   :start-at: def define_beam_random_variables
   :end-at: return


Let :math:`\V{z}=(R,E,X,Y), \mathbf{w}=(w,t)` denote the random and design variables respectively. We want to minimize weight :math:`wt` subject subject to constraints based upon the following functions defining failure

.. math::

   \begin{align*}
   f_1(\V{z},\mathbf{w})=1 - \frac{6L}{{\color{red}{R}}wt} \left(\frac{X}{w}+\frac{Y}{t}\right) \ge 0& &
  f_2(\V{z},\mathbf{w})=1 - \frac{4L^3}{{\color{red}{2.2535}}E w t } \sqrt{\frac{X^2}{w^4}+\frac{Y^2}{t^4}} \ge 0
  \end{align*}

We specify the objective and constraints with

..
  .. literalinclude:: ../../pyapprox/examples/cantilever_beam.py
   :start-at: def beam_obj
   :end-before: def setup_beam_design

Determinstic optimization
+++++++++++++++++++++++++
First we use deterministic optimiation to design the beam by solving the following optimization problem

.. math::
   :nowrap:
   
   \begin{align*}
   &\argmin_{w,t}\;  wt\\
   &f_1(\V{z})\ge 0\\
   &f_2(\V{z})\ge 0\\
   &1 \le w \le 4 \quad 1\le t \le 4
   \end{align*}

Deterministic optimization requires defining the nominal values of the random parameters. Here we use the means of the random variables.

We can find the deterministic optima using

..
  .. literalinclude:: ../../pyapprox/examples/cantilever_beam.py
   :start-at: def setup_beam_design
   :end-before: def find_uncertainty_aware_beam_design

Lets put this altogether. The aforementioned steps can be combined and run using the following 
	      
..
  .. plot::
   :include-source:

   from pyapprox.examples.cantilever_beam import *
   objective,constraints,constraint_functions,uq_samples,res,opt_history = \
       find_deterministic_beam_design()
   plot_beam_design(
       beam_obj,constraints,constraint_functions,uq_samples,
       res.x,res,opt_history,'deterministic')
   plt.show()

The first three rows of plots are:

#. From left to right, the contraints :math:`f_1,f_2` and objective as functions of the design variables. The black lines represent the constraints and the number dots are iterations of the optimization algorithm. It can be seen that the initial design is infeasible, i.e. it violates the constraints. From this initial design optimization marches towards the constraint boundaries until the constraints are satisfied.
   
#. The PDFs of the contstraint functions at the optima. Shaded areas to left of the horizontal line represents failure.
   
#. The CDFs of the constraints at the optima. Shaded areas to left of the horizontal line represents failure. There is clearly a large probability that the constraints will be violated when uncertainty is taken into account.

		

Design under uncertainty
++++++++++++++++++++++++
It is better to incoporate uncertainty into the design process. In the following we will solve the so called reliability based desgin optimization (RBDO) problem:

.. math::
   :nowrap:
   
   \begin{align*}
   &\argmin_{w,t}\; wt\\
   &P(f_1(\V{z})\le0)\le \delta_1\\
   &P(f_2(\V{z})\le0)\le \delta_2\\
   &1 \le w \le 4 \quad 1\le t \le 4
   \end{align*}

To setup the optimization problem use

..
  .. literalinclude:: ../../pyapprox/examples/cantilever_beam.py
   :start-at: def find_uncertainty_aware_beam_design
   :end-at: return objective

  .. plot::
   :include-source:

   from pyapprox.examples.cantilever_beam import *
   objective,constraints,constraint_functions,uq_samples,res,opt_history = \
      find_uncertainty_aware_beam_design()
   plot_beam_design(
      beam_obj,constraints,constraint_functions,uq_samples,
      res.x,res,opt_history,'DUU')
   plt.show()

The final design is heavier than the deterministic design, but by construction the optimization problem enforces the probability of failure is small at the optima. Weight has been sacrificied for improved reliability/safety.

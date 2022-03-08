Benchmarks
==========
The :mod:`pyapprox.benchmarks.benchmarks` provides a number of benchmarks commonly used to evaluate the performance of quadrature, sensitivity analysis, inference and design algorithms.

Following shows how to use the common interface, provided by :class:`pyapprox.benchmarks.benchmarks.Benchmark`, to access the data necessary
to run a benchmark.

To demonstrate the benchmark class consider the problem of estimating sensitivity indices of the Ishigami function

.. math:: f(z) = \sin(z_1)+a\sin^2(z_2) + bz_3^4\sin(z_0)

The mean, variance, main effect and total effect sensitivity indices are well known for this problem.

The following sets up a :class:`pyapprox.benchmarks.benchmarks.Benchmark` object which returns the Ishigami function its Jacobian, Hessian the joint density of the input variables :math:`z` and the various sensitivity indices. The attributes of the benchmark can be accessed using the member `keys()`

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark("ishigami",a=7,b=0.1)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'jac', 'hess', 'variable', 'mean', 'variance', 'main_effects', 'total_effects', 'sobol_indices'])
    >>> print(hess(np.zeros(3)))
    [[-0.  0.  0.]
    [ 0. 14.  0.]
    [ 0.  0.  0.]]

The various attributes of the benchmark can be accessed easily. For example
above we evaluated the Hessian at the point :math:`z=(0,0,0)^T`.

The following tabulates the benchmarks provided in :mod:`pyapprox.benchmarks.benchmarks`. Each benchmark can be instantiated with using `setup_benchmark(name,**kwargs)` Follow the links to find details on the options available for each benchmark which are specified via `kwargs`.

Sensitivity Analysis
--------------------

:func:`pyapprox.benchmarks.benchmarks.setup_ishigami_function`
:func:`pyapprox.benchmarks.benchmarks.setup_sobol_g_function`
:func:`pyapprox.benchmarks.benchmarks.setup_oakley_function`

Quadrature
----------
:func:`pyapprox.benchmarks.benchmarks.setup_genz_function`

Inference
---------
:func:`pyapprox.benchmarks.benchmarks.setup_rosenbrock_function`

Multi-fidelity Modeling
-----------------------
:func:`pyapprox_dev.fenics_models.advection_diffusion_wrappers.setup_advection_diffusion_benchmark`
:func:`pyapprox_dev.fenics_models.advection_diffusion_wrappers.setup_advection_diffusion_source_inversion_benchmark`
:func:`pyapprox_dev.fenics_models.helmholtz_benchmarks.setup_mfnets_helmholtz_benchmark`




Benchmarks
==========
The :mod:`pyapprox.benchmarks` provides a number of benchmarks commonly used to evaluate the performance of quadrature, sensitivity analysis, inference and design algorithms.

Sensitivity Analysis
--------------------
:func:`~pyapprox.benchmarks.IshigamiBenchmark`
:func:`~pyapprox.benchmarks.SobolGBenchmark`
:func:`~pyapprox.benchmarks.OakleyBenchmark`

Quadrature
----------
:func:`~pyapprox.benchmarks.GenzBenchmark`

Optimization
------------
:func:`~pyapprox.benchmarks.RosenbrockUnconstrainedOptimizationBenchmark`
:func:`~pyapprox.benchmarks.RosenbrockConstrainedOptimizationBenchmark`
:func:`~pyapprox.benchmarks.CantileverBeamDeterminsticOptimizationBenchmark`
:func:`~pyapprox.benchmarks.CantileverBeamUncertainOptimizationBenchmark`
:func:`~pyapprox.benchmarks.EvtushenkoConstrainedOptimizationBenchmark`

Inference
---------
:func:`~pyapprox.benchmarks.PyApproxPaperAdvectionDiffusionKLEInversionBenchmark`

Multi-fidelity Estimation
-------------------------
:func:`~pyapprox.benchmarks.PolynomialModelEnsembleBenchmark`
:func:`~pyapprox.benchmarks.TunableModelEnsembleBenchmark`
:func:`~pyapprox.benchmarks.ShortColumnModelEnsembleBenchmark`
:func:`~pyapprox.benchmarks.MultiOutputModelEnsembleBenchmark`
:func:`~pyapprox.benchmarks.PSDMultiOutputModelEnsembleBenchmark`
:func:`~pyapprox.benchmarks.MultiLevelCosineBenchmark`


Optimal Experimental Design
---------------------------
:func:`~pyapprox.benchmarks.LotkaVolterraOEDBenchmark`

Operator Learning
-----------------
:func:`~pyapprox.benchmarks.TransientViscousBurgers1DOperatorBenchmark`
:func:`~pyapprox.benchmarks.SteadyDarcy2DOperatorBenchmark`

Miscalleneous
-------------
:func:`~pyapprox.benchmarks.LotkaVolterraBenchmark`
:func:`~pyapprox.benchmarks.ChemicalReactionBenchmark`
:func:`~pyapprox.benchmarks.CoupledSpringsBenchmark`
:func:`~pyapprox.benchmarks.HastingsEcologyBenchmark`
:func:`~pyapprox.benchmarks.PistonBenchmark`
:func:`~pyapprox.benchmarks.WingWeightBenchmark`
:func:`~pyapprox.benchmarks.NonlinearSystemOfEquationsBenchmark`




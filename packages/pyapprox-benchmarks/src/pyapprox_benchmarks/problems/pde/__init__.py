"""PDE problem classes for benchmarks."""

from pyapprox.util.optional_deps import package_available

__all__: list[str]

if package_available("skfem"):
    from pyapprox_benchmarks.problems.pde.burgers import (
        build_periodic_burgers_opinf_problem,
    )
    from pyapprox_benchmarks.problems.pde.chafee_infante import (
        build_chafee_infante_opinf_problem,
    )
    from pyapprox_benchmarks.problems.pde.opinf_problem import (
        PDEOpInfProblem,
    )

    __all__ = [
        "PDEOpInfProblem",
        "build_chafee_infante_opinf_problem",
        "build_periodic_burgers_opinf_problem",
    ]
else:
    __all__ = []

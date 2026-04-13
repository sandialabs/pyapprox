"""Genz family benchmarks — analytical integrals."""

from typing import Generic, List

from pyapprox.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.genz import (
    CornerPeakFunction,
    GaussianPeakFunction,
    OscillatoryFunction,
    ProductPeakFunction,
)
from pyapprox_benchmarks.problems.function_over_domain import (
    FunctionOverDomainProblem,
)


def _get_genz_coefficients(nvars: int, decay: str = "none") -> List[float]:
    """Compute standard Genz coefficients with specified decay."""
    if decay == "none":
        c = [(i + 0.5) / nvars for i in range(nvars)]
    elif decay == "quadratic":
        c = [1.0 / (i + 1) ** 2 for i in range(nvars)]
    elif decay == "quartic":
        c = [1.0 / (i + 1) ** 4 for i in range(nvars)]
    else:
        raise ValueError(f"Unknown decay type: {decay}")
    total = sum(c)
    return [ci / total for ci in c]


class GenzOscillatoryBenchmark(Generic[Array]):
    """Genz oscillatory benchmark — analytical integral.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables (default 2).
    decay : str
        Coefficient decay type (default "none").
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
        decay: str = "none",
    ) -> None:
        c = _get_genz_coefficients(nvars, decay)
        w = [0.25] * nvars
        func = OscillatoryFunction(bkd, c=c, w=w)
        domain = BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            f"genz_oscillatory_{nvars}d", func, domain,
            description=f"{nvars}D oscillatory Genz function for quadrature",
        )
        self._integral = func.integrate()

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def integral(self) -> float:
        return self._integral


class GenzProductPeakBenchmark(Generic[Array]):
    """Genz product peak benchmark — analytical integral.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables (default 2).
    c : list[float] or None
        Width coefficients. Default: [2.0, 3.0, ...].
    w : list[float] or None
        Peak location. Default: center [0.5, ...].
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
        c: List[float] | None = None,
        w: List[float] | None = None,
    ) -> None:
        if c is None:
            c = [float(i + 2) for i in range(nvars)]
        if w is None:
            w = [0.5] * nvars
        func = ProductPeakFunction(bkd, c=c, w=w)
        domain = BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            f"genz_product_peak_{nvars}d", func, domain,
            description=f"{nvars}D product peak Genz function for quadrature",
        )
        self._integral = func.integrate()

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def integral(self) -> float:
        return self._integral


class GenzCornerPeakBenchmark(Generic[Array]):
    """Genz corner peak benchmark — analytical integral.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables (default 2).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
    ) -> None:
        c = _get_genz_coefficients(nvars)
        func = CornerPeakFunction(bkd, c=c)
        domain = BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            f"genz_corner_peak_{nvars}d", func, domain,
            description=f"{nvars}D corner peak Genz function for quadrature",
        )
        self._integral = func.integrate()

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def integral(self) -> float:
        return self._integral


class GenzGaussianPeakBenchmark(Generic[Array]):
    """Genz Gaussian peak benchmark — analytical integral.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables (default 2).
    c : list[float] or None
        Width coefficients. Default: [2.0, 3.0, ...].
    w : list[float] or None
        Peak location. Default: center [0.5, ...].
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
        c: List[float] | None = None,
        w: List[float] | None = None,
    ) -> None:
        if c is None:
            c = [float(i + 2) for i in range(nvars)]
        if w is None:
            w = [0.5] * nvars
        func = GaussianPeakFunction(bkd, c=c, w=w)
        domain = BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            f"genz_gaussian_peak_{nvars}d", func, domain,
            description=f"{nvars}D Gaussian peak Genz function for quadrature",
        )
        self._integral = func.integrate()

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def integral(self) -> float:
        return self._integral

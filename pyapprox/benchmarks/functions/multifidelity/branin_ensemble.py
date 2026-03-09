"""Multi-fidelity Branin function ensemble.

Implements a hierarchy of Branin function variants with decreasing
fidelity, useful for testing multi-fidelity surrogate methods (MFNets,
co-kriging, etc.).

The high-fidelity model is the standard Branin function. Lower-fidelity
models use perturbed parameters and an additive shift, producing
correlated but cheaper approximations.

References
----------
Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., &
Karniadakis, G. E. (2017). "Nonlinear information fusion algorithms
for data-efficient multi-fidelity modelling."
Proceedings of the Royal Society A, 473(2198), 20160751.
"""
#TODO: need to add html link to reference https://doi.org/10.1098/rspa.2016.0751


import math
from typing import Generic, List, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class BraninModelFunction(Generic[Array]):
    """Parametric Branin function for multi-fidelity hierarchies.

    f(x1, x2) = a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s + shift

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    a : float
        Quadratic coefficient. Standard: 1.0.
    b : float
        x1^2 coefficient inside the quadratic. Standard: 5.1/(4*pi^2).
    c : float
        x1 coefficient inside the quadratic. Standard: 5/pi.
    r : float
        Constant inside the quadratic. Standard: 6.
    s : float
        Cosine amplitude and offset. Standard: 10.
    t : float
        Cosine damping factor. Standard: 1/(8*pi).
    shift : float
        Additive constant shift. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: float = 1.0,
        b: float = 5.1 / (4 * math.pi**2),
        c: float = 5.0 / math.pi,
        r: float = 6.0,
        s: float = 10.0,
        t: float = 1.0 / (8 * math.pi),
        shift: float = 0.0,
    ) -> None:
        self._bkd = bkd
        self._a = a
        self._b = b
        self._c = c
        self._r = r
        self._s = s
        self._t = t
        self._shift = shift

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the Branin variant.

        Parameters
        ----------
        samples : Array
            Input samples of shape ``(2, nsamples)``.
            x1 in [-5, 10], x2 in [0, 15].

        Returns
        -------
        Array
            Values of shape ``(1, nsamples)``.
        """
        # TODO: Check this recovers each of the three formula
        # in the cited paper
        x1 = samples[0:1, :]
        x2 = samples[1:2, :]
        bkd = self._bkd

        term = x2 - self._b * x1**2 + self._c * x1 - self._r
        result = (
            self._a * term**2
            + self._s * (1 - self._t) * bkd.cos(x1)
            + self._s
            + self._shift
        )
        return result


class BraninEnsemble(Generic[Array]):
    """Three-model multi-fidelity Branin ensemble.

    Model 0 (high fidelity): Standard Branin function.
    Model 1 (medium fidelity): Perturbed parameters (modified b, c, shift).
    Model 2 (low fidelity): Larger perturbations and additive shift.

    The perturbations are chosen so that all three models share the same
    qualitative landscape (three basins) but differ in the precise
    location and depth of the minima. This produces strong but imperfect
    correlation between fidelity levels.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nmodels = 3
        self._nqoi = 1

        b_std = 5.1 / (4 * math.pi**2)
        c_std = 5.0 / math.pi
        t_std = 1.0 / (8 * math.pi)

        # High fidelity: standard Branin
        self._models: List[BraninModelFunction[Array]] = [
            BraninModelFunction(bkd),
            # Medium fidelity: slightly perturbed b, c, and a small shift
            BraninModelFunction(
                bkd,
                b=b_std * 1.2,
                c=c_std * 0.9,
                t=t_std * 1.1,
                shift=2.0,
            ),
            # Low fidelity: larger perturbations and bigger shift
            BraninModelFunction(
                bkd,
                b=b_std * 0.8,
                c=c_std * 1.2,
                t=t_std * 0.8,
                shift=5.0,
            ),
        ]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nmodels(self) -> int:
        return self._nmodels

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return self._nqoi

    def models(self) -> Sequence[BraninModelFunction[Array]]:
        return self._models

    def costs(self) -> Array:
        """Return per-model costs.

        Costs decrease by a factor of 10 per level.

        Returns
        -------
        Array
            Costs of shape ``(3,)``.
        """
        return self._bkd.array([1.0, 0.1, 0.01])

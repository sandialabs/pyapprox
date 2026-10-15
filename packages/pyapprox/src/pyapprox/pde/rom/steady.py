r"""Steady solves of a reduced residual.

Solves :math:`\hat{F}(z, t, p) = 0` for the codes with the library's
Newton solver, for any ``ReducedResidualProtocol``: exact, affine or
hyper-reduced. Every linear solve is :math:`r \times r` and dense. The
full state is recovered as :math:`u = g(t) + \phi(z)`.
"""

from typing import Generic

from pyapprox.pde.rom.lift import LiftProtocol
from pyapprox.pde.rom.projection import ReducedResidualProtocol
from pyapprox.pde.rom.trial_space import TrialSpaceProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class ReducedNewtonResidual(Generic[Array]):
    """A reduced residual at fixed time and parameters, for Newton.

    Satisfies ``NewtonSolverResidualProtocol``. ``linsolve`` returns
    ``J^{-1} r``, which the solver subtracts. ``is_one_step_solvable``
    is False: a single exact step is right only for a residual known to
    be affine in the codes, and a second iteration costs an
    ``r x r`` solve.
    """

    def __init__(
        self,
        reduced: ReducedResidualProtocol[Array],
        time: float,
        params: Array,
    ) -> None:
        self._reduced = reduced
        self._time = time
        self._params = params

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._reduced.bkd()

    def __call__(self, iterate: Array, /) -> Array:
        return self._reduced.residual(iterate, self._time, self._params)

    def linsolve(self, state: Array, prev_residual: Array) -> Array:
        jacobian = self._reduced.state_jacobian(state, self._time, self._params)
        return self.bkd().solve(jacobian, prev_residual)

    def is_one_step_solvable(self) -> bool:
        return False


def solve_reduced_steady(
    reduced: ReducedResidualProtocol[Array],
    params: Array,
    init_codes: Array,
    time: float = 0.0,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    maxiters: int = 20,
) -> Array:
    """Solve the reduced residual for the codes.

    Parameters
    ----------
    reduced : ReducedResidualProtocol
        The reduced residual.
    params : Array
        Shape: (nparams,).
    init_codes : Array
        Initial iterate. Shape: (ncodes,).
    time : float
        The time the data is frozen at; steady problems use the default.

    Returns
    -------
    Array
        The codes. Shape: (ncodes,).

    Raises
    ------
    TypeError
        If ``reduced`` does not satisfy its protocol.
    RuntimeError
        If Newton does not converge.
    """
    if not isinstance(reduced, ReducedResidualProtocol):
        raise TypeError(
            "reduced must satisfy ReducedResidualProtocol, got "
            f"{type(reduced).__name__}"
        )
    solver = NewtonSolver(ReducedNewtonResidual(reduced, time, params))
    solver.set_options(maxiters=maxiters, atol=atol, rtol=rtol)
    return solver.solve(init_codes)


def full_state(
    trial_space: TrialSpaceProtocol[Array],
    lift: LiftProtocol[Array],
    codes: Array,
    time: float = 0.0,
) -> Array:
    r"""Return :math:`g(t) + \phi(z)`. Shape: (ncodes,) -> (nstates,)."""
    return lift.lift(time) + trial_space.decode(codes[:, None])[:, 0]

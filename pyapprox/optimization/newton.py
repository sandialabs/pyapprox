from abc import ABC, abstractmethod

from pyapprox.util.backends.template import BackendMixin, Array


class ResidualEquation(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    def bkd(self) -> BackendMixin:
        return self._bkd

    def _check_iterate(self, iterate: Array) -> None:
        if iterate.shape != (self.nstates(),):
            raise ValueError(
                f"init_iterate has shape {iterate.shape} but must "
                f"have shape {(self.nstates(), )}"
            )

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _value(self, iterate) -> Array:
        raise NotImplementedError

    def __call__(self, iterate: Array) -> Array:
        self._check_iterate(iterate)
        value = self._value(iterate)
        # value is not an iterate but must have the same size
        # maybe relax this assumption
        self._check_iterate(value)
        return value

    @abstractmethod
    def _solve(self, init_iterate: Array) -> Array:
        raise NotImplementedError

    def solve(self, init_iterate: Array) -> Array:
        self._check_iterate(init_iterate)
        iterate = self._solve(init_iterate)
        self._check_iterate(iterate)
        return iterate

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class ResidualEquationWithStateJacobian(ResidualEquation):

    def _state_jacobian(self, iterate: Array) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(self.__call__, iterate)

    def state_jacobian(self, iterate: Array) -> Array:
        self._check_iterate(iterate)
        jac = self._state_jacobian(iterate)
        if jac.shape != (self.nstates(), self.nstates()):
            raise RuntimeError(
                f"{jac.shape=} but must be {(self.nstates(), self.nstates())}"
            )
        return jac


class NewtonSolver:
    def __init__(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        linesearch_maxiters: float = 5,
    ):
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size
        self._atol = atol
        self._rtol = rtol
        self._linesearch_maxiters = linesearch_maxiters

    def set_residual(self, residual: "NewtonResidual") -> None:
        if not isinstance(residual, NewtonResidual):
            raise ValueError("residual must be an instance of NewtonResidual")
        self._residual = residual
        self._bkd = residual._bkd

    def _update_sol(self, prev_sol: Array, delta: Array) -> Array:
        return prev_sol - self._step_size * delta

    def solve(self, init_iterate: Array) -> Array:
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D array")
        if not hasattr(self, "_residual"):
            raise ValueError("must call set_residual")
        sol = init_iterate
        residual = self._residual(sol)
        residual_norms = [self._bkd.norm(residual)]
        it = 0
        if self._verbosity > 1:
            print("Iter", it, "rnorm", residual_norms[0])
        while True:
            prev_sol = sol
            prev_residual = residual
            sol = self._update_sol(
                prev_sol, self._residual.linsolve(sol, prev_residual)
            )
            residual = self._residual(sol)
            residual_norm = self._bkd.norm(residual)
            residual_norms.append(residual_norm)

            it += 1
            if self._verbosity > 1:
                print("Iter", it, "rnorm", residual_norm)
            if not self._bkd.isfinite(residual_norm):
                raise RuntimeError("Residual is no longer finite")
            if residual_norm <= self._atol + self._rtol * residual_norms[0]:
                exit_msg = (
                    f"Tolerance {self._atol+self._rtol*residual_norms[0]} "
                    "reached"
                )
                break
            if it >= self._maxiters:
                exit_msg = (
                    f"Max iterations {self._maxiters} reached.\n"
                    f"Rel residual norm is {residual_norm} "
                    f"Needs to be {self._atol+self._rtol*residual_norms[0]}"
                )
                raise RuntimeError(exit_msg)
        if self._verbosity > 0:
            print(exit_msg)
        return sol

    def __repr__(self) -> str:
        return (
            "{0}(maxiters={1}, verbosity={2}, step_size={3}, atol={4}, "
            "rol={5})"
        ).format(
            self.__class__.__name__,
            self._maxiters,
            self._verbosity,
            self._step_size,
            self._atol,
            self._rtol,
        )

    def residual(self) -> "NewtonResidual":
        return self._residual


class NewtonResidual(ResidualEquationWithStateJacobian):

    def linsolve(self, iterate: Array, res: Array) -> Array:
        jac = self.state_jacobian(iterate)
        return self._bkd.solve(jac, res)

    def default_solver(self) -> NewtonSolver:
        return NewtonSolver()

    def _solve(self, iterate: Array) -> Array:
        if not hasattr(self, "_solver"):
            self.set_solver(self.default_solver())
        return self._solver.solve(iterate)

    def set_solver(self, solver: NewtonSolver) -> None:
        self._solver = solver
        self._solver.set_residual(self)


class BisectionSearch:
    def __init__(
        self,
        maxiters: int = 30,
        verbosity: int = 2,
        atol: float = 1e-7,
    ):
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._atol = atol

    def set_residual(self, residual: ResidualEquation):
        if not isinstance(residual, ResidualEquation):
            raise ValueError("residual must be an instance of NewtonResidual")
        self._residual = residual
        self._bkd = residual._bkd

    def _bisection_search(self, lb: Array, ub: Array) -> Array:
        it = 0
        residual_lb = self._residual(lb)
        residual_ub = self._residual(ub)
        if self._bkd.any(
            self._bkd.sign(residual_lb) == self._bkd.sign(residual_ub)
        ):
            raise RuntimeError
        while it < self._maxiters:
            center = (lb + ub) / 2
            residual_center = self._residual(center)
            residual_norm = self._bkd.norm(residual_center)
            if self._verbosity > 0:
                print("iter {0}: {1}".format(it, residual_norm))
            if residual_norm < self._atol:
                return center
            idx1 = self._bkd.where(
                self._bkd.sign(residual_center) == self._bkd.sign(residual_lb)
            )[0]
            idx2 = self._bkd.where(
                self._bkd.sign(residual_center) != self._bkd.sign(residual_lb)
            )[0]
            lb[idx1] = center[idx1]
            ub[idx2] = center[idx2]
            it += 1
        return center

    def solve(self, bounds: Array) -> Array:
        if bounds.shape[1] != 2:
            raise ValueError("Must provide upper and lower bound")
        return self._bisection_search(*bounds.T)


class BisectionResidual(ResidualEquation):
    def default_solver(self) -> BisectionSearch:
        return BisectionSearch()

    def _solve(self, iterate: Array) -> Array:
        if not hasattr(self, "_solver"):
            self.set_solver(self.default_solver())
        return self._solver.solve(iterate)

    def set_solver(self, solver: BisectionSearch) -> None:
        self._solver = solver
        self._solver.set_residual(self)

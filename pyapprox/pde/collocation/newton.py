from abc import ABC, abstractmethod


class NewtonResidual(ABC):
    def __init__(self, backend):
        self._bkd = backend

    @abstractmethod
    def __call__(self, iterate):
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, iterate):
        raise NotImplementedError

    @abstractmethod
    def linsolve(self, iterate, res):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class NewtonSolver:
    def __init__(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1,
        atol: float = 1e-7,
        rtol: float = 1e-7,
    ):
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size
        self._atol = atol
        self._rtol = rtol

    def set_residual(self, residual: NewtonResidual):
        if not isinstance(residual, NewtonResidual):
            raise ValueError("residual must be an instance of NewtonResidual")
        self._residual = residual
        self._bkd = residual._bkd

    def solve(self, init_guess):
        if init_guess.ndim != 1:
            raise ValueError("init_guess must be 1D array")
        sol = self._bkd.copy(init_guess)
        residual = self._residual(sol)
        residual_norms = []
        it = 0
        while True:
            sol = sol - self._step_size * self._residual.linsolve(
                sol, residual
            )
            residual = self._residual(sol)
            residual_norm = self._bkd.norm(residual)
            residual_norms.append(residual_norm)
            it += 1
            if self._verbosity > 1:
                print("Iter", it, "rnorm", residual_norm)
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

    def __repr__(self):
        return (
            "{0}(maxiters={1}, verbosity={2}, step_size={3}, atol={4},"
            "rol={5})"
        ).format(
            self.__class__.__name__,
            self._maxiters,
            self._verbosity,
            self._step_size,
            self._atol,
            self._rtol,
        )

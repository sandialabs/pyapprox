from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class NewtonResidual(ABC):
    def __init__(self, backend):
        self._bkd = backend

    def adjoint_implemented():
        return False

    @abstractmethod
    def __call__(self, iterate: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, iterate: Array) -> Array:
        raise NotImplementedError

    def linsolve(self, iterate: Array, res: Array) -> Array:
        jac = self.jacobian(iterate)
        return self._bkd.solve(jac, res)

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def _param_jacobian(self, sol: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        raise NotImplementedError

    def param_jacobian(self, sol: Array) -> Array:
        if sol.ndim != 1:
            raise ValueError("sol must be a 1d Array")
        jac = self._param_jacobian(sol)
        if jac.ndim != 2 or jac.shape[0] != sol.shape[0]:
            raise RuntimeError(f"jac has the wrong shape {jac.shape}")
        return jac


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
        if not hasattr(self, "_residual"):
            raise ValueError("must call set_residual")
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


class Functional(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _value(self, sol: Array) -> Array:
        raise NotImplementedError

    def __call__(self, sol: Array) -> Array:
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        val = self._value(sol)
        if val.ndim != 1:
            raise RuntimeError(f"{self} must return a 1D array")
        return val

    def __repr__(self):
        return "{0}(nstates={1}, nparams={2})".format(
            self.__class__.__name__, self.nstates(), self.nparams()
        )


class AdjointFunctional(Functional):
    @abstractmethod
    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def qoi_sol_jacobian(self, sol: Array) -> Array:
        """Gradient of qoi with respect to solution"""
        if sol.ndim != 1 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 1d Array")
        jac = self._qoi_sol_jacobian(sol)
        if jac.shape != (sol.shape[0],):
            raise RuntimeError("jac has the wrong shape")
        return jac

    @abstractmethod
    def _qoi_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def qoi_param_jacobian(self, sol: Array) -> Array:
        """Gradient of QoI with respect to parameters"""
        if sol.ndim != 1 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 1d Array")
        jac = self._qoi_param_jacobian(sol)
        if jac.ndim != 1 or jac.shape[0] != self.nparams():
            raise RuntimeError("jac has the wrong shape")
        return jac

    def set_param(self, param: Array):
        if param.ndim != 1:
            raise ValueError("param must be a 1D Array")
        self._param = param


class AdjointSolver:
    def __init__(
            self, newton_solver: NewtonSolver, functional: AdjointFunctional
    ):
        self._bkd = newton_solver._bkd
        self._residual = newton_solver._residual
        self.newton_solver = newton_solver
        self.functional = functional

    def solve_adjoint(self, sol: Array) -> Array:
        drdu = self._residual.jacobian(sol)
        dqdu = self.functional.qoi_sol_jacobian(sol)
        return self._bkd.solve(drdu.T, -dqdu)

    def gradient(self, sol: Array) -> Array:
        adj_sol = self.solve_adjoint(sol)
        drdp = self._residual.param_jacobian(sol)
        return self.functional.qoi_param_jacobian(sol) + adj_sol @ drdp

    def __repr__(self):
        return "{0}(functional={1})".format(
            self.__class__.__name__, self.functional
        )

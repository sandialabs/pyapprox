# from abc import ABC, abstractmethod

# from pyapprox.util.backends.template import BackendMixin, Array


# class ResidualEquation(ABC):
#     def __init__(self, backend: BackendMixin):
#         self._bkd = backend

#     def bkd(self) -> BackendMixin:
#         return self._bkd

#     def _check_iterate(self, iterate: Array) -> None:
#         if iterate.shape != (self.nstates(),):
#             raise ValueError(
#                 f"init_iterate has shape {iterate.shape} but must "
#                 f"have shape {(self.nstates(), )}"
#             )

#     @abstractmethod
#     def nstates(self) -> int:
#         raise NotImplementedError

#     @abstractmethod
#     def _value(self, iterate: Array) -> Array:
#         raise NotImplementedError

#     def value(self, iterate: Array) -> Array:
#         self._check_iterate(iterate)
#         value = self._value(iterate)
#         # value is not an iterate but must have the same size
#         # maybe relax this assumption
#         self._check_iterate(value)
#         return value

#     def __call__(self, iterate: Array) -> Array:
#         return self.value(iterate)

#     @abstractmethod
#     def _solve(self, init_iterate: Array) -> Array:
#         raise NotImplementedError

#     def solve(self, init_iterate: Array) -> Array:
#         self._check_iterate(init_iterate)
#         iterate = self._solve(init_iterate)
#         self._check_iterate(iterate)
#         return iterate

#     def __repr__(self) -> str:
#         return "{0}".format(self.__class__.__name__)


# class ResidualEquationWithJacobian(ResidualEquation):
#     def use_auto_differentiation(self) -> bool:
#         return False

#     def _check_automatic_differentiation(self) -> None:
#         if not self._bkd.jacobian_implemented():
#             raise NotImplementedError("Automatic differentiation not enabled")
#         if not self.use_auto_differentiation():
#             raise RuntimeError(
#                 f"{self}.use_auto_differentiation() returns False.\n"
#                 "Set it to return True if all functions this class"
#                 "requires use a backend that supports auto diffentiation.\n"
#                 "Otherwise, implement the jacobian with the anlaytical"
#                 "expression."
#             )

#     def _jacobian(self, iterate: Array) -> Array:
#         self._check_automatic_differentiation()
#         return self._bkd.jacobian(self.__call__, iterate)

#     def jacobian(self, iterate: Array) -> Array:
#         self._check_iterate(iterate)
#         jac = self._jacobian(iterate)
#         if jac.shape != (self.nstates(), self.nstates()):
#             raise RuntimeError(
#                 f"{jac.shape=} but must be {(self.nstates(), self.nstates())}"
#             )
#         return jac


# class NewtonResidual(ResidualEquationWithJacobian):
#     def linsolve(self, iterate: Array, res: Array) -> Array:
#         jac = self.jacobian(iterate)
#         return self._bkd.solve(jac, res)

#     def default_solver(self) -> "NewtonSolver":
#         return NewtonSolver()

#     def set_solver(self, solver: "NewtonSolver") -> None:
#         self._solver = solver
#         self._solver.set_residual(self)

#     def _solve(self, iterate: Array) -> Array:
#         if not hasattr(self, "_solver"):
#             self.set_solver(self.default_solver())
#         return self._solver.solve(iterate)


from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class NewtonSolverResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def __call__(self, iterate: Array) -> Array: ...

    def linsolve(self, sol: Array, prev_residual: Array) -> Array: ...


class NewtonSolver(Generic[Array]):
    def __init__(self, residual: NewtonSolverResidualProtocol[Array]) -> None:
        if not isinstance(residual, NewtonSolverResidualProtocol):
            raise ValueError(
                "residual must satisfy NewtonSolverResidualProtocol"
            )
        self._residual = residual
        self._bkd = residual.bkd()
        self.set_options()

    def set_options(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        linesearch_maxiters: float = 5,
    ) -> None:
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size
        self._atol = atol
        self._rtol = rtol
        self._linesearch_maxiters = linesearch_maxiters

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _update_iterate(self, prev_iterate: Array, delta: Array) -> Array:
        return prev_iterate - self._step_size * delta

    def solve(self, init_iterate: Array) -> Array:
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D array")
        if not hasattr(self, "_residual"):
            raise ValueError("must call set_residual")
        iterate = init_iterate
        residual = self._residual(iterate)
        residual_norms = [self._bkd.norm(residual)]
        it = 0
        if self._verbosity > 1:
            print("Iter", it, "rnorm", residual_norms[0])
        while True:
            prev_iterate = iterate
            prev_residual = residual
            iterate = self._update_iterate(
                prev_iterate, self._residual.linsolve(iterate, prev_residual)
            )
            residual = self._residual(iterate)
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
        return iterate

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

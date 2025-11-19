from abc import ABC, abstractmethod
from typing import Tuple

from pyapprox.util.backends.template import BackendMixin, Array


class NewtonResidual(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    def adjoint_implemented():
        return False

    @abstractmethod
    def __call__(self, iterate: Array) -> Array:
        raise NotImplementedError

    def _jacobian(self, iterate: Array) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(self.__call__, iterate)

    def jacobian(self, iterate: Array) -> Array:
        jac = self._jacobian(iterate)
        if jac.ndim != 2 or jac.shape[0] != iterate.shape[0]:
            raise RuntimeError(
                "jac must be 2D with 1 column but has the wrong shape {0} "
                "but iterate shape was {1}".format(jac.shape, iterate.shape)
            )
        return jac

    def linsolve(self, iterate: Array, res: Array) -> Array:
        jac = self.jacobian(iterate)
        return self._bkd.solve(jac, res)

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class ParameterizedNewtonResidual(NewtonResidual):
    @abstractmethod
    def _set_parameters(self, param: Array) -> None:
        raise NotImplementedError

    def set_parameters(self, param: Array) -> None:
        self._param = param
        self._set_parameters(param)

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    def get_parameters(self) -> Array:
        if not hasattr(self, "_parameters"):
            raise AttributeError("must call set_parameters")
        return self._parameters


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

    def set_residual(self, residual: NewtonResidual) -> None:
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

    def residual(self) -> NewtonResidual:
        return self._residual


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

    def set_residual(self, residual: NewtonResidual):
        if not isinstance(residual, NewtonResidual):
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


class BoundedNewtonResidual(NewtonResidual):
    def __init__(self, residual: NewtonResidual, bounds: Tuple[float, float]):
        # for now assume same bounds on all elements of the residual
        super().__init__(residual._bkd)
        self._bounds = self._bkd.asarray(bounds)
        self._residual = residual

    def _to_canonical(self, iterate: Array) -> Array:
        # x in [a, b]
        # z = (x-a)/(b-a)
        # y = -log(z-1)+log(z)
        a, b = self._bounds
        if self._bkd.any(iterate > b) or self._bkd.any(iterate < a):
            raise RuntimeError("iterates exceed bounds")
        z = (iterate - a) / (b - a)
        return -self._bkd.log(1 - z) + self._bkd.log(z)

    def _from_canonical(self, can_iterate: Array) -> Array:
        # canonical y in [-infty, infty]
        # z = 1/(1+exp(-y)) in [0, 1]
        # x = z * (b-a) + a in [a, b]
        a, b = self._bounds
        z = 1.0 / (1.0 + self._bkd.exp(-can_iterate))
        return z * (b - a) + a

    def _tranform_jacobian(self, can_iterate: Array) -> Array:
        a, b = self._bounds
        sigmoid = 1.0 / (1.0 + self._bkd.exp(-can_iterate))
        return sigmoid * (1 - sigmoid) * (b - a)

    def __call__(self, can_iterate: Array) -> Array:
        iterate = self._from_canonical(can_iterate)
        return self._residual(iterate)

    def _jacobian(self, can_iterate: Array) -> Array:
        iterate = self._from_canonical(can_iterate)
        return self._residual.jacobian(iterate) * self._tranform_jacobian(
            can_iterate
        )

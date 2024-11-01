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

    def _param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        raise NotImplementedError

    def param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        hvp = self._param_param_hvp(fwd_sol, adj_sol, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_param_hvp must return 1D array")
        return hvp

    def _state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        raise NotImplementedError

    def state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        hvp = self._state_state_hvp(fwd_sol, adj_sol, wvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_sol.shape[0]:
            raise RuntimeError("_state_state_hvp must return 1D array")
        return hvp

    def _param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        raise NotImplementedError

    def param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        hvp = self._param_state_hvp(fwd_sol, adj_sol, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_state_hvp must return 1D array")
        return hvp

    def _state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        raise NotImplementedError

    def state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        hvp = self._state_param_hvp(fwd_sol, adj_sol, vvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_sol.shape[0]:
            raise RuntimeError("_state_param_hvp must return 1D array")
        return hvp


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

    def _qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        raise NotImplementedError

    def qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        hvp = self._qoi_param_param_hvp(sol, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_param_param_hvp must return 1D array")
        return hvp

    def _qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        raise NotImplementedError

    def qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        hvp = self._qoi_state_state_hvp(sol, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_state_hvp must return 1D array")
        return hvp

    def _qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        raise NotImplementedError

    def qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        hvp = self._qoi_state_param_hvp(sol, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_param_hvp must return 1D array")
        return hvp

    def _qoi_param_state_hvp(self, sol: Array, wvec: Array) -> Array:
        raise NotImplementedError

    def qoi_param_state_hvp(self, sol: Array, wvec: Array) -> Array:
        hvp = self._qoi_param_state_hvp(sol, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_param_state_hvp must return 1D array")
        return hvp


class AdjointSolver:
    def __init__(
        self, newton_solver: NewtonSolver, functional: AdjointFunctional
    ):
        self._bkd = newton_solver._bkd
        self._residual = newton_solver._residual
        self._newton_solver = newton_solver
        self._functional = functional

        self._fwd_sol_param = None
        self._adj_sol_param = None

    def set_param(self, param: Array):
        self._param = param
        self._residual.set_param(param)
        self._functional.set_param(param)

    def set_initial_iterate(self, iterate: Array):
        if iterate.ndim != 1:
            raise ValueError("iterate must return 1D array")
        self._init_iterate = iterate

    def forward_solve(self):
        if not hasattr(self, "_init_iterate"):
            raise RuntimeError("must call set_initial_iterate")
        self._fwd_sol = self._newton_solver.solve(self._init_iterate)
        self._fwd_sol_param = self._bkd.copy(self._param)
        return self._fwd_sol

    def solve_adjoint(self) -> Array:
        if self._fwd_sol_param is None or not self._bkd.allclose(
            self._fwd_sol_param, self._param, atol=1e-15, rtol=1e-15
        ):
            self.forward_solve()
        self._drdy = self._residual.jacobian(self._fwd_sol)
        dqdy = self._functional.qoi_sol_jacobian(self._fwd_sol)
        self._adj_sol = self._bkd.solve(self._drdy.T, -dqdy)
        self._adj_sol_param = self._bkd.copy(self._param)
        return self._adj_sol

    def gradient(self) -> Array:
        self.solve_adjoint()
        self._drdp = self._residual.param_jacobian(self._fwd_sol)
        return (
            self._functional.qoi_param_jacobian(self._fwd_sol)
            + self._adj_sol @ self._drdp
        )

    def forward_hessian_solve(self, vvec: Array) -> Array:
        self._drdp = self._residual.param_jacobian(self._fwd_sol)
        return self._bkd.solve(self._drdy, self._drdp @ vvec)

    def _lagrangian_state_state_hvp(self, wvec: Array) -> Array:
        # L_yy.w, w = wvec
        return self._functional.qoi_state_state_hvp(
            self._fwd_sol, wvec
        ) + self._residual.state_state_hvp(self._fwd_sol, self._adj_sol, wvec)

    def _lagrangian_state_param_hvp(self, vvec: Array) -> Array:
        # L_yp.v
        return self._functional.qoi_state_param_hvp(
            self._fwd_sol, vvec
        ) + self._residual.state_param_hvp(self._fwd_sol, self._adj_sol, vvec)

    def _lagrangian_param_state_hvp(self, wvec: Array) -> Array:
        # L_py.w, w = wvec
        return self._functional.qoi_param_state_hvp(
            self._fwd_sol, wvec
        ) + self._residual.param_state_hvp(self._fwd_sol, self._adj_sol, wvec)

    def _lagrangian_param_param_hvp(self, vvec: Array) -> Array:
        # L_pp.v
        return self._functional.qoi_param_param_hvp(
            self._fwd_sol, vvec
        ) + self._residual.param_param_hvp(self._fwd_sol, self._adj_sol, vvec)

    def adjoint_hessian_solve(self, wvec: Array, vvec: Array) -> Array:
        return self._bkd.solve(
            self._drdy.T,
            self._lagrangian_state_state_hvp(wvec)
            - self._lagrangian_state_param_hvp(vvec),
        )

    def apply_hessian(self, vvec: Array) -> Array:
        if self._adj_sol_param is None or not self._bkd.allclose(
            self._adj_sol_param, self._param, atol=1e-15, rtol=1e-15
        ):
            self.solve_adjoint()

        wvec = self.forward_hessian_solve(vvec)
        svec = self.adjoint_hessian_solve(wvec, vvec)
        return (
            self._drdp.T @ svec
            - self._lagrangian_param_state_hvp(wvec)
            + self._lagrangian_param_param_hvp(vvec)
        )

    def __repr__(self):
        return "{0}(functional={1})".format(
            self.__class__.__name__, self.functional
        )

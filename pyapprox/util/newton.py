from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


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
            raise RuntimeError(f"jac has the wrong shape {jac.shape}")
        return jac

    def linsolve(self, iterate: Array, res: Array) -> Array:
        jac = self.jacobian(iterate)
        return self._bkd.solve(jac, res)

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class ParameterizedNewtonResidualMixin(ABC):

    @abstractmethod
    def set_param(self, param: Array):
        raise NotImplementedError
        self._param = param

    def _residual_param_wrapper(self, sol: Array, param: Array) -> Array:
        self.set_param(param)
        return self(sol)

    def _param_jacobian(self, sol: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._residual_param_wrapper, sol), self._param
        )

    def param_jacobian(self, sol: Array) -> Array:
        if sol.ndim != 1:
            raise ValueError("sol must be a 1d Array")
        jac = self._param_jacobian(sol)
        if jac.ndim != 2 or jac.shape != (sol.shape[0], self._param.shape[0]):
            raise RuntimeError(
                "jac has the wrong shape {0} should be {1}".format(
                    jac.shape, (sol.shape[0], self._param.shape[0])
                )
            )
        return jac

    def _adjoint_dot_residual_param_wrapper(
        self, adj_sol: Array, fwd_sol: Array, param: Array
    ):
        self.set_param(param)
        return adj_sol @ self(fwd_sol)

    def _adjoint_dot_residual_state_wrapper(
        self, adj_sol: Array, fwd_sol: Array
    ):
        return adj_sol @ self(fwd_sol)

    def _param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(
                self._adjoint_dot_residual_param_wrapper, adj_sol, fwd_sol
            ),
            self._param,
            vvec,
        )

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
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(self._adjoint_dot_residual_state_wrapper, adj_sol),
            fwd_sol,
            wvec,
        )

    def state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        hvp = self._state_state_hvp(fwd_sol, adj_sol, wvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_sol.shape[0]:
            raise RuntimeError("_state_state_hvp must return 1D array")
        return hvp

    def _adjoint_dot_residual_state_jvp(self, adj_sol, wvec, fwd_sol, param):
        self.set_param(param)
        return self._bkd.jvp(
            partial(self._adjoint_dot_residual_state_wrapper, adj_sol),
            fwd_sol,
            wvec,
        )

    def _param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            partial(
                self._adjoint_dot_residual_state_jvp, adj_sol, wvec, fwd_sol
            ),
            self._param,
        )

    def param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        hvp = self._param_state_hvp(fwd_sol, adj_sol, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_state_hvp must return 1D array")
        return hvp

    def _adjoint_dot_residual_param_jvp(self, adj_sol, vvec, param, fwd_sol):
        return self._bkd.jvp(
            partial(
                self._adjoint_dot_residual_param_wrapper, adj_sol, fwd_sol
            ),
            param,
            vvec,
        )

    def _state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            partial(
                self._adjoint_dot_residual_param_jvp,
                adj_sol,
                vvec,
                self._param,
            ),
            fwd_sol,
        )

    def state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        hvp = self._state_param_hvp(fwd_sol, adj_sol, vvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_sol.shape[0]:
            raise RuntimeError("_state_param_hvp must return 1D array")
        return hvp

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError


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

    def set_residual(self, residual: NewtonResidual):
        if not isinstance(residual, NewtonResidual):
            raise ValueError("residual must be an instance of NewtonResidual")
        self._residual = residual
        self._bkd = residual._bkd

    def _linesearch(
        self, prev_sol: Array, prev_residual: Array, prev_residual_norm: float
    ) -> Array:
        # bisection-based linesearch
        step_size = self._step_size / 2
        ii = 0
        while ii < self._linesearch_maxiters:
            sol = prev_sol - step_size * self._residual.linsolve(
                prev_sol, prev_residual
            )
            residual = self._residual(sol)
            residual_norm = self._bkd.norm(residual)
            if self._verbosity > 1:
                print("\t Linesearch Iter", ii, "rnorm", residual_norm)
            if residual_norm < prev_residual_norm:
                return sol, residual, residual_norm
            step_size /= 2.0
            ii += 1
        raise RuntimeError("Max linesearch iterations reached")

    def _update_sol(self, prev_sol: Array, delta: Array) -> Array:
        return prev_sol - self._step_size * delta

    def solve(self, init_guess: Array) -> Array:
        if init_guess.ndim != 1:
            raise ValueError("init_guess must be 1D array")
        if not hasattr(self, "_residual"):
            raise ValueError("must call set_residual")
        sol = init_guess  # self._bkd.copy(init_guess)
        residual = self._residual(sol)
        residual_norms = [self._bkd.norm(residual)]
        it = 0
        if self._verbosity > 1:
            print("Iter", it, "rnorm", residual_norms[0])
        while True:
            prev_sol = sol  # self._bkd.copy(sol)
            prev_residual = residual
            # sol = prev_sol - self._step_size * self._residual.linsolve(
            #     sol, prev_residual
            # )
            sol = self._update_sol(
                prev_sol, self._residual.linsolve(sol, prev_residual)
            )
            residual = self._residual(sol)
            residual_norm = self._bkd.norm(residual)
            # linesearch is not tested
            # if residual_norm > residual_norms[-1]:
            #     sol, residual, residual_norm = self._linesearch(
            #         prev_sol, prev_residual, residual_norms[-1]
            #     )
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


class Functional(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nunique_functional_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _value(self, sol: Array) -> Array:
        raise NotImplementedError

    def __call__(self, sol: Array) -> Array:
        # there seems to be an inconsistency of using sol.ndim == 2
        # here and ndim ==1 below.  I think this is currently time dependent
        # sols are 2D but steady sols are 2d
        if sol.ndim != 1 or sol.shape[0] != self.nstates():
            print(sol.shape, self.nstates())
            raise ValueError("sol has the wrong shape")
        val = self._value(sol)
        if val.ndim != 1 or val.shape[0] != self.nqoi():
            raise RuntimeError(f"{self} must return a 1D array")
        return val

    def __repr__(self):
        return "{0}(nstates={1}, nparams={2}, nqoi={3})".format(
            self.__class__.__name__,
            self.nstates(),
            self.nparams(),
            self.nqoi(),
        )


class AdjointFunctional(Functional):
    def _qoi_state_jacobian(self, sol: Array) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(self._value, sol)

    def qoi_state_jacobian(self, sol: Array) -> Array:
        """Gradient of qoi with respect to solution"""
        if sol.ndim != 1 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 1d Array")
        jac = self._qoi_state_jacobian(sol)
        if jac.shape != (self.nqoi(), sol.shape[0]):
            raise RuntimeError(
                "jac shape {0} should be {1}".format(
                    jac.shape, (self.nqoi(), sol.shape[0])
                )
            )
        return jac

    def _qoi_param_wrapper(self, sol: Array, param: Array) -> Array:
        self.set_param(param)
        return self._value(sol)

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_param_wrapper, sol), self._param
        )

    def qoi_param_jacobian(self, sol: Array) -> Array:
        """Gradient of QoI with respect to parameters"""
        if sol.ndim != 1 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 1d Array")
        jac = self._qoi_param_jacobian(sol)
        # make sure scalar jacobians get returned as 2D array with one row
        jac = self._bkd.atleast2d(jac)
        if jac.shape != (self.nqoi(), self.nparams()):
            raise RuntimeError("jac has the wrong shape")
        return jac

    def set_param(self, param: Array):
        if param.ndim != 1:
            raise ValueError("param must be a 1D Array")
        self._param = param

    def _qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(self._qoi_param_wrapper, sol), self._param, vvec
        )

    def qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        hvp = self._qoi_param_param_hvp(sol, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_param_param_hvp must return 1D array")
        return hvp

    def _qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(self._value, sol, wvec)

    def qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        hvp = self._qoi_state_state_hvp(sol, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_state_hvp must return 1D array")
        return hvp

    def _qoi_param_jvp(self, vvec, param, fwd_sol):
        return self._bkd.jvp(
            partial(self._qoi_param_wrapper, fwd_sol), self._param, vvec
        )

    def _qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_param_jvp, vvec, self._param), sol
        )[0]

    def qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        hvp = self._qoi_state_param_hvp(sol, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_param_hvp must return 1D array")
        return hvp

    def _qoi_state_jvp(self, wvec, fwd_sol, param):
        self.set_param(param)
        return self._bkd.jvp(self._value, fwd_sol, wvec)

    def _qoi_param_state_hvp(self, sol: Array, wvec: Array) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_state_jvp, wvec, sol), self._param
        )[0]

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
        self.set_functional(functional)

        self._fwd_sol_param = None
        self._adj_sol_param = None

    def set_functional(self, functional: AdjointFunctional):
        if functional is not None and not isinstance(
            functional, AdjointFunctional
        ):
            raise TypeError(
                "functional must be an instance of AdjointFunctional"
            )
        self._functional = functional

    def set_param(self, param: Array):
        self._param = param
        self._residual.set_param(param)
        if hasattr(self, "_functional") and self._functional is not None:
            self._functional.set_param(param)

    def set_initial_iterate(self, iterate: Array):
        if iterate.ndim != 1:
            raise ValueError("iterate must return 1D array")
        self._init_iterate = iterate

    def forward_solve(self):
        if not hasattr(self, "_init_iterate"):
            raise AttributeError("must call set_initial_iterate")
        self._fwd_sol = self._newton_solver.solve(self._init_iterate)
        self._fwd_sol_param = self._bkd.copy(self._param)
        return self._fwd_sol

    def solve_adjoint(self) -> Array:
        if self._fwd_sol_param is None or not self._bkd.allclose(
            self._fwd_sol_param, self._param, atol=3e-16, rtol=3e-16
        ):
            self.forward_solve()
        if self._functional.nqoi() != 1:
            raise ValueError(
                "Adjoint can only be applied to a scalar Functional"
            )
        self._drdy = self._residual.jacobian(self._fwd_sol)
        dqdy = self._functional.qoi_state_jacobian(self._fwd_sol)
        self._adj_sol = self._bkd.solve(self._drdy.T, -dqdy[0])
        self._adj_sol_param = self._bkd.copy(self._param)
        return self._adj_sol

    def solve_sensitivities(self) -> Array:
        if self._fwd_sol_param is None or not self._bkd.allclose(
            self._fwd_sol_param, self._param, atol=1e-15, rtol=1e-15
        ):
            self.forward_solve()
        drdy = self._residual.jacobian(self._fwd_sol)
        drdp = self._residual.param_jacobian(self._fwd_sol)
        sens = self._bkd.solve(drdy, -drdp)
        return sens

    def parameter_jacobian(self):
        # compute parameter jacobian using forward sensitivities
        # useful when then number of QoI is commensurate with the
        # number of parameters
        sens = self.solve_sensitivities()
        dqdy = self._functional.qoi_state_jacobian(self._fwd_sol)
        dqdp = self._functional.qoi_param_jacobian(self._fwd_sol)
        return dqdy @ sens + dqdp

    def gradient(self) -> Array:
        # compute the gradient of a single QoI
        self.solve_adjoint()
        self._drdp = self._residual.param_jacobian(self._fwd_sol)
        return (
            self._functional.qoi_param_jacobian(self._fwd_sol)[0]
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

        qps_hvp = self._functional.qoi_param_state_hvp(self._fwd_sol, wvec)
        if qps_hvp.ndim != 1:
            raise RuntimeError("qps_hvp must be a 1D array")
        rps_hvp = self._residual.param_state_hvp(
            self._fwd_sol, self._adj_sol, wvec
        )
        if rps_hvp.ndim != 1:
            raise RuntimeError("rps_hvp must be a 1D array")
        return qps_hvp + rps_hvp

    def _lagrangian_param_param_hvp(self, vvec: Array) -> Array:
        # L_pp.v

        qpp_hvp = self._functional.qoi_param_param_hvp(self._fwd_sol, vvec)
        if qpp_hvp.ndim != 1:
            raise RuntimeError("qpp_hvp must be a 1D array")
        rpp_hvp = self._residual.param_param_hvp(
            self._fwd_sol, self._adj_sol, vvec
        )
        if rpp_hvp.ndim != 1:
            raise RuntimeError(
                "rpp_hvp returned by {0} must be a 1D array".format(
                    self._residual
                )
            )
        return qpp_hvp + rpp_hvp

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
        lps_hvp = self._lagrangian_param_state_hvp(wvec)
        lpp_hvp = self._lagrangian_param_param_hvp(vvec)
        hvp = self._drdp.T @ svec - lps_hvp + lpp_hvp
        return hvp

    def __repr__(self):
        return "{0}(functional={1})".format(
            self.__class__.__name__, self._functional
        )


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

from abc import ABC, abstractmethod
from functools import partial

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.newnewton import ParameterizedNewtonResidual


class AdjointInterfaceConstraintEquation(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _solve(self, init_state: Array, param: Array) -> Array:
        raise NotImplementedError

    def solve(self, init_state: Array, param: Array) -> Array:
        if init_state.shape != (self.nstates(),):
            raise ValueError(
                f"init_state had shape {init_state.shape} but "
                f"must have shape {(self.nstates(),)}"
            )
        if param.shape != (self.nvars(),):
            raise ValueError(
                f"param had shape {param.shape} but "
                f"must have shape {(self.nvars(),)}"
            )
        return self._solve(init_state, param)

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(partial(self.solve, state), param)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        if state.shape != (self.nstates(),):
            raise ValueError(f"state must have shape {(self.nstates(),)}")
        if param.shape != (self.nvars(),):
            raise ValueError(f"param must have shape {(self.nvars(),)}")
        jac = self._param_jacobian(state, param)
        if jac.ndim != 2 or jac.shape != (self.nstates(), self.nvars()):
            raise RuntimeError(
                "jac has the wrong shape {0} should be {1}".format(
                    jac.shape, (self.nstates(), self.nvars())
                )
            )
        return jac

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to state"""
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(partial(self.solve, state), param)

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to state"""
        if state.shape != (self.nstates(),):
            raise ValueError(
                f"state has shape {state.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        if param.shape != (self.nvars(),):
            raise ValueError(
                f"param has shape {param.shape} but must have "
                f"shape {(self.nvars(),)}"
            )
        jac = self._state_jacobian(state, param)
        if jac.ndim != 2 or jac.shape != (self.nstates(), self.nstates()):
            raise RuntimeError(
                "jac has the wrong shape {0} should be {1}".format(
                    jac.shape, (self.nstates(), self.nstates())
                )
            )
        return jac


class Functional(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nunique_vars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _value(self, state: Array) -> Array:
        raise NotImplementedError

    def __call__(self, state: Array) -> Array:
        # there seems to be an inconsistency of using state.ndim == 2
        # here and ndim ==1 below.  I think this is currently time dependent
        # states are 2D but steady states are 2d
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError(
                "state has the wrong shape: {0} but nstates is {1}".format(
                    state.shape, self.nstates()
                )
            )
        val = self._value(state)
        if val.ndim != 1 or val.shape[0] != self.nqoi():
            raise RuntimeError(f"{self} must return a 1D array")
        return val

    def __repr__(self):
        return "{0}(nstates={1}, nvars={2}, nqoi={3})".format(
            self.__class__.__name__,
            self.nstates(),
            self.nvars(),
            self.nqoi(),
        )


class AdjointFunctional(Functional):
    def jacobian_implemented() -> bool:
        return False

    def apply_hessian_implemented() -> bool:
        return False

    def _qoi_state_jacobian(self, state: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the state.
        """
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(self._value, state)

    def qoi_state_jacobian(self, state: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the state.
        """
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError("state must be a 1d Array")
        jac = self._qoi_state_jacobian(state)
        if jac.shape != (self.nqoi(), state.shape[0]):
            raise RuntimeError(
                "jac shape {0} should be {1}".format(
                    jac.shape, (self.nqoi(), state.shape[0])
                )
            )
        return jac

    def _qoi_param_wrapper(self, state: Array, param: Array) -> Array:
        self.set_param(param)
        return self._value(state)

    def _qoi_param_jacobian(self, state: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the parameters
        """
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_param_wrapper, state), self._param
        )

    def qoi_param_jacobian(self, state: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the parameters
        """
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError("state must be a 1d Array")
        jac = self._qoi_param_jacobian(state)
        # make sure scalar jacobians get returned as 2D array with one row
        jac = self._bkd.atleast2d(jac)
        if jac.shape != (self.nqoi(), self.nvars()):
            raise RuntimeError("jac has the wrong shape")
        return jac

    def set_param(self, param: Array):
        if param.ndim != 1:
            raise ValueError("param must be a 1D Array")
        self._param = param

    def _qoi_param_param_hvp(self, state: Array, vvec: Array) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(self._qoi_param_wrapper, state), self._param, vvec
        )

    def qoi_param_param_hvp(self, state: Array, vvec: Array) -> Array:
        hvp = self._qoi_param_param_hvp(state, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_param_param_hvp must return 1D array")
        return hvp

    def _qoi_state_state_hvp(self, state: Array, wvec: Array) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(self._value, state, wvec)

    def qoi_state_state_hvp(self, state: Array, wvec: Array) -> Array:
        hvp = self._qoi_state_state_hvp(state, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_state_hvp must return 1D array")
        return hvp

    def _qoi_param_jvp(self, vvec, param, fwd_state):
        return self._bkd.jvp(
            partial(self._qoi_param_wrapper, fwd_state), self._param, vvec
        )

    def _qoi_state_param_hvp(self, state: Array, vvec: Array) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_param_jvp, vvec, self._param), state
        )[0]

    def qoi_state_param_hvp(self, state: Array, vvec: Array) -> Array:
        hvp = self._qoi_state_param_hvp(state, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_state_param_hvp must return 1D array")
        return hvp

    def _qoi_state_jvp(self, wvec, fwd_state, param):
        self.set_param(param)
        return self._bkd.jvp(self._value, fwd_state, wvec)

    def _qoi_param_state_hvp(self, state: Array, wvec: Array) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._qoi_state_jvp, wvec, state), self._param
        )[0]

    def qoi_param_state_hvp(self, state: Array, wvec: Array) -> Array:
        hvp = self._qoi_param_state_hvp(state, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_qoi_param_state_hvp must return 1D array")
        return hvp

    def nqoi(self) -> int:
        return 1


class AdjointInterfaceData:
    def __init__(self, backend: BackendMixin):
        self._bkd = backend
        self._attribute_names = [
            "_drdy",
            "_drdq",
            "_dqdy",
            "_dqdp",
            "_fwd_state",
        ]

    def set_parameter(self, param: Array) -> None:
        self._param = param
        self._clear()

    def _clear(self) -> None:
        for attr_name in self._attribute_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def has_parameter(self, param: Array):
        if not hasattr(self, "_param"):
            return False
        if self._bkd.allclose(param, self._param, atol=3e-16, rtol=3e-16):
            return True
        return False

    def set_forward_state(self, fwd_state: Array) -> None:
        self._fwd_state = fwd_state

    def has_forward_state(self) -> bool:
        if not hasattr(self, "_fwd_state"):
            return False
        return True

    def get_forward_state(self, fwd_state: Array) -> None:
        if not self.has_forward_state():
            raise AttributeError("must call set_parameter")
        return self._fwd_state

    def set_constraint_equation_state_jacobian(self, drdy: Array) -> None:
        self._drdy = drdy

    def set_constraint_equation_param_jacobian(self, drdp: Array) -> None:
        self._drdp = drdp

    def set_qoi_state_jacobian(self, dqdy: Array) -> None:
        self._dqdy = dqdy

    def set_qoi_param_jacobian(self, dqdp: Array) -> None:
        self._dqdp = dqdp

    def get_constraint_equation_state_jacobian(self) -> Array:
        if not hasattr(self, "_drdy"):
            raise AttributeError(
                "must call set_constraint_equation_state_jacobian"
            )
        return self._drdy

    def get_constraint_equation_param_jacobian(self) -> Array:
        if not hasattr(self, "_drdp"):
            raise AttributeError(
                "must call set_constraint_equation_param_jacobian"
            )
        return self._drdp

    def get_qoi_state_jacobian(self) -> Array:
        if not hasattr(self, "_dqdy"):
            raise AttributeError("must call set_qoi_param_jacobian")
        return self._dqdy

    def get_qoi_param_jacobian(self) -> Array:
        if not hasattr(self, "_dqdp"):
            raise AttributeError("must call set_qoi_param_jacobian")
        return self._dqdp


class AdjointInterface:
    def __init__(
        self,
        constraint_equation: AdjointInterfaceConstraintEquation,
        functional: AdjointFunctional,
    ):
        if not isinstance(
            constraint_equation, AdjointInterfaceConstraintEquation
        ):
            raise TypeError(
                "constraint_equation must be an instance of "
                "AdjointInterfaceConstraintEquation"
            )
        self._bkd = constraint_equation._bkd
        self._constraint_equation = constraint_equation
        self._adjoint_data = AdjointInterfaceData(self._bkd)

        if not isinstance(functional, AdjointFunctional):
            raise TypeError("functional must be an instance AdjointFunctional")
        if not self._bkd.bkd_equal(self._bkd, functional._bkd):
            raise TypeError(
                "constraint_equation bkd does not match functional backend"
            )
        self._functional = functional

    def solve_adjoint_equation(self, fwd_state: Array, param: Array) -> Array:
        drdy = self._constraint_equation.state_jacobian(fwd_state, param)
        dqdy = self._functional.qoi_state_jacobian(fwd_state, param)
        adj_state = self._bkd.solve(drdy.T, -dqdy[0])
        self._adjoint_data.set_constraint_equation_state_jacobian(drdy)
        self._adjoint_data.set_qoi_state_jacobian(dqdy)
        return adj_state

    def _get_forward_state(self, init_fwd_state: Array, param: Array) -> Array:
        if (
            not self._adjoint_data.has_parameter(param)
            or not self._adjoint_data.has_fwd_state()
        ):
            self._adjoint_data._clear()
            fwd_state = self._constraint_equation.solve(init_fwd_state, param)
            self._adjoint_data.set_forward_state(fwd_state)
        return self._adjoint_data.get_forward_state(fwd_state)

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        fwd_state = self._get_forward_state(init_fwd_state, param)
        adj_state = self.solve_adjoint_equation(fwd_state, param)
        drdp = self._constraint_equation.constraint_equation_param_jacobian(
            self._fwd_state
        )
        self._adjoint_data.set_constraint_equation_param_jacobian(drdp)
        jacobian = (
            self._functional.qoi_param_jacobian(fwd_state)[0]
            + adj_state @ drdp
        )[None, :]
        return jacobian


class GradientEnabledParameterizedNewtonResidual(ParameterizedNewtonResidual):
    def _residual_param_wrapper(self, state: Array, param: Array) -> Array:
        self.set_param(param)
        return self(state)

    def _param_jacobian(self, state: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            partial(self._residual_param_wrapper, state), self._param
        )

    def param_jacobian(self, state: Array) -> Array:
        if state.ndim != 1:
            raise ValueError("state must be a 1d Array")
        jac = self._param_jacobian(state)
        if jac.ndim != 2 or jac.shape != (
            state.shape[0],
            self._param.shape[0],
        ):
            raise RuntimeError(
                "jac has the wrong shape {0} should be {1}".format(
                    jac.shape, (state.shape[0], self._param.shape[0])
                )
            )
        return jac


class HVPEnabledParameterizedNewtonResidual(
    GradientEnabledParameterizedNewtonResidual
):
    def _adjoint_dot_residual_param_wrapper(
        self, adj_state: Array, fwd_state: Array, param: Array
    ):
        self.set_param(param)
        return adj_state @ self(fwd_state)

    def _adjoint_dot_residual_state_wrapper(
        self, adj_state: Array, fwd_state: Array
    ):
        return adj_state @ self(fwd_state)

    def _param_param_hvp(
        self, fwd_state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(
                self._adjoint_dot_residual_param_wrapper, adj_state, fwd_state
            ),
            self._param,
            vvec,
        )

    def param_param_hvp(
        self, fwd_state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        hvp = self._param_param_hvp(fwd_state, adj_state, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_param_hvp must return 1D array")
        return hvp

    def _state_state_hvp(
        self, fwd_state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            partial(self._adjoint_dot_residual_state_wrapper, adj_state),
            fwd_state,
            wvec,
        )

    def state_state_hvp(
        self, fwd_state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        hvp = self._state_state_hvp(fwd_state, adj_state, wvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError("_state_state_hvp must return 1D array")
        return hvp

    def _adjoint_dot_residual_state_jvp(
        self, adj_state, wvec, fwd_state, param
    ):
        self.set_param(param)
        return self._bkd.jvp(
            partial(self._adjoint_dot_residual_state_wrapper, adj_state),
            fwd_state,
            wvec,
        )

    def _param_state_hvp(
        self, fwd_state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            partial(
                self._adjoint_dot_residual_state_jvp,
                adj_state,
                wvec,
                fwd_state,
            ),
            self._param,
        )

    def param_state_hvp(
        self, fwd_state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        hvp = self._param_state_hvp(fwd_state, adj_state, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_state_hvp must return 1D array")
        return hvp

    def _adjoint_dot_residual_param_jvp(
        self, adj_state, vvec, param, fwd_state
    ):
        return self._bkd.jvp(
            partial(
                self._adjoint_dot_residual_param_wrapper, adj_state, fwd_state
            ),
            param,
            vvec,
        )

    def _state_param_hvp(
        self, fwd_state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            partial(
                self._adjoint_dot_residual_param_jvp,
                adj_state,
                vvec,
                self._param,
            ),
            fwd_state,
        )

    def state_param_hvp(
        self, fwd_state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        hvp = self._state_param_hvp(fwd_state, adj_state, vvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError("_state_param_hvp must return 1D array")
        return hvp


class AdjointNewtonSolver:
    def set_functional(self, functional: AdjointFunctional):
        if not isinstance(functional, AdjointFunctional):
            raise TypeError(
                "functional must be an instance of AdjointFunctional"
            )
        self._functional = functional

    def set_initial_iterate(self, init_iterate: Array) -> Array:
        # Unlike NewtonSolver we need to set init iterate here
        # instead of passing it into solve so that we can automatically
        # invoke solve when requesting adjoint solution
        self._init_iterate = init_iterate

    def get_initial_iterate(self) -> Array:
        if not hasattr(self, "_init_iterate"):
            raise AttributeError("must call set_initial_iterate")
        return self._init_iterate

    def solve(self) -> Array:
        self._params = self._bkd.copy(self._residual.parameters())
        self._fwd_state = super().solve(self.get_initial_iterate())
        return self._fwd_state

    def solve_adjoint(self) -> Array:
        if not hasattr(self, "_functional"):
            raise AttributeError("must call set_functional")
        if not hasattr(self, "_params") is None or not self._bkd.allclose(
            self._params, self._residual.parameters(), atol=3e-16, rtol=3e-16
        ):
            self.solve()
        self._drdy = self._residual.jacobian(self._fwd_state)
        dqdy = self._functional.qoi_state_jacobian(self._fwd_state)
        self._adj_state = self._bkd.solve(self._drdy.T, -dqdy[0])
        self._adj_state_param = self._bkd.copy(self._param)
        return self._adj_state

    def solve_sensitivities(self) -> Array:
        if not hasattr(self, "_params") is None or not self._bkd.allclose(
            self._params, self._residual.parameters(), atol=3e-16, rtol=3e-16
        ):
            self.solve()
        drdy = self._residual.jacobian(self._fwd_state)
        drdp = self._residual.param_jacobian(self._fwd_state)
        sens = self._bkd.solve(drdy, -drdp)
        return sens

    def parameter_jacobian(self):
        # compute parameter jacobian using forward sensitivities
        # useful when then number of QoI is commensurate with the
        # number of parameters
        sens = self.solve_sensitivities()
        dqdy = self._functional.qoi_state_jacobian(self._fwd_state)
        dqdp = self._functional.qoi_param_jacobian(self._fwd_state)
        return dqdy @ sens + dqdp

    def gradient(self) -> Array:
        # compute the gradient of a single QoI
        self.solve_adjoint()
        self._drdp = self._residual.param_jacobian(self._fwd_state)
        return (
            self._functional.qoi_param_jacobian(self._fwd_state)[0]
            + self._adj_state @ self._drdp
        )

    def forward_hessian_solve(self, vvec: Array) -> Array:
        self._drdp = self._residual.param_jacobian(self._fwd_state)
        return self._bkd.solve(self._drdy, self._drdp @ vvec)

    def _lagrangian_state_state_hvp(self, wvec: Array) -> Array:
        # L_yy.w, w = wvec
        return self._functional.qoi_state_state_hvp(
            self._fwd_state, wvec
        ) + self._residual.state_state_hvp(
            self._fwd_state, self._adj_state, wvec
        )

    def _lagrangian_state_param_hvp(self, vvec: Array) -> Array:
        # L_yp.v
        return self._functional.qoi_state_param_hvp(
            self._fwd_state, vvec
        ) + self._residual.state_param_hvp(
            self._fwd_state, self._adj_state, vvec
        )

    def _lagrangian_param_state_hvp(self, wvec: Array) -> Array:
        # L_py.w, w = wvec

        qps_hvp = self._functional.qoi_param_state_hvp(self._fwd_state, wvec)
        if qps_hvp.ndim != 1:
            raise RuntimeError("qps_hvp must be a 1D array")
        rps_hvp = self._residual.param_state_hvp(
            self._fwd_state, self._adj_state, wvec
        )
        if rps_hvp.ndim != 1:
            raise RuntimeError("rps_hvp must be a 1D array")
        return qps_hvp + rps_hvp

    def _lagrangian_param_param_hvp(self, vvec: Array) -> Array:
        # L_pp.v

        qpp_hvp = self._functional.qoi_param_param_hvp(self._fwd_state, vvec)
        if qpp_hvp.ndim != 1:
            raise RuntimeError("qpp_hvp must be a 1D array")
        rpp_hvp = self._residual.param_param_hvp(
            self._fwd_state, self._adj_state, vvec
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
        if self._adj_state_param is None or not self._bkd.allclose(
            self._adj_state_param, self._param, atol=1e-15, rtol=1e-15
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

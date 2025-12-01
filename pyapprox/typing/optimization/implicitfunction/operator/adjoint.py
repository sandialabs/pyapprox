from abc import ABC, abstractmethod
from typing import Optional
import unittest  # Enable check_derivatives with good error messages

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.interface.model import GradientCheckMixin
from pyapprox.optimization.newton import ResidualEquation, NewtonResidual


class ParameterizedResidualEquation(ResidualEquation):
    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _set_parameters(self, param: Array) -> None:
        raise NotImplementedError

    def _check_parameters(self, param: Array) -> None:
        if param.shape != (self.nvars(),):
            raise ValueError(
                f"param has shape {param.shape} but must have "
                "shape {(self.nvars(),)}"
            )

    def set_parameters(self, param: Array) -> None:
        self._check_parameters(param)
        self._param = param
        self._set_parameters(param)

    def _check_has_parameters(self) -> None:
        if not hasattr(self, "_param"):
            raise AttributeError("must call set_parameters")

    def get_parameters(self) -> Array:
        self._check_has_parameters()
        return self._param

    def _check_state_param_shapes(
        self, init_state: Array, param: Array
    ) -> None:
        if init_state.shape != (self.nstates(),):
            raise ValueError(
                f"init_state had shape {init_state.shape} but "
                f"must have shape {(self.nstates(),)}"
            )
        self._check_parameters(param)

    @abstractmethod
    def _value(self, iterate: Array, param: Array) -> Array:
        raise NotImplementedError

    def value(self, init_state: Array) -> Array:
        self._check_has_parameters()
        self._check_iterate(init_state)
        val = self._value(init_state, self.get_parameters())
        if val.shape != (self.nstates(),):
            raise ValueError(
                f"value has shape {val.shape} but must be {(self.nstates(),)}"
            )
        return val

    def solve(self, init_state: Array, param: Array) -> Array:
        # used for checking derivatives
        self._check_iterate(init_state)
        self.set_parameters(param)
        return self._solve(init_state)


class AdjointResidualEquation(
    ParameterizedResidualEquation, GradientCheckMixin
):
    def use_auto_differentiation(self) -> bool:
        """
        Set to true if the user wants to use automatic differentiation.
        Even if this is true, an error will be thrown if the backend
        does not support it.
        """
        return False

    def _check_automatic_differentiation(self) -> None:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError("Automatic differentiation not enabled")
        if not self.use_auto_differentiation():
            raise RuntimeError(
                f"{self}.use_auto_differentiation() returns False.\n"
                "Set it to return True if all functions this class"
                "requires use a backend that supports auto diffentiation.\n"
                "Otherwise, implement the functions with the anlaytical"
                "expressions."
            )

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        self._check_automatic_differentiation()
        return self._bkd.jacobian(lambda p: self._value(state, p), param)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        self._check_state_param_shapes(state, param)
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
        self._check_automatic_differentiation()
        return self._bkd.jacobian(lambda y: self._value(y, param), state)

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Gradient of residual with respect to state"""
        self._check_state_param_shapes(state, param)
        jac = self._state_jacobian(state, param)
        if jac.ndim != 2 or jac.shape != (self.nstates(), self.nstates()):
            raise RuntimeError(
                "jac has the wrong shape {0} should be {1}".format(
                    jac.shape, (self.nstates(), self.nstates())
                )
            )
        return jac

    def check_state_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_jacobian")
        return self._check_apply(
            state[:, None],
            "J_y",
            lambda y: self._value(y[:, 0], param),
            lambda y, v: self._state_jacobian(y[:, 0], param) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def check_param_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_jacobian")
        return self._check_apply(
            param[:, None],
            "J_p",
            lambda p: self._value(state, p[:, 0]),
            lambda p, v: self.param_jacobian(state, p[:, 0]) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def __repr__(self) -> str:
        return "{0}(nstates={1}, nvars={2})".format(
            self.__class__.__name__, self.nstates(), self.nvars()
        )


class AdjointResidualEquationWithHessian(AdjointResidualEquation):
    def _adjoint_dot_residual_param_wrapper(
        self, fwd_state: Array, param: Array, adj_state: Array
    ) -> Array:
        return adj_state @ self._value(fwd_state, param)

    def _adjoint_dot_residual_state_wrapper(
        self,
        fwd_state: Array,
        param: Array,
        adj_state: Array,
    ) -> Array:
        return adj_state @ self._value(fwd_state, param)

    def _param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError

        # TODO: if user provides a param_jacobian function but not
        # state_state_hvp, then it would be faster than what is below
        # to compute jacobian of the param_jacobian.
        # But if not provided and jacobians are nested torch
        # requires create_graph=True so only want to nest if user provides
        # the function without autograd
        return self._bkd.hvp(
            lambda p: adj_state @ self._value(fwd_state, p),
            param,
            vvec,
        )

    def param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        hvp = self._param_param_hvp(fwd_state, param, adj_state, vvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_param_hvp must return 1D array")
        return hvp

    def check_param_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_param_hvp")
        return self._check_apply(
            param[:, None],
            "H_pp",
            lambda p: adj_state[None, :] @ self.param_jacobian(state, p[:, 0]),
            lambda p, v: self.param_param_hvp(
                state, p[:, 0], adj_state, v[:, 0]
            ),
            fd_eps,
            direction,
            relative,
            disp,
        )

    def _state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        # TODO: if user provides a state_jacobian function but not
        # state_state_hvp, then it would be faster than what is below
        # to compute jacobian of the state_jacobian
        # But if not provided and jacobians are nested torch
        # requires create_graph=True so only want to nest if user provides
        # the function without autograd
        return self._bkd.hvp(
            lambda y: adj_state @ self._value(y, param),
            fwd_state,
            wvec,
        )

    def state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        hvp = self._state_state_hvp(fwd_state, param, adj_state, wvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError("_state_state_hvp must return 1D array")
        return hvp

    def check_state_state_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_state_hvp")
        return self._check_apply(
            state[:, None],
            "H_yy",
            lambda y: adj_state[None, :] @ self.state_jacobian(y[:, 0], param),
            lambda y, w: self.state_state_hvp(
                y[:, 0], param, adj_state, w[:, 0]
            ),
            fd_eps,
            direction,
            relative,
            disp,
        )

    # def _adjoint_dot_residual_state_jvp(
    #     self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    # ):
    #     return self._bkd.jvp(
    #         partial(
    #             self._adjoint_dot_residual_state_wrapper,
    #             param=param,
    #             adj_state=adj_state,
    #         ),
    #         fwd_state,
    #         wvec,
    #     )

    def _param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            lambda p: adj_state @ self.state_jacobian(fwd_state, p) @ wvec,
            # partial(
            #    self._adjoint_dot_residual_state_jvp,
            #    fwd_state,
            #    adj_state=adj_state,
            #    wvec=wvec,
            # ),
            param,
        )

    def param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        hvp = self._param_state_hvp(fwd_state, param, adj_state, wvec)
        if hvp.ndim != 1 or hvp.shape[0] != param.shape[0]:
            raise RuntimeError(
                f"_param_state_hvp returned shape {hvp.shape} but "
                f"must have shape (param.shape[0],)"
            )
        return hvp

    def check_param_state_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_state_hvp")
        return self._check_apply(
            state[:, None],
            "H_py",
            lambda y: adj_state[None, :] @ self.param_jacobian(y[:, 0], param),
            lambda y, w: self.param_state_hvp(
                y[:, 0], param, adj_state, w[:, 0]
            ),
            fd_eps,
            direction,
            relative,
            disp,
        )

    # def _adjoint_dot_residual_param_jvp(
    #     self, adj_state, vvec, param, fwd_state
    # ):
    #     return self._bkd.jvp(
    #         partial(
    #             self._adjoint_dot_residual_param_wrapper, adj_state, fwd_state
    #         ),
    #         param,
    #         vvec,
    #     )

    def _state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # if using torch requires result of jvp to be differentiable
        return self._bkd.jacobian(
            lambda y: adj_state @ self.param_jacobian(y, param) @ vvec,
            # partial(
            #     self._adjoint_dot_residual_param_jvp,
            #     adj_state,
            #     vvec,
            #     self._param,
            # ),
            fwd_state,
        )

    def state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        hvp = self._state_param_hvp(fwd_state, param, adj_state, vvec)
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError(
                f"_state_param_hvp returned shape {hvp.shape} but "
                f"must have shape (fwd_state.shape[0],)"
            )
        return hvp

    def check_state_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_param_hvp")
        return self._check_apply(
            param[:, None],
            "H_yp",
            lambda p: adj_state[None, :] @ self.state_jacobian(state, p[:, 0]),
            lambda p, v: self.state_param_hvp(
                state, p[:, 0], adj_state, v[:, 0]
            ),
            fd_eps,
            direction,
            relative,
            disp,
        )


class Functional(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend
        if self.nunique_vars() >= self.nvars():
            raise ValueError(
                f"{self.nunique_vars()=} must be smaller than {self.nvars()=}"
            )

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
    def _value(self, state: Array, param: Array) -> Array:
        raise NotImplementedError

    def __call__(self, state: Array, param: Array) -> Array:
        # there seems to be an inconsistency of using state.ndim == 2
        # here and ndim ==1 below.  I think this is currently time dependent
        # states are 2D but steady states are 2d
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError(
                "state has the wrong shape: {0} but nstates is {1}".format(
                    state.shape, self.nstates()
                )
            )
        val = self._bkd.atleast1d(self._value(state, param))
        if val.ndim != 1 or val.shape[0] != self.nqoi():
            raise RuntimeError(
                f"{self}: value has shape {val.shape} but must have "
                f"shape {(self.nqoi(),)}"
            )
        return val

    def __repr__(self) -> str:
        return "{0}(nstates={1}, nvars={2}, nqoi={3})".format(
            self.__class__.__name__,
            self.nstates(),
            self.nvars(),
            self.nqoi(),
        )


class VectorAdjointFunctional(Functional, GradientCheckMixin):
    def use_auto_differentiation(self) -> bool:
        """
        Set to true if the user wants to use automatic differentiation.
        Even if this is true, an error will be thrown if the backend
        does not support it.
        """
        return False

    def _check_automatic_differentiation(self) -> None:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError("Automatic differentiation not enabled")
        if not self.use_auto_differentiation():
            raise RuntimeError(
                f"{self}.use_auto_differentiation() returns False.\n"
                "Set it to return True if all functions this class"
                "requires use a backend that supports auto diffentiation.\n"
                "Otherwise, implement the functions with the anlaytical"
                "expressions."
            )

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the state.
        """
        self._check_automatic_differentiation()
        return self._bkd.jacobian(lambda y: self(y, param), state)

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the state.
        """
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError("state must be a 1d Array")
        jac = self._state_jacobian(state, param)
        if jac.shape != (self.nqoi(), self.nstates()):
            raise RuntimeError(
                "jac shape {0} should be {1}".format(
                    jac.shape, (self.nqoi(), self.nstates())
                )
            )
        return jac

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the parameters
        """
        self._check_automatic_differentiation()
        return self._bkd.jacobian(lambda p: self(state, param), param)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        The Jacobian of the QoI with respect to the parameters
        """
        if state.ndim != 1 or state.shape[0] != self.nstates():
            raise ValueError("state must be a 1d Array")
        jac = self._param_jacobian(state, param)
        # make sure scalar jacobians get returned as 2D array with one row
        jac = self._bkd.atleast2d(jac)
        if jac.shape != (self.nqoi(), self.nvars()):
            raise RuntimeError(
                f"jac has shape {jac.shape} bust must have "
                f"shape {(self.nqoi(), self.nvars())}"
            )
        return jac

    def check_state_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_jacobian")
        return self._check_apply(
            state[:, None],
            "J_state",
            lambda y: self(y[:, 0], param),
            lambda y, v: self.state_jacobian(y[:, 0], param) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def check_param_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_jacobian")
        return self._check_apply(
            param[:, None],
            "J_param",
            lambda p: self(state, p[:, 0]),
            lambda p, v: self.param_jacobian(state, p[:, 0]) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )


class ScalarAdjointFunctional(VectorAdjointFunctional):
    def nqoi(self) -> int:
        return 1


class ScalarAdjointFunctionalWithHessian(ScalarAdjointFunctional):
    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError

        # TODO: if user provides a state_jacobian function but not
        # state_state_hvp, then it would be faster than what is below
        # to compute jacobian of the state_jacobian
        # But if not provided and jacobians are nested torch
        # requires create_graph=True so only want to nest if user provides
        # the function without autograd
        # return self._bkd.jacobian(
        #    lambda p: self.param_jacobian(state, p)[0] @ vvec, param
        # )
        return self._bkd.hvp(lambda p: self(state, p), param, vvec)

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        hvp = self._param_param_hvp(state, param, vvec)
        if hvp.shape != (param.shape[0],):
            raise RuntimeError(
                f"_param_param_hvp had shape {hvp.shape} "
                f"but must have shape {(param.shape[0],)}"
            )
        return hvp

    def check_param_param_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_param_hvp")
        return self._check_apply(
            param[:, None],
            "H_pp",
            lambda p: self.param_jacobian(state, p[:, 0]),
            lambda p, v: self.param_param_hvp(state, p[:, 0], v[:, 0]),
            fd_eps,
            direction,
            relative,
            disp,
        )

    def _state_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        # TODO: if user provides a state_jacobian function but not
        # state_state_hvp, then it would be faster than what is below
        # to compute jacobian of the state_jacobian
        # But if not provided and jacobians are nested torch
        # requires create_graph=True so only want to nest if user provides
        # the function without autograd
        # return self._bkd.jacobian(
        #     lambda y: self.state_jacobian(y, param)[0] @ wvec, state
        # )
        return self._bkd.hvp(lambda y: self(y, param)[0], state, wvec)

    def state_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        hvp = self._state_state_hvp(state, param, wvec)
        if hvp.shape != (state.shape[0],):
            raise RuntimeError(
                f"_state_state_hvp had shape {hvp.shape} "
                f"but must have shape {(state.shape[0],)}"
            )
        return hvp

    def check_state_state_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_state_hvp")
        return self._check_apply(
            state[:, None],
            "H_yy",
            lambda y: self.state_jacobian(y[:, 0], param),
            lambda y, w: self.state_state_hvp(y[:, 0], param, w[:, 0]),
            fd_eps,
            direction,
            relative,
            disp,
        )

    # def _param_jvp(self, vvec, param, fwd_state):
    #     return self._bkd.jvp(
    #         partial(self._param_wrapper, fwd_state), self._param, vvec
    #     )

    def _state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            lambda y: self.param_jacobian(y, param) @ vvec,
            # partial(self._param_jvp, vvec, self._param),
            state,
        )[0]

    def state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        hvp = self._state_param_hvp(state, param, vvec)
        if hvp.ndim != 1:
            raise RuntimeError(
                f"_state_param_hvp has shape {hvp.shape} "
                f"but must have shape {(state.shape[0],)}"
            )
        return hvp

    def check_state_param_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_state_param_hvp")
        return self._check_apply(
            param[:, None],
            "H_yp",
            lambda p: self.state_jacobian(state, p[:, 0]),
            lambda p, v: self.state_param_hvp(state, p[:, 0], v[:, 0]),
            fd_eps,
            direction,
            relative,
            disp,
        )

    # def _state_jvp(self, wvec, fwd_state, param):
    #     self.set_param(param)
    #     return self._bkd.jvp(self._value, fwd_state, wvec)

    def _param_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        # return self._bkd.jacobian(
        #    partial(self._state_jvp, wvec, state), self._param
        # )[0]
        return self._bkd.jacobian(
            lambda p: self.state_jacobian(state, p) @ wvec,
            # partial(self._param_jvp, vvec, self._param),
            param,
        )[0]

    def param_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        hvp = self._param_state_hvp(state, param, wvec)
        if hvp.ndim != 1:
            raise RuntimeError("_param_state_hvp must return 1D array")
        return hvp

    def check_param_state_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_param_state_hvp")
        return self._check_apply(
            state[:, None],
            "H_py",
            lambda y: self.param_jacobian(y[:, 0], param),
            lambda y, w: self.param_state_hvp(y[:, 0], param, w[:, 0]),
            fd_eps,
            direction,
            relative,
            disp,
        )


class ScalarAdjointOperator(GradientCheckMixin):
    def __init__(
        self,
        residual_eq: AdjointResidualEquation,
        functional: ScalarAdjointFunctional,
    ):
        self._bkd = residual_eq._bkd
        self._residual_eq = residual_eq
        if not self._bkd.bkd_equal(self._bkd, functional._bkd):
            raise TypeError(
                "residual_eq bkd does not match functional backend"
            )
        self._functional = functional
        self._adjoint_data = AdjointOperatorData(self._bkd)

    def adjoint_data(self) -> AdjointOperatorData:
        return self._adjoint_data

    def nvars(self) -> int:
        return self._residual_eq.nvars()

    def value(self, init_fwd_state: Array, param: Array) -> Array:
        fwd_state = self._get_forward_state(init_fwd_state, param)
        return self._functional(fwd_state, param)

    def solve_adjoint_equation(self, fwd_state: Array, param: Array) -> Array:
        drdy = self._residual_eq.state_jacobian(fwd_state, param)
        dqdy = self._functional.state_jacobian(fwd_state, param)
        adj_state = self._bkd.solve(drdy.T, -dqdy[0])
        self._adjoint_data.set_adjoint_state(adj_state)
        self._adjoint_data.set_residual_eq_state_jacobian(drdy)
        self._adjoint_data.set_qoi_state_jacobian(dqdy)
        return adj_state

    def _get_forward_state(self, init_fwd_state: Array, param: Array) -> Array:
        if (
            True
            or not self._adjoint_data.has_parameter(param)
            or not self._adjoint_data.has_fwd_state()
        ):
            self._adjoint_data._clear()
            fwd_state = self._residual_eq.solve(init_fwd_state, param)
            self._adjoint_data.set_forward_state(fwd_state)
        return self._adjoint_data.get_forward_state(fwd_state)

    def _get_residual_eq_param_jacobian(
        self, fwd_state: Array, param: Array
    ) -> Array:
        if not self._adjoint_data.has_residual_eq_param_jacobian():
            drdp = self._residual_eq.param_jacobian(fwd_state, param)
            self._adjoint_data.set_residual_eq_param_jacobian(drdp)
        return self._adjoint_data.get_residual_eq_param_jacobian()

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        fwd_state = self._get_forward_state(init_fwd_state, param)
        adj_state = self.solve_adjoint_equation(fwd_state, param)
        drdp = self._get_residual_eq_param_jacobian(fwd_state, param)
        jacobian = (
            self._functional.param_jacobian(fwd_state, param)[0]
            + adj_state @ drdp
        )[None, :]
        return jacobian

    def check_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_jacobian")
        return self._check_apply(
            param[:, None],
            "J",
            lambda p: self._functional(
                self._residual_eq.solve(state, p[:, 0]), p[:, 0]
            ),
            lambda p, v: self.jacobian(state, p[:, 0]) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def _assert_derivatives_close(self, errors: Array, tol: float) -> None:
        if self._bkd.min(errors) == self._bkd.max(errors):
            assert self._bkd.min(errors) == 0.0
        else:
            self._unittest.assertLessEqual(
                self._bkd.min(errors) / self._bkd.max(errors), tol
            )

    def get_derivative_tolerances(self, tol: float) -> Array:
        nchecks = 5
        return self._bkd.full((nchecks,), tol)

    def check_derivatives(
        self,
        init_state: Array,
        param: Array,
        tols: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> None:
        # Create an instance of TestCase
        self._unittest = unittest.TestCase()
        # check first order derivatives of contraint equation
        errors = self._residual_eq.check_state_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[0])
        errors = self._residual_eq.check_param_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[1])

        # check first order derivatives of functional
        errors = self._functional.check_state_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[2])
        # exact jac is zero so set relative to zero to avoid divide by zero
        errors = self._functional.check_param_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[3])

        # check Jacobian
        errors = self.check_jacobian(init_state, param, fd_eps, disp=disp)
        self._assert_derivatives_close(errors, tols[4])


class ScalarAdjointOperatorWithHessian(ScalarAdjointOperator):
    def __init__(
        self,
        residual_eq: AdjointResidualEquationWithHessian,
        functional: ScalarAdjointFunctionalWithHessian,
    ):
        # calling
        # super().__init__(residual_eq, functional)
        # causes type errors (see below) so have to use code below

        self._bkd = residual_eq._bkd
        # explicitly type residual_eq to avoid typing errors like:
        # "AdjointResidualEquation" has no attribute "state_state_hvp"
        self._residual_eq: AdjointResidualEquationWithHessian = residual_eq
        if not self._bkd.bkd_equal(self._bkd, functional._bkd):
            raise TypeError(
                "residual_eq bkd does not match functional backend"
            )
        # explicitly type residual_eq to avoid typing errors like:
        # "ScalarAdjointFunctional" has no attribute "param_param_hvp"
        self._functional: ScalarAdjointFunctionalWithHessian = functional
        self._adjoint_data = AdjointOperatorData(self._bkd)

    def _forward_hessian_solve(
        self,
        fwd_state: Array,
        param: Array,
        drdy: Array,
        drdp: Array,
        vvec: Array,
    ) -> Array:
        return self._bkd.solve(drdy, drdp @ vvec)

    def _lagrangian_state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        # L_yy.w, w = wvec
        return self._functional.state_state_hvp(
            fwd_state, param, wvec
        ) + self._residual_eq.state_state_hvp(
            fwd_state, param, adj_state, wvec
        )

    def _lagrangian_state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        # L_yp.v
        return self._functional.state_param_hvp(
            fwd_state, param, vvec
        ) + self._residual_eq.state_param_hvp(
            fwd_state, param, adj_state, vvec
        )

    def _lagrangian_param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        # L_py.w, w = wvec

        qps_hvp = self._functional.param_state_hvp(fwd_state, param, wvec)
        if qps_hvp.ndim != 1:
            raise RuntimeError("qps_hvp must be a 1D array")
        rps_hvp = self._residual_eq.param_state_hvp(
            fwd_state, param, adj_state, wvec
        )
        if rps_hvp.ndim != 1:
            raise RuntimeError("rps_hvp must be a 1D array")
        return qps_hvp + rps_hvp

    def _lagrangian_param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        # L_pp.v

        qpp_hvp = self._functional.param_param_hvp(fwd_state, param, vvec)
        if qpp_hvp.ndim != 1:
            raise RuntimeError("qpp_hvp must be a 1D array")
        rpp_hvp = self._residual_eq.param_param_hvp(
            fwd_state, param, adj_state, vvec
        )
        if rpp_hvp.ndim != 1:
            raise RuntimeError(
                "rpp_hvp returned by {0} must be a 1D array".format(
                    self._residual_eq
                )
            )
        return qpp_hvp + rpp_hvp

    def _adjoint_hessian_solve(
        self,
        fwd_state: Array,
        param: Array,
        adj_state: Array,
        drdy: Array,
        wvec: Array,
        vvec: Array,
    ) -> Array:
        return self._bkd.solve(
            drdy.T,
            self._lagrangian_state_state_hvp(fwd_state, param, adj_state, wvec)
            - self._lagrangian_state_param_hvp(
                fwd_state, param, adj_state, vvec
            ),
        )

    def _get_adjoint_state(self, init_fwd_state: Array, param: Array) -> Array:
        if (
            not self._adjoint_data.has_parameter(param)
            or not self._adjoint_data.has_adjoint_state()
        ):
            fwd_state = self._get_forward_state(init_fwd_state, param)
            adj_state = self.solve_adjoint_equation(fwd_state, param)
            self._adjoint_data.set_adjoint_state(adj_state)
        return self._adjoint_data.get_adjoint_state()

    def apply_hessian(
        self, init_fwd_state: Array, param: Array, vvec: Array
    ) -> Array:
        self._residual_eq._check_state_param_shapes(init_fwd_state, param)
        if vvec.shape != (self.nvars(),):
            raise ValueError(
                "vvec has shape {vvec.shape} but must be {(self.nvars(),)}"
            )

        # load or compute forward state
        fwd_state = self._get_forward_state(init_fwd_state, param)

        # load or compute adj state
        adj_state = self._get_adjoint_state(init_fwd_state, param)

        # load drdy, which is guaranteed to be created here because
        # solve_adjoint has been called by get_adjoint_state
        drdy = self._adjoint_data.get_residual_eq_state_jacobian()

        # load or compute drdp, which may exist if jacobian previously called
        drdp = self._get_residual_eq_param_jacobian(fwd_state, param)

        # Compute forward hessian state
        wvec = self._forward_hessian_solve(fwd_state, param, drdy, drdp, vvec)

        # Compute adjoint hessian state
        svec = self._adjoint_hessian_solve(
            fwd_state, param, adj_state, drdy, wvec, vvec
        )
        lps_hvp = self._lagrangian_param_state_hvp(
            fwd_state, param, adj_state, wvec
        )
        lpp_hvp = self._lagrangian_param_param_hvp(
            fwd_state, param, adj_state, vvec
        )
        hvp = drdp.T @ svec - lps_hvp + lpp_hvp
        return hvp

    def check_apply_hessian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_apply_hessian")
        return self._check_apply(
            param[:, None],
            "Hv",
            lambda p: self.jacobian(
                self._residual_eq.solve(state, p[:, 0]), p[:, 0]
            ),
            lambda p, v: self.apply_hessian(state, p[:, 0], v[:, 0]),
            fd_eps,
            direction,
            relative,
            disp,
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self._residual_eq, self._functional
        )

    def get_derivative_tolerances(self, tol: float) -> Array:
        nchecks = 14
        return self._bkd.full((nchecks,), tol)

    def check_derivatives(
        self,
        init_state: Array,
        param: Array,
        tols: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> None:
        super().check_derivatives(init_state, param, tols, fd_eps, disp)
        # check second order derivatives of contraint equation
        adj_state = self.adjoint_data().get_adjoint_state()
        errors = self._residual_eq.check_param_param_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[5])
        errors = self._residual_eq.check_state_state_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[6])
        errors = self._residual_eq.check_param_state_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[7])
        errors = self._residual_eq.check_state_param_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[8])

        # check second order derivaties of functional
        errors = self._functional.check_param_param_hvp(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[9])
        errors = self._functional.check_state_state_hvp(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[10])
        errors = self._functional.check_state_param_hvp(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[11])
        errors = self._functional.check_param_state_hvp(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[12])

        # check Hessian
        errors = self.check_apply_hessian(init_state, param, fd_eps, disp=disp)
        self._assert_derivatives_close(errors, tols[13])


class VectorAdjointOperator(GradientCheckMixin):
    def __init__(
        self,
        residual_eq: AdjointResidualEquation,
        functional: VectorAdjointFunctional,
    ):
        if not isinstance(residual_eq, AdjointResidualEquation):
            raise TypeError(
                "residual_eq must be an instance of " "AdjointResidualEquation"
            )
        self._bkd = residual_eq._bkd
        self._residual_eq = residual_eq
        self._adjoint_data = AdjointOperatorData(self._bkd)

        if not isinstance(functional, VectorAdjointFunctional):
            raise TypeError("functional must be an instance AdjointFunctional")
        if not self._bkd.bkd_equal(self._bkd, functional._bkd):
            raise TypeError(
                "residual_eq bkd does not match functional backend"
            )
        self._functional = functional

    def _get_forward_state(self, init_fwd_state: Array, param: Array) -> Array:
        if (
            not self._adjoint_data.has_parameter(param)
            or not self._adjoint_data.has_forward_state()
        ):
            self._adjoint_data._clear()
            fwd_state = self._residual_eq.solve(init_fwd_state, param)
            self._adjoint_data.set_forward_state(fwd_state)
        return self._adjoint_data.get_forward_state(fwd_state)

    def _sensitivities(self, fwd_state: Array, param: Array) -> Array:
        """
        Jacobian of the state with respect the parameters
        """
        drdy = self._residual_eq.state_jacobian(fwd_state, param)
        drdp = self._residual_eq.param_jacobian(fwd_state, param)
        sens = self._bkd.solve(drdy, -drdp)
        return sens

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        # compute parameter jacobian using forward sensitivities
        # useful when then number of QoI is commensurate with the
        # number of parameters
        fwd_state = self._get_forward_state(init_fwd_state, param)
        sens = self._sensitivities(fwd_state, param)
        dqdy = self._functional.state_jacobian(fwd_state, param)
        dqdp = self._functional.param_jacobian(fwd_state, param)
        return dqdy @ sens + dqdp

    def check_jacobian(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        disp: bool = False,
    ) -> Array:
        if disp:
            print(f"{self}.check_jacobian")
        return self._check_apply(
            param[:, None],
            "J",
            lambda p: self._functional(
                self._residual_eq.solve(state, p[:, 0]), p[:, 0]
            ),
            lambda p, v: self.jacobian(state, p[:, 0]) @ v,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def _assert_derivatives_close(self, errors: Array, tol: float) -> None:
        if self._bkd.min(errors) == self._bkd.max(errors):
            assert self._bkd.min(errors) == 0.0
        else:
            self._unittest.assertLessEqual(
                self._bkd.min(errors) / self._bkd.max(errors), tol
            )

    def get_derivative_tolerances(self, tol: float) -> Array:
        nchecks = 4
        return self._bkd.full((nchecks,), tol)

    def check_derivatives(
        self,
        init_state: Array,
        param: Array,
        tols: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> None:
        # Create an instance of TestCase
        self._unittest = unittest.TestCase()
        # check first order derivatives of contraint equation
        errors = self._residual_eq.check_state_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[0])
        errors = self._residual_eq.check_param_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[1])
        errors = self.check_jacobian(init_state, param, fd_eps, disp=disp)
        self._assert_derivatives_close(errors, tols[3])

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self._residual_eq, self._functional
        )


class NewtonResidualWithGradient(NewtonResidual, AdjointResidualEquation):
    def _jacobian(self, state: Array) -> Array:
        """
        jacobian used by solvers, e.g. NewtonSolver
        """
        return self.state_jacobian(state, self.get_parameters())


class NewtonResidualWithHessian(
    NewtonResidual, AdjointResidualEquationWithHessian
):
    def _jacobian(self, state: Array) -> Array:
        """
        jacobian used by solvers, e.g. NewtonSolver
        """
        return self.state_jacobian(state, self.get_parameters())

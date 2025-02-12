from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from scipy.optimize import LinearConstraint

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.interface.model import SingleSampleModel
from pyapprox.optimization.pya_minimize import (
    ConstrainedOptimizer, ScipyConstrainedOptimizer
)


class LocalOEDCriterionMixin(SingleSampleModel):
    def __init__(
        self,
        design_factors: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self.set_design_factors(design_factors, noise_mult)
        self._jacobian_implemented = True
        self._stored_design_prob_measure = dict()

    def set_design_factors(self, design_factors: Array, noise_mult: Array):
        if design_factors.ndim != 2:
            raise ValueError("design_factors must be a 3D array")
        ndesign_pts, ndesign_vars = design_factors.shape
        if noise_mult is not None and (
            noise_mult.ndim != 1 or noise_mult.shape != (ndesign_pts,)
        ):
            raise ValueError("noise multiplier has the wrong shape")
        self._design_factors = design_factors
        self._noise_mult = noise_mult

        # We assume that the noise of each observation at a design point
        # is independent of all observations at the design point and other
        # design points
        M0k, M1k = self._individual_design_matrices()
        if M0k.shape != (ndesign_pts, ndesign_vars, ndesign_vars):
            raise RuntimeError(
                "M0k returned by individual_design_matrices has wrong shape"
            )
        if M1k.shape != (ndesign_pts, ndesign_vars, ndesign_vars):
            raise RuntimeError(
                "M1k returned by individual_design_matrices has wrong shape"
            )
        self._M0k, self._M1k = M0k, M1k

    def _M0(self, design_prob_measure: Array) -> Array:
        if self._design_measure_changed("M0", design_prob_measure):
            M0 = self._bkd.einsum(
                "i,ijk->jk", design_prob_measure[:, 0], self._M0k
            )
            self._stored_design_prob_measure["M0"] = (design_prob_measure, M0)
        return self._stored_design_prob_measure["M0"][1]

    def _M1(self, design_prob_measure: Array) -> Array:
        if self._design_measure_changed("M1", design_prob_measure):
            M1 = self._bkd.einsum(
                "i,ijk->jk", design_prob_measure[:, 0], self._M1k
            )
            self._stored_design_prob_measure["M1"] = (design_prob_measure, M1)
        return self._stored_design_prob_measure["M1"][1]

    def _design_measure_changed(
        self, name: str, design_prob_measure: Array
    ) -> bool:
        return True  # hack
        if name not in self._stored_design_prob_measure:
            return True
        return not self._bkd.allclose(
            self._stored_design_prob_measure[name][0],
            design_prob_measure,
            atol=1e-15,
            rtol=1e-15,
        )

    def _M1inv(self, design_prob_measure: Array) -> Array:
        if self._design_measure_changed("M1inv", design_prob_measure):
            M1inv = self._bkd.inv(self._M1(design_prob_measure))
            self._stored_design_prob_measure["M1inv"] = (
                design_prob_measure,
                M1inv,
            )
        return self._stored_design_prob_measure["M1inv"][1]

    def ndesign_candidates(self) -> int:
        """The number of candidate design configurations"""
        return self._design_factors.shape[0]

    def nvars(self) -> int:
        """Number of uncertain model parameters"""
        return self._design_factors.shape[0]

    def nqoi(self) -> int:
        """The dimension of the vector returned by objective function."""
        return 1

    def is_homoscedastic(self) -> bool:
        return self._noise_mult is None


class LocalOEDRegressionMixin(ABC):
    @abstractmethod
    def _individual_design_matrices(self) -> Tuple[Array, Array]:
        raise NotImplementedError


class LstSqLocalOEDRegressionMixin(LocalOEDRegressionMixin):
    def _individual_design_matrices(self) -> Tuple[Array, Array]:
        M1k = self._bkd.einsum(
            "ij,il->ijl", self._design_factors, self._design_factors
        )
        if self.is_homoscedastic():
            return M1k, M1k
        M0k = self._bkd.einsum(
            "ij,il->ijl",
            self._design_factors,
            self._noise_mult[:, None] ** 2 * self._design_factors,
        )
        return M0k, M1k


class QuantileLocalOEDRegressionMixin(LocalOEDRegressionMixin):
    def _individual_design_matrices(self) -> Tuple[Array, Array]:
        M0k = self._bkd.einsum(
            "ij,il->ijl", self._design_factors, self._design_factors
        )
        if self.is_homoscedastic():
            return M0k, M0k
        M1k = self._bkd.einsum(
            "ij,il->ijl",
            self._design_factors,
            1 / self._noise_mult[..., None] * self._design_factors,
        )
        return M0k, M1k


class DOptimalMixin:
    def __init__(
        self,
        design_factors: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(design_factors, noise_mult, backend)

    def _evaluate(self, design_prob_measure: Array) -> Array:
        M1 = self._M1(design_prob_measure)
        if self.is_homoscedastic():
            # log(det(M1_inv)) = log(1/det(M1)) = -log(det(M1))
            return self._bkd.zeros((1, 1)) - self._bkd.log(self._bkd.det(M1))
        M0 = self._M0(design_prob_measure)
        M1_inv = self._bkd.inv(M1)
        gamma = M0 @ M1_inv
        return self._bkd.zeros((1, 1)) + self._bkd.log(
            self._bkd.det(M1_inv @ gamma)
        )

    def _jacobian(self, design_prob_measure: Array) -> Array:
        if self.is_homoscedastic():
            return self._homoscedastic_jacobian(design_prob_measure)
        return self._hetroscedastic_jacobian(design_prob_measure)

    def _hessian(self, design_prob_measure: Array) -> Array:
        if not self.is_homoscedastic():
            raise RuntimeError(
                "Hessian not supported for heteroscedastic noise"
            )
        return self._homoscedastic_hessian(design_prob_measure)

    @abstractmethod
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        raise NotImplementedError

    def _homoscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M1 = self._M1(design_prob_measure)
        M1_inv = self._bkd.inv(M1)
        temp = self._design_factors.T * (M1_inv @ self._design_factors.T)
        return -self._bkd.sum(temp, axis=0)[None, :]


class DOptimalLstSqCriterion(
    LstSqLocalOEDRegressionMixin, DOptimalMixin, LocalOEDCriterionMixin
):
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M1 = self._M1(design_prob_measure)
        M1_inv = self._bkd.inv(M1)
        M0_inv = self._bkd.inv(self._M0(design_prob_measure))
        return (
            -2
            * self._bkd.sum(
                self._design_factors.T * (M1_inv @ self._design_factors.T),
                axis=0,
            )[None, :]
            + self._bkd.sum(
                (self._noise_mult[:, None] * self._design_factors).T
                * (
                    M0_inv
                    @ (self._noise_mult[:, None] * self._design_factors).T
                ),
                axis=0,
            )[None, :]
        )


class DOptimalQuantileCriterion(
    QuantileLocalOEDRegressionMixin, DOptimalMixin, LocalOEDCriterionMixin
):
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M1 = self._M1(design_prob_measure)
        M1_inv = self._bkd.inv(M1)
        M0_inv = self._bkd.inv(self._M0(design_prob_measure))
        return (
            -2
            * self._bkd.sum(
                self._design_factors.T
                * (
                    M1_inv
                    @ ((self._design_factors / self._noise_mult[:, None]).T)
                ),
                axis=0,
            )[None, :]
            + self._bkd.sum(
                self._design_factors.T * (M0_inv @ self._design_factors.T),
                axis=0,
            )[None, :]
        )


from pyapprox.pde.collocation.adjoint_models import SteadyAdjointModel
from pyapprox.pde.collocation.newton import (
    NewtonResidual,
    ParameterizedNewtonResidualMixin,
    AdjointFunctional,
)


class LocalOEDAdjointFunctional(AdjointFunctional):
    def __init__(self, criterion: LocalOEDCriterionMixin):
        self._bkd = criterion._bkd
        self._crit = criterion

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._crit._design_factors.shape[1]

    def nparams(self) -> int:
        return self._crit._design_factors.shape[0]

    def nunique_functional_params(self) -> int:
        return 0


class LocalOEDAdjointFunctional(LocalOEDAdjointFunctional):
    def _value(self, states: Array) -> Array:
        M0 = self._crit._M0(self._param[:, None])
        return states[None, :] @ M0 @ states

    def _qoi_state_jacobian(self, states: Array) -> Array:
        M0 = self._crit._M0(self._param[:, None])
        return (2 * M0 @ states)[None, :]

    def _qoi_param_jacobian(self, states: Array) -> Array:
        return self._bkd.einsum("i,ji->j", states, self._crit._M0k @ states)

    def _qoi_param_param_hvp(self, state: Array, vvec: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))

    def _qoi_state_state_hvp(self, state: Array, wvec: Array) -> Array:
        return 2 * self._crit._M0(self._param[:, None]) @ wvec

    def _qoi_state_param_hvp(self, state: Array, vvec: Array) -> Array:
        return 2 * state @ self._bkd.einsum("i,ijk->jk", vvec, self._crit._M0k)

    def _qoi_param_state_hvp(self, state: Array, wvec: Array) -> Array:
        return 2 * self._bkd.einsum("ijk,k->ij", self._crit._M0k, wvec) @ state


class LocalOEDParameterizedNewtonResidual(
    ParameterizedNewtonResidualMixin, NewtonResidual
):
    def __init__(self, backend: LinAlgMixin):
        self._bkd = backend

    def set_vector(self, vec: Array):
        self._vec = vec

    def nstates(self) -> int:
        return self._crit._design_factors.shape[1]

    def nvars(self) -> int:
        return self._crit._design_factors.shape[0]

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError(
                "param has the wrong shape {0} should be {1}".format(
                    param.shape, (self.nvars(), 1)
                )
            )
        self._param = param

    def set_criterion(self, criterion: LocalOEDCriterionMixin):
        self._crit = criterion

    def linsolve(self, iterate: Array, res: Array) -> Array:
        return self._crit._M1inv(self._param[:, None]) @ res

    def __call__(self, iterate: Array) -> Array:
        return self._jacobian(iterate) @ iterate - self._vec

    def _jacobian(self, iterate: Array) -> Array:
        return self._crit._M1(self._param[:, None])

    def _param_jacobian(self, states: Array) -> Array:
        return self._bkd.einsum("i,jik->kj", states, self._crit._M1k)

    def _state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))

    def _param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        return self._bkd.einsum("i,ji->j", adj_sol, self._crit._M1k @ wvec)

    def _state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        return self._bkd.einsum("i,ij->j", vvec, self._crit._M1k @ adj_sol)


class LocalOEDAdjointModel(SteadyAdjointModel):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True
        self._apply_hessian_implemented = True

    def nvars(self) -> int:
        self._residual.nvars()


class LocalOEDCriterionAdjointMixin:
    def __init__(
        self,
        design_factors: Array,
        vecs: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        # Adjoint calculations require M0 to be computed even
        # when noise is None
        if noise_mult is None:
            noise_mult = backend.ones((design_factors.shape[0],))
        super().__init__(design_factors, noise_mult, backend)
        if vecs.ndim != 2 or vecs.shape[1] != design_factors.shape[1]:
            raise ValueError("vecs has the wrong shape")
        self._vecs = vecs
        self._residual = LocalOEDParameterizedNewtonResidual(self._bkd)
        self._residual.set_criterion(self)
        functional = LocalOEDAdjointFunctional(self)
        self._adj_model = LocalOEDAdjointModel(
            self._residual,
        )
        self._adj_model._adjoint_solver.set_initial_iterate(
            self._bkd.zeros((self._residual.nstates(),))
        )
        self._adj_model.set_functional(functional)
        self._apply_hessian_implemented = (
            self._adj_model._apply_hessian_implemented
        )

    def _evaluate(self, sample: Array) -> Array:
        val = 0
        for vec in self._vecs:
            self._residual.set_vector(vec)
            # must make adj solver recompute forward solution
            self._adj_model._adjoint_solver._fwd_sol_param = None
            self._adj_model._adjoint_solver._adj_sol_param = None
            val += self._adj_model._evaluate(sample)
        return val

    def _jacobian(self, sample: Array) -> Array:
        jac = 0
        for vec in self._vecs:
            self._residual.set_vector(vec)
            # must make adj solver recompute forward solution
            self._adj_model._adjoint_solver._fwd_sol_param = None
            self._adj_model._adjoint_solver._adj_sol_param = None
            jac += self._adj_model.jacobian(sample)
        return jac

    def _apply_hessian(self, sample: Array, vvec: Array) -> Array:
        hvp = 0
        for vec in self._vecs:
            self._residual.set_vector(vec)
            # must make adj solver recompute forward solution
            self._adj_model._adjoint_solver._fwd_sol_param = None
            self._adj_model._adjoint_solver._adj_sol_param = None
            hvp += self._adj_model.apply_hessian(sample, vvec)
        return hvp


class COptimalLstSqCriterion(
    LstSqLocalOEDRegressionMixin,
    LocalOEDCriterionAdjointMixin,
    LocalOEDCriterionMixin,
):
    def __init__(
        self,
        design_factors: Array,
        vec: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(design_factors, vec[None, :], noise_mult, backend)


class COptimalQuantileCriterion(
    QuantileLocalOEDRegressionMixin,
    LocalOEDCriterionAdjointMixin,
    LocalOEDCriterionMixin,
):
    def __init__(
        self,
        design_factors: Array,
        vec: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(design_factors, vec[None, :], noise_mult, backend)


class AOptimalLstSqCriterion(
    LstSqLocalOEDRegressionMixin,
    LocalOEDCriterionAdjointMixin,
    LocalOEDCriterionMixin,
):
    def __init__(
        self,
        design_factors: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        vecs = backend.eye(design_factors.shape[1])
        super().__init__(design_factors, vecs, noise_mult, backend)


class AOptimalQuantileCriterion(
    QuantileLocalOEDRegressionMixin,
    LocalOEDCriterionAdjointMixin,
    LocalOEDCriterionMixin,
):
    def __init__(
        self,
        design_factors: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        vecs = backend.eye(design_factors.shape[1])
        super().__init__(design_factors, vecs, noise_mult, backend)


class IOptimalLstSqCriterion(
        LstSqLocalOEDRegressionMixin,
        LocalOEDCriterionAdjointMixin,
        LocalOEDCriterionMixin,
):
    def __init__(
        self,
        design_factors: Array,
        pred_factors: Array,
        pred_prob_measure: Array = None,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self._integrate_pred_factors(pred_factors, pred_prob_measure)
        vecs = self._Bchol.T
        super().__init__(design_factors, vecs, noise_mult, backend)

    def _integrate_pred_factors(
            self, pred_factors: Array, pred_prob_measure: Array
    ):
        npred_pts = pred_factors.shape[0]
        if pred_prob_measure is None:
            pred_prob_measure = self._bkd.full((npred_pts,), 1/npred_pts)
        if pred_prob_measure.shape != (npred_pts,):
            raise ValueError("pred_prob_measure has the wrong shape")
        self._pred_prob_measure = pred_prob_measure
        self._pred_factors = pred_factors
        self._Bmat = pred_factors.T @ (
            pred_prob_measure[:, None] * pred_factors
        )
        self._Bchol = self._bkd.cholesky(self._Bmat)


class MiniMaxObjective(SingleSampleModel):
    def nvars(self) -> int:
        return 1

    def _evaluate(self, sample: Array) -> Array:
        return sample[:1, 0]

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.hstack(
            (self._bkd.ones((1,)), self._bkd.zeros((sample.shape[0]-1,)))
        )

    def _hessian(self, sample: Array) -> Array:
        return self._bkd.hstack(
            (self._bkd.ones((1,)), self._bkd.zeros((sample.shape[0]-1,)))
        )


class LocalOptimalExperimentalDesign:
    def __init__(self, criterion: LocalOEDCriterionMixin):
        if not isinstance(criterion, LocalOEDCriterionMixin):
            raise ValueError(
                "crit must be an instance of LocalOEDCriterionMixin"
            )
        self._crit = criterion
        self._bkd = self._crit._bkd

    def set_optimizer(self, optimizer: ConstrainedOptimizer):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be instance of ConstrainedOptimizer"
            )
        self._optimizer = optimizer
        self._optimizer.set_objective_function(self._crit)
        self._optimizer.set_bounds(
            self._bkd.stack(
                (
                    self._bkd.zeros((self._crit.nvars(),)),
                    self._bkd.full((self._crit.nvars(),), np.inf),
                ),
                axis=1,
            )
        )
        linear_con = LinearConstraint(
            self._bkd.ones((1, self._crit.nvars())),
            1.0,
            1.0,
            keep_feasible=True,
        )
        self._optimizer.set_constraints([linear_con])

    def default_optimizer(self) -> ScipyConstrainedOptimizer:
        return ScipyConstrainedOptimizer()

    def construct(self, init_iterate: Array = None) -> Array:
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        if init_iterate is None:
            init_iterate = self._bkd.full(
                (self._crit.nvars(), 1), 1./self._crit.nvars()
            )
        if init_iterate.shape != (self._crit.nvars(), 1):
            raise ValueError("init_iterate has the wrong shape")
        self._res = self._optimizer.minimize(init_iterate)
        return self._res.x

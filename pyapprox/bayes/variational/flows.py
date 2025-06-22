from abc import ABC, abstractmethod
from typing import List, Optional

from pyapprox.variables.joint import JointVariable
from pyapprox.interface.model import Model
from pyapprox.optimization.minimize import MultiStartOptimizer
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import (
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)


class FlowLayer(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _map_from_latent(self, usamples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _map_to_latent(self, samples: Array) -> Array:
        raise NotImplementedError


class RealNVPLayer(FlowLayer):
    def __init__(self, shapes, mask: Array):
        super().__init__(shapes._bkd)
        self._shapes = shapes
        if mask.shape != (self.nvars(),):
            raise ValueError("mask has the wrong shape")
        if mask.dtype != self._bkd.bool_type():
            raise ValueError("mask must be a boolean array")
        self._mask = mask
        self._mask_complement = ~mask
        self._ntransformed_vars = self._bkd.where(self._mask_complement)[
            0
        ].shape[0]
        self._hyp_list = self._shapes._hyp_list

    def nvars(self) -> int:
        return self._shapes.nqoi()

    def _map_from_latent(self, usamples: Array, return_logdet: bool) -> Array:
        # extract the variable dimensions that are not transformed
        shapes = self._shapes(usamples[self._mask])
        shift = shapes[:, : self._ntransformed_vars].T
        scale = shapes[:, self._ntransformed_vars :].T
        samples = self._bkd.copy(usamples)
        samples[self._mask_complement] = (
            usamples[self._mask_complement] * self._bkd.exp(scale) + shift
        )
        if not return_logdet:
            return samples
        return samples, self._map_from_latent_jacobian_log_determinant(shapes)

    def _map_to_latent(self, samples: Array, return_logdet: bool) -> Array:
        shapes = self._shapes(samples[self._mask])
        # shifts stored in first half of columns
        shift = shapes[:, : self._ntransformed_vars].T
        # scales stored in second half of columns
        scale = shapes[:, self._ntransformed_vars :].T
        usamples = self._bkd.copy(samples)
        usamples[self._mask_complement] = (
            samples[self._mask_complement] - shift
        ) * self._bkd.exp(-scale)
        if not return_logdet:
            return usamples
        return usamples, self._map_to_latent_jacobian_log_determinant(shapes)

    def _map_from_latent_jacobian_log_determinant(
        self, shapes: Array
    ) -> Array:
        shift = shapes[:, : self.nvars()].T
        return self._bkd.sum(shift, axis=0)

    def _map_to_latent_jacobian_log_determinant(self, shapes: Array) -> Array:
        return -self._map_from_latent_jacobian_log_determinant(shapes)


class Flow:
    def __init__(
        self, latent_variable: JointVariable, layers: List[FlowLayer]
    ):
        if not isinstance(latent_variable, JointVariable):
            raise ValueError("latent_variable must be a JointVariable")
        self._bkd = latent_variable._bkd
        self._latent_variable = latent_variable
        for layer in layers:
            if not isinstance(layer, FlowLayer):
                print(layers)
                raise ValueError("layer must be an intance of FlowLayer")
            if layer.nvars() != layers[0].nvars():
                raise ValueError(
                    "layers must have the same number of variables"
                )
        self._layers = layers
        self._hyp_list = sum(layer._hyp_list for layer in layers)

    def logpdf(self, samples: Array) -> Array:
        vals = 0.0
        samples = self._bkd.copy(samples)
        for layer in reversed(self._layers):
            samples, layer_logdet = layer._map_to_latent(samples, True)
            vals += layer_logdet
        vals = vals[:, None] + self._latent_variable.logpdf(samples)
        return vals

    def rvs(self, nsamples: int) -> Array:
        samples = self._latent_variable.rvs(nsamples)
        for layer in self._layers:
            samples = layer._map_to_latent(samples, False)
        return samples

    def nvars(self) -> int:
        return self._layers[0].nvars()

    def fit(self, samples: Array, iterate: Optional[Array] = None):
        self._samples = samples
        loss = FlowLoss()
        loss.set_flow(self)
        if not hasattr(self, "_optimizer"):
            self._optimizer = self.default_optimizer()
        self._optimizer.set_objective_function(loss)
        self._optimizer.set_bounds(self._hyp_list.get_active_opt_bounds())
        if iterate is None:
            iterate = self._hyp_list.get_active_opt_params()[:, None]
        self._opt_result = self._optimizer.minimize(iterate)
        print(self._opt_result)
        self._hyp_list.set_active_opt_params(self._opt_result.x[:, 0])

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        """
        Set the optimization algorithm.

        Parameters
        ----------
        optimizer : MultiStartOptimizer
            The optimization algorithm.
        """
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer

    def default_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        iterate_gen: OptimizerIterateGenerator = None,
        method: str = "L-BFGS-B",
    ) -> MultiStartOptimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        # L-BFGS-Bseems to require less iterations than trust-constr when
        # building GPs
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=method,
        )
        local_optimizer.set_verbosity(verbosity - 1)
        ms_optimizer = MultiStartOptimizer(
            local_optimizer, ncandidates=ncandidates
        )
        if iterate_gen is None:
            iterate_gen = self._default_iterator_gen()
        if not isinstance(iterate_gen, OptimizerIterateGenerator):
            raise ValueError(
                "iterate_gen must be an instance of OptimizerIterateGenerator"
            )
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(verbosity)
        return ms_optimizer

    def _default_iterator_gen(self):
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            self._hyp_list.nactive_vars(), backend=self._bkd
        )
        iterate_gen.set_bounds(
            self._bkd.to_numpy(self._hyp_list.get_active_opt_bounds())
        )
        return iterate_gen


class RealNVP(Flow):
    """
    Real Non Volume Preserving (RealNVP) Flow
    """

    def __init__(
        self, latent_variable: JointVariable, layers: List[FlowLayer]
    ):
        for layer in layers:
            if not isinstance(layer, RealNVPLayer):
                raise ValueError("layer must be an intance of RealNVPLayer")
        super().__init__(latent_variable, layers)


class FlowLoss(Model):
    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._flow._hyp_list.nactive_vars()

    def set_flow(self, flow: Flow):
        self._flow = flow
        self._bkd = self._flow._bkd

    def _values(self, active_opt_params: Array) -> Array:
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        nsamples = self._flow._samples.shape[1]
        weights = self._bkd.full((nsamples, 1), 1.0 / nsamples)
        return -weights.T @ self._flow.logpdf(self._flow._samples)

    def _jacobian(self, active_opt_params: Array):
        return self._bkd.jacobian(
            lambda p: self._values(p[:, None])[:, 0], active_opt_params[:, 0]
        )

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from pyapprox.util.visualization import get_meshgrid_samples
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
        self._loss = FlowLoss()
        self._loss.set_flow(self)

    def logpdf(self, samples: Array) -> Array:
        vals = 0.0
        samples = self._bkd.copy(samples)
        for layer in reversed(self._layers):
            samples, layer_logdet = layer._map_to_latent(samples, True)
            vals += layer_logdet
        vals = vals[:, None] + self._latent_variable.logpdf(samples)
        return vals

    def pdf(self, samples: Array) -> Array:
        return self._bkd.exp(self.logpdf(samples))

    def _map_from_latent(self, usamples: Array) -> Array:
        samples = usamples
        for layer in self._layers:
            samples = layer._map_from_latent(samples, False)
        return samples

    def _map_to_latent(self, samples: Array) -> Array:
        for layer in reversed(self._layers):
            samples = layer._map_to_latent(samples, False)
        return samples

    def rvs(self, nsamples: int) -> Array:
        return self._map_from_latent(self._latent_variable.rvs(nsamples))

    def nvars(self) -> int:
        return self._layers[0].nvars()

    def fit(self, samples: Array, iterate: Optional[Array] = None):
        self._loss.set_samples(samples)
        if not hasattr(self, "_optimizer"):
            self._optimizer = self.default_optimizer()
        self._optimizer.set_objective_function(self._loss)
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
        # if not isinstance(optimizer, MultiStartOptimizer):
        #     raise ValueError(
        #         "optimizer {0} must be instance of MultiStartOptimizer".format(
        #             optimizer
        #         )
        #     )
        self._optimizer = optimizer

    def xdefault_optimizer(self):
        from pyapprox.optimization.minimize import ChainedOptimizer
        from pyapprox.optimization.scipy import (
            ScipyConstrainedDifferentialEvolutionOptimizer,
        )

        opt1 = ScipyConstrainedDifferentialEvolutionOptimizer(
            opts={"maxiter": 20}
        )
        opt2 = ScipyConstrainedOptimizer()
        return ChainedOptimizer(opt1, opt2)

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

    # todo plot code is copy from joint.py merge codes, perhaps make flow
    # a joint variable
    def get_plot_axis(self, figsize=(8, 6), surface=False):
        if self.nvars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def _plot_pdf_1d(self, ax, npts_1d: Array, plot_limits: Array, **kwargs):
        plot_xx = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        ax.plot(plot_xx[0], self.pdf(plot_xx), **kwargs)

    def meshgrid_samples(
        self, plot_limits: Array, npts_1d: Union[Array, int] = 51
    ) -> Array:
        if self.nvars() != 2:
            raise RuntimeError("nvars !=2.")
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        return X, Y, pts

    def plot_pdf(
        self, ax, plot_limits: Array, npts_1d: Union[Array, int] = 51, **kwargs
    ):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot PDF when nvars >= 3.")
        if len(plot_limits) != self.nvars() * 2:
            raise ValueError("plot_limits has the wrong shape")
        if not isinstance(npts_1d, list):
            npts_1d = [npts_1d] * self.nvars()
        if self.nvars() == 1:
            return self._plot_pdf_1d(ax, npts_1d, plot_limits, **kwargs)
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        Z = self._bkd.detach(self._bkd.reshape(self.pdf(pts), X.shape))
        if kwargs.get("levels", None) is None:
            if ax.name != "3d":
                raise ValueError(
                    "levels not specified so trying to plot surface but not"
                    " given 3d axis"
                )
            return ax.plot_surface(X, Y, Z, **kwargs)
        return ax.contourf(X, Y, Z, **kwargs)


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

    def set_samples(self, samples: Array):
        self._samples = samples

    def _values(self, active_opt_params: Array) -> Array:
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        nsamples = self._samples.shape[1]
        weights = self._bkd.full((nsamples, 1), 1.0 / nsamples)
        return -weights.T @ self._flow.logpdf(self._samples)

    def _jacobian(self, active_opt_params: Array):
        return self._bkd.jacobian(
            lambda p: self._values(p[:, None])[:, 0], active_opt_params[:, 0]
        )

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.variables.joint import JointVariable
from pyapprox.interface.model import Model
from pyapprox.optimization.minimize import (
    ConstrainedMultiStartOptimizer,
    Constraint,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import (
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.surrogates.affine.multiindex import anova_level_indices


class FlowLayer(ABC):
    def __init__(self, nvars: int, nlabels: int, backend: BackendMixin):
        self._nvars = nvars
        self._nlabels = nlabels
        self._bkd = backend

    def nvars(self) -> int:
        return self._nvars

    def nlabels(self) -> int:
        return self._nlabels

    @abstractmethod
    def _map_from_latent(self, usamples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _map_to_latent(self, samples: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nlabels={2})".format(
            self.__class__.__name__, self.nvars(), self.nlabels()
        )


class RealNVPLayer(FlowLayer):
    def __init__(self, nvars, shapes, mask: Array, nlabels: int = 0):
        super().__init__(nvars, nlabels, shapes._bkd)
        if mask.shape != (self.nvars(),):
            raise ValueError(f"{mask.shape=} must be {(self.nvars(),)}")
        if mask.dtype != self._bkd.bool_type():
            raise ValueError("mask must be a boolean array")
        self._mask = mask
        self._mask_complement = ~mask
        self._ntransformed_vars = self._bkd.where(self._mask_complement)[
            0
        ].shape[0]

        # shapes is assumed to have signature=shapes([samples, labels])
        self._mask_w_labels = self._bkd.hstack(
            (self._mask, self._bkd.ones((nlabels,), dtype=bool))
        )
        self._mask_complement_wo_labels = self._bkd.hstack(
            (self._mask_complement, self._bkd.zeros((nlabels,), dtype=bool))
        )
        self._shapes = shapes
        if self._shapes.nvars() != self._ntransformed_vars + nlabels:
            raise ValueError(
                "shapes must be a model with model.nvars()={0}".format(
                    self._ntransformed_vars + nlabels
                )
                + f" but {shapes.nvars()=}"
            )
        self._hyp_list = self._shapes._hyp_list

    def _map_from_latent(self, usamples: Array, return_logdet: bool) -> Array:
        # extract the variable dimensions that are not transformed
        shapes = self._shapes(usamples[self._mask_w_labels])
        shift = shapes[:, : self._ntransformed_vars].T
        scale = shapes[:, self._ntransformed_vars :].T
        samples = self._bkd.copy(usamples)
        samples[self._mask_complement_wo_labels] = (
            usamples[self._mask_complement_wo_labels] * self._bkd.exp(scale)
            + shift
        )
        if not return_logdet:
            return samples
        return samples, self._map_from_latent_jacobian_log_determinant(shapes)

    def _map_to_latent(self, samples: Array, return_logdet: bool) -> Array:
        shapes = self._shapes(samples[self._mask_w_labels])
        # shifts stored in first half of columns
        shift = shapes[:, : self._ntransformed_vars].T
        # scales stored in second half of columns
        scale = shapes[:, self._ntransformed_vars :].T
        # print(shift[:, :5], "shift")
        if not self._bkd.all(self._bkd.isfinite(self._bkd.exp(scale))):
            II = self._bkd.where(~self._bkd.isfinite(self._bkd.exp(scale)))[1]
            print(self._bkd.exp(scale[:, II]))
            print(scale[:, II])
            print(samples[:, II])
            print(self._shapes.get_coefficients())
            raise ValueError("Not all values were finite")
        usamples = self._bkd.copy(samples)
        usamples[self._mask_complement_wo_labels] = (
            samples[self._mask_complement_wo_labels] - shift
        ) * self._bkd.exp(-scale)
        # use * exp(-scale) instead of 1/exp(scale)
        if not return_logdet:
            return usamples
        return usamples, self._map_to_latent_jacobian_log_determinant(shapes)

    def _map_from_latent_jacobian_log_determinant(
        self, shapes: Array
    ) -> Array:
        scale = shapes[:, self._ntransformed_vars :].T
        return self._bkd.sum(scale, axis=0)

    def _map_to_latent_jacobian_log_determinant(self, shapes: Array) -> Array:
        return -self._map_from_latent_jacobian_log_determinant(shapes)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, shapes={2})".format(
            self.__class__.__name__, self.nvars(), self._shapes
        )


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

            if layer.nlabels() != layers[0].nlabels():
                raise ValueError("layers must have the same number of labels")
        self._layers = layers
        self._hyp_list = sum(layer._hyp_list for layer in layers)
        self._loss = FlowLoss()
        self._loss.set_flow(self)

    def logpdf(self, samples: Array) -> Array:
        """
        Compute the logarithm of the PDF of the target variable.

        Parameters
        ----------
        samples: Array (nvars+nlabels, nsamples)
            Array containing samples from the target distribution and
            conditional labels, e.g. vstack((target_samples, labels)).

        Returns
        -------
        logpdf_vals: Array (nsamples, 1)
            Logratithm of the target PDF evaluated at samples.
        """
        logdet = 0.0
        samples = self._bkd.copy(samples)
        for layer in reversed(self._layers):
            samples, layer_logdet = layer._map_to_latent(samples, True)
            logdet += layer_logdet
        return logdet[:, None] + self._latent_variable.logpdf(
            samples[: self.nvars()]
        )

    def pdf(self, samples: Array) -> Array:
        """
        Compute the the PDF of the target variable.

        Parameters
        ----------
        samples: Array (nvars+nlabels, nsamples)
            Array containing samples from the target distribution and
            conditional labels, e.g. vstack((target_samples, labels)).

        Returns
        -------
        pdf_vals: Array (nsamples, 1)
            The target PDF evaluated at samples.
        """
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

    def append_labels(self, samples: Array, labels: Array):
        nsamples = samples.shape[1]
        if labels.shape[1] == 1:
            labels = self._bkd.tile(labels, (nsamples,))
        if labels.shape != (self.nlabels(), nsamples):
            raise ValueError(
                f"{labels.shape=} but must be " f"{(self.nlabels(), nsamples)}"
            )
        return self._bkd.vstack((samples, labels))

    def rvs(self, nsamples: int, labels: Optional[Array] = None) -> Array:
        """
        Randomly draw samples from the target variable.

        Parameters
        ----------
        nsamples: int
            The number of samples to draw.

        labels: Array (nlabel_vars, nsamples)
            Labels to condition the target variable.

        Returns
        -------
        samples: Array (nvars, nsamples)
            Array containing samples from the (conditional) target distribution
            conditional on the labels.
        """
        latent_samples = self._latent_variable.rvs(nsamples)
        if labels is None and self.nlabels() > 0:
            raise ValueError("Must specificy labels")
        if labels is not None:
            latent_samples = self.append_labels(latent_samples, labels)
        return self._map_from_latent(latent_samples)[: self.nvars()]

    def nvars(self) -> int:
        """
        Return the dimension of the (conditional) target variable.
        """
        return self._layers[0].nvars()

    def nlabels(self) -> int:
        """
        Return the dimension of the conditioning labels.
        """
        return self._layers[0].nlabels()

    def fit(
        self,
        samples: Array,
        iterate: Optional[Array] = None,
        weights: Optional[Array] = None,
    ):
        """
        Fit the flow to samples from the target variable.

        Parameters
        ----------
        samples: Array (nvars+nlabels, nsamples)
            Array containing samples from the target distribution and
            conditional labels, e.g. vstack((target_samples, labels)).

        iterate: Array (nactive_hyperparams, 1)
            The iterate used to initialize the optimization

        weights: Array (nsamples, 1)
            Quadrature weights associated with the samples (one-to-one-mapping)
            Defaults to an array of Monte Carlo weights, 1/nsamples.
        """
        self._loss.set_samples(samples)
        if weights is not None:
            self._loss.set_weights(weights)
        if not hasattr(self, "_optimizer"):
            self._optimizer = self.default_optimizer()
        self._optimizer.set_objective_function(self._loss)
        self._optimizer.set_bounds(self._hyp_list.get_active_opt_bounds())
        if iterate is None:
            iterate = self._hyp_list.get_active_opt_params()[:, None]
        self._opt_result = self._optimizer.minimize(iterate)
        # print(self._opt_result)
        self._hyp_list.set_active_opt_params(self._opt_result.x[:, 0])

    def set_optimizer(self, optimizer: ConstrainedMultiStartOptimizer):
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

    def default_optimizer(
        self,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        globalmaxiter: int = 20,
        globaltol: float = 1.0e-2,
        method: str = "L-BFGS-B",
    ):
        from pyapprox.optimization.minimize import ChainedOptimizer
        from pyapprox.optimization.scipy import (
            ScipyConstrainedDifferentialEvolutionOptimizer,
        )

        opt1 = ScipyConstrainedDifferentialEvolutionOptimizer(
            opts={"maxiter": globalmaxiter, "tol": globaltol}
        )
        opt1.set_verbosity(verbosity - 1)
        opt2 = ScipyConstrainedOptimizer()
        opt2.set_options(gtol=gtol, maxiter=maxiter, method=method)
        opt2.set_verbosity(verbosity - 1)
        return ChainedOptimizer(opt1, opt2)

    def default_multistart_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        iterate_gen: OptimizerIterateGenerator = None,
        method: str = "trust-constr",
        exit_hard: bool = True,
    ) -> ConstrainedMultiStartOptimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=method,
        )
        local_optimizer.set_verbosity(verbosity - 1)
        ms_optimizer = ConstrainedMultiStartOptimizer(
            local_optimizer, ncandidates=ncandidates, exit_hard=exit_hard
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

    def _plot_pdf_1d(
        self, ax, npts_1d: Array, plot_limits: Array, label: Array, **kwargs
    ):
        pts = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        if label is not None:
            pts = self.append_labels(pts, label)
        ax.plot(pts[0], self.pdf(pts), **kwargs)

    def meshgrid_samples(
        self, plot_limits: Array, npts_1d: Union[Array, int] = 51
    ) -> Array:
        if self.nvars() != 2:
            raise RuntimeError("nvars !=2.")
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        return X, Y, pts

    def plot_pdf(
        self,
        ax,
        plot_limits: Array,
        npts_1d: Union[Array, int] = 51,
        label: Optional[Array] = None,
        **kwargs,
    ):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot PDF when nvars >= 3.")
        if len(plot_limits) != self.nvars() * 2:
            raise ValueError("plot_limits has the wrong shape")
        if not isinstance(npts_1d, list):
            npts_1d = [npts_1d] * self.nvars()
        if label is None and self.nlabels() > 0:
            raise ValueError("Must specificy a label")
        if label is not None and label.shape != (self.nlabels(), 1):
            raise ValueError("label shape must be {(self.nlabels(), 1)}")
        if self.nvars() == 1:
            return self._plot_pdf_1d(ax, npts_1d, plot_limits, **kwargs)
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        if label is not None:
            pts = self.append_labels(pts, label)
        Z = self._bkd.detach(self._bkd.reshape(self.pdf(pts), X.shape))
        if kwargs.get("levels", None) is None:
            if ax.name != "3d":
                raise ValueError(
                    "levels not specified so trying to plot surface but not"
                    " given 3d axis"
                )
            return ax.plot_surface(X, Y, Z, **kwargs)
        return ax.contourf(X, Y, Z, **kwargs)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nlabels={2}, nlayers={3})".format(
            self.__class__.__name__,
            self.nvars(),
            self.nlabels(),
            len(self._layers),
        )


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

    def set_weights(self, weights: Array):
        if weights.shape != (self.get_samples().shape[1], 1):
            raise ValueError(
                "weights.shape was {0} but must be {1}".format(
                    weights.shape, (self.get_samples().shape[1], 1)
                )
            )
        self._weights = weights

    def get_weights(self) -> Array:
        if hasattr(self, "_weights"):
            return self._weights
        nsamples = self.get_samples().shape[1]
        return self._bkd.full((nsamples, 1), 1.0 / nsamples)

    def set_samples(self, samples: Array):
        self._samples = samples

    def get_samples(self) -> Array:
        if not hasattr(self, "_samples"):
            raise ValueError("must call set_samples()")
        return self._samples

    def _values(self, active_opt_params: Array) -> Array:
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        return -self.get_weights().T @ self._flow.logpdf(self._samples)

    def _jacobian(self, active_opt_params: Array):
        return self._bkd.jacobian(
            lambda p: self._values(p[:, None])[:, 0], active_opt_params[:, 0]
        )

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def plot_cross_section(self, ax, id1: int, id2: int, npts_1d=51, **kwargs):
        bounds = self._flow._hyp_list.get_active_opt_bounds()
        plot_limits = self._bkd.hstack((bounds[id1], bounds[id2]))
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        active_opt_params = self._bkd.copy(
            self._flow._hyp_list.get_active_opt_params()
        )[:, None]
        active_pt = self._bkd.copy(active_opt_params[[id1, id2], 0])
        vals = []
        for pt in pts.T:
            active_opt_params[[id1, id2], 0] = pt
            vals.append(self._values(active_opt_params))
        Z = self._bkd.reshape(self._bkd.vstack(vals)[:, 0], X.shape)
        im = ax.contourf(X, Y, Z, **kwargs)
        ax.plot(*active_pt, "ko", ms=20)
        return im

    def get_all_variable_pairs(self) -> Array:
        variable_pairs = self._bkd.asarray(
            anova_level_indices(self.nvars(), 2)
        )
        # make first column values vary fastest so we plot lower triangular
        # matrix of subplots
        variable_pairs[:, 0], variable_pairs[:, 1] = (
            self._bkd.copy(variable_pairs[:, 1]),
            self._bkd.copy(variable_pairs[:, 0]),
        )
        return variable_pairs

    def plot_cross_sections(
        self,
        variable_pairs: List[Tuple[int, int]] = None,
        npts_1d=51,
        **kwargs,
    ):
        nfig_rows, nfig_cols = self.nvars(), self.nvars()
        if variable_pairs is None:
            variable_pairs = self.get_all_variable_pairs()
        if variable_pairs.shape[1] != 2:
            raise ValueError("Variable pairs has the wrong shape")
        variable_pairs = variable_pairs[28:30]
        fig, axs = plt.subplots(nfig_rows, nfig_cols, sharex="col")
        for ax_row in axs:
            for ax in ax_row:
                ax.axis("off")
        ims = []
        for ii, pair in enumerate(variable_pairs):
            print(f"plotting cross section {pair}")
            if pair[0] == pair[1]:
                continue
            im = self.plot_cross_section(
                axs[pair[0]][pair[1]], pair[0], pair[1]
            )
            ims.append(im)
        return axs, im


class RealNVPScalingConstraint(Constraint):
    def __init__(
        self,
        ntrain_samples: int,
        backend: BackendMixin,
        keep_feasible: bool = False,
    ):
        self._ntrain_samples = ntrain_samples
        super().__init__(None, keep_feasible, backend)

    def set_flow(self, flow: Flow):
        self._flow = flow
        self._bkd = self._flow._bkd

    def nvars(self) -> int:
        return self._flow._hyp_list.nactive_vars()

    def nqoi(self) -> int:
        return self._ntrain_samples * len(self._flow._layers)

    def _values(self, active_opt_params: Array) -> Array:
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        samples = self._bkd.copy(self._flow._loss._samples)
        layer_scales = []
        for layer in reversed(self._flow._layers):
            shapes = layer._shapes(samples[layer._mask_w_labels])
            # TODO I dont think this will work when n_transformed_variables > 1
            # likely need to flatten shapes
            layer_scales.append(shapes[:, layer._ntransformed_vars :].T)
            samples, layer_logdet = layer._map_to_latent(samples, True)
        layer_scales = self._bkd.hstack(layer_scales)
        # print(layer_scales.max(), "constraint")
        return layer_scales

    def jacobian_implemented(self) -> bool:
        return True

    def _jacobian_incomplete(self, active_opt_params: Array) -> Array:
        raise NotImplementedError
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        samples = self._bkd.copy(self._flow._loss._samples)
        layer_scale_jacs = []
        for layer in reversed(self._flow._layers):
            scales_jac = layer._shapes.basis()(samples[layer._mask_w_labels])
            print(scales_jac)
            # TODO need chain rule trough layers
            layer_scale_jacs.append(scales_jac)
            samples, layer_logdet = layer._map_to_latent(samples, True)
        layer_scale_jacs = self._bkd.vstack(layer_scale_jacs)
        import torch

        torch.set_printoptions(linewidth=1000)
        print(super()._jacobian(active_opt_params))
        assert self._bkd.allclose(
            layer_scale_jacs, super()._jacobian(active_opt_params)
        )
        return layer_scale_jacs

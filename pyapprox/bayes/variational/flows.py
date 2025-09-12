from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    IdentityHyperParameterTransform,
    LogHyperParameterTransform,
)


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
    def ntransformed_vars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _map_from_latent(self, usamples: Array, return_logdet: bool) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _map_to_latent(self, samples: Array, return_logdet: bool) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nlabels={2})".format(
            self.__class__.__name__, self.nvars(), self.nlabels()
        )


class ScaleAndShiftFlowLayer(FlowLayer):
    def __init__(
        self,
        nvars: int,
        nlabels: int,
        backend: BackendMixin,
        # scale: Array = 2.0,
        # shift: Array = -1.0,
        scale: Array = 1.0,
        shift: Array = -0.5,
        scale_labels: bool = False,
    ):
        super().__init__(nvars, nlabels, backend)
        scale_transform = LogHyperParameterTransform(backend=self._bkd)
        self._ntransfomed_vars = nvars + nlabels if scale_labels else nvars
        self._scale = HyperParameter(
            "scale",
            self._ntransfomed_vars,
            scale,
            (0.0, np.inf),
            scale_transform,
            fixed=True,
            backend=self._bkd,
        )
        shift_transform = IdentityHyperParameterTransform(backend=self._bkd)
        self._shift = HyperParameter(
            "shift",
            self._ntransfomed_vars,
            shift,
            (-np.inf, np.inf),
            shift_transform,
            fixed=True,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._scale, self._shift])

    def ntransformed_vars(self) -> int:
        return self._nvars - self._nlabels

    def _map_to_latent(self, samples: Array, return_logdet: bool) -> Array:
        """
        Transform samples from original space to latent space [0,1]
        then scale and shift according to hyperparameters.
        """
        if not hasattr(self, "_lower_bounds"):
            self._lower_bounds = self._bkd.min(samples, axis=1)[
                : self._ntransfomed_vars
            ]
            upper_bounds = self._bkd.max(samples, axis=1)[
                : self._ntransfomed_vars
            ]
            self._ranges = upper_bounds - self._lower_bounds

        # map samples to [0,1]
        usamples1 = self._bkd.copy(samples)
        usamples1[: self._ntransfomed_vars] = (
            1.0
            / self._ranges[:, None]
            * (samples[: self._ntransfomed_vars] - self._lower_bounds[:, None])
        )
        # scale and shift usamples from [0,1] to best learned canonical domain
        # return self._scale.get_values()

        # Following works with torch autograd but is slower than below
        # Since we assume that scale and shift are fixed we can comment it out
        # usamples = self._bkd.empty(usamples1.shape)
        # usamples[: self._ntransfomed_vars] = (
        #     self._scale.get_values()[0] * usamples1[: self._ntransfomed_vars]
        #     + self._shift.get_values()[:, None]
        # )
        # usamples[self._ntransfomed_vars :] = usamples1[
        #     self._ntransfomed_vars :
        # ]

        # Following DOES NOT work with torch autograd but is fastesr than above
        # Since we assume that scale and shift are fixed we can use it here
        usamples = usamples1
        usamples[: self._ntransfomed_vars] = (
            self._scale.get_values()[:, None]
            * usamples[: self._ntransfomed_vars]
            + self._shift.get_values()[:, None]
        )
        if not return_logdet:
            return usamples
        # logdet should never involve label variable so use [:nvars]

        # Works with torch autograd
        jacobian_logdet = self._bkd.tile(
            self._bkd.sum(
                self._bkd.log(self._scale.get_values() / self._ranges)[
                    : self._nvars
                ]
            ),
            (samples.shape[1],),
        )

        # Does not works with torch autograd
        # jacobian_logdet = self._bkd.full(
        #     (samples.shape[1],),
        #     self._bkd.sum(
        #         self._bkd.log(self._scale.get_values() / self._ranges)[
        #             : self._nvars
        #         ]
        #     ),
        # )
        return usamples, jacobian_logdet

    def _map_from_latent(self, usamples: Array, return_logdet: bool) -> Array:
        """
        Transform samples from latent space 0,1], scaled and shifted
        according to hyperparameters, back to original space.
        """
        # Ensure lower_bounds and scale are computed
        if self._lower_bounds is None:
            raise ValueError("self._map_from_latent must be called first.")

        # scale and shift usamples from best learned canonical domain to [0,1]
        usamples = self._bkd.copy(usamples)
        usamples[: self._ntransfomed_vars] = (
            1.0
            / self._scale.get_values()[:, None]
            * (
                usamples[: self._ntransfomed_vars]
                - self._shift.get_values()[:, None]
            )
        )

        # Compute original samples from latent samples in [0,1]
        samples = self._bkd.copy(usamples)
        # samples = usamples
        samples[: self._ntransfomed_vars] = (
            self._ranges[:, None] * usamples[: self._ntransfomed_vars]
            + self._lower_bounds[:, None]
        )
        if not return_logdet:
            return samples

        # Works with torch autograd
        jacobian_logdet = self._bkd.tile(
            self._bkd.sum(
                -self._bkd.log(self._scale.get_values() / self._ranges)[
                    : self._nvars
                ]
            ),
            (samples.shape[1],),
        )

        # Does not work with torch autograd
        # jacobian_logdet = self._bkd.full(
        #     (samples.shape[1],),
        #     self._bkd.sum(
        #         -self._bkd.log(self._scale.get_values() / self._ranges)[
        #             : self._nvars
        #         ]
        #     ),
        # )
        return samples, jacobian_logdet


class RealNVPLayer(FlowLayer):
    def __init__(self, nvars, shapes, mask: Array, nlabels: int = 0):
        super().__init__(nvars, nlabels, shapes._bkd)
        if mask.shape != (self.nvars(),):
            raise ValueError(f"{mask.shape=} must be {(self.nvars(),)}")
        if mask.dtype != self._bkd.bool_type():
            raise ValueError("mask must be a boolean array")
        # mask defined over just nvars
        self._mask = mask
        # mask complement defined over just nvars
        self._mask_complement = ~mask
        self._ntransformed_vars = self._bkd.where(self._mask_complement)[
            0
        ].shape[0]

        # shapes is assumed to have signature=shapes([samples, labels])
        # mask_w_labels defined over nvars and labels (TODO change name as it
        # is confusing)
        self._mask_w_labels = self._bkd.hstack(
            (self._mask, self._bkd.ones((nlabels,), dtype=bool))
        )
        # mask_wo_labels defined over nvars and labels (TODO change name as it
        # is confusing)
        self._mask_wo_labels = self._bkd.hstack(
            (self._mask, self._bkd.zeros((nlabels,), dtype=bool))
        )
        # mask_complement_w_labels defined over nvars and labels
        self._mask_complement_w_labels = self._bkd.hstack(
            (self._mask_complement, self._bkd.ones((nlabels,), dtype=bool))
        )
        # mask_complement_wo_labels defined over nvars and labels
        self._mask_complement_wo_labels = self._bkd.hstack(
            (self._mask_complement, self._bkd.zeros((nlabels,), dtype=bool))
        )
        self._shapes = shapes
        self._hyp_list = self._shapes._hyp_list

    def ntransformed_vars(self) -> int:
        return self._ntransformed_vars

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
        if not self._bkd.all(self._bkd.isfinite(self._bkd.exp(scale))):
            II = self._bkd.where(~self._bkd.isfinite(self._bkd.exp(scale)))[1]
            print(self._bkd.exp(scale[:, II]))
            print(scale[:, II], "s")
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

    def _jacobian_scale_wrt_scale_hyperparameters(
        self, samples: Array
    ) -> Array:
        return self._shapes.basis()(samples[self._mask_w_labels])

    def _jacobian_scale_wrt_hyperparameters(self, samples: Array) -> Array:
        scale_jac = self._jacobian_scale_wrt_scale_hyperparameters(samples)
        jac = self._bkd.zeros((scale_jac.shape[0], 2 * scale_jac.shape[1]))
        jac[:, 1::2] = scale_jac
        return jac

    def _jacobian_scale_wrt_samples(self, samples: Array) -> Array:
        bexp = self._shapes
        active_samples = samples[self._mask_w_labels]
        basis_jacobian_wrt_active_samples_wo_labels = bexp.basis().jacobian(
            active_samples
        )[..., : self._ntransformed_vars]
        scale_coefs = bexp.get_coefficients()[:, self._ntransformed_vars :]
        jac = self._bkd.einsum(
            "ijk,jl->ikl",
            basis_jacobian_wrt_active_samples_wo_labels,
            scale_coefs,
        )
        # flatten assumes that scale for each variable returned by
        # scale are concatenated together

        # jac = self._bkd.diag(jac.flatten())

        # return as diagonal to speed up its application to a matrix
        jac = jac.flatten()[:, None]

        # print(jac)
        # def fun(x):
        #     x = x[None, :]
        #     x = self._bkd.vstack((x, active_samples[x.shape[0] :]))
        #     return bexp(x)[:, self._ntransformed_vars :].T[0]

        # jac1 = self._bkd.jacobian(fun, active_samples[self._mask][0])
        # print(self._bkd.diag(jac1))
        # assert self._bkd.allclose(jac, jac1)
        return jac

    def _jacobian_samples_wrt_hyperparameters(self, samples: Array) -> Array:
        shapes = self._shapes(samples[self._mask_w_labels])
        scale = shapes[:, self._ntransformed_vars :].T
        print(
            self._shapes.basis()(samples[self._mask_w_labels]).shape,
            self._bkd.exp(-scale).T.shape,
        )
        jac_wrt_shift_params = -self._bkd.exp(-scale).T * self._shapes.basis()(
            samples[self._mask_w_labels]
        )
        shift = shapes[:, : self._ntransformed_vars].T
        jac_wrt_scale_params = (
            -(samples[self._mask_complement_wo_labels] - shift)
            * self._bkd.exp(-scale)
        ).T * self._jacobian_scale_wrt_scale_hyperparameters(samples)
        jac = self._bkd.empty(
            (
                jac_wrt_shift_params.shape[0],
                jac_wrt_shift_params.shape[1] + jac_wrt_scale_params.shape[1],
            )
        )
        jac[:, ::2] = jac_wrt_shift_params
        jac[:, 1::2] = jac_wrt_scale_params

        # def fun(p):
        #     self._shapes._hyp_list.set_active_opt_params(p)
        #     samples1, layer_logdet = self._map_to_latent(samples, True)
        #     return samples1

        # # assert False
        # jac1 = self._bkd.jacobian(
        #     fun, self._shapes._hyp_list.get_active_opt_params()
        # )[
        #     0
        # ]  # zero works only for shapes with one qoi
        # assert self._bkd.allclose(jac, jac1)
        return jac


class Flow:
    def __init__(
        self, latent_variable: JointVariable, layers: List[FlowLayer]
    ):
        if not isinstance(latent_variable, JointVariable):
            raise ValueError("latent_variable must be a JointVariable")
        self._bkd = latent_variable._bkd
        self._latent_variable = latent_variable
        if layers[0].nvars() != latent_variable.nvars():
            raise ValueError(
                "layers must have the same number of variables as latent_variable"
            )
        for ii, layer in enumerate(layers):
            if not isinstance(layer, FlowLayer):
                raise ValueError("layer must be an intance of FlowLayer")
            if layer.nvars() != layers[0].nvars():
                raise ValueError(
                    "layers must have the same number of variables"
                )
            if layer.nlabels() != layers[0].nlabels():
                raise ValueError("layers must have the same number of labels")

        self._layers = layers
        import torch

        torch.autograd.set_detect_anomaly(True)
        self._hyp_list = sum(
            layer._hyp_list
            for layer in layers
            if not isinstance(layer, ScaleAndShiftFlowLayer)
        )
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
        for ii, layer in enumerate(layers):
            if not isinstance(layer, (RealNVPLayer, ScaleAndShiftFlowLayer)):
                raise ValueError(
                    "layer must be an intance of RealNVPLayer or "
                    "ScaleAndShiftFlowLayer"
                )
            if ii != len(layers) - 1 and isinstance(
                layer, ScaleAndShiftFlowLayer
            ):
                # this error is because analytical gradient computation
                # depends on this assumption
                raise ValueError(
                    "ScaleAndShiftFlowLayer only allowed for last layer"
                )

            if (
                isinstance(layer, RealNVPLayer)
                and ii > 0
                and layer._shapes.nvars()
                != layers[ii - 1].ntransformed_vars() + layers[0].nlabels()
            ):
                raise ValueError(
                    "shapes must be a model with model.nvars()={0}".format(
                        layers[ii - 1].ntransformed_vars()
                        + layers[0].nlabels()
                    )
                    + f" but {layer._shapes.nvars()=}"
                )
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
        # print(self._logpdf_jacobian(active_opt_params))
        # print(
        #     self._bkd.jacobian(
        #         lambda p: self._values(p[:, None])[:, 0],
        #         active_opt_params[:, 0],
        #     )
        # )
        # print(
        #     self._logpdf_jacobian(active_opt_params)
        #     - self._bkd.jacobian(
        #         lambda p: self._values(p[:, None])[:, 0],
        #         active_opt_params[:, 0],
        #     ),
        #     "DIFF",
        # )
        assert self._bkd.allclose(
            self._logpdf_jacobian(active_opt_params),
            self._bkd.jacobian(
                lambda p: self._values(p[:, None])[:, 0],
                active_opt_params[:, 0],
            ),
        )
        return self._logpdf_jacobian(active_opt_params)
        # return self._bkd.jacobian(
        #     lambda p: self._values(p[:, None])[:, 0], active_opt_params[:, 0]
        # )

    # def _jacobian_pass(self, active_opt_params: Array) -> Array:
    #     self._flow._hyp_list.set_active_opt_params(active_opt_params)
    #     samples = self._bkd.copy(self._samples)
    #     for layer in reversed(self._flow._layers):
    #         samples, layer_logdet = layer._map_to_latent(samples, True)
    #         # dv2dc2_custom = layer._jacobian_scale_wrt_hyperparameters(samples2)
    #         # dx1dc2_custom = layer._jacobian_samples_wrt_hyperparameters(samples2)
    #     return samples

    def _x(self, ii, samples):
        samples = self._bkd.copy(samples)
        for layer in reversed(self._flow._layers[ii:]):
            samples, layer_logdet = layer._map_to_latent(samples, True)
        return layer._map_to_latent(samples, True)[0]

    def _dxdh(self, p):
        self._flow._hyp_list.set_active_opt_params(p)
        return self._x(self._samples)

    def _layer_logdet_jac(self, ii, active_opt_params: Array) -> Array:
        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            samples = self._bkd.copy(self._samples)
            for layer in reversed(self._flow._layers[ii:]):
                samples, layer_logdet = layer._map_to_latent(samples, True)
            return layer._map_to_latent(samples, True)[1]

        # def _dsdx(samples):
        #     shapes = self._flow._layers[ii]._shapes(
        #         samples[layer._mask_w_labels]
        #     )
        #     scale = shapes[:, layer._ntransformed_vars :].T
        #     return scale

        # # l: logdet
        # # s: scale
        # # x: samples
        # # h: hyperparameters

        # # def _dxdh(p):
        # #     self._flow._hyp_list.set_active_opt_params(p)
        # #     samples = self._bkd.copy(self._samples)
        # #     for layer in reversed(self._flow._layers[ii:]):
        # #         samples = layer._map_to_latent(samples, False)
        # #     shapes = self._flow._layers[ii]._shapes(
        # #         samples[layer._mask_w_labels]
        # #     )
        # #     scale = shapes[:, layer._ntransformed_vars :].T
        # #     print(scale.shape)
        # #     return self._bkd.sum(scale, axis=0)

        # if ii == 1:
        #     layer = self._flow._layers[ii]
        #     usamples = _dxdh(active_opt_params[:, 0])
        #     dxdh = self._bkd.jacobian(_dxdh, active_opt_params[:, 0]).reshape(
        #         (20, 32)
        #     )
        #     dsdx = self._bkd.jacobian(_dsdx, usamples).reshape((2, 5, 20))
        #     dldx = self._bkd.sum(dsdx, axis=0)
        #     # [ntransformed_vars*nsamples]
        #     print(self._samples.shape, "x")
        #     print(dxdh, "dxdh")
        #     print(dsdx.shape, "dsdx")
        #     print(dldx.shape, "dldx")
        #     print(dldx @ dxdh)
        #     print(self._bkd.jacobian(fun, active_opt_params[:, 0]))
        #     assert False
        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _logdet_jac(self, active_opt_params: Array) -> Array:
        samples = self._bkd.copy(self._samples)
        logdet_jac = 0.0
        ii = len(self._flow._layers) - 1
        for layer in reversed(self._flow._layers):
            logdet_jac += self._layer_logdet_jac(ii, active_opt_params)
            samples = layer._map_to_latent(samples, False)
            ii -= 1
        return logdet_jac

    def _dxdp(self, p):
        self._flow._hyp_list.set_active_opt_params(p)
        samples = self._bkd.copy(self._samples)
        usamples = self._flow._map_to_latent(samples)
        return usamples

    def _dsdp(self, layer, active_opt_params, samples):
        # derivative of exp(-scale)
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            scale = bexp(active_samples)[:, layer._ntransformed_vars :].T
            return self._bkd.exp(-scale)

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _dmdp(self, layer, active_opt_params, samples):
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            shift = bexp(active_samples)[:, : layer._ntransformed_vars].T
            return shift

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _dsdx(self, layer, active_opt_params, samples):
        # jacobian of exp(-scale) with respect to flattened samples
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(x):
            self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
            scale = bexp(x.reshape(active_samples.shape))[
                :, layer._ntransformed_vars :
            ].T
            return self._bkd.exp(-scale)

        jac = self._bkd.jacobian(fun, active_samples.flatten())
        # jac is mainly zeros ith row contains nactive sample nonzero
        #  values at i::nsamples
        return jac

    def _dmdx(self, layer, active_opt_params, samples):
        # jacobian of shift mu with respect to flattened samples
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(x):
            # self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
            shift = bexp(x.reshape(active_samples.shape))[
                :, : layer._ntransformed_vars
            ].T
            return shift

        jac = self._bkd.jacobian(fun, active_samples.flatten())
        return jac
        # return self._bkd.stack([self._bkd.diag(mat) for mat in jac], axis=0)

    def _layer_active_hyperparam_indices(self):
        layer_active_hyperparam_indices = []
        lb = 0
        for xlayer in self._flow._layers:
            ub = lb + xlayer._hyp_list.nactive_vars()
            layer_active_hyperparam_indices.append(self._bkd.arange(lb, ub))
            lb = ub
        return layer_active_hyperparam_indices

    def _dxdp_layer(self, layer, ii, samples, p):

        import torch

        torch.set_printoptions(linewidth=1000)  # , threshold=10000)

        def f(p):
            self._flow._hyp_list.set_active_opt_params(p)
            # will contain jac for both masked and mask complement dims
            kk = 0
            usamples = self._bkd.copy(self._samples)
            for xlayer in reversed(self._flow._layers):
                if kk <= ii:
                    usamples = xlayer._map_to_latent(usamples, False)
                else:
                    break
                kk += 1
            return usamples

        autojac = self._bkd.jacobian(f, p[:, 0])

        # TODO do this only once every time loss is created (as long as active params)
        # does not change
        nlayers = len(self._flow._layers)
        layer_active_hyperparam_indices = (
            self._layer_active_hyperparam_indices()
        )

        # ii: reverse layer ordering, i.e. output layer is layer 0
        # s: exp(-sigma) , m: mu (shift) p: params, x: samples
        shapes = layer._shapes(samples[layer._mask_w_labels])
        shift = shapes[:, : layer._ntransformed_vars].T
        scale = shapes[:, layer._ntransformed_vars :].T
        delta = self._bkd.exp(-scale)
        diff = samples[layer._mask_complement_wo_labels] - shift

        jac = self._bkd.zeros(
            (
                self._flow.nvars() + self._flow.nlabels(),
                samples.shape[1],
                self._flow._hyp_list.nactive_vars(),
            )
        )
        # compute jacobians with respect to hyperparameters of current layer
        dsdp = self._dsdp(layer, p, samples)
        dmdp = self._dmdp(layer, p, samples)
        jac_ii = (diff[..., None] * dsdp - delta[..., None] * dmdp)[
            ..., layer_active_hyperparam_indices[nlayers - ii - 1]
        ]
        print("########", ii, layer._mask)
        # numpy and torch reduce singleton dimension of when either of the
        # index arrays has only one entry
        # jac[indices1, :, indices2]
        # so do the following
        print(layer_active_hyperparam_indices)
        lb = layer_active_hyperparam_indices[nlayers - ii - 1][0]
        ub = layer_active_hyperparam_indices[nlayers - ii - 1][-1] + 1
        jac[layer._mask_complement_wo_labels, :, lb:ub] = jac_ii

        # compute jacobians with respect to hyperparameters of each parent
        # layer (layers closer to output layer are parents)
        # todo allow for multiple fixed layers for now only allow
        # for first layer to have no active hyperparams
        for jj in range(ii):
            if len(layer_active_hyperparam_indices[nlayers - jj - 1]) == 0:
                continue
            lb = layer_active_hyperparam_indices[nlayers - jj - 1][0]
            ub = layer_active_hyperparam_indices[nlayers - jj - 1][-1] + 1
            print(layer._mask)
            print(
                self._sample_jacobians[ii - 1][
                    layer._mask_w_labels, :, lb:ub
                ].shape
            )
            print(jac[layer._mask_w_labels, :, lb:ub].shape)
            jac[layer._mask_wo_labels, :, lb:ub] = self._sample_jacobians[
                ii - 1
            ][layer._mask_wo_labels, :, lb:ub]

            # grad of shift with respect to flattened active samples
            dmdx = self._dmdx(layer, p, samples)
            # grad of scale with respect to flattened active samples
            dsdx = self._dsdx(layer, p, samples)
            # grad of ALL unflattened samples
            dxdp_jj = self._sample_jacobians[ii - 1][
                ..., layer_active_hyperparam_indices[nlayers - jj - 1]
            ]
            shape = (
                *diff.shape,
                self._bkd.where(layer._mask_w_labels)[0].shape[0],
                samples.shape[1],
            )
            if True:
                print("@@@@@@", jj)
                print(samples.shape, "samples")
                print(diff.shape, "diff")
                print(delta.shape, "delta")
                print(dsdx.shape, "dsdx")
                print(dxdp_jj.shape, "dxdp")
                print(shape)
                # print(dxdp_jj[layer._mask])
            # scale and shift depend of samples[layer._mask] at this layer
            # thus they depend on samples[prev_layer._mask_complement]
            # at previous layer
            jac_jj = self._bkd.einsum(
                "dnkn,knp->dnp",
                self._bkd.reshape(diff[..., None] * dsdx, shape),
                dxdp_jj[layer._mask_w_labels],
            ) - self._bkd.einsum(
                "dnkn,knp->dnp",
                self._bkd.reshape(delta[..., None] * dmdx, shape),
                dxdp_jj[layer._mask_w_labels],
            )
            print(jj)
            print(self._sample_jacobians[0])
            print(dxdp_jj[layer._mask_complement_w_labels], "dxdp_jj_active")
            jac_jj += (
                delta[..., None] * dxdp_jj[layer._mask_complement_wo_labels]
            )
            jac[layer._mask_complement_wo_labels, :, lb:ub] = jac_jj

        if ii >= 1:

            print(ii)
            print("$")
            print(jac.detach(), jac.shape, "myjac")
            print("$")
            print(autojac, autojac.shape, "autojac")
            print(autojac - jac.detach())
            # print(self._bkd.jacobian(f, p[:, 0]))
        assert self._bkd.allclose(jac, autojac), ii
        return jac

    def _dxdp_custom(self, p):
        samples = self._bkd.copy(self._samples)
        self._flow._hyp_list.set_active_opt_params(p[:, 0])
        ii = 0  # counter for layers with hyperparameters
        self._sample_jacobians = []
        for layer in reversed(self._flow._layers):
            if isinstance(layer, ScaleAndShiftFlowLayer):
                # do not increment ii because this layer does not have
                # tunable hyperparameters
                self._sample_jacobians.append(None)
                samples = layer._map_to_latent(samples, False)
                ii += 1
                continue

            print("L", layer)
            jac = self._dxdp_layer(layer, ii, samples, p)
            self._sample_jacobians.append(jac)
            samples = layer._map_to_latent(samples, False)
            ii += 1

        return self._sample_jacobians[-1]

    def _logpdf_jac(self, active_opt_params: Array) -> Array:
        def fun(p):
            usamples = self._dxdp(p)
            return self._flow._latent_variable.logpdf(
                usamples[: self._flow.nvars()]
            )[:, 0]

        # jacobian of log pdf l with respect to samples x
        # usamples can be stored so it is only computed once
        usamples = self._dxdp(active_opt_params[:, 0])
        dldx = self._flow._latent_variable.logpdf_jacobian(
            usamples[: self._flow.nvars()]
        )
        # jacobian of samples x with respect to parameters p
        dxdp = self._dxdp_custom(active_opt_params)[: self._flow.nvars()]
        # dxdp = self._bkd.jacobian(self._dxdp, active_opt_params[:, 0])
        # d: nvars, n: nsamples, p: nhyperparams
        dldx = self._bkd.reshape(dldx, (dxdp.shape[0], dxdp.shape[1]))
        return self._bkd.einsum("dn, dnp -> np", dldx, dxdp)
        # return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _logpdf_jacobian(self, active_opt_params: Array):
        jac = self._logdet_jac(active_opt_params) + self._logpdf_jac(
            active_opt_params
        )
        return -self.get_weights().T @ jac

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    # def apply_hessian_implemented(self) -> bool:
    #     return self._bkd.hvp_implemented()

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
        return self._ntrain_samples * sum(
            layer._ntransformed_vars for layer in reversed(self._flow._layers)
        )

    def _scales_values(self, active_opt_params: Array) -> Array:
        self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
        samples = self._bkd.copy(self._flow._loss._samples)
        layer_scales = []
        for layer in reversed(self._flow._layers):
            shapes = layer._shapes(samples[layer._mask_w_labels])
            # TODO I dont think this will work when n_transformed_variables > 1
            # likely need to flatten shapes
            layer_scales.append(
                shapes[:, layer._ntransformed_vars :].T.flatten()
            )
            samples, layer_logdet = layer._map_to_latent(samples, True)
        layer_scales = self._bkd.hstack(layer_scales)
        # print(layer_scales.max(), "constraint")
        return layer_scales[None, :]

    def _values(self, active_opt_params: Array) -> Array:
        scale_vals = self._scales_values(active_opt_params)
        return scale_vals  # pairs with _all_scale_values_jacobian nqoi
        # ensure average scale is reasonable
        # return (
        #     scale_vals.sum(axis=1)[None, :] / scale_vals.shape[1]
        # )  # needs nqoi=1

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _dv1dx1(self, samples):
        layer = self._flow._layers[0]
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(x):
            x = x[None, :]
            x = self._bkd.vstack((x, active_samples[x.shape[0] :]))
            return bexp(x)[:, layer._ntransformed_vars :].T[0]

        return self._bkd.jacobian(fun, active_samples[layer._mask][0])

    def _dx1dc2(self, active_opt_params):
        layer = self._flow._layers[-1]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            samples1, layer_logdet = layer._map_to_latent(
                self._flow._loss._samples, True
            )
            return samples1

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _dvdc(self, layer, active_opt_params, samples):
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            return bexp(active_samples)[:, layer._ntransformed_vars :].T[0]

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _jacobian(self, active_opt_params: Array) -> Array:
        return self._all_scale_values_jacobian(active_opt_params)

    def _all_scale_values_jacobian(self, active_opt_params: Array) -> Array:
        # custom derivatives will likely only work in 2D
        layer2 = self._flow._layers[1]
        samples2 = self._flow._loss._samples
        layer = self._flow._layers[-1]
        samples1, layer_logdet = layer._map_to_latent(
            self._flow._loss._samples, True
        )
        dv2dc2_custom = layer2._jacobian_scale_wrt_hyperparameters(samples2)
        dx1dc2_custom = layer2._jacobian_samples_wrt_hyperparameters(samples2)
        layer1 = self._flow._layers[0]
        dv1dx1_custom = layer1._jacobian_scale_wrt_samples(samples1)
        dv1dc1_custom = layer1._jacobian_scale_wrt_hyperparameters(samples1)
        dv1dc2_custom = dv1dx1_custom * dx1dc2_custom
        # import torch

        # torch.set_printoptions(linewidth=1000)
        # dv2dc2 = self._dvdc(layer2, active_opt_params, samples2)[
        #     :, active_opt_params.shape[0] // 2 :
        # ]
        # dv1dc1 = self._dvdc(layer1, active_opt_params, samples1)[
        #     :, : active_opt_params.shape[0] // 2
        # ]
        # dv1dc2 = (self._dv1dx1(samples1) @ self._dx1dc2(active_opt_params)[0])[
        #     :, active_opt_params.shape[0] // 2 :
        # ]
        # assert self._bkd.allclose(dv2dc2_custom, dv2dc2)
        # assert self._bkd.allclose(
        #     self._bkd.diag(dv1dx1_custom[:, 0]), self._dv1dx1(samples1)
        # )
        # assert self._bkd.allclose(
        #     dx1dc2_custom,
        #     self._dx1dc2(active_opt_params)[0][
        #         :, active_opt_params.shape[0] // 2 :
        #     ],
        # )
        # assert self._bkd.allclose(dv1dc1_custom, dv1dc1)
        # assert self._bkd.allclose(dv1dc2_custom, dv1dc2)
        jacobian = self._bkd.block(
            [
                [self._bkd.zeros(dv2dc2_custom.shape), dv2dc2_custom],
                [dv1dc1_custom, dv1dc2_custom],
            ]
        )
        # super_jac = super()._jacobian(active_opt_params)
        # assert self._bkd.allclose(jacobian, super_jac)
        return jacobian

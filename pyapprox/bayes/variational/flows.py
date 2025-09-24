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
    ChainRuleArrays,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import (
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.surrogates.affine.multiindex import anova_level_indices
from pyapprox.surrogates.affine.basisexp import BasisExpansion
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


class RealNVPShapesMapMixin:

    def _jacobian_shift_wrt_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError

        active_samples_and_labels = samples[mask_w_labels]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o

        def fun(p):
            self._hyp_list.set_active_opt_params(p)
            return self(active_samples_and_labels)[:, :ninactive_vars]

        return self._bkd.jacobian(fun, active_opt_params)

    def _jacobian_scale_wrt_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError

        active_samples_and_labels = samples[mask_w_labels]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o

        def fun(p):
            self._hyp_list.set_active_opt_params(p)
            return self(active_samples_and_labels)[:, ninactive_vars:]

        return self._bkd.jacobian(fun, active_opt_params)

    def _jacobian_scale_wrt_samples(
        self, samples: Array, mask: Array, mask_w_labels: Array
    ) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError

        active_samples_and_labels = samples[mask_w_labels]
        active_samples = samples[mask]  # shape (n_i, n)
        labels = active_samples_and_labels[active_samples.shape[0] :]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o

        def fun(x):
            x = self._bkd.vstack((x.T, labels))
            return self(x)[:, ninactive_vars:]

        # shape (N, n_o, n_i)
        return self._bkd.sum(self._bkd.jacobian(fun, active_samples.T), axis=2)

    def _jacobian_shift_wrt_samples(
        self, samples: Array, mask: Array, mask_w_labels: Array
    ) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError

        active_samples_and_labels = samples[mask_w_labels]
        active_samples = samples[mask]  # shape (n_i, n)
        labels = active_samples_and_labels[active_samples.shape[0] :]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o

        def fun(x):
            x = self._bkd.vstack((x.T, labels))
            return self(x)[:, :ninactive_vars]

        # shape (N, n_o, n_i)
        return self._bkd.sum(self._bkd.jacobian(fun, active_samples.T), axis=2)


class RealNVPShapesBasisExpansionMap(BasisExpansion, RealNVPShapesMapMixin):
    def __init__(self, bexp: BasisExpansion):
        super().__init__(
            bexp.basis(),
            bexp._solver,
            bexp.nqoi(),
            bexp._coef.get_bounds(),
            fixed=bexp._coef.nactive_vars() == 0,
        )
        self.set_coefficients(self._bkd.copy(bexp.get_coefficients()))

    def _zero_compressed_jac(self, n: int, o: int, p: int) -> Array:
        if not hasattr(self, "_zeros_comp") or self._zeros_comp.shape != (
            n,
            o,
            p,
        ):
            # the nonzero entries will always be in the same place
            # so only need to create array once
            self._zeros_comp = self._bkd.zeros((n, o, p))
        return self._zeros_comp

    def _zero_shift_jac(self, n: int, o: int, p: int) -> Array:
        if not hasattr(self, "_zeros_shift") or self._zeros_shift.shape != (
            n,
            o,
            p,
        ):
            # the nonzero entries will always be in the same place
            # so only need to create array once
            self._zeros_shift = self._bkd.zeros((n, o, p))
        return self._zeros_shift

    def _zero_scale_jac(self, n: int, o: int, p: int) -> Array:
        if not hasattr(self, "_zeros_scale") or self._zeros_scale.shape != (
            n,
            o,
            p,
        ):
            # the nonzero entries will always be in the same place
            # so only need to create array once
            self._zeros_scale = self._bkd.zeros((n, o, p))
        return self._zeros_scale

    def _jacobian_shift_wrt_shift_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        active_samples = samples[mask_w_labels]
        basis_mat = self.basis()(active_samples)
        n = active_samples.shape[1]
        o = self._bkd.where(~mask)[0].shape[0]
        # only half the active params are for the scale
        # jac = self._bkd.zeros((n, o, active_opt_params.shape[0] // 2))
        jac = self._zero_compressed_jac(n, o, active_opt_params.shape[0] // 2)
        lb = 0
        for ii in range(o):
            ub = lb + basis_mat.shape[1]
            jac[:, ii, lb:ub] = basis_mat
            lb = ub
        return jac

    def _jacobian_scale_wrt_scale_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        active_samples = samples[mask_w_labels]
        basis_mat = self.basis()(active_samples)
        n = active_samples.shape[1]
        o = self._bkd.where(~mask)[0].shape[0]
        # only half the active params are for the scale
        # jac = self._bkd.zeros((n, o, active_opt_params.shape[0] // 2))
        jac = self._zero_compressed_jac(n, o, active_opt_params.shape[0] // 2)
        lb = 0
        for ii in range(o):
            ub = lb + basis_mat.shape[1]
            jac[:, ii, lb:ub] = basis_mat
            lb = ub
        return jac

    def _jacobian_shift_wrt_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        shift_jac = self._jacobian_shift_wrt_shift_hyperparameters(
            samples, active_opt_params, mask, mask_w_labels
        )
        # jac = self._bkd.zeros((*shift_jac.shape[:2], 2 * shift_jac.shape[2]))
        jac = self._zero_shift_jac(
            *shift_jac.shape[:2], 2 * shift_jac.shape[2]
        )
        o = self._bkd.where(~mask)[0].shape[0]
        length = shift_jac.shape[2] // o
        for ii in range(o):
            jac[:, ii, ii :: 2 * o] = shift_jac[
                :, ii, ii * length : (ii + 1) * length
            ]
        return jac

    def _jacobian_scale_wrt_hyperparameters(
        self,
        samples: Array,
        active_opt_params: Array,
        mask: Array,
        mask_w_labels: Array,
    ) -> Array:
        scale_jac = self._jacobian_scale_wrt_scale_hyperparameters(
            samples, active_opt_params, mask, mask_w_labels
        )
        # jac = self._bkd.zeros((*scale_jac.shape[:2], 2 * scale_jac.shape[2]))
        jac = self._zero_scale_jac(
            *scale_jac.shape[:2], 2 * scale_jac.shape[2]
        )
        o = self._bkd.where(~mask)[0].shape[0]
        length = scale_jac.shape[2] // o
        for ii in range(o):
            jac[:, ii, o + ii :: 2 * o] = scale_jac[
                :, ii, ii * length : (ii + 1) * length
            ]
        return jac

    def _jacobian_scale_wrt_samples(
        self, samples: Array, mask: Array, mask_w_labels: Array
    ) -> Array:
        active_samples = samples[mask_w_labels]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o
        return self._many_jacobian(active_samples)[:, ninactive_vars:]

    def _jacobian_shift_wrt_samples(
        self, samples: Array, mask: Array, mask_w_labels: Array
    ) -> Array:
        active_samples = samples[mask_w_labels]
        ninactive_vars = self._bkd.where(~mask)[0].shape[0]  # n_o
        return self._many_jacobian(active_samples)[:, :ninactive_vars]


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

        if not isinstance(shapes, RealNVPShapesMapMixin):
            raise ValueError(
                "shapes must be an instance of RealNVPShapesMapMixin"
            )

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

    def _jacobian_scale_wrt_hyperparameters(self, samples: Array) -> Array:
        scale_jac = self._shapes._jacobian_scale_wrt_hyperparameters(
            samples,
            self._shapes._hyp_list.get_active_opt_params(),
            self._mask,
            self._mask_w_labels,
        )
        return scale_jac

    def _jacobian_scale_wrt_samples(self, samples: Array) -> Array:
        scale_jac = self._shapes._jacobian_scale_wrt_samples(
            samples, self._mask, self._mask_w_labels
        )
        return scale_jac

    def _jacobian_shift_wrt_hyperparameters(self, samples: Array) -> Array:
        return self._shapes._jacobian_shift_wrt_hyperparameters(
            samples,
            self._shapes._hyp_list.get_active_opt_params(),
            self._mask,
            self._mask_w_labels,
        )

    def _jacobian_shift_wrt_samples(self, samples: Array) -> Array:
        return self._shapes._jacobian_shift_wrt_samples(
            samples, self._mask, self._mask_w_labels
        )


class Flow(ABC):
    def __init__(self, source_variable: JointVariable):
        if not isinstance(source_variable, JointVariable):
            raise ValueError("source_variable must be a JointVariable")
        self._bkd = source_variable._bkd
        self._source_variable = source_variable
        self._loss = FlowLoss()
        self._loss.set_flow(self)

    def nvars(self) -> int:
        """
        Return the dimension of the (conditional) target variable.
        """
        return self._source_variable.nvars()

    @abstractmethod
    def nlabels(self) -> int:
        """
        Return the dimension of the conditioning labels.
        """
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, samples: Array) -> Array:
        raise NotImplementedError

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

    def append_labels(self, samples: Array, labels: Array):
        nsamples = samples.shape[1]
        if labels.shape[1] == 1:
            labels = self._bkd.tile(labels, (nsamples,))
        if labels.shape != (self.nlabels(), nsamples):
            raise ValueError(
                f"{labels.shape=} but must be " f"{(self.nlabels(), nsamples)}"
            )
        return self._bkd.vstack((samples, labels))

    @abstractmethod
    def _map_from_latent(self, usamples: Array) -> Array:
        raise NotImplementedError

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
        latent_samples = self._source_variable.rvs(nsamples)
        if labels is None and self.nlabels() > 0:
            raise ValueError("Must specificy labels")
        if labels is not None:
            latent_samples = self.append_labels(latent_samples, labels)
        return self._map_from_latent(latent_samples)[: self.nvars()]

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
        return "{0}(nvars={1}, nlabels={2})".format(
            self.__class__.__name__, self.nvars(), self.nlabels()
        )


class DiscreteFlow(Flow):
    def __init__(
        self, source_variable: JointVariable, layers: List[FlowLayer]
    ):
        if not isinstance(source_variable, JointVariable):
            raise ValueError("source_variable must be a JointVariable")
        self._bkd = source_variable._bkd
        self._source_variable = source_variable
        if layers[0].nvars() != source_variable.nvars():
            raise ValueError(
                "layers must have the same number of variables as source_variable"
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
        return logdet[:, None] + self._source_variable.logpdf(
            samples[: self.nvars()]
        )

    def _map_from_latent(self, usamples: Array) -> Array:
        samples = usamples
        for layer in self._layers:
            samples = layer._map_from_latent(samples, False)
        return samples

    def _map_to_latent(self, samples: Array) -> Array:
        for layer in reversed(self._layers):
            samples = layer._map_to_latent(samples, False)
        return samples

    def nlabels(self) -> int:
        """
        Return the dimension of the conditioning labels.
        """
        return self._layers[0].nlabels()


class RealNVP(DiscreteFlow):
    """
    Real Non Volume Preserving (RealNVP) Flow
    """

    def __init__(
        self, source_variable: JointVariable, layers: List[FlowLayer]
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
        super().__init__(source_variable, layers)

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nlabels={2}, nlayers={3})".format(
            self.__class__.__name__,
            self.nvars(),
            self.nlabels(),
            len(self._layers),
        )


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
        # assert self._bkd.allclose(
        #     self._logpdf_jacobian(active_opt_params),
        #     self._bkd.jacobian(
        #         lambda p: self._values(p[:, None])[:, 0],
        #         active_opt_params[:, 0],
        #     ),
        # )
        import time

        t0 = time.time()
        print("WARNING using autograd")
        auto_jac = self._bkd.jacobian(
            lambda p: self._values(p[:, None])[:, 0],
            active_opt_params[:, 0],
        )
        jac = auto_jac
        # jac = self._logpdf_jacobian(active_opt_params)
        print(time.time() - t0, "seconds")
        return jac

    def _layer_logdet_jac_auto(self, ii, active_opt_params: Array) -> Array:
        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            samples = self._bkd.copy(self._samples)
            for layer in reversed(self._flow._layers[ii:]):
                samples, layer_logdet = layer._map_to_latent(samples, True)
            return layer._map_to_latent(samples, True)[1]

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _layer_logdet_jac(self, ii, active_opt_params: Array) -> Array:
        # assumes that _jacobian_samples_wrt_hyperparameters has been called
        # scale_jacobians are in reverse order for layers, e.g. 2, 1, 0
        nlayers = len(self._scale_jacobians)
        jac = -self._bkd.sum(self._scale_jacobians[nlayers - ii - 1], axis=1)
        # autojac = self._layer_logdet_jac_auto(ii, active_opt_params)
        # assert self._bkd.allclose(jac, autojac)
        return jac

    def _logdet_jac(self, active_opt_params: Array) -> Array:
        samples = self._bkd.copy(self._samples)
        logdet_jac = 0.0
        ii = len(self._flow._layers) - 1
        for layer in reversed(self._flow._layers):
            if layer._hyp_list.nactive_vars() > 0:
                logdet_jac += self._layer_logdet_jac(ii, active_opt_params)
            samples = layer._map_to_latent(samples, False)
            ii -= 1
        return logdet_jac

    def _d_scale_d_layerp(self, layer, active_opt_params, samples):
        # derivative of scale only with respect to parameters
        # of current layer.
        # When using autograd we must pass in samples for the layer
        # rather than computing them from self._samples

        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            scale = bexp(active_samples)[:, layer._ntransformed_vars :].T
            return scale

        jac = self._bkd.jacobian(fun, active_opt_params[:, 0])
        import torch

        jac = torch.swapaxes(jac, 0, 1)
        return jac

    def _d_omega_d_layerp(self, layer, active_opt_params, samples):
        # derivative of omega=exp(-scale) only with respect to parameters
        # of current layer.
        # When using autograd we must pass in samples for the layer
        # rather than computing them from self._samples

        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            scale = bexp(active_samples)[:, layer._ntransformed_vars :].T
            return self._bkd.exp(-scale)

        jac = self._bkd.jacobian(fun, active_opt_params[:, 0])
        import torch

        jac = torch.swapaxes(jac, 0, 1)
        return jac

    def _d_shift_d_layerp(self, layer, active_opt_params, samples):
        # derivative of shift only with respect to parameters
        # of current layer.
        # When using autograd we must pass in samples for the layer
        # rather than computing them from self._samples
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            shift = bexp(active_samples)[:, : layer._ntransformed_vars].T
            return shift

        jac = self._bkd.jacobian(fun, active_opt_params[:, 0])
        import torch

        return torch.swapaxes(jac, 0, 1)

    def _domega_dx(self, layer, active_opt_params, samples):
        # jacobian of exp(-scale) with respect to flattened samples
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(x):
            self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
            scale = bexp(x.T)[:, layer._ntransformed_vars :].T
            return self._bkd.exp(-scale)

        jac = self._bkd.jacobian(fun, active_samples.T)
        # jac is mainly zeros ith row contains nactive sample nonzero
        #  values at i::nsamples
        import torch

        # .sum compresses diagonal elements of jac
        return torch.swapaxes(jac, 0, 1).sum(dim=2)

    def _dshift_dx(self, layer, active_opt_params, samples):
        # jacobian of shift mu with respect to flattened samples
        bexp = layer._shapes
        active_samples = samples[layer._mask_w_labels]

        def fun(x):
            # self._flow._hyp_list.set_active_opt_params(active_opt_params[:, 0])
            shift = bexp(x.T)[:, : layer._ntransformed_vars].T
            return shift

        jac = self._bkd.jacobian(fun, active_samples.T)
        import torch

        # .sum compresses diagonal elements of jac
        return torch.swapaxes(jac, 0, 1).sum(dim=2)

    def _layer_active_hyperparam_indices(self):
        layer_active_hyperparam_indices = []
        lb = 0
        for xlayer in self._flow._layers:
            ub = lb + xlayer._hyp_list.nactive_vars()
            layer_active_hyperparam_indices.append(self._bkd.arange(lb, ub))
            lb = ub
        return layer_active_hyperparam_indices

    def _xlayer_arg_p(self, layer, ii, p):
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
        return usamples  # shape (n_i, N)

    def _dxlayer_dp(self, layer, ii, params):
        autojac = self._bkd.jacobian(
            lambda p: self._xlayer_arg_p(layer, ii, p).T, params
        )  # shape (N, n_i, N, n_p)
        return autojac

    def _dslayer_dp(self, layer, ii, params):
        bexp = layer._shapes

        def fun(p):
            usamples = self._xlayer_arg_p(layer, ii - 1, p)
            active_samples = usamples[layer._mask_w_labels]
            self._flow._hyp_list.set_active_opt_params(p)
            compressed_scale = bexp(active_samples)[
                :, layer._ntransformed_vars :
            ].T
            scale = self._bkd.zeros(
                usamples.shape[0], compressed_scale.shape[1]
            )
            scale[layer._mask_complement_w_labels] = compressed_scale
            return scale

        jac = self._bkd.jacobian(fun, params)
        import torch

        jac = torch.swapaxes(jac, 0, 1)
        return jac

    def _layer_jacobian_samples_wrt_hyperparameters(
        self, layer, ii, samples, p
    ):

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
        omega = self._bkd.exp(-scale)
        diff = samples[layer._mask_complement_wo_labels] - shift

        nsamples = samples.shape[1]
        n = nsamples
        o = omega.shape[0]
        i = samples[layer._mask_w_labels].shape[0]
        # idx_ii = layer_active_hyperparam_indices[nlayers - ii - 1]

        sample_jac = self._bkd.zeros(
            (
                nsamples,
                self._flow.nvars() + self._flow.nlabels(),
                self._flow._hyp_list.nactive_vars(),
            )
        )
        scale_jac = self._bkd.copy(sample_jac)
        #      scale_jac = self._bkd.zeros(
        #         nsamples,
        #         self._flow.nvars() + self._flow.nlabels(),
        #         self._flow._hyp_list.nactive_vars(),
        #     )
        # )
        # compute jacobians with respect to hyperparameters of current layer
        ds_dp = layer._jacobian_scale_wrt_hyperparameters(samples)
        # d_omega_d_layerp = self._bkd.zeros(n, o, p.shape[0])
        # d_scale_d_layerp = self._bkd.zeros(n, o, p.shape[0])
        # d_scale_d_layerp[:, :, idx_ii] = ds_dp
        # d_omega_d_layerp[:, :, idx_ii] = -omega.T[..., None] * ds_dp
        # d_omega_d_layerp1 = self._d_omega_d_layerp(layer, p, samples)
        # assert self._bkd.allclose(d_omega_d_layerp, d_omega_d_layerp1)

        # d_shift_d_layerp = self._bkd.zeros(n, o, p.shape[0])
        # d_shift_d_layerp[:, :, idx_ii] = (
        #     layer._jacobian_shift_wrt_hyperparameters(samples)
        # )
        # d_shift_d_layerp1 = self._d_shift_d_layerp(layer, p, samples)
        # assert self._bkd.allclose(d_shift_d_layerp, d_shift_d_layerp1)

        sample_jac_ii = (
            # diff.T[..., None] * d_omega_d_layerp[..., idx_ii]
            -diff.T[..., None] * omega.T[..., None] * ds_dp
            # - omega.T[..., None] * d_shift_d_layerp[..., idx_ii]
            - omega.T[..., None]
            * layer._jacobian_shift_wrt_hyperparameters(samples)
        )

        scale_jac_ii = ds_dp  # d_scale_d_layerp[..., idx_ii]

        # numpy and torch reduce singleton dimension of when either of the
        # index arrays has only one entry
        # jac[indices1, :, indices2]
        # so do the following
        lb = layer_active_hyperparam_indices[nlayers - ii - 1][0]
        ub = layer_active_hyperparam_indices[nlayers - ii - 1][-1] + 1
        # TODO: store only nonzero elements of sample and scale jac1
        # i.e. do not store jac for parameters of non-parent layers
        sample_jac[:, layer._mask_complement_wo_labels, lb:ub] = sample_jac_ii
        scale_jac[:, layer._mask_complement_wo_labels, lb:ub] = scale_jac_ii

        # compute jacobians with respect to hyperparameters of each parent
        # layer (layers closer to output layer are parents)
        # todo allow for multiple fixed layers for now only allow
        # for first layer to have no active hyperparams
        for jj in range(ii):
            if len(layer_active_hyperparam_indices[nlayers - jj - 1]) == 0:
                continue
            lb = layer_active_hyperparam_indices[nlayers - jj - 1][0]
            ub = layer_active_hyperparam_indices[nlayers - jj - 1][-1] + 1
            sample_jac[:, layer._mask_wo_labels, lb:ub] = (
                self._sample_jacobians[ii - 1][:, layer._mask_wo_labels, lb:ub]
            )
            # scale_jac[:, layer._mask_wo_labels, lb:ub] = self._scale_jacobians[
            #     ii - 1
            # ][:, layer._mask_wo_labels, lb:ub]

            # grad of shift with respect to flattened active samples
            # shape (n,o,i)
            dshift_dx = layer._jacobian_shift_wrt_samples(samples)
            # dshift_dx1 = self._dshift_dx(layer, p, samples)
            # assert self._bkd.allclose(dshift_dx, dshift_dx1)

            # grad of scale with respect to flattened active samples
            dscale_dx = layer._jacobian_scale_wrt_samples(samples)

            # grad of ALL unflattened samples
            # shape (n,i,p)
            dxdp_jj = self._sample_jacobians[ii - 1][
                ..., layer_active_hyperparam_indices[nlayers - jj - 1]
            ]

            chain_rule = ChainRuleArrays(False, False, self._bkd)

            chain_rule.set_arrays(
                (n, i), (n, o), dxdp_jj[:, layer._mask_w_labels], dscale_dx
            )
            dscale_dp = chain_rule((n, dxdp_jj.shape[-1]))
            domega_dp = -omega.T[..., None] * dscale_dp
            chain_rule.set_arrays(
                (n, i), (n, o), dxdp_jj[:, layer._mask_w_labels], dshift_dx
            )
            dshift_dp = chain_rule((n, dxdp_jj.shape[-1]))

            sample_jac_jj = (
                diff.T[..., None] * domega_dp - omega.T[..., None] * dshift_dp
            )
            scale_jac_jj = dscale_dp

            sample_jac_jj += (
                omega.T[..., None]
                * dxdp_jj[:, layer._mask_complement_wo_labels]
            )
            sample_jac[:, layer._mask_complement_wo_labels, lb:ub] = (
                sample_jac_jj
            )
            scale_jac[:, layer._mask_complement_wo_labels, lb:ub] = (
                scale_jac_jj
            )

        # sample_autojac = self._dxlayer_dp(layer, ii, p[:, 0])
        # assert self._bkd.allclose(sample_jac, sample_autojac), ii
        # scale_autojac = self._dslayer_dp(layer, ii, p[:, 0])
        # assert self._bkd.allclose(scale_jac, scale_autojac), ii
        return sample_jac, scale_jac

    def _jacobian_samples_wrt_hyperparameters(self, p):
        samples = self._bkd.copy(self._samples)
        self._flow._hyp_list.set_active_opt_params(p[:, 0])
        ii = 0  # counter for layers with hyperparameters
        self._sample_jacobians = []
        self._scale_jacobians = []
        for layer in reversed(self._flow._layers):
            if isinstance(layer, ScaleAndShiftFlowLayer):
                # do not increment ii because this layer does not have
                # tunable hyperparameters
                self._sample_jacobians.append(None)
                self._scale_jacobians.append(None)
                samples = layer._map_to_latent(samples, False)
                ii += 1
                continue

            sample_jac, scale_jac = (
                self._layer_jacobian_samples_wrt_hyperparameters(
                    layer, ii, samples, p
                )
            )
            self._sample_jacobians.append(sample_jac)
            self._scale_jacobians.append(scale_jac)
            samples = layer._map_to_latent(samples, False)
            ii += 1

        return self._sample_jacobians[-1]

    def _logpdf_jac_auto(self, active_opt_params: Array) -> Array:
        def fun(p):
            self._flow._hyp_list.set_active_opt_params(p)
            samples = self._bkd.copy(self._samples)
            usamples = self._flow._map_to_latent(samples)
            return self._flow._source_variable.logpdf(
                usamples[: self._flow.nvars()]
            )[:, 0]

        return self._bkd.jacobian(fun, active_opt_params[:, 0])

    def _logpdf_jac(self, active_opt_params: Array) -> Array:
        # jacobian of log pdf l with respect to samples x
        # usamples can be stored so it is only computed once
        usamples = usamples = self._flow._map_to_latent(self._samples)
        dldx = self._flow._source_variable.logpdf_jacobian(
            usamples[: self._flow.nvars()]
        )
        # jacobian of samples x with respect to parameters p
        dxdp = self._jacobian_samples_wrt_hyperparameters(active_opt_params)[
            :, : self._flow.nvars()
        ]
        # dxdp = self._bkd.jacobian(self._dxdp, active_opt_params[:, 0])
        # d: nvars, n: nsamples, p: nhyperparams
        dldx = self._bkd.reshape(dldx, (-1, self._samples.shape[1])).T
        jac = self._bkd.einsum("nd, ndp -> np", dldx, dxdp)
        # assert self._bkd.allclose(
        #     jac, self._logpdf_jac_auto(active_opt_params)
        # )
        return jac

    def _logpdf_jacobian(self, active_opt_params: Array):
        # logdet_jac must be callsed after logpdf_jac.
        # The latter computes scale_jacs needs for logdet_jac
        jac = self._logpdf_jac(active_opt_params) + self._logdet_jac(
            active_opt_params
        )
        jac = -self.get_weights().T @ jac
        return jac

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

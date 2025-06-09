from abc import ABC, abstractmethod
import warnings
from typing import Tuple, Union

import numpy as np

from pyapprox.variables.joint import JointVariable
from pyapprox.interface.model import SingleSampleModel
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.optimization.risk import (
    CholeskyBasedGaussianExactKLDivergence,
    IndependentGaussianExactKLDivergence,
)
from pyapprox.expdesign.sequences import LowDiscrepancySequence
from pyapprox.bayes.likelihood import ModelBasedLogLikelihoodMixin
from pyapprox.optimization.minimize import (
    MultiStartOptimizer,
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    CholeskyHyperParameter,
)
from pyapprox.variables.marginals import (
    BetaMarginal,
    UniformMarginal,
    GaussianMarginal,
)
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    IndependentMultivariateGaussian,
)
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    DirichletVariable,
)
from pyapprox.surrogates.affine.basis import QuadratureRule

# TODO implement diagonal plus low rank Gaussian covariance based divergence see
# https://proceedings.neurips.cc/paper_files/paper/2020/file/310cc7ca5a76a446f85c1a0d641ba96d-Paper.pdf
# TODO: Implement KLDivergence when second gaussian is IID standard Normal


class LatentVariableGenerator(ABC):
    def __init__(self, nvars: int, backend: BackendMixin):
        self._bkd = backend
        self._nvars = nvars

    @abstractmethod
    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def _rvs(self, nsamples: int) -> Array:
        raise NotImplementedError

    def _check_samples_weights(
        self, nsamples: int, samples: Array, weights: Array
    ):
        if samples.shape != (self._nvars, nsamples):
            raise RuntimeError(
                "samples has the wrong shape. Was {0} should be {1}".format(
                    samples.shape, (self._nvars, nsamples)
                )
            )
        if weights.shape != (nsamples, 1):
            raise RuntimeError("samples has the wrong shape")

    def __call__(self, nsamples: int) -> Tuple[Array, Array]:
        samples, weights = self._samples_weights(nsamples)
        self._check_samples_weights(nsamples, samples, weights)
        return samples, weights

    def rvs(self, nsamples: int) -> Array:
        samples, weights = self._rvs(nsamples)
        if samples.shape != (self._nvars, nsamples):
            raise RuntimeError("samples has the wrong shape")
        return samples

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def nvars(self) -> int:
        return self._nvars


class MonteCarloIndependentLatentVariableGenerator(LatentVariableGenerator):
    def __init__(self, variable: IndependentMarginalsVariable):
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an IndependentMarginalsVariable"
            )
        self._variable = variable
        super().__init__(variable.nvars(), variable._bkd)
        self._samples = self._bkd.zeros((self.nvars(), 0))

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        if self._samples.shape[1] < nsamples:
            self._samples = self._bkd.hstack(
                (self._samples, self._variable.rvs(nsamples))
            )
        return self._samples[:, :nsamples], self._bkd.full(
            (nsamples, 1), 1 / nsamples
        )

    def _rvs(self, nsamples: int) -> Array:
        return self._variable.rvs(nsamples)


class LowDiscrepanySequenceIndependentLatentVariableGenerator(
    LatentVariableGenerator
):
    def __init__(self, sequence: LowDiscrepancySequence):
        if not isinstance(sequence, LowDiscrepancySequence):
            raise ValueError("sequence must be a LowDiscrepancySequence")
        self._seq = sequence
        super().__init__(sequence.nvars(), sequence._bkd)
        self._samples = self._bkd.zeros((self.nvars(), 0))

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        if self._samples.shape[1] < nsamples:
            self._samples = self._seq.rvs(nsamples)
        return self._samples[:, :nsamples], self._bkd.full(
            (nsamples, 1), 1 / nsamples
        )

    def _rvs(self, nsamples: int) -> Array:
        return self._seq._variable.rvs(nsamples)


class QuadratureRuleLatentVariableGenerator(LatentVariableGenerator):
    def __init__(self, quad_rule: QuadratureRule):
        if not isinstance(quad_rule, QuadratureRule):
            raise ValueError("quad_rule must be a QaudratureRule")
        self._quad_rule = quad_rule
        super().__init__(quad_rule.nvars(), quad_rule._bkd)

    def _check_samples_weights(
        self, nsamples: int, samples: Array, weights: Array
    ):
        if not hasattr(nsamples, "shape"):
            nsamples = self._bkd.array([nsamples] * self.nvars())
        super()._check_samples_weights(
            self._bkd.prod(nsamples), samples, weights
        )

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        # assumes same number of samples used for each dimension
        return self._quad_rule([nsamples] * self.nvars())

    def _rvs(self, nsamples: int) -> Array:
        return self._quad_rule._variable.rvs(nsamples)


class VariationalPosterior(ABC):
    def __init__(
        self,
        latent_generator: LatentVariableGenerator,
        nlatent_samples: int,
    ):
        if not isinstance(latent_generator, LatentVariableGenerator):
            raise ValueError(
                "latent_generator must be a LatentVariableGenerator"
            )
        self._bkd = latent_generator._bkd
        self._nvars = latent_generator.nvars()
        self._latent_generator = latent_generator
        self._latent_samples, self._latent_weights = self._latent_generator(
            nlatent_samples
        )

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def __repr__(self) -> str:
        return "{0}({1}, nvars={2}, bkd={3})".format(
            self.__class__.__name__,
            self._hyp_list._short_repr(),
            self._nvars,
            self._bkd.__name__,
        )

    @abstractmethod
    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        raise NotImplementedError

    def rvs(self, nsamples: int) -> Array:
        return self._map_from_latent_samples(
            self._latent_generator.rvs(nsamples)
        )


class CholeskyGaussianVariationalPosterior(VariationalPosterior):
    def __init__(
        self,
        prior: DenseCholeskyMultivariateGaussian,
        nlatent_samples: int,
        flattened_cholesky_values: Array,  # only entries on and below diagonal
        mean_values: Array = None,
        cholesky_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        mean_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [GaussianMarginal(0.0, 1.0, backend)] * nvars, backend=backend
            )
            latent_generator = MonteCarloIndependentLatentVariableGenerator(
                latent_variable
            )
        super().__init__(latent_generator, nlatent_samples)
        if mean_values is None:
            mean_values = backend.zeros((nvars))
        self._mean = HyperParameter(
            "mean",
            nvars,
            mean_values,
            mean_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._cholesky = CholeskyHyperParameter(
            "cholesky",
            nvars,
            flattened_cholesky_values,
            cholesky_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._mean, self._cholesky])
        self._setup_divergence(prior)

    def mean(self) -> Array:
        return self._mean.get_values()[:, None]

    def covariance(self) -> Array:
        chol = self._cholesky.get_cholesky_factor()
        return chol @ chol.T

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        chol = self._cholesky.get_cholesky_factor()
        return self.mean() + chol @ latent_samples

    def update(self):
        mean1 = self._mean.get_values()[:, None]
        chol1 = self._cholesky.get_cholesky_factor()
        self._divergence.set_left_distribution(mean1, chol1)

    def _setup_divergence(self, prior: IndependentMultivariateGaussian):
        self._divergence = CholeskyBasedGaussianExactKLDivergence(
            prior.nvars(), prior._bkd
        )
        self._divergence.set_right_distribution(
            prior.mean(), prior.covariance()
        )


class IndependentGaussianVariationalPosterior(VariationalPosterior):
    def __init__(
        self,
        prior: IndependentMultivariateGaussian,
        nlatent_samples: int,
        std_diag_values: Array,
        mean_values: Array = None,
        std_diag_bounds: Union[Tuple[float, float], Array] = (0, np.inf),
        mean_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [GaussianMarginal(0.0, 1.0, backend)] * nvars, backend=backend
            )
            latent_generator = MonteCarloIndependentLatentVariableGenerator(
                latent_variable
            )
        super().__init__(latent_generator, nlatent_samples)
        if mean_values is None:
            mean_values = backend.zeros((nvars))
        self._mean = HyperParameter(
            "mean",
            nvars,
            mean_values,
            mean_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._std_diag = HyperParameter(
            "std_diag",
            nvars,
            std_diag_values,
            std_diag_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._mean, self._std_diag])
        self._setup_divergence(prior)

    def mean(self) -> Array:
        return self._mean.get_values()[:, None]

    def covariance(self) -> Array:
        return self._bkd.diag(self._std_diag.get_values() ** 2)

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        return (
            self.mean() + self._std_diag.get_values()[:, None] * latent_samples
        )

    def _setup_divergence(self, prior: IndependentMultivariateGaussian):
        self._divergence = IndependentGaussianExactKLDivergence(
            prior.nvars(), prior._bkd
        )
        self._divergence.set_right_distribution(
            prior.mean(), prior.covariance_diagonal()[:, None]
        )

    def update(self):
        mean1 = self._mean.get_values()[:, None]
        diag1 = self._std_diag.get_values()[:, None] ** 2
        self._divergence.set_left_distribution(mean1, diag1)


class IndependentBetaVariationalPosterior(VariationalPosterior):
    def __init__(
        self,
        prior: IndependentMarginalsVariable,
        nlatent_samples: int,
        ashape_values: Array,
        bshape_values: Array,
        bounds: Array,
        ashape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        bshape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [UniformMarginal(0, 1, backend)] * nvars, backend=backend
            )
            # self._latent_generator = MonteCarloIndependentLatentVariableGenerator(
            #     latent_variable
            # )
            from pyapprox.expdesign.sequences import HaltonSequence

            sequence = HaltonSequence(
                nvars, start_idx=1, variable=latent_variable, bkd=backend
            )
            latent_generator = (
                LowDiscrepanySequenceIndependentLatentVariableGenerator(
                    sequence
                )
            )
        super().__init__(latent_generator, nlatent_samples)
        self._ashapes = HyperParameter(
            "ashapes",
            nvars,
            ashape_values,
            ashape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._bshapes = HyperParameter(
            "bshapes",
            nvars,
            bshape_values,
            bshape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._ashapes, self._bshapes])
        if bounds.shape != (nvars, 2):
            raise ValueError("Bounds has the wrong shape")
        # nquad samples effects accuracy of variational inference gradients
        marginals = [
            BetaMarginal(
                ashape_values[ii],
                bshape_values[ii],
                *bounds[ii],
                nquad_samples=1001,
                backend=self._bkd,
            )
            for ii in range(nvars)
        ]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )
        self._prior = prior

    def update(self):
        ashapes = self._ashapes.get_values()[:, None]
        bshapes = self._bshapes.get_values()[:, None]
        for ii, marginal in enumerate(self.marginals()):
            marginal.set_shapes(ashapes[ii], bshapes[ii])

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        return self._bkd.stack(
            [
                marginal.ppf(latent_samples[ii])
                for ii, marginal in enumerate(self._variable.marginals())
            ],
            axis=0,
        )

    def marginals(self):
        return self._variable.marginals()

    def _divergence(self):
        return self._variable.kl_divergence(self._prior)[:, None]


class DirichletVariationalPosterior(VariationalPosterior):
    def __init__(
        self,
        prior: DirichletVariable,
        nlatent_samples: int,
        ashape_values: Array,
        ashape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [UniformMarginal(0, 1, backend)] * nvars, backend=backend
            )
            # self._latent_generator = MonteCarloIndependentLatentVariableGenerator(
            #     latent_variable
            # )
            from pyapprox.expdesign.sequences import HaltonSequence

            sequence = HaltonSequence(
                nvars, start_idx=1, variable=latent_variable, bkd=backend
            )
            latent_generator = (
                LowDiscrepanySequenceIndependentLatentVariableGenerator(
                    sequence
                )
            )
        super().__init__(latent_generator, nlatent_samples)
        self._ashapes = HyperParameter(
            "ashapes",
            nvars,
            ashape_values,
            ashape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._ashapes])
        # nquad samples effects accuracy of variational inference gradients
        self._variable = DirichletVariable(
            self._ashapes.get_values(), backend=self._bkd
        )
        self._prior = prior

    def update(self):
        self._variable.set_shapes(self._ashapes.get_values())

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        # map uniform samples to gamma samples
        print(latent_samples.shape, self._variable._gamma_variable.nvars())
        gamma_samples = self._bkd.stack(
            [
                marginal.ppf(latent_samples[ii])
                for ii, marginal in enumerate(
                    self._variable._gamma_variable.marginals()
                )
            ],
            axis=0,
        )
        # map gamma samples to dirichlet samples
        return gamma_samples / self._bkd.sum(gamma_samples, axis=0)[None, :]

    def _divergence(self):
        return self._bkd.atleast2d(self._variable.kl_divergence(self._prior))


class NegELBO(SingleSampleModel):
    def __init__(
        self,
        loglike: ModelBasedLogLikelihoodMixin,
        posterior: VariationalPosterior,
        nsamples: int = 1000,
    ):
        super().__init__(posterior._bkd)
        if not isinstance(posterior, VariationalPosterior):
            raise ValueError(
                "posterior must be an instance of VariationalPosterior"
            )
        self._posterior = posterior
        self._loglike = loglike
        self._hyp_list = (
            self._posterior.hyp_list()  # + self._loglike.hyplist()
        )

    def nvars(self) -> int:
        return self._hyp_list.nactive_vars()

    def nqoi(self) -> int:
        return 1

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _jacobian(self, param: Array) -> Array:
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[:, 0], param[:, 0]
        )

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _evaluate(self, params: Array) -> Array:
        self._hyp_list.set_active_opt_params(params[:, 0])
        self._posterior.update()
        samples = self._posterior._map_from_latent_samples(
            self._posterior._latent_samples
        )
        # TODO make this a sample average objective
        # expected_loglike = self._bkd.mean(self._loglike(samples))
        weights = self._posterior._latent_weights
        expected_loglike = self._loglike(samples)[:, 0] @ weights[:, 0]
        return -(expected_loglike - self._posterior._divergence())


class VariationalInverseProblem:
    def __init__(
        self,
        prior: JointVariable,
        loglike: ModelBasedLogLikelihoodMixin,
        posterior: VariationalPosterior,
    ):
        self._bkd = posterior._bkd
        self._prior = prior
        self._neg_elbo = NegELBO(loglike, posterior)
        self._optimizer = None

    def fit(self, iterate: Array = None):
        if self._neg_elbo.hyp_list().nactive_vars() == 0:
            warnings.warn("No active parameters so fit was not called")
            return
        if self._optimizer is None:
            self.set_optimizer(self.default_optimizer())
        if iterate is None:
            # iterate = self._optimizer._initial_interate_gen()
            iterate = self._neg_elbo.hyp_list().get_active_opt_params()[
                :, None
            ]
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self._neg_elbo.hyp_list().set_active_opt_params(active_opt_params)

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer
        self._optimizer.set_objective_function(self._neg_elbo)
        self._optimizer.set_bounds(
            self._neg_elbo.hyp_list().get_active_opt_bounds()
        )

    def _default_iterator_gen(self):
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            self._neg_elbo.hyp_list().nactive_vars(), backend=self._bkd
        )
        iterate_gen.set_bounds(
            self._bkd.to_numpy(
                self._neg_elbo.hyp_list().get_active_opt_bounds()
            )
        )
        return iterate_gen

    def default_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        iterate_gen: OptimizerIterateGenerator = None,
        local_method="L-BFGS-B",
    ) -> MultiStartOptimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=local_method,
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

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            self._prior,
            self._neg_elbo._loglike,
            self._neg_elbo._posterior,
        )


# TODO: read Variational Gaussian Copula Inference.
# http://proceedings.mlr.press/v51/han16.pdf

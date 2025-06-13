from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from scipy import stats


from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.nonlinear.classifiers import LogisticClassifier
from pyapprox.optimization.minimize import Optimizer, OptimizationResult


class Generator(ABC):
    def __init__(
        self, nlatent_vars: int, nobs: int, backend: BackendMixin = TorchMixin
    ):
        self._bkd = backend
        self._nlatent_vars = nlatent_vars
        self._nobs = nobs

    @abstractmethod
    def _transform_latent_samples(self, latent_samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _latent_rvs(self, nsamples: int) -> Array:
        raise NotImplementedError

    def rvs(self, nsamples: int, conditional_vars: Array = None) -> Array:
        """Generate fake samples."""
        if conditional_vars is None:
            return self._transform_latent_samples(self._latent_rvs(nsamples))

        if nsamples != conditional_vars.shape[1]:
            raise ValueError(
                "Number of samples requested {0} {1}".format(
                    nsamples,
                    "does not match number of conditional samples {0}".format(
                        conditional_vars.shape[1]
                    ),
                )
            )
        samples = self._transform_latent_samples(
            self._bkd.vstack((self._latent_rvs(nsamples), conditional_vars))
        )
        return samples

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def nvars(self) -> int:
        return self._nlatent_vars + self._nobs

    def nlatent_vars(self) -> int:
        return self._nlatent_vars

    def nobs(self) -> int:
        return self._nobs


class GaussianLatentMixin:
    def _latent_rvs(self, nsamples: int) -> Array:
        """
        Generate latent samples that will be transformed by the
        generator.
        """
        return self._bkd.asarray(
            np.random.normal(0, 1, (self.nlatent_vars(), nsamples))
        )


class WeinerChaosMixin:
    def _setup_expansion(self, nterms_1d: List[int]):
        if self.nvars() != len(nterms_1d):
            print(self.nvars(), nterms_1d)
            raise ValueError("must specify nterms_1d for each dimension")
        marginals = [stats.norm(0, 1)] * self.nvars()
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=self._bkd
            )
            for marginal in marginals
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices(nterms_1d)
        nqoi = 1
        self._bexp = PolynomialChaosExpansion(basis, None, nqoi=nqoi)
        self._hyp_list = self._bexp.hyp_list()


class WeinerChaosGenerator(WeinerChaosMixin, GaussianLatentMixin, Generator):
    def __init__(
        self,
        nlatent_vars: int,
        nobs: int,
        nterms_1d: List[int],
        backend: BackendMixin = TorchMixin,
    ):
        super().__init__(nlatent_vars, nobs, backend)
        self._setup_expansion(nterms_1d)

    def _transform_latent_samples(self, latent_samples: Array) -> Array:
        return self._bexp(latent_samples).T


class Discriminator(ABC):
    def __init__(self, backend: BackendMixin = TorchMixin):
        self._bkd = backend

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LogisticClassifierDiscriminator(
    LogisticClassifier, WeinerChaosMixin, Discriminator
):
    def __init__(
        self, nterms_1d: List[int], backend: BackendMixin = TorchMixin
    ):
        self._bkd = backend
        self._nvars = len(nterms_1d)
        self._setup_expansion(nterms_1d)
        super().__init__(self._bexp)

    def __call__(self, samples: Array) -> Array:
        return super().__call__(samples)

    def nvars(self) -> int:
        return self._nvars


class GradientDescent(Optimizer):
    def __init__(
        self,
        epochs: int = 20,
        learn_rate: float = 1e-3,
        backend: BackendMixin = TorchMixin,
    ):
        """
        Use the Adam optimizer
        """
        super().__init__()
        self._epochs = epochs
        self._learn_rate = learn_rate
        self._bkd = backend

    def _step_from_objective(
        self, objective, iterate: Array
    ) -> Tuple[float, Array]:
        val = objective(iterate)
        grad = objective.jacobian(iterate)
        iterate -= self._learn_rate * grad
        return val, iterate

    def _step(self, iterate: Array) -> Tuple[float, Array]:
        self._val, self._grad = self._step_from_objective(
            self._objective, iterate
        )

    def _prepare_result(self, iterate: Array) -> OptimizationResult:
        result = OptimizationResult()
        result["x"] = iterate
        result["fun"] = self._val
        result["gnorm"] = self._bkd.norm(self._grad)
        return result

    def _minimize(self, iterate: Array) -> OptimizationResult:
        self._it = 0
        while self._it < self._epochs:
            self._step(iterate)
            self._it += 1
        return self._prepare_result(iterate)


class GenerativeAdvesarialGradientDescent(GradientDescent):
    def _subsample_real_samples(self) -> Array:
        super()._subsample_real_samples()
        self._batch_real_samples = self._disc._bkd.vstack(
            (
                self._batch_real_samples,
                self._conditional_vars[:, self._batch_indices],
            )
        )

    def set_objective_functions(self, gen_model: "GenerativeAdvesarialModel"):
        self._gen_objective = GenerativeAdvesarialGeneratorLoss(gen_model)
        self._gen_objective.set_data(
            gen_model._real_samples, gen_model._conditional_samples
        )
        self._disc_objective = GenerativeAdvesarialDiscriminatorLoss(gen_model)
        self._disc_objective.set_data(
            gen_model._real_samples, gen_model._conditional_samples
        )

    def _update_generator(self) -> bool:
        # can customize to change only certain number of iterations
        if self._it % 1 == 0:
            return True
        return False

    def _prepare_result(self, iterate: Array) -> OptimizationResult:
        result = OptimizationResult()
        result["x"] = iterate
        result["disc_fun"] = self._disc_loss
        result["gen_fun"] = self._gen_loss
        result["disc_norm"] = self._bkd.norm(self._disc_grad)
        result["gen_norm"] = self._bkd.norm(self._gen_grad)
        return result

    def _step(self, iterate: Array) -> Tuple[float, Array]:
        # update the discriminator
        self._disc_loss, self._disc_grad = self._step_from_objective(
            self._disc_objective, iterate
        )
        if not self._update_generator():
            return
        self._gen_loss, self._gen_grad = self._step_from_objective(
            self._gen_objective, iterate
        )

        if self._verbosity > 1 and self._it % 10 == 0:
            print(f"Epoch: {self._it} Generator Loss.: {self._gen_loss}")
            print(f"Epoch: {self._it} Disc. Loss: {self._disc_loss}")

    def prepare_result(self, iterate: Array) -> OptimizationResult:
        result = OptimizationResult()
        result["x"] = iterate
        result["fun"] = self._disc_loss + self._gen_loss
        result["disc_loss"] = self.disc_loss
        result["disc_gnorm"] = self._bkd.norm(self._disc_grad)
        result["gen_loss"] = self.gen_loss
        result["gen_gnorm"] = self._bkd.norm(self._gen_grad)
        return result


class GenerativeAdvesarialModel(ABC):
    def __init__(self, backend: BackendMixin = TorchMixin):
        self._bkd = backend
        self.setup_generator_and_discriminator()
        self._hyp_list = self._disc._hyp_list + self._gen._hyp_list

    @abstractmethod
    def setup_generator_and_discriminator(self):
        raise NotImplementedError

    def _initial_interate_gen(self) -> Array:
        nparams = (
            self._gen._hyp_list.nactive_vars()
            + self._disc._hyp_list.nactive_vars()
        )
        return self._bkd.zeros((nparams, 1))

    def _fit(self, iterate: Array):
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        if iterate is None:
            # todo may need to eventually make this property of
            # optimizer like other surrogates if using new opt formulations
            # the current one does not require this
            iterate = self._initial_interate_gen()
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self._hyp_list.set_active_opt_params(active_opt_params)

    def set_optimizer(self, optimizer: GenerativeAdvesarialGradientDescent):
        # todo allow user to change optimizer based on formulation
        if not isinstance(optimizer, GenerativeAdvesarialGradientDescent):
            raise ValueError(
                "optimizer must be instance of "
                "GenerativeAdvesarialGradientDescent"
            )
        self._optimizer = optimizer

    def fit(
        self,
        real_samples: Array,
        conditional_samples: Array = None,
        iterate: Array = None,
    ):
        self._real_samples = real_samples
        self._conditional_samples = conditional_samples
        self._optimizer.set_objective_functions(self)
        self._fit(iterate)

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self._gen, self._disc
        )

    def _generate_fake_samples(
        self, nsamples, conditional_samples: Array
    ) -> Array:
        if (
            conditional_samples is not None
            and conditional_samples.shape[1] != nsamples
        ):
            raise RuntimeError("conditional samples has the wrong shape")
        fake_samples = self._gen.rvs(nsamples, conditional_samples)
        if conditional_samples is None:
            return fake_samples
        return self._bkd.vstack((fake_samples, conditional_samples))


def _binary_cross_entropy_loss(
    pred_values: Array, obs_values: Array, bkd
) -> Array:
    if pred_values.shape != obs_values.shape:
        raise ValueError("pred_values.shape!=obs_values.shape")
    if pred_values.shape[1] != 1:
        raise ValueError("pred_values must be 2d matrix with one column")
    nsamples = obs_values.shape[0]
    return -(
        bkd.sum(
            obs_values * bkd.log(pred_values)
            + (1.0 - obs_values) * bkd.log(1.0 - pred_values),
            axis=0,
        )
        / nsamples
    )


# Todo derive from loss.py LossFunction
class GenerativeAdvesarialGeneratorLoss:
    def __init__(self, gen_model: GenerativeAdvesarialModel):
        self._gen_model = gen_model
        self._bkd = gen_model._bkd

    def set_data(self, real_samples: Array, conditional_samples: Array):
        if (
            conditional_samples is not None
            and real_samples.shape[1] != conditional_samples.shape[1]
        ):
            raise ValueError(
                "shapes of real and fake samples must match, but were"
                "{0} and {1}".format(
                    real_samples.shape, conditional_samples.shape
                )
            )
        self._real_samples = real_samples
        self._conditional_samples = conditional_samples

    def __call__(self, active_opt_params: Array) -> float:
        self._gen_model._gen._hyp_list.set_active_opt_params(
            active_opt_params[:, 0]
        )
        # generate fake samples
        nsamples = self._real_samples.shape[1]
        fake_samples = self._gen_model._generate_fake_samples(
            nsamples, self._conditional_samples
        )
        # predict the fake labels using the discriminator
        pred_labels = self._gen_model._disc(fake_samples)
        print(pred_labels[:4, 0], fake_samples)
        real_labels = self._bkd.ones((pred_labels.shape[0], 1))
        return _binary_cross_entropy_loss(pred_labels, real_labels, self._bkd)

    def jacobian(self, active_opt_params: Array) -> Array:
        return self._bkd.jacobian(
            lambda p: self.__call__(p)[0], active_opt_params
        )


class GenerativeAdvesarialDiscriminatorLoss:
    def __init__(self, gen_model: GenerativeAdvesarialModel):
        self._gen_model = gen_model
        self._bkd = self._gen_model._bkd

    def set_data(self, real_samples: Array, conditional_samples: Array):
        if (
            conditional_samples is not None
            and real_samples.shape[1] != conditional_samples.shape[1]
        ):
            raise ValueError(
                "shapes of real and fake samples must match, but were"
                "{0} and {1}".format(
                    real_samples.shape, conditional_samples.shape
                )
            )
        self._real_samples = real_samples
        self._conditional_samples = conditional_samples

    def __call__(self, active_opt_params: Array) -> float:
        self._gen_model._disc._hyp_list.set_active_opt_params(
            active_opt_params[:, 0]
        )
        # generate fake samples
        nsamples = self._real_samples.shape[1]
        fake_samples = self._gen_model._generate_fake_samples(
            nsamples, self._conditional_samples
        )
        # predict the real labels using the discriminator
        real_samples = self._bkd.vstack(
            (self._real_samples, self._conditional_samples)
        )
        pred_real_labels = self._gen_model._disc(real_samples)
        real_labels = self._bkd.ones((real_samples.shape[1], 1))
        disc_real_loss = _binary_cross_entropy_loss(
            pred_real_labels, real_labels, self._bkd
        )
        # predict the fake labels using the discriminator
        pred_fake_labels = self._gen_model._disc(fake_samples)
        fake_labels = self._bkd.zeros((nsamples, 1))
        disc_fake_loss = _binary_cross_entropy_loss(
            pred_fake_labels, fake_labels, self._bkd
        )
        # compute the GAN loss
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        return disc_loss

    def jacobian(self, active_opt_params: Array) -> Array:
        return self._bkd.jacobian(
            lambda p: self.__call__(p)[0], active_opt_params
        )


class LogisticGenerativeAdvesarialModel(GenerativeAdvesarialModel):
    def __init__(
        self,
        nlatent_vars: int,
        nobs: int,
        gen_nterms_1d: List[int],
        disc_nterms_1d: List[int],
        backend: BackendMixin = TorchMixin,
    ):
        self._bkd = backend
        self._nlatent_vars = nlatent_vars
        self._nobs = nobs
        self._gen_nterms_1d = gen_nterms_1d
        self._disc_nterms_1d = disc_nterms_1d
        super().__init__(backend)

    def setup_generator_and_discriminator(self):
        self._gen = WeinerChaosGenerator(
            self._nlatent_vars, self._nobs, self._gen_nterms_1d, self._bkd
        )
        self._disc = LogisticClassifierDiscriminator(
            self._disc_nterms_1d, self._bkd
        )

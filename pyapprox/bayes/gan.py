from typing import Tuple

import numpy as np

from abc import ABC, abstractmethod
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from scipy import stats
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.bases.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.bases.classifiers import LogisticClassifier


class Generator(ABC):
    def __init__(self, backend: LinAlgMixin = TorchLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def _transform_latent_samples(self, latent_samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _latent_rvs(self, nsamples: int) -> Array:
        raise NotImplementedError

    def rvs(self, nsamples: int, conditional_vars: Array = None) -> Array:
        """Generate fake samples."""
        if conditional_vars is None:
            return self._transfrom_latent_samples(self._latent_rvs(nsamples))

        if nsamples != conditional_vars.shape[1]:
            raise ValueError(
                "Number of samples requested {0} {1}".format(
                    nsamples,
                    "does not match number of conditional samples {0}".format(
                        conditional_vars.shape[1]
                    )
                )
            )
        samples = self._transfrom_latent_samples(
            self._bkd.vstack((self._latent_rvs(nsamples), conditional_vars))
        )
        return samples

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class GaussianLatentMixin:
    def _latent_rvs(self, nsamples: int) -> Array:
        """Generate latent samples that will be transformed by the generator.
        """
        return self._bkd.atleast2d(
            np.random.normal(0, 1, (self._nlatent_vars, nsamples)))


class WeinerChaosMixin:
    def _setup_expansion(self, nterms_1d):
        nvars = len(nterms_1d)
        marginals = [stats.norm(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=self._bkd)
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=self._bkd
            )
            for marginal in marginals
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices([nterms_1d] * variable.num_vars())
        nqoi = 1
        self._bexp = PolynomialChaosExpansion(basis, None, nqoi=nqoi)
        self.hyp_list = self._bexp.hyp_list


class WeinerChaosGenerator(WeinerChaosMixin, GaussianLatentMixin, Generator):
    def __init__(self, nterms_1d):
        super().__init__()
        self.setup_basis_expansion(nterms_1d)

    def _transform_latent_samples(self, latent_samples: Array) -> Array:
        return self._bexp(latent_samples)


class Discriminator(ABC):
    def __init__(self, backend: LinAlgMixin = TorchLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LogisticClassifierDiscriminator(Discriminator, LogisticClassifier):
    pass


# Todo derive from loss.py LossFunction
class GenerativeAdvesarialLoss(ABC):
    def set_data(
            self, real_samples: Array,
            fake_samples: Array,
            conditional_samples: Array
    ):
        if real_samples.shape != fake_samples.shape:
            raise ValueError("shapes of real and fake samples must match")
        self._real_samples = real_samples
        self._fake_samples = fake_samples

    def generator_loss(self) -> float:
        pred_labels = self._disc(self._fake_samples)
        return self._loss_function(pred_labels, self._real_labels)

    def discriminator_loss(self) -> float:
        # predict the real labels using the discriminator
        pred_real_labels = self._disc(self._real_samples)
        real_labels = self._bkd.ones(self._real_samples.shape[0])
        disc_real_loss = self._loss_function(pred_real_labels, real_labels)
        # predict the fake labels using the discriminator
        pred_fake_labels = self._disc(self._fake_samples)
        fake_labels = self._bkd.zeros(self._fake_samples.shape[0])
        disc_fake_loss = self._loss_function(pred_fake_labels, fake_labels)
        # compute the GAN loss
        disc_loss = (disc_real_loss + disc_fake_loss)/2
        return disc_loss

    def _generate_fake_samples(self) -> Array:
        fake_samples = self._gen.rvs(
            self._batchsize, self._conditional_samples[:, self._batch_indices])
        return fake_samples


from pyapprox.optimization.pya_minimize import Optimizer, OptimizationResult
class GradientDecent(Optimizer):
    def __init__(self, epochs=20, learn_rate=1e-3):
        '''
        Use the Adam optimizer
        '''
        super().__init__()
        self._epochs = epochs
        self._learn_rate = learn_rate

    def _step_from_objective(self, objective, iterate: Array):
        val = objective(iterate)
        grad = objective.jacobian(iterate)
        iterate -= self._learn_rate * grad
        return val, iterate

    def step(self, iterate: Array) -> Tuple[float, Array]:
        self._val, self._grad = self._step_from_objective(
            self._objective, iterate
        )

    def preapare_result(self, iterate: Array) -> OptimizationResult:
        result = OptimizationResult()
        result["x"] = iterate
        result["fun"] = self._val
        result["gnorm"] = self._bkd.norm(self._grad)
        return result

    def _minimize(self, iterate):
        self._it = 0
        while self._it < self._maxiters:
            self.step(iterate)
            self._it += 1
        return self.prepare_result(iterate)


class GenerativeAdvesarialGradientDecent(GradientDecent):
    def update_generator(self) -> bool:
        # can customize to change only certain number of iterations
        if self._it % 1 == 0:
            return True
        return False

    def step(self, iterate: Array) -> Tuple[float, Array]:
        # update the discriminator
        self._disc_loss, self._disc_grad = self._step_from_objective(
            self._disc_objective, iterate
        )
        if not self._update_generator():
            return
        self._gen_loss, self._gen_grad = self._step_from_objective(
            self.gen_objective, iterate
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
    @abstractmethod
    def setup_generator_and_discriminator(self):
        raise NotImplementedError

    def _fit(self):
        pass

    def fit(self, samples: Array):
        self._samples = samples
        self._fit()

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self._gen, self._disc)


class LogisticGenerativeAdvesarialModel(GenerativeAdvesarialModel):
    def setup_generator_and_discriminator(self):
        self._gen = WeinerChaosGenerator()
        self._disc = LogisticClassifierDiscriminator()

from typing import Optional, Tuple
from abc import abstractmethod, ABC

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.bayes.variational.flows import Flow
from pyapprox.variables.joint import (
    JointVariable,
    IndependentGroupsVariable,
    IndependentMarginalsVariable,
)
from pyapprox.variables.marginals import UniformMarginal
from pyapprox.util.newton import NewtonSolver
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    TimeIntegratorNewtonResidual,
    # HeunResidual,
    ForwardEulerResidual,
)
from pyapprox.surrogates.affine.basisexp import BasisExpansion
from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import ModelBasedLogLikelihoodMixin
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)


class VelocityField(TransientNewtonResidual):
    def __init__(self, nstates: int, nlabels: int, backend: BackendMixin):
        super().__init__(backend)
        self._nlabels = nlabels
        self._nstates = nstates

    def nlabels(self) -> int:
        return self._nlabels

    def nstates(self) -> int:
        return self._nstates

    def set_time(self, time: float):
        self._time = time

    def set_label(self, label: Array):
        if label.shape != (self.nlabels(),):
            raise ValueError(
                f"label had shape {label.shape} but must be {(self.nlabels(),)}"
            )
        self._label = label

    @abstractmethod
    def _value(self, state: Array) -> Array:
        raise NotImplementedError

    def __call__(self, state: Array) -> Array:
        if state.ndim != 1:
            raise ValueError("state.ndim must equal 1")
        values = self._value(state)
        if values.ndim != 1:
            raise ValueError(
                f"self._value must return 1D array with shape {state.shape}"
            )
        return values

    def _expand_state(self, state) -> Array:
        if self.nlabels() > 0:
            return self._bkd.hstack(
                (self._bkd.asarray(self._time)[None], state, self._label)
            )
        return self._bkd.hstack((self._bkd.asarray(self._time)[None], state))


class ReverseVelocityField(VelocityField):
    def __init__(self, vel_field: VelocityField):
        if not isinstance(vel_field, VelocityField):
            raise ValueError("vel_field must be an instance of VelocityField")
        self._vel_field = vel_field
        super().__init__(
            vel_field.nstates(), vel_field.nlabels(), vel_field._bkd
        )

    def set_time(self, time: float):
        super().set_time(time)
        self._vel_field.set_time(time)

    def set_label(self, label: Array):
        super().set_label(label)
        self._vel_field.set_label(label)

    def _value(self, state: Array) -> Array:
        time = self._vel_field._time
        # mimic reversing in time
        self._vel_field.set_time(1.0 - time)
        vel_field = self._vel_field(state[:-1])
        jac = self._vel_field.jacobian(state[:-1])
        trace = self._bkd.trace(jac)
        self._vel_field.set_time(time)
        result = self._bkd.hstack((vel_field, -trace))
        # return -result to mimic stepping back in time by deltat
        # may not work for all timestepping schemes
        return -result


class BasisExpansionVelocityField(VelocityField):
    def __init__(self, bexp: BasisExpansion, nlabels: int):
        if not isinstance(bexp, BasisExpansion):
            raise ValueError("bexp must be an instance of BasisExpansion")
        # first variable that bexp accepts is time so nstates is nvars-1
        if bexp.nvars() != bexp.nqoi() + nlabels + 1:
            raise ValueError(
                f"bexp.nvars() = {bexp.nvars()} != "
                f"bexp.nqoi()+nlabels+1 = {bexp.nqoi()+nlabels+1}"
            )
        super().__init__(bexp.nqoi(), nlabels, bexp._bkd)
        self._bexp = bexp

    def _value(self, state: Array) -> Array:
        return self._bexp(self._expand_state(state)[:, None])[0]

    def _jacobian(self, state: Array) -> Array:
        jac = self._bexp.jacobian(self._expand_state(state)[:, None])
        # ignore derivative with respect to time
        return jac[:, 1:]

    def get_basis_expansion(self) -> BasisExpansion:
        return self._bexp


class FlowODE:
    def __init__(
        self,
        time_residual: TimeIntegratorNewtonResidual,
        deltat: float,
        newton_solver: NewtonSolver = None,
    ):
        self._time_int = ImplicitTimeIntegrator(
            time_residual,
            0.0,
            1.0,
            deltat,
            newton_solver=newton_solver,
            verbosity=0,
        )

    def _set_deltat(self, deltat: float):
        self._time_int._deltat = deltat

    def __call__(self, source_sample: Array) -> Array:
        result = self._time_int.solve(source_sample)
        return result[0][:, -1]

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class FlowMatchingPathSampler(ABC):
    def __init__(self, source_variable: JointVariable):
        self._bkd = source_variable._bkd
        self._source_variable = source_variable
        time_variable = IndependentMarginalsVariable(
            [UniformMarginal(0.0, 1.0, backend=self._bkd)], backend=self._bkd
        )
        variable_groups = [time_variable, source_variable]
        self._joint_variable = IndependentGroupsVariable(variable_groups)

    def nlabels(self) -> int:
        if self._latent_data_variable is not None:
            return self._latent_data_variable.nvars()
        return 0

    def __repr__(self) -> str:
        return "{0}({1}, nlabels={2})".format(
            self.__class__.__name__(), self._joint_variable, self.nlabels()
        )

    def get_source_variable(self) -> JointVariable:
        return self._source_variable

    def get_samples(self) -> Tuple[Array, Array, Array, Array]:
        return (
            self._intermediate_times,
            self._source_samples,
            self._target_samples,
            self._labels,
        )

    @abstractmethod
    def get_weights(self) -> Array:
        raise NotImplementedError()

    def generate_path_samples(self) -> Tuple[Array, Array]:
        intermediate_times, source_samples, target_samples, labels = (
            self.get_samples()
        )
        intermedieate_samples = (
            1.0 - intermediate_times
        ) * source_samples + intermediate_times * target_samples
        time_derivs = target_samples - source_samples
        if labels is None:
            return (
                self._bkd.vstack((intermediate_times, intermedieate_samples)),
                time_derivs,
            )
        return (
            self._bkd.vstack(
                (intermediate_times, intermedieate_samples, labels)
            ),
            time_derivs,
        )


class FixedDataFlowMatchingPathSampler(FlowMatchingPathSampler):
    def __init__(
        self,
        source_variable: JointVariable,
        target_samples: Array,
        labels: Optional[Array] = None,
    ):
        super().__init__(source_variable)
        self._target_samples = target_samples
        self._labels = labels
        self._generate_samples()

    def _generate_samples(self):
        joint_samples = self._joint_variable.rvs(self._target_samples.shape[1])
        self._intermediate_times = joint_samples[:1]
        self._source_samples = joint_samples[1:]

    def get_weights(self) -> Array:
        nsamples = self._target_samples.shape[1]
        return self._bkd.full((nsamples, 1), 1.0 / nsamples)


class ModelBasedFlowMatchingPathSampler(FlowMatchingPathSampler):
    """
    Generate samples using models
    """

    def __init__(
        self,
        source_variable: JointVariable,
        prior_variable: JointVariable,
        nsamples: int,
        latent_data_variable: Optional[JointVariable] = None,
        qoi_model: Optional[Model] = None,
        loglike: Optional[ModelBasedLogLikelihoodMixin] = None,
    ):
        super().__init__(source_variable)
        if not isinstance(prior_variable, JointVariable):
            raise ValueError(
                "prior_variable must be an instance of JointVariable"
            )
        self._prior_variable = prior_variable
        variable_groups = [self._joint_variable, prior_variable]
        if latent_data_variable is not None:
            variable_groups.append(latent_data_variable)

        self._joint_variable = IndependentGroupsVariable(variable_groups)
        if latent_data_variable is not None and not isinstance(
            latent_data_variable, JointVariable
        ):
            raise ValueError(
                "latent_data_variable must be an instance of JointVariable"
            )
        self._latent_data_variable = latent_data_variable
        if qoi_model is not None and not isinstance(qoi_model, Model):
            raise ValueError("qoi_model must be an instance of model")
        self._qoi_model = qoi_model
        if loglike is not None and not isinstance(
            loglike, ModelBasedLogLikelihoodMixin
        ):
            raise ValueError(
                "loglike must be an instance of ModelBasedLogLikelihoodMixin"
            )
        if (loglike is None and latent_data_variable is not None) or (
            loglike is not None and latent_data_variable is None
        ):
            raise ValueError(
                "must specify both loglike and latent_data_variable"
            )
        self._loglike = loglike
        self._nsamples = nsamples
        self._generate_samples()

    @abstractmethod
    def _sample_joint_variable(self, nsamples: int) -> Array:
        raise NotImplementedError

    def _generate_samples(self):
        joint_samples = self._sample_joint_variable(self._nsamples)
        (
            self._intermediate_times,
            self._source_samples,
            prior_samples,
            latent_data_samples,
        ) = self._split_samples(joint_samples)
        if self._qoi_model is not None:
            self._target_samples = self._qoi_model(prior_samples)
        else:
            self._target_samples = prior_samples

        if self.nlabels() == 0:
            self._labels = None
            return

        # generate labels
        obs_model_samples = self._loglike._model(prior_samples).T
        self._labels = self._loglike._rvs_from_likelihood_samples(
            obs_model_samples, latent_data_samples
        )

    def _split_samples(
        self, joint_samples: Array
    ) -> Tuple[Array, Array, Array, Array]:
        time_samples = joint_samples[:1]
        source_samples = joint_samples[1 : 1 + self._source_variable.nvars()]
        idx1 = 1 + self._source_variable.nvars()
        idx2 = idx1 + self._prior_variable.nvars()
        prior_samples = joint_samples[idx1:idx2]
        if self._latent_data_variable is not None:
            latent_data_samples = joint_samples[idx2:]
        else:
            latent_data_samples = None
        return time_samples, source_samples, prior_samples, latent_data_samples

    def get_weights(self) -> Array:
        return self._bkd.full((self._nsamples, 1), 1.0 / self._nsamples)


class MonteCarloModelBasedFlowMatchingPathSampler(
    ModelBasedFlowMatchingPathSampler
):
    def _sample_joint_variable(self, nsamples: int) -> Array:
        return self._joint_variable.rvs(nsamples)


class TensorProductGaussQuadratureModelBasedFlowMatchingPathSampler(
    ModelBasedFlowMatchingPathSampler
):
    def __init__(
        self,
        source_variable: JointVariable,
        prior_variable: JointVariable,
        n1d_samples: int,
        latent_data_variable: Optional[JointVariable] = None,
        qoi_model: Optional[Model] = None,
        loglike: Optional[ModelBasedLogLikelihoodMixin] = None,
    ):
        self._n1d_samples = n1d_samples
        super().__init__(
            source_variable,
            prior_variable,
            source_variable._bkd.sum(n1d_samples),
            latent_data_variable,
            qoi_model,
            loglike,
        )

    def _sample_joint_variable(self, nsamples: int) -> Array:
        quad_rule = setup_tensor_product_gauss_quadrature_rule(
            self._joint_variable
        )
        samples, self._weights = quad_rule(self._n1d_samples)
        return samples

    def get_weights(self) -> Array:
        return self._weights


class ContinuousNormalizingFlow(Flow):
    def __init__(
        self,
        path_sampler: FlowMatchingPathSampler,
        vel_field: VelocityField,
        deltat: float,
        nlabels: int = 0,
        time_residual_cls: TimeIntegratorNewtonResidual = ForwardEulerResidual,
    ):
        self._set_path_sampler(path_sampler)
        super().__init__(path_sampler.get_source_variable())
        if not isinstance(vel_field, VelocityField):
            raise ValueError("vel_field must be an instance of VelocityField")
        self._vel_field = vel_field
        if not self._bkd.bkd_equal(self._bkd, vel_field._bkd):
            raise ValueError(
                "backend of joint variable and vel_field must be the same"
            )
        self._from_latent_ode_model = FlowODE(
            time_residual_cls(self._vel_field), deltat
        )
        self._reverse_vel_field = ReverseVelocityField(self._vel_field)
        self._to_latent_ode_model = FlowODE(
            time_residual_cls(self._reverse_vel_field), deltat
        )
        if vel_field.nstates() != path_sampler._source_variable.nvars():
            raise ValueError(
                f"{vel_field.nstates()=} but should be "
                f"{path_sampler._source_variable.nvars()}"
            )
        self._nlabels = nlabels

    def _set_deltat(self, deltat: float):
        # adjust deltat if accuracy requirements change
        self._to_latent_ode_model._set_deltat(deltat)
        self._from_latent_ode_model._set_deltat(deltat)

    def _set_path_sampler(self, path_sampler: FlowMatchingPathSampler):
        if not isinstance(path_sampler, FlowMatchingPathSampler):
            raise ValueError(
                "path_sampler must be an instance of "
                "FlowMatchingPathSampler"
            )
        self._path_sampler = path_sampler

    def nlabels(self) -> int:
        return self._nlabels

    def _map_from_latent_single_sample(self, source_sample: Array) -> Array:
        # source_sample = [target_samples, labels]
        self._vel_field.set_label(source_sample[self.nvars() :])
        return self._from_latent_ode_model(source_sample[: self.nvars()])

    def _map_from_latent_many_sample(self, source_samples: Array):
        # source_samples = [source_samples, labels]
        results = []
        nfailed = 0
        for cnt, source_sample in enumerate(source_samples.T):
            try:
                results.append(
                    self._map_from_latent_single_sample(source_sample)
                )
            except RuntimeError as e:
                # print(e)
                print(f"sample {cnt} failed at {self._vel_field._time}")
                nfailed += 1
        print(f"{nfailed} samples failed")
        return self._bkd.stack(results, axis=1)

    def _map_to_latent_single_sample(self, sample: Array) -> Array:
        # sample = [target_samples, labels]
        self._vel_field.set_label(sample[self.nvars() :])
        init_state = self._bkd.hstack(
            (sample[: self.nvars()], self._bkd.zeros((1,)))
        )
        result = self._to_latent_ode_model(init_state)
        sample = result[:-1]
        divergence = result[-1:]
        return sample, divergence

    def _map_to_latent_many_sample(self, samples: Array) -> Array:
        # samples = [target_samples, labels]
        # results = [
        #    self._map_to_latent_single_sample(sample) for sample in samples.T
        # ]
        nfailed = 0
        results = []
        for cnt, source_sample in enumerate(samples.T):
            try:
                results.append(
                    self._map_to_latent_single_sample(source_sample)
                )
            except RuntimeError as e:
                # print(e)
                print(f"sample {cnt} failed at {self._vel_field._time}")
                nfailed += 1
        print(f"{nfailed} samples failed")
        # samples = self._bkd.asarray([result[0] for result in results])
        samples = self._bkd.stack([result[0] for result in results], axis=1)
        logpdf_vals = self._bkd.stack(
            [result[1] for result in results], axis=0
        )
        return samples, logpdf_vals

    def _map_to_latent(self, source_samples: Array) -> Array:
        # samples = [target_samples, labels]
        return self._map_to_latent_many_sample(source_samples)[0]

    def _map_from_latent(self, source_samples: Array) -> Array:
        # source_samples = [source_samples, labels]
        return self._map_from_latent_many_sample(source_samples)

    def logpdf(self, samples: Array) -> Array:
        # samples = [target_samples, labels]
        source_samples, div = self._map_to_latent_many_sample(samples)
        return self._source_variable.logpdf(source_samples) - div

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nlabels={2}, deltat={3})".format(
            self.__class__.__name__,
            self.nvars(),
            self.nlabels(),
            self._to_latent_ode_model._time_int._deltat,
        )


class BasisExpansionContinuousNormalizingFlow(ContinuousNormalizingFlow):
    def __init__(
        self,
        source_variable: JointVariable,
        vel_field: VelocityField,
        deltat: float,
        nlabels: int = 0,
        time_residual_cls: TimeIntegratorNewtonResidual = ForwardEulerResidual,
    ):
        if not isinstance(vel_field, BasisExpansionVelocityField):
            raise ValueError(
                "vel_field must be an instance of BasisExpansionVelocityField"
                f"but was {type(vel_field)}"
            )
        super().__init__(
            source_variable, vel_field, deltat, nlabels, time_residual_cls
        )

    def fit(self):
        """
        Fit the flow to samples from the target variable.
        """
        path_samples, time_derivs = self._path_sampler.generate_path_samples()
        basis_mat = self._vel_field._bexp.basis()(path_samples)
        print(f"Solving linear system with matrix shape {basis_mat.shape}")
        self._vel_field._bexp._solver.set_weights(
            self._path_sampler.get_weights()
        )
        coef = self._vel_field._bexp._solver.solve(basis_mat, time_derivs.T)
        self._vel_field._bexp.set_coefficients(coef)

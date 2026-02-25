"""Adaptive GP builder for iterative sample selection and GP fitting."""

from typing import Callable, Generic, Optional, cast

from pyapprox.surrogates.gaussianprocess.adaptive.protocols import (
    AdaptiveSamplerProtocol,
    SamplingScheduleProtocol,
)
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.input_transform import (
    InputAffineTransformProtocol,
)
from pyapprox.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.surrogates.kernels.protocols import Kernel, KernelProtocol
from pyapprox.util.backends.protocols import Array, Backend


class AdaptiveGPBuilder(Generic[Array]):
    """Builder that iteratively selects samples and fits GPs.

    The builder is NOT a GP. It produces an ``ExactGaussianProcess`` at
    each step; users interact with the returned GP directly for
    predictions, derivatives, and statistics.

    Parameters
    ----------
    kernel : Kernel[Array]
        Covariance kernel used when creating each new GP.
    sampler : AdaptiveSamplerProtocol[Array]
        Adaptive sampler operating in scaled space.
    bkd : Backend[Array]
        Backend for numerical computations.
    input_transform : InputAffineTransformProtocol[Array] | None
        Transform from user space to scaled space.
    output_transform : OutputAffineTransformProtocol[Array] | None
        Transform for output scaling.
    noise_variance : float
        Observation noise added to the GP nugget.
    """

    def __init__(
        self,
        kernel: Kernel[Array],
        sampler: AdaptiveSamplerProtocol[Array],
        bkd: Backend[Array],
        input_transform: Optional[InputAffineTransformProtocol[Array]] = None,
        output_transform: Optional[OutputAffineTransformProtocol[Array]] = None,
        noise_variance: float = 1e-6,
    ) -> None:
        if not isinstance(sampler, AdaptiveSamplerProtocol):
            raise TypeError(
                f"sampler must satisfy AdaptiveSamplerProtocol, "
                f"got {type(sampler).__name__}"
            )
        self._kernel = kernel
        self._sampler = sampler
        self._bkd = bkd
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._noise_variance = noise_variance
        self._X_user: Optional[Array] = None
        self._y_user: Optional[Array] = None
        self._current_gp: Optional[ExactGaussianProcess[Array]] = None
        self._last_samples_user: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def current_gp(self) -> Optional[ExactGaussianProcess[Array]]:
        """Return the current fitted GP, or None if not yet fitted."""
        return self._current_gp

    def training_data(self) -> Optional[tuple[Array, Array]]:
        """Return accumulated training data (X_user, y_user), or None."""
        if self._X_user is None or self._y_user is None:
            return None
        return (self._X_user, self._y_user)

    def step_samples(self, nsamples: int) -> Array:
        """Select new sample locations.

        Samples are selected in scaled space by the sampler, then
        converted to user space for return.

        Parameters
        ----------
        nsamples : int
            Number of new samples to select.

        Returns
        -------
        samples_user : Array
            New sample locations in user space, shape (nvars, nsamples).
        """
        samples_scaled = self._sampler.select_samples(nsamples)
        if self._input_transform is not None:
            samples_user = self._input_transform.inverse_transform(
                samples_scaled
            )
        else:
            samples_user = samples_scaled
        self._last_samples_user = samples_user
        return samples_user

    def step_values(
        self,
        new_values_user: Array,
        optimize: bool = True,
    ) -> ExactGaussianProcess[Array]:
        """Accumulate new values and fit a new GP.

        Parameters
        ----------
        new_values_user : Array
            Function values at the most recently selected samples,
            shape (nqoi, nsamples_new) in user space.
        optimize : bool
            Whether to optimize hyperparameters.

        Returns
        -------
        gp : ExactGaussianProcess[Array]
            A new fitted GP with transforms applied.
        """
        if self._last_samples_user is None:
            raise RuntimeError(
                "Must call step_samples() before step_values()"
            )
        new_X_user = self._last_samples_user

        # Accumulate data
        bkd = self._bkd
        if self._X_user is None:
            self._X_user = new_X_user
            self._y_user = new_values_user
        else:
            assert self._y_user is not None
            self._X_user = bkd.hstack([self._X_user, new_X_user])
            self._y_user = bkd.hstack([self._y_user, new_values_user])

        # Create and fit a new GP
        gp = self._create_and_fit_gp(optimize)
        self._current_gp = gp

        # Update sampler with new kernel after optimization
        self._sampler.set_kernel(cast(KernelProtocol[Array], gp.kernel()))

        return gp

    def step(
        self,
        function: Callable[[Array], Array],
        nsamples: int,
    ) -> tuple[Array, ExactGaussianProcess[Array]]:
        """Select samples, evaluate function, and fit GP.

        Parameters
        ----------
        function : Callable[[Array], Array]
            Function mapping (nvars, nsamples) -> (nqoi, nsamples).
        nsamples : int
            Number of new samples.

        Returns
        -------
        samples : Array
            Selected samples in user space, shape (nvars, nsamples).
        gp : ExactGaussianProcess[Array]
            Fitted GP.
        """
        samples = self.step_samples(nsamples)
        values = function(samples)
        gp = self.step_values(values, optimize=True)
        return samples, gp

    def run(
        self,
        function: Callable[[Array], Array],
        schedule: SamplingScheduleProtocol,
    ) -> ExactGaussianProcess[Array]:
        """Run the full adaptive loop until the schedule is exhausted.

        Parameters
        ----------
        function : Callable[[Array], Array]
            Function mapping (nvars, nsamples) -> (nqoi, nsamples).
        schedule : SamplingScheduleProtocol
            Controls how many samples to add at each step.

        Returns
        -------
        gp : ExactGaussianProcess[Array]
            Final fitted GP.
        """
        gp: Optional[ExactGaussianProcess[Array]] = None
        while not schedule.is_exhausted():
            nsamples = schedule.nnew_samples()
            _, gp = self.step(function, nsamples)
        assert gp is not None, "Schedule produced no steps"
        return gp

    def _create_and_fit_gp(
        self, optimize: bool
    ) -> ExactGaussianProcess[Array]:
        """Create a new ExactGaussianProcess and fit it."""
        assert self._X_user is not None
        assert self._y_user is not None

        nvars = self._X_user.shape[0]
        gp = ExactGaussianProcess(
            kernel=self._kernel,
            nvars=nvars,
            bkd=self._bkd,
            nugget=self._noise_variance,
        )

        if not optimize:
            gp.hyp_list().set_all_inactive()

        gp.fit(
            self._X_user,
            self._y_user,
            output_transform=self._output_transform,
            input_transform=self._input_transform,
        )
        return gp

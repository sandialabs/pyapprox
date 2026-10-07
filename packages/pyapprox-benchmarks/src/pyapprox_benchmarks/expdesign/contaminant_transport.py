r"""Contaminant transport prediction OED problem.

A plume released in the unit square is carried by a divergence-free flow
and spread by an uncertain diffusivity:

.. math::

    \partial_t u + \beta\cdot\nabla u - \nabla\cdot(e^{\kappa}\nabla u) = 0,
    \qquad u(x, 0) = u_0(x),

with zero diffusive flux on the whole boundary (the non-conservative form
with no boundary conditions), so mass leaves only by advection.

- **Log-diffusivity.** :math:`\kappa = \kappa_0 + W\zeta`, with
  :math:`\zeta \sim N(0, I)` and :math:`W` the full-rank factor of a
  bi-Laplacian prior with Robin boundaries (lumped white-noise mass, so
  :math:`\zeta` has one entry per mesh node).
- **Velocity.** :math:`\beta = \nabla^\perp\psi` for the stream function
  :math:`\psi = \psi_{\mathrm{mean}} + \sum_{ij}\sqrt{\lambda_{ij}}\,\eta_{ij}
  \sin(i\pi x_1)\sin(j\pi x_2)`, :math:`\lambda_{ij} = \sigma_\psi^2
  (i^2+j^2)^{-p}`. An injected velocity model decides whether
  :math:`\eta` is fixed or appended to the inputs as a nuisance.
- **Observations.** Point values on a sensor grid at a few times, sensor
  index fastest.
- **QoI.** The advective outflux :math:`u\,\beta\cdot n` at the right
  boundary nodes at the observation times, boundary node fastest. Every
  stream-function mode vanishes on the boundary, so :math:`\beta\cdot n` on
  the right edge is the mean flow's and does not depend on :math:`\eta`.

The inputs are :math:`\xi = (\zeta, \eta)` with prior :math:`N(0, I)`;
:math:`\eta` is present only for an uncertain velocity.

The structured mesh fixes the direction of every triangle's diagonal, so
the mesh is symmetric under the point reflection through the center but
not under reflection in either axis.
"""

import math
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldVelocity,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.stream_function import (
    StreamFunctionProtocol,
    StreamFunctionVelocityMap,
)
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.bilaplacian import BiLaplacianPrior
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.observations import (
    nodal_outflux_functional,
    probe_observation_functional,
)
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.problems.inverse import BayesianInferenceProblem
from pyapprox_benchmarks.problems.oed import PredictionOEDProblem


class CellsPlusUniformStreamFunction(Generic[Array]):
    r"""The stream function :math:`\sin\pi x_1 \sin\pi x_2 + x_2`.

    A single recirculating cell superposed on a unit uniform flow in
    :math:`x_1`.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def gradient(self, points: Array) -> Array:
        """Gradient at ``points`` (2, npts). Shape: (2, npts)"""
        bkd = self._bkd
        x, y = points[0], points[1]
        return bkd.stack(
            [
                math.pi * bkd.cos(math.pi * x) * bkd.sin(math.pi * y),
                math.pi * bkd.sin(math.pi * x) * bkd.cos(math.pi * y) + 1.0,
            ],
            axis=0,
        )


class GaussianPlume(Generic[Array]):
    r"""The initial condition :math:`\exp(-\|S^{-1}(x - c)\|^2/(2w^2))`.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    center : Tuple[float, float]
        Plume center :math:`c`.
    scales : Tuple[float, float]
        Axis scales :math:`S = \mathrm{diag}(s_1, s_2)`.
    width : float
        Width :math:`w`.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        center: Tuple[float, float] = (0.5, 0.25),
        scales: Tuple[float, float] = (1.0, 0.5),
        width: float = 0.3,
    ) -> None:
        self._bkd = bkd
        self._center = center
        self._scales = scales
        self._width = width

    def __call__(self, points: Array) -> Array:
        """Values at ``points`` (2, npts). Shape: (npts,)"""
        dx = (points[0] - self._center[0]) / self._scales[0]
        dy = (points[1] - self._center[1]) / self._scales[1]
        r2 = dx**2 + dy**2
        return self._bkd.exp(-r2 / (2.0 * self._width**2))


class StreamFunctionFlow(Generic[Array]):
    r"""A random stream-function flow on the unit square.

    :math:`\psi = \psi_{\mathrm{mean}} + \sum_{i,j=1}^{n}
    \sqrt{\lambda_{ij}}\,\eta_{ij}\sin(i\pi x_1)\sin(j\pi x_2)`, with
    :math:`\lambda_{ij} = \sigma^2 (i^2 + j^2)^{-p}` and standardized
    coefficients :math:`\eta_{ij}`, ``i`` outer and ``j`` inner.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mean_stream_function : StreamFunctionProtocol[Array]
        :math:`\psi_{\mathrm{mean}}`.
    nmodes_per_dim : int
        :math:`n`, so there are :math:`n^2` coefficients.
    std : float
        :math:`\sigma`, the strength of the perturbation.
    decay : float
        :math:`p`.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mean_stream_function: StreamFunctionProtocol[Array],
        nmodes_per_dim: int = 6,
        std: float = 1.5,
        decay: float = 2.5,
    ) -> None:
        if nmodes_per_dim < 1:
            raise ValueError(
                f"nmodes_per_dim must be positive, got {nmodes_per_dim}"
            )
        if std < 0.0:
            raise ValueError(f"std must be nonnegative, got {std}")
        self._bkd = bkd
        self._mean_stream_function = mean_stream_function
        self._nmodes_per_dim = nmodes_per_dim
        index = np.arange(1, nmodes_per_dim + 1, dtype=np.float64)
        self._spectrum = bkd.asarray(
            std**2 / np.add.outer(index**2, index**2) ** decay
        )

    def nmodes(self) -> int:
        """Number of stream-function coefficients."""
        return self._nmodes_per_dim**2

    def mean_stream_function(self) -> StreamFunctionProtocol[Array]:
        """The mean stream function."""
        return self._mean_stream_function

    def velocity_map(
        self, vel_basis: VectorLagrangeBasis[Array]
    ) -> StreamFunctionVelocityMap[Array]:
        """Map from coefficients to the velocity DOFs of ``vel_basis``."""
        return StreamFunctionVelocityMap(
            self._bkd,
            vel_basis.dof_coordinates()[:, 0::2],
            self._spectrum,
            self._mean_stream_function,
            vel_basis.component_layout(),
        )


@runtime_checkable
class VelocityModelProtocol(Protocol, Generic[Array]):
    """How the stream-function coefficients enter the inputs."""

    def flow(self) -> StreamFunctionFlow[Array]:
        """The flow whose coefficients this model sets."""
        ...

    def nvars(self) -> int:
        """Number of inputs appended to the log-diffusivity inputs."""
        ...

    def coefficients(self, inputs: Array) -> Array:
        """Map the appended inputs (nvars,) to coefficients (nmodes,)."""
        ...


class FixedVelocity(Generic[Array]):
    """A known velocity: the coefficients are fixed and append no inputs.

    Zero coefficients give the mean flow; the true coefficients give the
    oracle.

    Parameters
    ----------
    flow : StreamFunctionFlow[Array]
        The flow.
    coefficients : Array
        The standardized coefficients. Shape: (nmodes,)
    """

    def __init__(
        self, flow: StreamFunctionFlow[Array], coefficients: Array
    ) -> None:
        if tuple(coefficients.shape) != (flow.nmodes(),):
            raise ValueError(
                f"coefficients must have shape ({flow.nmodes()},), got "
                f"{tuple(coefficients.shape)}"
            )
        self._flow = flow
        self._coefficients = coefficients

    def flow(self) -> StreamFunctionFlow[Array]:
        return self._flow

    def nvars(self) -> int:
        return 0

    def coefficients(self, inputs: Array) -> Array:
        return self._coefficients


class UncertainVelocity(Generic[Array]):
    """An uncertain velocity: the coefficients are nuisance inputs.

    Parameters
    ----------
    flow : StreamFunctionFlow[Array]
        The flow. Its ``std`` is the nuisance strength.
    """

    def __init__(self, flow: StreamFunctionFlow[Array]) -> None:
        self._flow = flow

    def flow(self) -> StreamFunctionFlow[Array]:
        return self._flow

    def nvars(self) -> int:
        return self._flow.nmodes()

    def coefficients(self, inputs: Array) -> Array:
        return inputs


class ContaminantTransportOEDProblem(Generic[Array]):
    r"""Contaminant transport prediction OED problem.

    See the module docstring for the model. No ground truth, so this is a
    Problem. NumPy backend only (the Galerkin solver uses skfem).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    velocity_model : VelocityModelProtocol[Array]
        Whether the velocity is fixed or a nuisance.
    nelements_per_dim : int
        Elements per side of the structured triangular mesh.
    deltat : float
        Crank--Nicolson time step.
    final_time : float
        Final time.
    observation_times : Sequence[float]
        Times at which the sensors and the outflux are recorded; each must
        be a multiple of ``deltat``.
    nsensors_per_dim : int
        The sensors form a grid on ``linspace(0, 1, nsensors_per_dim)²``.
    noise_std : float
        Observation noise standard deviation.
    log_diffusivity_mean : float
        :math:`\kappa_0`.
    correlation_length : float
        Correlation length of the log-diffusivity prior.
    log_diffusivity_std : float
        Free-space standard deviation of the log-diffusivity prior.
    initial_condition : Callable[[Array], Array], optional
        Map from points (2, npts) to values (npts,). Default:
        :class:`GaussianPlume`.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        velocity_model: VelocityModelProtocol[Array],
        *,
        nelements_per_dim: int = 39,
        deltat: float = 0.01,
        final_time: float = 1.0,
        observation_times: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        nsensors_per_dim: int = 6,
        noise_std: float = 1e-2,
        log_diffusivity_mean: float = -3.0,
        correlation_length: float = math.sqrt(0.1),
        log_diffusivity_std: float = 1.0
        / math.sqrt(4.0 * math.pi * (10.0 / 39.0) * (100.0 / 39.0)),
        initial_condition: Optional[Callable[[Array], Array]] = None,
    ) -> None:
        if not isinstance(velocity_model, VelocityModelProtocol):
            raise TypeError(
                "velocity_model must satisfy VelocityModelProtocol, got "
                f"{type(velocity_model).__name__}"
            )
        self._bkd = bkd
        self._velocity_model = velocity_model
        self._observation_steps = self._steps(
            observation_times, deltat, final_time
        )
        if initial_condition is None:
            initial_condition = GaussianPlume(bkd)

        self._mesh = StructuredMesh2D(
            nelements_per_dim,
            nelements_per_dim,
            [(0.0, 1.0), (0.0, 1.0)],
            bkd,
            element_type="tri",
        )
        self._basis = LagrangeBasis(self._mesh, degree=1)
        self._vel_basis = VectorLagrangeBasis(self._mesh, degree=1)
        nnodes = self._basis.ndofs()

        self._log_prior = BiLaplacianPrior.from_correlation_length(
            self._basis, correlation_length, log_diffusivity_std, bkd
        )
        self._log_mean = bkd.full((nnodes,), log_diffusivity_mean)
        self._log_factor = self._log_prior.covariance_factor()
        diffusivity_map = TransformedFieldMap(
            MeshKLEFieldMap(bkd, self._log_mean, self._log_factor),
            transform=bkd.exp,
            transform_deriv=bkd.exp,
            bkd=bkd,
            transform_deriv2=bkd.exp,
        )
        self._nparams = diffusivity_map.nvars()
        self._velocity_map = velocity_model.flow().velocity_map(
            self._vel_basis
        )
        nvars = self._nparams + velocity_model.nvars()

        physics = AdvectionDiffusionReaction(
            basis=self._basis,
            diffusivity=NodalFieldDiffusion(
                self._basis, dofs=np.ones(nnodes)
            ),
            bkd=bkd,
            velocity=NodalFieldVelocity(
                self._vel_basis,
                bkd.to_numpy(self._velocity(bkd.zeros((nvars, 1)))),
            ),
        )
        if velocity_model.nvars() > 0:
            parameterization = AdvectionDiffusionParameterization(
                physics,
                diffusivity_map=diffusivity_map,
                velocity_map=self._velocity_map,
                bkd=bkd,
            )
        else:
            parameterization = AdvectionDiffusionParameterization(
                physics, diffusivity_map=diffusivity_map, bkd=bkd
            )
        system = compose_galerkin_system(physics, [])
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
            newton_tol=1e-10,
            newton_maxiter=10,
            lumped_mass=False,
            verbosity=0,
        )
        coords = self._basis.dof_coordinates()
        init_state = initial_condition(coords)

        grid = np.linspace(0.0, 1.0, nsensors_per_dim)
        self._sensor_locations = bkd.asarray(
            np.vstack(
                [
                    np.repeat(grid, nsensors_per_dim),
                    np.tile(grid, nsensors_per_dim),
                ]
            )
        )
        self._obs_functional = probe_observation_functional(
            self._basis, self._sensor_locations, self._observation_steps, nvars
        )
        coords_np = bkd.to_numpy(coords)
        right = bkd.to_numpy(self._basis.get_dofs("right"))
        right = right[np.argsort(coords_np[1, right])]
        self._outflow_dofs = bkd.asarray(right)
        mean_gradient = velocity_model.flow().mean_stream_function().gradient(
            coords[:, self._outflow_dofs]
        )
        self._qoi_functional = nodal_outflux_functional(
            self._outflow_dofs,
            mean_gradient[1],
            self._observation_steps,
            nnodes,
            nvars,
            bkd,
        )
        self._obs_model = GalerkinTransientForwardModel(
            system,
            parameterization,
            init_state,
            config,
            bkd,
            functional=self._obs_functional,
        )
        self._qoi_model = GalerkinTransientForwardModel(
            system,
            parameterization,
            init_state,
            config,
            bkd,
            functional=self._qoi_functional,
        )

        self._prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)], bkd
        )
        nobs = self._obs_functional.nqoi()
        self._noise_variances = bkd.full((nobs,), noise_std**2)
        inference = BayesianInferenceProblem(
            obs_map=self._obs_model,
            prior=self._prior,
            noise_variances=self._noise_variances,
            bkd=bkd,
        )
        self._problem = PredictionOEDProblem(
            inference,
            self._qoi_model,
            self._design_conditions(deltat),
            bkd,
            name="contaminant_transport",
            description=(
                "Plume transport with a bi-Laplacian log-diffusivity and a "
                "stream-function velocity; point sensors, right-boundary "
                "outflux"
            ),
        )

    @staticmethod
    def _steps(
        times: Sequence[float], deltat: float, final_time: float
    ) -> List[int]:
        nsteps = int(round(final_time / deltat))
        if not math.isclose(nsteps * deltat, final_time, rel_tol=1e-10):
            raise ValueError(
                f"final_time {final_time} is not a multiple of deltat {deltat}"
            )
        steps = []
        for time in times:
            step = int(round(time / deltat))
            if not math.isclose(step * deltat, time, rel_tol=1e-10):
                raise ValueError(
                    f"observation time {time} is not a multiple of deltat "
                    f"{deltat}"
                )
            if step < 1 or step > nsteps:
                raise ValueError(
                    f"observation time {time} is outside (0, {final_time}]"
                )
            steps.append(step)
        return steps

    def _design_conditions(self, deltat: float) -> Array:
        """Rows x, y, t of each observation, sensor index fastest."""
        bkd = self._bkd
        nsensors = int(self._sensor_locations.shape[1])
        times = bkd.asarray(
            np.repeat(
                np.asarray(self._observation_steps, dtype=np.float64)
                * deltat,
                nsensors,
            )
        )
        locations = bkd.hstack(
            [self._sensor_locations] * len(self._observation_steps)
        )
        return bkd.vstack([locations, times[None, :]])

    def _velocity(self, sample: Array) -> Array:
        """Velocity DOFs for one input sample (nvars, 1). Shape: (nveldofs,)"""
        nuisance = sample[self._nparams :, 0]
        return self._velocity_map(self._velocity_model.coefficients(nuisance))

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> PredictionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    def obs_map(self) -> GalerkinTransientForwardModel[Array]:
        """Map inputs (nvars, nsamples) to observations (nobs, nsamples)."""
        return self._obs_model

    def qoi_map(self) -> GalerkinTransientForwardModel[Array]:
        """Map inputs (nvars, nsamples) to the outflux (nqoi, nsamples)."""
        return self._qoi_model

    def prior(self) -> IndependentJoint[Array]:
        """Standard Gaussian prior over the inputs."""
        return self._prior

    def noise_variances(self) -> Array:
        """Observation noise variances. Shape: (nobs,)"""
        return self._noise_variances

    def nparams(self) -> int:
        """Number of log-diffusivity inputs (the leading block of xi)."""
        return self._nparams

    def nvars(self) -> int:
        """Total number of inputs, log-diffusivity then nuisance."""
        return self._nparams + self._velocity_model.nvars()

    def log_diffusivity(self, samples: Array) -> Array:
        r"""Nodal :math:`\kappa = \kappa_0 + W\zeta` of input samples.

        Parameters
        ----------
        samples : Array
            Inputs; only the leading ``nparams`` rows are read. Shape:
            (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nnodes, nsamples)
        """
        zeta = samples[: self._nparams]
        return self._log_mean[:, None] + self._log_factor @ zeta

    def log_diffusivity_prior(self) -> BiLaplacianPrior[Array]:
        """The bi-Laplacian prior of the log-diffusivity, whose
        ``covariance()`` is the exact nodal prior covariance."""
        return self._log_prior

    def evaluate_both(self, samples: Array) -> Tuple[Array, Array]:
        """Observations and outflux from one solve per sample.

        Parameters
        ----------
        samples : Array
            Inputs. Shape: (nvars, nsamples)

        Returns
        -------
        obs_values : Array
            Shape: (nobs, nsamples)
        qoi_values : Array
            Shape: (nqoi, nsamples)
        """
        obs_values, qoi_values = [], []
        for ii in range(samples.shape[1]):
            sample = samples[:, ii : ii + 1]
            states, _ = self._obs_model.forward_solve(sample)
            obs_values.append(self._obs_functional(states, sample))
            qoi_values.append(self._qoi_functional(states, sample))
        return self._bkd.hstack(obs_values), self._bkd.hstack(qoi_values)

    def solve_trajectory(self, sample: Array) -> Tuple[Array, Array]:
        """Nodal states (nnodes, nsteps + 1) and times (nsteps + 1,)."""
        return self._obs_model.forward_solve(sample)

    def velocity_field(self, sample: Array) -> Array:
        """Velocity at the mesh nodes for one sample. Shape: (2, nnodes)"""
        dofs = self._velocity(sample)
        return self._bkd.transpose(self._bkd.reshape(dofs, (-1, 2)))

    def mesh(self) -> StructuredMesh2D[Array]:
        """The structured triangular mesh."""
        return self._mesh

    def basis(self) -> LagrangeBasis[Array]:
        """The P1 basis of the concentration and log-diffusivity."""
        return self._basis

    def sensor_locations(self) -> Array:
        """Sensor coordinates. Shape: (2, nsensors)"""
        return self._sensor_locations

    def observation_steps(self) -> List[int]:
        """Time steps at which sensors and outflux are recorded."""
        return list(self._observation_steps)

    def outflow_nodes(self) -> Array:
        """Right-boundary node indices, sorted by height. Shape: (nout,)"""
        return self._outflow_dofs

    def truth(self, seed: int) -> Tuple[Array, Array]:
        """Draw data-generating inputs from the full prior.

        Draws the log-diffusivity inputs and all stream-function
        coefficients whatever the velocity model, so problems with fixed
        and uncertain velocity can be compared on the same data.

        Returns
        -------
        zeta : Array
            Log-diffusivity inputs. Shape: (nparams, 1)
        eta : Array
            Stream-function coefficients. Shape: (nmodes,)
        """
        rng = np.random.default_rng(seed)
        zeta = rng.standard_normal((self._nparams, 1))
        eta = rng.standard_normal(self._velocity_model.flow().nmodes())
        return self._bkd.asarray(zeta), self._bkd.asarray(eta)

    def groups_by_sensor(self) -> List[List[int]]:
        """Observation indices of each sensor across all times."""
        nsensors = int(self._sensor_locations.shape[1])
        ntimes = len(self._observation_steps)
        return [
            [jj * nsensors + ss for jj in range(ntimes)]
            for ss in range(nsensors)
        ]

    def groups_by_time(self) -> List[List[int]]:
        """Observation indices of all sensors at each time."""
        nsensors = int(self._sensor_locations.shape[1])
        ntimes = len(self._observation_steps)
        return [
            [jj * nsensors + ss for ss in range(nsensors)]
            for jj in range(ntimes)
        ]

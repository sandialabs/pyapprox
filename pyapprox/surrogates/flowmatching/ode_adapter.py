"""ODE adapter for flow matching vector fields.

Provides FlowODEResidual to adapt a trained vector field to the
ODEResidualProtocol, and integrate_flow to integrate the flow ODE
for a batch of initial conditions.

.. todo::
    The batched integration functions (_integrate_euler_batch,
    _integrate_heun_batch) duplicate stepping logic from the existing
    TimeIntegrator / explicit stepper infrastructure.  A better design
    would extend TimeIntegrator (or add an ODEResidualProtocol variant)
    to natively support batched state arrays ``(d, nsamples)`` so that
    new stepper classes automatically get vectorised integration without
    adding per-stepper dispatch here.  The current approach is a
    pragmatic short-term fix for the O(nsamples) Python-loop bottleneck.
"""

from typing import Generic, Optional, Type

from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.pde.time.implicit_steppers.integrator import (
    TimeIntegrator,
)
from pyapprox.util.backends.protocols import Array, Backend


class FlowODEResidual(Generic[Array]):
    """Adapts a flow matching vector field to ODEResidualProtocol.

    Wraps a trained VF v(t, x [, c]) so it can be used with the
    existing time integration infrastructure.

    The ODE is: dx/dt = v(t, x [, c])

    State shape: ``(d,)`` — single sample. Caller loops over batch.

    Parameters
    ----------
    vf : callable
        Vector field, ``(nvars_in, ns) -> (d, ns)``.
    bkd : Backend[Array]
        Computational backend.
    conditioning : Array, optional
        Conditioning variables for this sample, shape ``(m, 1)``.
    """

    def __init__(
        self,
        vf: object,
        bkd: Backend[Array],
        conditioning: Optional[Array] = None,
    ) -> None:
        self._vf = vf
        self._bkd = bkd
        self._conditioning = conditioning
        self._time: float = 0.0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation."""
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate dx/dt = v(t, x [, c]).

        Parameters
        ----------
        state : Array
            Current state, shape ``(d,)``.

        Returns
        -------
        Array
            Time derivative, shape ``(d,)``.
        """
        d = state.shape[0]
        t_col = self._bkd.array([[self._time]])  # (1, 1)
        x_col = self._bkd.reshape(state, (d, 1))  # (d, 1)

        if self._conditioning is not None:
            vf_input = self._bkd.vstack([t_col, x_col, self._conditioning])
        else:
            vf_input = self._bkd.vstack([t_col, x_col])

        result = self._vf(vf_input)  # type: ignore[operator]
        return result[:, 0]  # (d,)

    def jacobian(self, state: Array) -> Array:
        """Jacobian df/dy — not needed for explicit steppers.

        Returns zeros since explicit steppers don't use it.

        Parameters
        ----------
        state : Array
            Current state, shape ``(d,)``.

        Returns
        -------
        Array
            Zero matrix, shape ``(d, d)``.
        """
        d = state.shape[0]
        return self._bkd.zeros((d, d))

    def mass_matrix(self, nstates: int) -> Array:
        """Return identity mass matrix.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Identity matrix, shape ``(nstates, nstates)``.
        """
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix (identity) to vector.

        Parameters
        ----------
        vec : Array
            Vector, shape ``(d,)``.

        Returns
        -------
        Array
            Same vector unchanged, shape ``(d,)``.
        """
        return vec


def _batched_vf_eval(
    vf: object,
    t: float,
    x: Array,
    bkd: Backend[Array],
    c: Optional[Array] = None,
) -> Array:
    """Evaluate VF on a batch of samples.

    Parameters
    ----------
    vf : callable
        Vector field, ``(nvars_in, ns) -> (d, ns)``.
    t : float
        Current time.
    x : Array
        Current states, shape ``(d, nsamples)``.
    bkd : Backend[Array]
        Computational backend.
    c : Array, optional
        Conditioning variables, shape ``(m, nsamples)``.

    Returns
    -------
    Array
        VF values, shape ``(d, nsamples)``.
    """
    nsamples = x.shape[1]
    t_row = bkd.full((1, nsamples), t)
    if c is not None:
        vf_input = bkd.vstack([t_row, x, c])
    else:
        vf_input = bkd.vstack([t_row, x])
    return vf(vf_input)  # type: ignore[operator]


def _integrate_euler_batch(
    vf: object,
    x: Array,
    t_start: float,
    dt: float,
    n_steps: int,
    bkd: Backend[Array],
    c: Optional[Array] = None,
) -> Array:
    """Forward Euler integration, vectorized over samples."""
    t = t_start
    for _ in range(n_steps):
        k = _batched_vf_eval(vf, t, x, bkd, c)
        x = x + dt * k
        t += dt
    return x


def _integrate_heun_batch(
    vf: object,
    x: Array,
    t_start: float,
    dt: float,
    n_steps: int,
    bkd: Backend[Array],
    c: Optional[Array] = None,
) -> Array:
    """Heun (RK2) integration, vectorized over samples."""
    t = t_start
    for _ in range(n_steps):
        k1 = _batched_vf_eval(vf, t, x, bkd, c)
        k2 = _batched_vf_eval(vf, t + dt, x + dt * k1, bkd, c)
        x = x + 0.5 * dt * (k1 + k2)
        t += dt
    return x


def _integrate_persample_fallback(
    vf: object,
    x0_batch: Array,
    t_start: float,
    t_end: float,
    dt: float,
    bkd: Backend[Array],
    c: Optional[Array] = None,
    stepper_cls: Type = ForwardEulerResidual,
) -> Array:
    """Per-sample integration fallback for non-standard steppers."""
    nsamples = x0_batch.shape[1]
    results = []
    for i in range(nsamples):
        c_i = None
        if c is not None:
            c_i = bkd.reshape(c[:, i], (-1, 1))  # (m, 1)

        ode_res = FlowODEResidual(vf, bkd, conditioning=c_i)
        stepper = stepper_cls(ode_res)
        newton = NewtonSolver(stepper)
        newton.set_options(maxiters=1)

        integrator = TimeIntegrator(
            init_time=t_start,
            final_time=t_end,
            deltat=dt,
            newton_solver=newton,
            verbosity=0,
        )

        init_state = x0_batch[:, i]  # (d,)
        states, _ = integrator.solve(init_state)
        results.append(states[:, -1])

    return bkd.stack(results, axis=1)  # (d, nsamples)


def integrate_flow(
    vf: object,
    x0_batch: Array,
    t_start: float,
    t_end: float,
    n_steps: int,
    bkd: Backend[Array],
    c: Optional[Array] = None,
    stepper_cls: Type = ForwardEulerResidual,
) -> Array:
    """Integrate the flow ODE for a batch of initial conditions.

    Solves dx/dt = v(t, x [, c]) from t_start to t_end for each
    sample in x0_batch.

    For ForwardEulerResidual and HeunResidual, uses vectorized batch
    integration (one VF call per timestep for all samples). Falls back
    to per-sample integration for other stepper classes.

    Parameters
    ----------
    vf : callable
        Trained vector field, ``(nvars_in, ns) -> (d, ns)``.
    x0_batch : Array
        Initial conditions, shape ``(d, nsamples)``.
    t_start : float
        Start time.
    t_end : float
        End time.
    n_steps : int
        Number of time steps.
    bkd : Backend[Array]
        Computational backend.
    c : Array, optional
        Conditioning variables, shape ``(m, nsamples)``.
    stepper_cls : Type
        Time stepping residual class. Default: ForwardEulerResidual.

    Returns
    -------
    Array
        Final states, shape ``(d, nsamples)``.
    """
    dt = (t_end - t_start) / n_steps

    if stepper_cls is ForwardEulerResidual:
        return _integrate_euler_batch(
            vf,
            x0_batch,
            t_start,
            dt,
            n_steps,
            bkd,
            c,
        )
    elif stepper_cls is HeunResidual:
        return _integrate_heun_batch(
            vf,
            x0_batch,
            t_start,
            dt,
            n_steps,
            bkd,
            c,
        )
    else:
        return _integrate_persample_fallback(
            vf,
            x0_batch,
            t_start,
            t_end,
            dt,
            bkd,
            c,
            stepper_cls,
        )

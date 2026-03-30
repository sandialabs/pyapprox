"""Torch autograd tests for transient functionals.

Verifies that __call__ preserves the PyTorch computation graph so that
torch.autograd.functional.jacobian produces derivatives matching
DerivativeChecker convergence (error_ratio <= 2e-6).

The jacobian passed to DerivativeChecker is computed via autograd through
__call__. If __call__ breaks the computation graph, the autograd jacobian
will be wrong and the error ratio will be large.
"""

from typing import Callable, Optional, Tuple

import torch

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.time.functionals.mse import TransientMSEFunctional
from pyapprox.util.backends.torch import TorchBkd


def _autograd_jacobian(
    f: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    nqoi: int,
    nvars: int,
) -> torch.Tensor:
    """Compute jacobian of f via torch.autograd.functional.jacobian."""
    def f_flat(x: torch.Tensor) -> torch.Tensor:
        return f(x).reshape(nqoi)

    jac: torch.Tensor = torch.autograd.functional.jacobian(
        f_flat, sample.reshape(nvars)
    )
    return jac


class TestEndpointFunctionalAutograd:
    """Autograd derivative checker tests for EndpointFunctional."""

    def test_jacobian_wrt_sol(self, torch_bkd: TorchBkd) -> None:
        from pyapprox.pde.time.functionals.endpoint import EndpointFunctional

        bkd = torch_bkd
        nstates, ntimes, nparams = 3, 5, 2
        func = EndpointFunctional(state_idx=1, nstates=nstates,
                                  nparams=nparams, bkd=bkd)
        param = torch.randn(nparams, 1, dtype=torch.float64)
        nvars = nstates * ntimes

        def eval_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return func(sol_flat.reshape(nstates, ntimes), param)

        def jac_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return _autograd_jacobian(eval_fn, sol_flat, 1, nvars)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nvars, fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)
        sample = torch.randn(nvars, 1, dtype=torch.float64)
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 2e-6


class TestAllStatesEndpointFunctionalAutograd:
    """Autograd derivative checker tests for AllStatesEndpointFunctional."""

    def test_jacobian_wrt_sol(self, torch_bkd: TorchBkd) -> None:
        from pyapprox.pde.time.functionals.all_states_endpoint import (
            AllStatesEndpointFunctional,
        )

        bkd = torch_bkd
        nstates, ntimes, nparams = 3, 5, 2
        func = AllStatesEndpointFunctional(nstates=nstates, nparams=nparams,
                                           bkd=bkd)
        param = torch.randn(nparams, 1, dtype=torch.float64)
        nvars = nstates * ntimes

        def eval_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return func(sol_flat.reshape(nstates, ntimes), param)

        def jac_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return _autograd_jacobian(eval_fn, sol_flat, nstates, nvars)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=nstates, nvars=nvars, fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)
        sample = torch.randn(nvars, 1, dtype=torch.float64)
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 2e-6


class TestTransientMSEFunctionalAutograd:
    """Autograd derivative checker tests for TransientMSEFunctional."""

    def _make_functional(
        self, bkd: TorchBkd, noise_std: Optional[float] = 0.1,
    ) -> Tuple[TransientMSEFunctional[torch.Tensor], int, int]:
        nstates, nresidual_params = 2, 3
        obs_time_indices_0 = bkd.asarray([1, 3], dtype=int)
        obs_time_indices_1 = bkd.asarray([2, 4], dtype=int)
        obs_tuples = [
            (0, obs_time_indices_0),
            (1, obs_time_indices_1),
        ]
        func: TransientMSEFunctional[torch.Tensor] = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=nresidual_params,
            obs_tuples=obs_tuples,
            bkd=bkd,
            noise_std=noise_std,
        )
        observations = torch.randn(4, dtype=torch.float64)
        func.set_observations(observations)
        return func, nstates, nresidual_params

    def test_jacobian_wrt_sol_fixed_sigma(self, torch_bkd: TorchBkd) -> None:
        """Autograd jacobian dQ/dsol converges, fixed noise_std."""
        bkd = torch_bkd
        func, nstates, nresidual_params = self._make_functional(
            bkd, noise_std=0.1,
        )
        ntimes = 6
        nvars = nstates * ntimes
        param = torch.randn(nresidual_params, 1, dtype=torch.float64)

        def eval_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return func(sol_flat.reshape(nstates, ntimes), param)

        def jac_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return _autograd_jacobian(eval_fn, sol_flat, 1, nvars)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nvars, fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)
        sample = torch.randn(nvars, 1, dtype=torch.float64)
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 2e-6

    def test_jacobian_wrt_sol_sigma_param(self, torch_bkd: TorchBkd) -> None:
        """Autograd jacobian dQ/dsol converges, sigma as parameter."""
        bkd = torch_bkd
        func, nstates, nresidual_params = self._make_functional(
            bkd, noise_std=None,
        )
        ntimes = 6
        nvars = nstates * ntimes
        nparams = nresidual_params + 1
        param = torch.randn(nparams, 1, dtype=torch.float64)
        param[0, 0] = 0.5

        def eval_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return func(sol_flat.reshape(nstates, ntimes), param)

        def jac_fn(sol_flat: torch.Tensor) -> torch.Tensor:
            return _autograd_jacobian(eval_fn, sol_flat, 1, nvars)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nvars, fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)
        sample = torch.randn(nvars, 1, dtype=torch.float64)
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 2e-6

    def test_jacobian_wrt_param_sigma(self, torch_bkd: TorchBkd) -> None:
        """Autograd jacobian dQ/dparam converges, sigma as parameter."""
        bkd = torch_bkd
        func, nstates, nresidual_params = self._make_functional(
            bkd, noise_std=None,
        )
        ntimes = 6
        sol = torch.randn(nstates, ntimes, dtype=torch.float64)
        nparams = nresidual_params + 1

        def eval_fn(p: torch.Tensor) -> torch.Tensor:
            # Ensure positive sigma without in-place ops
            p_abs = torch.cat([p[:1].abs() + 0.1, p[1:]])
            return func(sol, p_abs.reshape(nparams, 1))

        def jac_fn(p: torch.Tensor) -> torch.Tensor:
            return _autograd_jacobian(eval_fn, p, 1, nparams)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nparams, fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)
        sample = torch.randn(nparams, 1, dtype=torch.float64)
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 2e-6

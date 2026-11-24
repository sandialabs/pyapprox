r"""Transient Stokes beside transient Darcy through cG(q) windows in time
(:mod:`tests._helpers.coupling_stokes_darcy_transient`): Taylor-Hood
Stokes as a port-loaded response and Q2 Darcy as a pinned one, on 4 x 4
cells each, 5 skeleton nodes, the interface pressure the skeleton
unknown, Stokes stepping 2 and Darcy 3 SDIRK3 steps per window.

- In every window the normal velocities tested against the constant
  (each DOF's tests summed) balance to the Newton tolerance, each side's
  alone not small: mass conservation per window.
- Against a fine coupled run on the same meshes (T = 0.2, 64 windows),
  windows H = 0.05 to 0.0125: Darcy converges at 2 under cG(1) (1.97,
  2.04) and 3 under cG(2) (2.88, 3.12); the Stokes velocity at 2 (1.92,
  2.11 under cG(2)) and its pressure at 1 (1.03, 1.20), the order
  reduction of an index-2 system under SDIRK3, whose stage order is 1.
- Against the exact solution at T = 0.2: dropping Beavers-Joseph-Saffman
  or flipping the Stokes normal leaves a velocity error at least ten
  times the coupled run's.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from tests._helpers.coupling_stokes_darcy_transient import TransientStokesDarcy

_NCELLS, _NSKEL, _FINAL = 4, 5, 0.2


@pytest.mark.parametrize("q", [1, 2])
def test_each_window_conserves_mass(q: int) -> None:
    run = TransientStokesDarcy(_NCELLS, _NSKEL, q, 4, _FINAL)
    assert run.stokes_sink.nsteps() == 8 and run.darcy_sink.nsteps() == 12
    for n in range(4):
        balance = np.zeros((run.nskel_dofs * q, 1))
        largest = 0.0
        for sd, recorded in zip(run.subdomains, run.recorded):
            output = recorded.outputs[n]
            largest = max(largest, float(np.abs(output).max()))
            balance = sd.transfer.gather_add(output, balance)
        assert largest > 1e-3
        conserved = balance.reshape(-1, q).sum(axis=1)
        assert float(np.abs(conserved).max()) < 1e-10


@pytest.mark.slowest_on("*")
@pytest.mark.parametrize(
    "q, darcy, velocity", [(1, 2.0, None), (2, 3.0, 2.0)], ids=["cG1", "cG2"]
)
def test_temporal_order(q: int, darcy: float, velocity: float | None) -> None:
    fine = TransientStokesDarcy(_NCELLS, _NSKEL, q, 64, _FINAL)
    split = fine.pressure_offset()
    fine_stokes, fine_darcy = fine.final_states()
    errors = []
    for nwindows in (4, 8, 16):
        stokes, darcy_state = TransientStokesDarcy(
            _NCELLS, _NSKEL, q, nwindows, _FINAL
        ).final_states()
        errors.append(
            (
                float(np.abs(darcy_state - fine_darcy).max()),
                float(np.abs(stokes[:split] - fine_stokes[:split]).max()),
            )
        )
    darcy_rate = np.log2(errors[-2][0] / errors[-1][0])
    assert abs(darcy_rate - darcy) < 0.25, f"Darcy rate {darcy_rate:.2f}, {errors}"
    if velocity is not None:
        rate = np.log2(errors[-2][1] / errors[-1][1])
        assert abs(rate - velocity) < 0.25, f"velocity rate {rate:.2f}, {errors}"


@pytest.mark.slow_on("*")
def test_wrong_coupling_controls_fail() -> None:
    right = TransientStokesDarcy(_NCELLS, _NSKEL, 2, 4, _FINAL).velocity_error()
    no_bjs = TransientStokesDarcy(
        _NCELLS, _NSKEL, 2, 4, _FINAL, bjs=False
    ).velocity_error()
    flipped = TransientStokesDarcy(
        _NCELLS, _NSKEL, 2, 4, _FINAL, stokes_sign=1.0
    ).velocity_error()
    assert no_bjs > 10.0 * right, f"BJS dropped {no_bjs:.2e}, coupled {right:.2e}"
    assert flipped > 10.0 * right, f"flipped {flipped:.2e}, coupled {right:.2e}"

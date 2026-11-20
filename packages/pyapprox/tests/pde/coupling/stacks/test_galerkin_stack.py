"""The skfem subdomain builder's refusal; NumPy only, as skfem is.

Its behavior is exercised by every skfem coupling test, which build their
subdomains through it (:func:`tests._helpers.coupling_skfem.pinned_piece`).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.pde.coupling.stacks.galerkin import skfem_pinned_response
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from tests._helpers.coupling_skfem import TensorQuadMesh, newton, poisson_system


def test_no_port_dofs_is_refused() -> None:
    grid = np.linspace(0.0, 1.0, 3)
    system = poisson_system(TensorQuadMesh(grid, grid, NumpyBkd()), [])
    with pytest.raises(ValueError, match="at least one port DOF"):
        skfem_pinned_response(system, [], RecomputeAlways(ScipySparseLU()), newton())

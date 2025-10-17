import unittest

import numpy as np

from pyapprox.pde.galerkin.parameterized import OctagonalHelmholtz


class TestParameterizedFiniteElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_octagonal_helmholtz(self):
        model = OctagonalHelmholtz(2, 5, 400)
        nfacets = model._boundary_facets()
        assert sum(nfacets) == model._basis.mesh.boundary_facets().shape[0]
        p0 = 20  # reference sound pressure of air in micro Pa

        # solve with default parameters
        pressure = model.solve()
        sound_pressure_level = model.pressure_to_sound_pressure_level(
            pressure, p0
        )
        assert np.allclose(
            10 * np.log10((pressure / p0) ** 2), sound_pressure_level
        )

        # solve with perturbed parameters
        model.set_params(np.array([400, 4000]))
        pressure_perturbed = model.solve()

        from skfem.visuals.matplotlib import plot, show, plt

        pressures = [pressure, pressure_perturbed]
        print(
            np.linalg.norm(pressures[0] - pressures[1])
            / np.linalg.norm(pressures[0])
        )
        axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
        for ii in range(2):
            plot(
                model.basis(),
                # sound_pressure_level,
                pressures[ii],
                ax=axs[ii],
                colorbar=True,
            )
        show()


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

import numpy as np

from pyapprox.pde.galerkin.parameterized import (
    OctagonalHelmholtz,
    ObstructedStokesFlow,
)


class TestParameterizedFiniteElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_octagonal_helmholtz(self):
        model = OctagonalHelmholtz(2, 0, 400)
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

        # regression test
        # print(np.array2string(pressure, separator=", "))
        ref_pressure = np.array(
            [
                84570.65588025,
                84570.65588025,
                46687.21381605,
                13049.14909565,
                -19516.76214234,
                -19516.76214234,
                13049.14909565,
                46687.21381605,
                67533.19533878,
                -40826.5499315,
                -34253.82523178,
                23492.05225877,
                -34253.82523178,
                23492.05225877,
                -892.35703249,
                20129.67047647,
                37839.23047102,
                -10191.70220192,
                50573.51245121,
                -22296.21762307,
                37839.23047102,
                -22296.21762307,
                -892.35703249,
                -10191.70220192,
                20129.67047647,
            ]
        )
        assert np.allclose(pressure, ref_pressure)

    def test_obstructed_stokes_flow(self):
        from skfem.visuals.matplotlib import plot, plt, show, draw

        model = ObstructedStokesFlow(3, True)
        model.set_params(np.array([10, 10]))
        sol = model.solve()
        model.plot_pressure(sol)
        axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
        model.plot_velocity_component(sol, 0, ax=axs[0])
        model.plot_velocity_component(sol, 1, ax=axs[1])
        model.plot_velocity_magnitude(sol)
        model.plot_velocity_field(sol)
        # model.plot_vorticity(sol)
        # draw(model._mesh)
        plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)

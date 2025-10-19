import unittest

import numpy as np

from pyapprox.pde.galerkin.parameterized import (
    OctagonalHelmholtz,
    ObstructedFlowDomain,
    ObstructedStokesFlow,
    KLEHyperParameters,
    ObstructedAdvectionDiffusion,
)
from pyapprox.util.backends.numpy import NumpyMixin
from skfem.visuals.matplotlib import plt


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

    def test_obstructed_flow_domain(self):
        L = 7
        domain_bounds = [0, L, 0, 1]
        nsubdomains_1d = [5, 3]
        intervals = [
            np.array(
                [
                    0,
                    2 * L / 7,
                    3 * L / 7,
                    4 * L / 7,
                    5 * L / 7,
                    L,
                ]
            ),
            np.linspace(*domain_bounds[2:], nsubdomains_1d[1] + 1),
        ]

        obstruction_indices = np.array([3, 6, 13], dtype=int)
        domain = ObstructedFlowDomain(*intervals, obstruction_indices)

        ref_connectivity = np.array(
            [
                # first row
                [0, 1, 7, 6],
                [1, 2, 8, 7],
                [2, 3, 9, 8],
                [3, 4, 10, 9],
                [4, 5, 11, 10],
                # second row
                [6, 7, 13, 12],
                [7, 8, 14, 13],
                [8, 9, 15, 14],
                [9, 10, 16, 15],
                [10, 11, 17, 16],
                # third row
                [12, 13, 19, 18],
                [13, 14, 20, 19],
                [14, 15, 21, 20],
                [15, 16, 22, 21],
                [16, 17, 23, 22],
            ],
            dtype=np.int64,
        ).T
        connectivity = domain._full_connectivity
        assert np.allclose(connectivity, ref_connectivity)
        assert np.allclose(
            domain._connectivity,
            np.delete(ref_connectivity, obstruction_indices, axis=1),
        )

        e = 1e-8
        ref_obstruction_bndry_dict = {
            "obs0": lambda x: (
                (x[0] >= (intervals[0][3] - e))
                & (x[0] <= (intervals[0][4] + e))
                & (x[1] >= (intervals[1][0] - e))
                & (x[1] <= (intervals[1][1] + e))
            ),
            "obs1": lambda x: (
                (x[0] >= (intervals[0][1] - e))
                & (x[0] <= (intervals[0][2] + e))
                & (x[1] >= (intervals[1][1] - e))
                & (x[1] <= (intervals[1][2] + e))
            ),
            "obs2": lambda x: (
                (x[0] >= (intervals[0][3] - e))
                & (x[0] <= (intervals[0][4] + e))
                & (x[1] >= (intervals[1][2] - e))
                & (x[1] <= (intervals[1][3] + e))
            ),
        }
        xx = NumpyMixin.cartesian_product(
            [np.linspace(0, L, 51), np.linspace(0, 1, 51)]
        )
        for key, fun in ref_obstruction_bndry_dict.items():
            assert np.allclose(domain._bndry_definitions[key](xx), fun(xx))

    def test_obstructed_stokes_flow(self):

        model = ObstructedStokesFlow(0, True)
        model.set_params(np.array([10, 10]))
        sol = model.solve()
        # print(np.array2string(sol, separator=", "))

        # regression test
        # fmt: off
        ref_sol = np.array(
            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.87500000e+00,  6.80819133e-01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.42388531e+00, -3.27332773e-01,
             2.50000000e+00,  9.14542804e-01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.33516438e+00,  6.93781172e-01,
             1.87500000e+00,  4.53114199e-01,  3.41642790e+00,  1.77833633e+00,
             3.67794990e+00,  5.97789135e-01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  3.63060275e+00, -2.00852339e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.09375000e+00,  3.51447458e-01,
             0.00000000e+00,  0.00000000e+00,  9.74481402e-01, -2.47087435e-01,
             1.23543323e+00, -7.30962748e-01,  0.00000000e+00,  0.00000000e+00,
             2.41259485e+00, -2.24830448e-01,  1.53200442e+00,  7.13869147e-01,
             4.68268269e-01,  1.68066409e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             -1.93148677e-02,  1.69005092e-02,  2.82003983e-01, -2.04567309e-01,
             1.07034176e+00,  7.36246906e-02,  2.34375000e+00,  9.48274335e-01,
             0.00000000e+00,  0.00000000e+00,  1.75630386e+00,  9.85015475e-01,
             0.00000000e+00,  0.00000000e+00,  1.31330653e+00,  1.81981311e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.17735337e+00,  4.90396821e-01,  2.32519967e+00, -1.98723912e-01,
             7.93594544e-01, -3.22740679e-01,  4.55950996e+00, -3.78834991e-01,
             4.43574806e+00, -3.86951785e-01,  2.15068241e+00, -2.48162807e-01,
             2.04333244e+00, -4.23465008e-02,  2.50300631e+00,  1.57545860e+00,
             2.34375000e+00,  8.76648840e-01,  0.00000000e+00,  0.00000000e+00,
             2.68788172e+00,  1.81322738e+00,  3.41045426e+00,  1.77921530e+00,
             3.23041603e-02, -7.85467943e-01,  4.04983732e+00,  1.09844421e+00,
             3.17490267e+00, -3.01814102e-01,  0.00000000e+00,  0.00000000e+00,
             2.26049673e+00, -1.15019422e+00,  0.00000000e+00,  0.00000000e+00,
             7.85112749e-01,  6.00733891e-01,  0.00000000e+00,  0.00000000e+00,
             5.83183900e-01,  3.75372007e-01,  8.36320753e-01, -5.00966757e-02,
             2.52077944e+00,  1.32115118e+00,  1.09375000e+00, -1.08807881e-01,
             3.49904000e+00,  1.05208504e+00,  1.67499218e+00,  1.79809106e-01,
             2.10353914e+00,  5.16926947e-01,  3.35735313e+00,  9.16200554e-02,
             2.67864071e+00,  8.23728231e-01,  3.26344051e+00,  1.28450371e+00,
             0.00000000e+00,  0.00000000e+00,  3.72777101e+00,  1.44440749e+00,
             5.03074633e+00,  1.65202971e+00,  2.44453336e+00, -1.83698528e+00,
             6.91292280e+00,  1.23276338e+00,  5.26344042e+00, -4.21687019e-01,
             5.01002677e+00, -1.22933324e+00,  3.64351660e+00,  3.36368481e-02,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  3.98376777e+01,  3.65030284e+01,
             3.15233680e+01,  4.48645572e+01, -1.22909386e-01,  3.49182217e-01,
             3.91307515e+01,  4.10793595e+01,  2.54093825e+01,  1.64096719e+01,
             -2.49222535e+00,  9.91228937e-01,  3.86324151e+01,  3.63381208e+01,
             1.70115764e+01,  2.07548499e+01, -1.01115978e+00,  6.28296026e-01,
             3.52616148e+01,  3.20239766e+01,  3.30209197e+01,  3.07703856e+01,
             -6.60568791e+00, -1.91325611e+00,  3.63093595e+01,  3.38395605e+01,
             2.68034923e+01,  2.42575224e+01,  1.32926646e+01, -1.37902807e+01]
        )
        # fmt: on
        assert np.allclose(sol, ref_sol)

        # tests plots run
        model.plot_pressure(sol)
        axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
        model.plot_velocity_component(sol, 0, ax=axs[0])
        model.plot_velocity_component(sol, 1, ax=axs[1])
        model.plot_velocity_magnitude(sol)
        model.plot_velocity_field(sol)
        # model.plot_vorticity(sol)
        # draw(model._mesh)
        # plt.show()

    def test_obstructed_advection_diffusion(self):
        nterms = 10
        final_time = 1.0
        kle_hyperparams = KLEHyperParameters(0.5, 1.0, np.inf, nterms)
        # sue coarse meshes and 1 timestep just for easier regression testing
        model = ObstructedAdvectionDiffusion(
            0, 0, 1.0, final_time, kle_hyperparams, True, np.array([10, 10])
        )

        def forcing(x):
            scale = 100.0
            loc = [0.15, 0.4]
            return np.exp(
                -scale * ((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2)
            )

        # comute forcing at quadrature point like KLE
        forcing_vals = model._basis.interpolate(
            model._basis.project(forcing)
        ).flatten()
        # make sure that log of forcing values is not zero
        forcing_vals = np.maximum(forcing_vals, 1e-8)
        # solve for KLE coefficients which are in log space
        params = NumpyMixin.lstsq(
            model.kle().weighted_eigenvectors(), np.log(forcing_vals)
        )
        model.set_params(params)
        sols, times = model.solve()

        print(np.array2string(sols, separator=", "))

        # regression test
        ref_sols = np.array(
            [
                [0.0, 0.00885335],
                [0.0, 0.01087657],
                [0.0, 0.01089406],
                [0.0, 0.00941075],
                [0.0, 0.00759339],
                [0.0, 0.0068866],
                [0.0, 0.01543014],
                [0.0, 0.01446798],
                [0.0, 0.01074722],
                [0.0, 0.01091576],
                [0.0, 0.01137842],
                [0.0, 0.00895938],
                [0.0, 0.02743359],
                [0.0, 0.02269296],
                [0.0, 0.01831084],
                [0.0, 0.01656176],
                [0.0, 0.01322811],
                [0.0, 0.01046788],
                [0.0, 0.01731121],
                [0.0, 0.01912476],
                [0.0, 0.01856387],
                [0.0, 0.01770564],
                [0.0, 0.01561217],
                [0.0, 0.01397999],
                [0.0, 0.00975487],
                [0.0, 0.01327846],
                [0.0, 0.01487903],
                [0.0, 0.01534775],
                [0.0, 0.01592607],
                [0.0, 0.01383978],
            ]
        )
        assert np.allclose(sols, ref_sols)
        assert np.allclose(times, [0.0, 1.0])

        # check plots run
        model.plot_forcing(params, colorbar=True)
        axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)[1]
        model.plot_kle_eigenvecs(np.array([0, nterms // 2, -1]), axs)
        axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
        model.plot_concentration_snapshots(sols, np.array([0, -1]), axs)
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        ani = model.animate_concentration_snapshots(fig, axs, sols)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

import numpy as np
from skfem import Basis

from pyapprox.pde.galerkin.parameterized import (
    OctagonalHelmholtz,
    ObstructedFlowDomain,
    ObstructedStokesFlow,
    KLEHyperParameters,
    ObstructedAdvectionDiffusion,
    FETransientOutputModel,
    FETransientSubdomainIntegralFunctional,
)
from pyapprox.pde.galerkin.util import (
    get_mesh,
    get_element,
    get_subdomain_basis,
)
from pyapprox.pde.timeintegration import (
    TransientObservationFunctional,
    CrankNicholsonResidual,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from skfem.visuals.matplotlib import plt


class TestParameterizedFiniteElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_FE_transient_subdomain_integral_functional(self):
        # override mesh to be square domain so we can compute
        # analytical value of functional from an imposed solution
        # (not computed with model) that does not require setting
        # BCs and other attributes. This is a hack to enable easy
        # testing. DO NOT reuse.
        mesh = get_mesh([0, 1, 0, 1], 1)
        mesh = mesh.with_subdomains(
            {"target_subdomain": lambda x: x[0] >= 1.0 / 2.0}
        )
        element = get_element(mesh, 2)
        basis = Basis(mesh, element)
        subdomain_basis = get_subdomain_basis(
            mesh, element, "target_subdomain"
        )
        functional = FETransientSubdomainIntegralFunctional(
            basis.N, 0, subdomain_basis, NumpyMixin
        )
        times = np.array([0.0, 0.5, 1.0])
        # use CrankNicholson so we can exactly integrate qoi which
        # is linear in time
        # qoi = \int_{1/2}^1\int_0^1 x^2(t+1) dxdt
        time_residual = CrankNicholsonResidual
        functional.set_quadrature_sample_weights(
            *time_residual.quadrature_samples_weights(
                np.asarray(times), NumpyMixin
            )
        )
        sols = np.stack(
            [
                basis.project(lambda x: (time + 1) * x[0] ** 2)
                for time in times
            ],
            axis=1,
        )
        qoi = functional(sols)
        # qoi = \int_{1/2}^1\int_0^1 x^2(t+1) dxdt
        exact_qoi = (1.0 / 3.0 - 1.0 / 24.0) * 3 / 2
        assert np.allclose(qoi, exact_qoi)

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
        model.set_params(np.array([10, 2, 2]))
        sol = model.solve()
        print(np.array2string(sol, separator=", "))

        # regression test
        # fmt: off
        ref_sol = np.array(
            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.87500000e-01,  4.41765474e-02,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.64065263e-01, -2.40456591e-02,
             2.50000000e-01,  2.22192791e-01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.73372502e-01, -2.57280399e-02,
             1.87500000e-01,  1.55414678e-01,  3.77681321e-01,  8.29179119e-02,
             3.54568797e-01, -1.31162950e-02,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  2.56346312e-01, -9.76411653e-02,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  1.09375000e-01,  3.02572654e-02,
             0.00000000e+00,  0.00000000e+00,  1.08110995e-01, -2.91512614e-02,
             1.30881103e-01, -5.73629714e-02,  0.00000000e+00,  0.00000000e+00,
             2.12091128e-01,  9.71271841e-03,  1.12077742e-01,  9.14211952e-02,
             3.49210363e-02,  1.50398433e-01,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             3.80904786e-02, -3.33291688e-02,  6.93052320e-02, -3.34851496e-02,
             9.17452334e-02,  5.38902870e-04,  2.34375000e-01,  1.35820777e-01,
             0.00000000e+00,  0.00000000e+00,  1.25831252e-01,  1.09581109e-01,
             0.00000000e+00,  0.00000000e+00,  1.26500053e-01,  1.06849430e-01,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             1.95619323e-01, -4.27961761e-02,  3.42072087e-01, -8.68017442e-02,
             1.06989397e-01, -2.11406689e-02,  4.94341917e-01, -6.68925111e-02,
             3.90191721e-01, -2.76797502e-02,  2.08020929e-01, -9.99912638e-03,
             1.97150974e-01, -3.93632629e-03,  1.77331881e-01,  1.81823388e-01,
             2.34375000e-01,  2.36239416e-01,  0.00000000e+00,  0.00000000e+00,
             2.73556805e-01,  2.20817019e-01,  3.69393359e-01,  1.05520989e-01,
             4.40645828e-02, -7.69886569e-02,  3.58469441e-01, -1.75922709e-02,
             2.41919247e-01, -1.18697595e-01,  0.00000000e+00,  0.00000000e+00,
             1.50927741e-01, -1.17231383e-01,  0.00000000e+00,  0.00000000e+00,
             1.46968879e-01, -2.27933449e-02,  0.00000000e+00,  0.00000000e+00,
             1.57736333e-01, -5.90856418e-02,  1.80499124e-01, -5.81442458e-02,
             2.85620482e-01,  1.49095647e-01,  1.09375000e-01,  3.26195151e-02,
             3.57853365e-01,  4.35242022e-03,  2.07982445e-01,  4.67471592e-02,
             2.71867351e-01,  6.19364367e-02,  2.42766228e-01, -8.31508271e-03,
             3.22797148e-01,  9.15201279e-02,  3.71439035e-01,  1.22864742e-01,
             0.00000000e+00,  0.00000000e+00,  3.94942423e-01,  1.15067346e-01,
             5.53009116e-01,  7.41846053e-02,  2.26056808e-01, -1.34048216e-01,
             6.40519572e-01, -2.26853308e-02,  4.45004097e-01, -1.22211645e-01,
             3.46985285e-01, -9.33791848e-02,  2.56152631e-01,  1.39006123e-02,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  3.49253271e+00,  2.99935954e+00,
             2.54367280e+00,  3.62257569e+00, -1.15025911e-01, -2.40248266e-02,
             3.32934379e+00,  3.44224534e+00,  2.11492260e+00,  1.37338303e+00,
             4.35651798e-02,  4.47821866e-02,  3.24886626e+00,  3.18548834e+00,
             1.97450552e+00,  1.95426535e+00,  1.70533003e-02, -3.96598904e-02,
             3.06353548e+00,  2.85033049e+00,  2.88193358e+00,  2.62342970e+00,
             7.92501884e-02, -7.18259479e-02,  3.25359735e+00,  2.85681679e+00,
             2.26350958e+00,  2.20802223e+00,  8.94561115e-01, -5.70678554e-01]
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
        model.plot_velocity_magnitude(sol)
        # model.plot_vorticity(sol)
        # draw(model._mesh)
        # plt.show()

    def test_obstructed_advection_diffusion_fixed_velocity_field(self):
        nterms = 10
        final_time = 1.0
        kle_hyperparams = KLEHyperParameters(0.5, 1.0, np.inf, nterms)
        # sue coarse meshes and 1 timestep just for easier regression testing
        model = ObstructedAdvectionDiffusion(
            0,
            0,
            1.0,
            final_time,
            kle_hyperparams,
            True,
            np.array([10, 2.0, 2.0]),
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

        # print(np.array2string(sols, separator=", "))

        # regression test
        ref_sols = np.array(
            [
                [0.0, 0.00665188],
                [0.0, 0.00646361],
                [0.0, 0.00585081],
                [0.0, 0.00520039],
                [0.0, 0.00126005],
                [0.0, 0.00123352],
                [0.0, 0.01278174],
                [0.0, 0.00951642],
                [0.0, 0.00512129],
                [0.0, 0.00407917],
                [0.0, 0.0023362],
                [0.0, 0.00124373],
                [0.0, 0.01847514],
                [0.0, 0.01132648],
                [0.0, 0.00565933],
                [0.0, 0.00426944],
                [0.0, 0.00246231],
                [0.0, 0.00150405],
                [0.0, 0.00755924],
                [0.0, 0.00673346],
                [0.0, 0.0055627],
                [0.0, 0.00431175],
                [0.0, 0.00251652],
                [0.0, 0.00167194],
                [0.0, 0.00442903],
                [0.0, 0.00481364],
                [0.0, 0.00436547],
                [0.0, 0.00360117],
                [0.0, 0.00289187],
                [0.0, 0.00164736],
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

    def test_obstructed_advection_diffusion_random_velocity_field(self):
        nterms = 10
        final_time = 1.0
        kle_hyperparams = KLEHyperParameters(0.5, 1.0, np.inf, nterms)
        # sue coarse meshes and 1 timestep just for easier regression testing
        model = ObstructedAdvectionDiffusion(
            0, 0, 1.0, final_time, kle_hyperparams, True
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
        kle_params = NumpyMixin.lstsq(
            model.kle().weighted_eigenvectors(), np.log(forcing_vals)
        )
        stokes_params = np.array([10.0, 2.0, 2.0])
        params = np.hstack((kle_params, stokes_params))
        model.set_params(params)
        sols, times = model.solve()

        # regression test
        # print(np.array2string(sols, separator=", "))
        ref_sols = np.array(
            [
                [0.0, 0.00665188],
                [0.0, 0.00646361],
                [0.0, 0.00585081],
                [0.0, 0.00520039],
                [0.0, 0.00126005],
                [0.0, 0.00123352],
                [0.0, 0.01278174],
                [0.0, 0.00951642],
                [0.0, 0.00512129],
                [0.0, 0.00407917],
                [0.0, 0.0023362],
                [0.0, 0.00124373],
                [0.0, 0.01847514],
                [0.0, 0.01132648],
                [0.0, 0.00565933],
                [0.0, 0.00426944],
                [0.0, 0.00246231],
                [0.0, 0.00150405],
                [0.0, 0.00755924],
                [0.0, 0.00673346],
                [0.0, 0.0055627],
                [0.0, 0.00431175],
                [0.0, 0.00251652],
                [0.0, 0.00167194],
                [0.0, 0.00442903],
                [0.0, 0.00481364],
                [0.0, 0.00436547],
                [0.0, 0.00360117],
                [0.0, 0.00289187],
                [0.0, 0.00164736],
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
        # ani = model.animate_concentration_snapshots(fig, axs, sols)
        ax = plt.figure().gca()
        model.stokes_model().plot_inlet_velocity_profile(ax)

        # fe models are implemented in python
        # test wrapper that takes in and outputs arrays from
        # a different backend, e.g. TorchMixin and
        # evaluations a funcional on the solution
        bkd = TorchMixin
        # decrease timestep so that the solver takes more steps
        model._solver._deltat = 0.25
        obs_time_tuples = [
            (ii, bkd.array([1, 3])) for ii in range(0, model.nmesh_pts(), 10)
        ]
        obs_model = FETransientOutputModel(model, bkd)
        functional = TransientObservationFunctional(
            model.nmesh_pts(),
            obs_model.nvars(),
            obs_time_tuples,
            backend=bkd,
        )
        obs_model.set_functional(functional)
        sample = bkd.asarray(params[:, None])
        qoi = obs_model(sample)

        # regression test
        # print(np.array2string(bkd.to_numpy(qoi), separator=", "))
        ref_qoi = bkd.array(
            [
                [
                    0.00027443,
                    0.00220279,
                    0.00011866,
                    0.00082548,
                    0.00071503,
                    0.00288328,
                ]
            ]
        )
        assert bkd.allclose(qoi, ref_qoi)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import numpy as np
from skfem import (
    Basis,
    MeshTri1,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    ElementQuad2,
    MeshQuad1,
    Mesh,
    MeshQuad,
)
from skfem.visuals.matplotlib import plot as skfemplot

from pyapprox.pde.galerkin.util import get_element
from pyapprox.pde.galerkin.physics import BoundaryConditions, Helmholtz, Stokes
from pyapprox.pde.galerkin.solvers import SteadyStatePDE
from pyapprox.pde.galerkin.functions import (
    FEMScalarFunctionFromCallable,
    FEMVectorFunctionFromCallable,
    FEMNonLinearOperatorFromCallable,
)
from pyapprox.util.backends.numpy import NumpyMixin


def get_vertices_of_polygon(ampothem, nedges):
    assert np.issubdtype(type(nedges), np.integer)
    circumradius = ampothem / np.cos(np.pi / nedges)
    vertices = []
    for t in np.linspace(0, 2 * np.pi, nedges + 1)[:-1] + np.pi / nedges:
        vertex = [circumradius * np.cos(t), circumradius * np.sin(t)]
        vertices.append(vertex)
    vertices = np.array(vertices).T
    return vertices


class SteadyParameterizedFEModel(ABC):
    @abstractmethod
    def _set_params(self, params: np.ndarray):
        raise NotImplementedError

    def set_params(self, params: np.ndarray):
        if params.shape != (self.nparams(),):
            raise ValueError(
                f"{params.shape=} but must be {(self.nparams(),)}"
            )
        self._set_params(params)

    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> np.ndarray:
        raise NotImplementedError


class OctagonalBoundaries:
    def __init__(self, tol=1e-10):
        self.tol = tol
        self.ampothem = 1.5
        self.nedges = 8
        self.nsegments_per_edge = 3
        self.cumulative_segment_sizes = [0.2, 0.8, 1.0]

    def _2d_bndry_segment_1(self, m, x1, y1, y2, tol, x):
        return (
            np.isclose(x[1] - m * (x[0] - x1) - y1, 0, tol)
            & ((m * (x[0] - x1) + y1) < max(y1, y2) + tol)
            & ((m * (x[0] - x1) + y1) > min(y1, y2) - tol)
        )

    def _2d_bndry_segment_2(self, x1, y1, y2, tol, x):
        return (
            np.isclose(x[0], x1, tol)
            & ((x[1] - y1) > -tol)
            & ((x[1] - y2) < tol)
        )

    def _2d_bndry_segment_3(self, x1, x2, y1, tol, x):
        return (
            np.isclose(x[1], y1, tol)
            & ((x[0] - x1) > -tol)
            & ((x[0] - x2) < tol)
        )

    def _get_2d_bndry_segment_fun(self, x1, y1, x2, y2, tol=1e-10):
        """
        Define boundary segment along the line between (x1,y1) and (x2,y2)
        Assumes x1,y1 x2,y2 come in clockwise order
        """
        tol = 1e-12
        if abs(x2 - x1) > tol and abs(y2 - y1) > tol:
            m = (y2 - y1) / (x2 - x1)
            bndry_fun = partial(self._2d_bndry_segment_1, m, x1, y1, y2, tol)
        elif abs(x2 - x1) < tol:
            II = np.argsort([y1, y2])
            x1, x2 = np.array([x1, x2])[II]
            y1, y2 = np.array([y1, y2])[II]
            bndry_fun = partial(self._2d_bndry_segment_2, x1, y1, y2, tol)
        else:
            II = np.argsort([x1, x2])
            x1, x2 = np.array([x1, x2])[II]
            y1, y2 = np.array([y1, y2])[II]
            bndry_fun = partial(self._2d_bndry_segment_3, x1, x2, y1, tol)
        return bndry_fun

    def _get_polygon_boundary_segments(
        self,
        ampothem,
        nedges,
        nsegments_per_edge=None,
        cumulative_segment_sizes=None,
    ):
        bndry_funs = []
        vertices = get_vertices_of_polygon(ampothem, nedges)
        if cumulative_segment_sizes is None:
            assert nsegments_per_edge is not None
            cumulative_segment_sizes = (
                np.arange(1, nsegments_per_edge + 1) / nsegments_per_edge
            )
        else:
            assert nsegments_per_edge is None or nsegments_per_edge == len(
                cumulative_segment_sizes
            )
            nsegments_per_edge = len(cumulative_segment_sizes)
        x1, y1 = vertices[:, -1]
        for ii in range(vertices.shape[1]):
            x2, y2 = vertices[:, ii]
            pt_begin, pt_end = np.array([x1, y1]), np.array([x2, y2])
            pt_diff = pt_end - pt_begin
            p1 = pt_begin
            for jj in range(nsegments_per_edge):
                p2 = pt_begin + pt_diff * cumulative_segment_sizes[jj]
                bndry_seg = self._get_2d_bndry_segment_fun(
                    p1[0], p1[1], p2[0], p2[1]
                )
                bndry_funs.append(bndry_seg)
                p1 = p2.copy()
            x1, y1 = x2, y2
        return bndry_funs

    def boundaries_dict(self):
        bndry_funs = self._get_polygon_boundary_segments(
            self.ampothem,
            self.nedges,
            self.nsegments_per_edge,
            self.cumulative_segment_sizes,
        )
        labels = []
        for ii in range(self.nedges):
            labels += [
                "seg_%d%d" % (ii, jj) for jj in range(self.nsegments_per_edge)
            ]
        bndrys_dict = dict(zip(labels, bndry_funs))
        return bndrys_dict


class OctagonalHelmholtz(SteadyParameterizedFEModel):
    def __init__(
        self, basis_degree: int, mesh_resolution: int, frequency: float = 400
    ):
        self._radius = 0.5
        self._frequency = frequency
        self._omega = 2.0 * np.pi * self._frequency
        self._sound_speed = np.array([343, 6320, 343])
        self._speaker_amplitudes = np.full((8,), 1.0)
        self._degree = basis_degree
        self._base_mesh, self._bndrys_dict = self._set_domain()
        self._mesh = self._get_mesh(mesh_resolution)
        self._elem = self._get_element(basis_degree)
        self._basis = Basis(self._mesh, self._elem)
        self._bndry_conds = self._get_boundary_conditions()

    def nparams(self) -> int:
        return 3

    def _set_params(self, params: np.ndarray):
        self._sound_speed[:] = params

    def _speaker_boundary_function(self, amplitude, x):
        # negative because Helmholtz uses AdvectionDiffusionReaction
        # which computes Neumann boundary as k\nabla u = f
        # and k=-1 for Helmholtz but we want
        # \nabla u = f
        return -1.204 * self._omega * amplitude + x[0] * 0

    def _cabinet_boundary_function(self, x):
        return x[0] * 0

    def _get_boundary_conditions(self):
        # Dirichlet
        labels = self._bndrys_dict.keys()
        N_bndry_funs = []
        cnt = 0
        for label in labels:
            if label[-1] == "1":
                N_bndry_funs.append(
                    FEMScalarFunctionFromCallable(
                        partial(
                            self._speaker_boundary_function,
                            self._speaker_amplitudes[cnt],
                        )
                    )
                )
                cnt += 1
            else:
                N_bndry_funs.append(
                    FEMScalarFunctionFromCallable(
                        self._cabinet_boundary_function
                    )
                )

        bndry_conds = BoundaryConditions(
            self._mesh,
            self._elem,
            self._basis,
            None,
            None,
            labels,
            N_bndry_funs,
        )

        return bndry_conds

    def _set_domain(self):
        coords, cells = self.create_octagonal_mesh()
        self._domain = OctagonalBoundaries()
        bndrys_dict = self._domain.boundaries_dict()
        base_mesh = MeshTri1(coords.T, cells.T).with_boundaries(
            bndrys_dict, boundaries_only=True
        )
        return base_mesh, bndrys_dict

    def _kappa_fun(self, x: np.ndarray) -> np.ndarray:
        # return x[0]*0
        tol = 1e-14
        kappas = self._omega / self._sound_speed
        c1 = 0
        c2 = c1 - self._radius / np.sqrt(8)
        vals = np.empty(x.shape[1:])
        # outer-most domain
        II = (x[0] - c1) * (x[0] - c1) + (x[1] - c1) * (
            x[1] - c1
        ) >= self._radius**2 + tol
        vals[II] = kappas[0]
        # middle domain
        JJ = (x[0] - c2) * (x[0] - c2) + (x[1] - c2) * (
            x[1] - c2
        ) >= self._radius**2 / 4 + tol
        vals[(~II) & JJ] = kappas[1]
        vals[(~II) & (~JJ)] = kappas[2]
        return vals**2

    def _diff_fun(self, x: np.ndarray) -> np.ndarray:
        return x[0] * 0.0 + 1.0

    def _react_fun(self, x: np.ndarray) -> np.ndarray:
        return self._kappa_fun(x)

    def _get_element(self, degree: int):
        if degree == 1:
            return ElementTriP1()
        if degree == 2:
            return ElementTriP2()
        raise ValueError("Degree must be in [1, 2]")

    def _get_mesh(self, resolution: int):
        # check resolution is an integer
        assert np.mod(resolution, 1) == 0
        resolution = int(resolution)
        mesh = self._base_mesh.refined(resolution).with_boundaries(
            self._bndrys_dict, boundaries_only=True
        )
        return mesh

    def solve(self) -> np.ndarray:
        wnum_op = FEMNonLinearOperatorFromCallable(
            lambda x, u: self._kappa_fun(x) * u,
            lambda x, u: self._kappa_fun(x) + 0 * u,
        )
        physics = Helmholtz(
            self._mesh, self._elem, self._basis, self._bndry_conds, wnum_op
        )
        solver = SteadyStatePDE(physics)

        sol = solver.solve(physics.init_guess())
        return sol

    def create_octagonal_mesh(self):
        vertices = get_vertices_of_polygon(1.5, 8).T
        center_vertex = np.array([[0.0, 0.0]])  # Central vertex at the origin
        vertices = np.vstack(
            [vertices, center_vertex]
        )  # Add central vertex to the list

        # Define the connectivity (triangulation) of the octagon
        # Each triangle connects the center vertex to two consecutive vertices
        # of the octagon
        # 8 triangles, connecting the center to the edges
        elements = np.array([[i, (i + 1) % 8, 8] for i in range(8)])
        # Create the mesh
        return vertices, elements

    def _boundary_facets(self, plot: bool = False):
        from skfem.visuals.matplotlib import draw
        import matplotlib.pyplot as plt

        nfacets = []
        ms = 5
        if plot:
            draw(self._mesh, ax=plt.figure(figsize=(8, 6)).gca())
            plt.plot(
                *self._basis.mesh.p[
                    :,
                    self._basis.mesh.facets[
                        :, self._basis.mesh.boundary_facets()
                    ],
                ].mean(axis=1),
                "ko",
                ms=ms,
            )
        for key in self._bndrys_dict.keys():
            midp = self._basis.mesh.p[:, self._basis.mesh.facets].mean(axis=1)
            facets = np.nonzero(self._bndrys_dict[key](midp))[0]
            facets = np.intersect1d(facets, self._basis.mesh.boundary_facets())
            nfacets.append(facets.shape[0])
            # plot mid points of facets on the boundary
            if not plot:
                continue
            plt.plot(
                *self._basis.mesh.p[
                    :, self._basis.mesh.facets[:, facets]
                ].mean(axis=1),
                "s",
                ms=ms / 2,
            )
        return nfacets

    def pressure_to_sound_pressure_level(
        self, pressure: np.ndarray, p0: float
    ) -> np.ndarray:
        p0 = 20  # reference sound pressure in micro Pa

        # change of base formula log10(a) = log(a)/log(10)
        # sound_pressure_level = 10*np.log((pressure/p0)**2)/np.log(10)
        sound_pressure_level = 10 * np.log10((pressure / p0) ** 2)
        return sound_pressure_level

    def basis(self):
        return self._basis


class ObstructedStokesFlow(SteadyParameterizedFEModel):
    def __init__(self, nrefine: int, navier_stokes: bool = False):
        self._set_mesh(nrefine)
        self._navier_stokes = navier_stokes
        self._set_boundary_conditions()

    def _init_gappy_quad_mesh(self, x: np.ndarray, y: np.ndarray) -> Mesh:
        if x.shape[0] != 6 or y.shape[0] != 4 or x.ndim > 1 or y.ndim > 1:
            raise ValueError("x and/or y has the wrong shape")
        p = NumpyMixin.cartesian_product([x, y])
        t = np.array(
            [
                [0, 1, 7, 6],
                [1, 2, 8, 7],
                [2, 3, 9, 8],
                [4, 5, 11, 10],
                [6, 7, 13, 12],
                [8, 9, 15, 14],
                [9, 10, 16, 15],
                [10, 11, 17, 16],
                [12, 13, 19, 18],
                [13, 14, 20, 19],
                [14, 15, 21, 20],
                [16, 17, 23, 22],
            ],
            dtype=np.int64,
        ).T
        mesh = MeshQuad1(p, t)
        return mesh

    def _gappy_bndry_tests(self, intervals):
        e = 1e-8
        return {
            "left": lambda x: np.isclose(x[0], intervals[0][0]),
            "right": lambda x: np.isclose(x[0], intervals[0][-1]),
            "bottom": lambda x: np.isclose(x[1], intervals[1][0]),
            "top": lambda x: np.isclose(x[1], intervals[1][-1]),
            "obs0": lambda x: (
                (x[0] >= (intervals[0][1] - e))
                & (x[0] <= (intervals[0][2] + e))
                & (x[1] >= (intervals[1][1] - e))
                & (x[1] <= (intervals[1][2] + e))
            ),
            "obs1": lambda x: (
                (x[0] >= (intervals[0][3] - e))
                & (x[0] <= (intervals[0][4] + e))
                & (x[1] >= (intervals[1][0] - e))
                & (x[1] <= (intervals[1][1] + e))
            ),
            "obs2": lambda x: (
                (x[0] >= (intervals[0][3] - e))
                & (x[0] <= (intervals[0][4] + e))
                & (x[1] >= (intervals[1][2] - e))
                & (x[1] <= (intervals[1][3] + e))
            ),
        }

    def _set_basis(self, use_quadmesh: bool):
        if use_quadmesh:
            self._element = {
                "u": ElementVector(get_element(self._mesh, 2)),
                "p": get_element(self._mesh, 1),
            }
            vel_component_elem = ElementQuad2()
        else:
            self._element = {
                "u": ElementVector(ElementTriP2()),
                "p": ElementTriP1(),
            }
            vel_component_elem = ElementTriP2()
        self._basis = {
            variable: Basis(self._mesh, e, intorder=4)
            for variable, e in self._element.items()
        }
        self._vel_component_basis = Basis(
            self._mesh, vel_component_elem, intorder=4
        )

    def _set_mesh(self, nrefine: int):
        self._L = 7.0
        use_quadmesh = False
        domain_bounds = [0, self._L, 0, 1]
        nsubdomains_1d = [5, 3]
        intervals = [
            np.array(
                [
                    0,
                    2 * self._L / 7,
                    3 * self._L / 7,
                    4 * self._L / 7,
                    5 * self._L / 7,
                    self._L,
                ]
            ),
            np.linspace(*domain_bounds[2:], nsubdomains_1d[1] + 1),
        ]
        MeshQuad.init_gappy = self._init_gappy_quad_mesh
        quad_mesh = MeshQuad.init_gappy(*intervals)
        if use_quadmesh:
            mesh = quad_mesh
        else:
            mesh = quad_mesh.to_meshtri()
        self._mesh = mesh.refined(nrefine).with_boundaries(
            self._gappy_bndry_tests(intervals)
        )
        self._set_basis(use_quadmesh)

    def nparams(self) -> int:
        return 2

    def _set_params(self, params: np.ndarray):
        self._params = params
        self._reynolds_num = params[0]
        self._inlet_vel_mag = params[1]

    def _zero_bndry_fun(self, x: np.ndarray) -> np.ndarray:
        vals = x[0] * 0.0
        return vals

    def _inlet_bndry_fun(self, x: np.ndarray) -> np.ndarray:
        """return the plane Poiseuille parabolic inlet profile"""
        vals = self._inlet_vel_mag * x[1] * (1.0 - x[1])
        return vals

    def _setup_horizontal_velocity_boundary_conditions(self):
        D_bndry_names = ["obs0", "obs1", "obs2", "bottom", "top"]
        no_slip_bndry_fun = self._zero_bndry_fun
        D_bndry_funs = [
            FEMScalarFunctionFromCallable(no_slip_bndry_fun)
            for name in D_bndry_names
        ]
        D_bndry_names += ["left"]
        D_bndry_funs += [FEMScalarFunctionFromCallable(self._inlet_bndry_fun)]
        return BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
            D_bndry_names,
            D_bndry_funs,
        )

    def _setup_vertical_velocity_boundary_conditions(self):
        D_bndry_names = ["obs0", "obs1", "obs2", "bottom", "top"]
        D_bndry_funs = [
            FEMScalarFunctionFromCallable(self._zero_bndry_fun)
            for name in D_bndry_names
        ]
        return BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
            D_bndry_names,
            D_bndry_funs,
        )

    def _setup_pressure_boundary_conditions(self):
        # do not apply any conditions
        return BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
        )

    def _set_boundary_conditions(self):
        # apply inlet velocity profile on left boundary
        # apply zero neumann on right boundary, i.e. do nothing and allow outflow
        # apply noslip velocity condition on all other boundaries
        # do not apply pressure bounary conditions
        self._bndry_conds = [
            self._setup_horizontal_velocity_boundary_conditions(),
            self._setup_vertical_velocity_boundary_conditions(),
            self._setup_pressure_boundary_conditions(),
        ]

    def solve(self) -> np.ndarray:
        if not hasattr(self, "_params"):
            raise AttributeError("must call set_params")
        physics = Stokes(
            self._mesh,
            self._element,
            self._basis,
            self._bndry_conds,
            self._navier_stokes,
            FEMVectorFunctionFromCallable(lambda x: x * 0, swapaxes=False),
            FEMScalarFunctionFromCallable(lambda x: x[0] * 0),
            viscosity=self._L / self._reynolds_num,
        )
        solver = SteadyStatePDE(physics)

        sol = solver.solve(physics.init_guess())
        return sol

    def split_solution(self, sol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vel, pres = np.split(sol, [self._basis["u"].N])
        return vel, pres

    def split_velocity(self, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nvars = self._mesh.p.shape[0]
        if nvars == 1:
            return vel
        if vel.shape != (self._basis["u"].N,):
            raise ValueError(
                f"velocity has shape {vel.shape} but should be "
                f"{(self._basis["u"].N,)}"
            )
        return vel[::2], vel[1::2]

    def plot_pressure(self, sol: np.ndarray, **kwargs):
        pres = self.split_solution(sol)[1]
        return skfemplot(self._basis["p"], pres, **kwargs)

    def plot_velocity_component(
        self, sol: np.ndarray, component_id: int, **kwargs
    ):
        if component_id >= self._mesh.p.shape[0]:
            raise ValueError("component_id must be less than nvars")
        vel = self.split_solution(sol)[0]
        vel_comp = self.split_velocity(vel)[component_id]
        return skfemplot(self._vel_component_basis, vel_comp, **kwargs)

    def plot_velocity_magnitude(self, sol: np.ndarray, **kwargs):
        vel = self.split_solution(sol)[0]
        vel_comps = self.split_velocity(vel)
        vel_mag = np.sqrt(sum([v**2 for v in vel_comps]))
        return skfemplot(self._vel_component_basis, vel_mag, **kwargs)

    def plot_velocity_field(self, sol: np.ndarray, **kwargs):
        vel = self.split_solution(sol)[0]
        return skfemplot(self._basis["u"], vel, **kwargs)

    def plot_vorticity(self, sol: np.ndarray):
        from skfem import asm, solve, condense
        from skfem.models.poisson import laplace
        from skfem.models.general import rot
        from matplotlib.tri import Triangulation
        import matplotlib.pyplot as plt

        ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        basis = self._basis
        mesh = self._mesh
        vel = self.split_solution(sol)[0]
        # plotting vorticity currently requires triangulation
        # and thus triangular elements
        basis["psi"] = basis["u"].with_element(ElementTriP2())
        # basis["psi"] = basis["u"].with_element(ElementQuad2())
        A = asm(laplace, basis["psi"])
        psi = basis["psi"].zeros()
        vorticity = asm(rot, basis["psi"], w=self._basis["u"].interpolate(vel))
        psi = solve(
            *condense(A, vorticity, D=self._basis["psi"].get_dofs("bottom"))
        )
        n_streamlines = 11
        contour = partial(
            ax.tricontour,
            Triangulation(*mesh.p, mesh.t.T),
            psi[basis["psi"].nodal_dofs.flatten()],
            linewidths=1.0,
        )
        for levels, color, style in [
            (
                np.linspace(0, 2 / 3, n_streamlines),
                "k",
                ["dashed"] + ["solid"] * (n_streamlines - 2) + ["dashed"],
            ),
            (np.linspace(2 / 3, max(psi), n_streamlines)[0:], "r", "solid"),
            (np.linspace(min(psi), 0, n_streamlines)[:-1], "g", "solid"),
        ]:
            contour(levels=levels, colors=color, linestyles=style)

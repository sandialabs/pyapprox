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
    ElementQuad1,
)
from skfem.visuals.matplotlib import plot as skfemplot

from pyapprox.pde.galerkin.util import get_element
from pyapprox.pde.galerkin.physics import (
    BoundaryConditions,
    Helmholtz,
    Stokes,
    NonLinearAdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.galerkin.functions import (
    FEMScalarFunctionFromCallable,
    FEMVectorFunctionFromCallable,
    FEMNonLinearOperatorFromCallable,
)
from pyapprox.surrogates.affine.kle import MeshKLE
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
        self._params = params
        self._set_params(params)

    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


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


class ObstructedFlowDomain:
    def __init__(self, xintervals, yintervals, obstruction_indices):
        self._intervals = [xintervals, yintervals]
        self._L = xintervals[-1]
        # vertices ordered x0, y0, x1, y1, ..., x1, y0, x1, y1 etc
        self._full_vertices = NumpyMixin.cartesian_product(
            [xintervals, yintervals]
        )
        # t (ordered clockwise starting from bottom left
        # e.g. when nx=6 and ny = 4, [0, 1, 7, 6] is bottom left corner
        self._full_connectivity = self._generate_connectivity(
            xintervals.shape[0], yintervals.shape[0]
        )
        self._obstruction_indices = obstruction_indices
        self._connectivity = self._add_obstructions(self._full_connectivity)
        self._bndry_definitions = self._setup_boundary_definitions()

    def _generate_connectivity(self, nx: int, ny: int) -> np.ndarray:
        """
        Generate connectivity array for a grid defined by nx and ny.

        Parameters:
            nx (int): Number of points in the x-direction.
            ny (int): Number of points in the y-direction.

        Returns:
            np.ndarray: Connectivity array `t` defining the grid elements.
        """
        # Total number of points
        # npoints = nx * ny

        # Initialize connectivity array
        t = []

        # Loop through rows and columns to generate connectivity
        for row in range(ny - 1):  # Iterate over rows
            for col in range(nx - 1):  # Iterate over columns
                # Calculate indices of the four corners of the current cell
                bottom_left = row * nx + col
                bottom_right = bottom_left + 1
                top_left = bottom_left + nx
                top_right = top_left + 1

                # Add the cell connectivity in clockwise order
                t.append([bottom_left, bottom_right, top_right, top_left])

        # Convert to numpy array and transpose
        t = np.array(t, dtype=np.int64).T
        return t

    def _add_obstructions(self, connectivity: np.ndarray) -> np.ndarray:
        mask = np.ones((connectivity.shape[1],), dtype=bool)
        mask[self._obstruction_indices] = False
        t = connectivity[:, mask]
        return t

    def setup_unrefined_quad_mesh(self) -> Mesh:
        mesh = MeshQuad1(self._full_vertices, self._connectivity)
        # from skfem.visuals.matplotlib import plot, plt, show, draw

        # refined_mesh = mesh.refined(2)
        # draw(refined_mesh)
        # for ii in range(self._obstruction_indices.shape[0]):
        #     idx = self._bndry_definitions[f"obs{2}"](refined_mesh.p)
        #     plt.plot(*refined_mesh.p[:, idx], "o")
        # # plt.plot(*self._full_vertices[:, self._connectivity], "*")
        # show()
        return mesh

    def _obstruction_boundary_definition(
        self, obstruction_idx: int, x: np.ndarray
    ) -> np.ndarray:
        eps = 1e-8
        vertex_indices = self._full_connectivity[:, obstruction_idx]
        vertices = self._full_vertices[:, vertex_indices]
        return (
            (x[0] >= (vertices[0, 0] - eps))  # left
            & (x[0] <= (vertices[0, 1] + eps))  # right
            & (x[1] >= (vertices[1, 0] - eps))  # bottom
            & (x[1] <= (vertices[1, 2] + eps))  # top
        )

    def _setup_boundary_definitions(self):
        # will not work if obstructions moved are in the corners
        # or domain becomes disconnected, e.g. two obstructions touch
        # at one mesh vertex
        # residual not coverged will be thrown
        if self._obstruction_indices.ndim != 1:
            raise ValueError("obstruction_indices must be a 1D array")
        if self._obstruction_indices.max() >= self._full_connectivity.shape[1]:
            raise ValueError(
                "obstruction_indices must be smaller than the total "
                f"number of subdomains: {self._full_connectivity.shape[1]}"
            )
        bndry_dict = {
            "left": lambda x: np.isclose(x[0], self._intervals[0][0]),
            "right": lambda x: np.isclose(x[0], self._intervals[0][-1]),
            "bottom": lambda x: np.isclose(x[1], self._intervals[1][0]),
            "top": lambda x: np.isclose(x[1], self._intervals[1][-1]),
        }
        for ii, idx in enumerate(self._obstruction_indices):
            bndry_dict[f"obs{ii}"] = partial(
                self._obstruction_boundary_definition, idx
            )

        # def f(x):
        #     # x = np.array(([7.0, 1.0], [0, 0], [7.0, 0.0])).T
        #     # print(x)
        #     on_right = np.isclose(x[0], self._intervals[0][-1])
        #     not_on_obstructions = [
        #         bndry_dict[f"obs{ii}"](x) == 0
        #         for ii in range(self._obstruction_indices.shape[0])
        #     ]
        #     from functools import reduce

        #     print(on_right)
        #     print(not_on_obstructions)
        #     print(on_right & reduce(np.logical_and, not_on_obstructions))
        #     return on_right & reduce(np.logical_and, (not_on_obstructions))

        return bndry_dict


class ObstructedStokesFlow(SteadyParameterizedFEModel):
    def __init__(
        self, nrefine: int, navier_stokes: bool = False, use_quadmesh=False
    ):
        self._use_quadmesh = use_quadmesh
        self._set_domain()
        self._set_mesh(nrefine)
        self._navier_stokes = navier_stokes
        self._set_boundary_conditions()

    # def _set_domain(self):
    #     L = 7
    #     domain_bounds = [0, L, 0, 1]
    #     nsubdomains_1d = [5, 3]
    #     intervals = [
    #         np.array(
    #             [
    #                 0,
    #                 2 * L / 7,
    #                 3 * L / 7,
    #                 4 * L / 7,
    #                 5 * L / 7,
    #                 L,
    #             ]
    #         ),
    #         np.linspace(*domain_bounds[2:], nsubdomains_1d[1] + 1),
    #     ]
    #     obstruction_indices = np.array([3, 6, 13], dtype=int)
    #     self._domain = ObstructedFlowDomain(*intervals, obstruction_indices)
    def _set_domain(self):
        L = 1  # 7 # 7 distorts MESHKLE that cannot enforce anisotropy
        domain_bounds = [0, L, 0, 1]
        nsubdomains_1d = [5, 4]
        intervals = [
            np.array(
                [
                    0,
                    2 * L / 7,
                    3 * L / 7,
                    4 * L / 7,
                    5 * L / 7,
                    # 6 * L / 7,
                    L,
                ]
            ),
            np.linspace(*domain_bounds[2:], nsubdomains_1d[1] + 1),
        ]
        obstruction_indices = np.array([3, 6, 13], dtype=int)
        # add this to ist above 6 * L / 7, and use
        # creates obstruction at right end of domain, but flow field
        # is not useful because flow slows down to much due to noslip
        # on right most obstruction
        # obstruction_indices = np.array([3, 7, 11, 15], dtype=int)
        self._domain = ObstructedFlowDomain(*intervals, obstruction_indices)

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
        self._vel_component_basis = self._basis["u"].with_element(
            vel_component_elem
        )

    def _set_mesh(self, nrefine: int):
        quad_mesh = self._domain.setup_unrefined_quad_mesh()
        if self._use_quadmesh:
            mesh = quad_mesh
        else:
            mesh = quad_mesh.to_meshtri()
        self._mesh = mesh.refined(nrefine).with_boundaries(
            self._domain._bndry_definitions
        )
        self._set_basis(self._use_quadmesh)

    def nparams(self) -> int:
        return 2

    def _set_params(self, params: np.ndarray):
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
            viscosity=self._domain._L / self._reynolds_num,
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


class KLEHyperParameters:
    def __init__(
        self, lenscale: float, sigma: float, matern_nu: float, nterms: int
    ):
        self.lenscale = lenscale
        self.sigma = sigma
        self.matern_nu = matern_nu
        self.nterms = nterms


class ObstructedAdvectionDiffusion(SteadyParameterizedFEModel):
    def __init__(
        self,
        nstokes_refine: int,
        nadvec_diff_refine: int,
        deltat: float,
        final_time: float,
        kle_hyperparams: KLEHyperParameters,
        navier_stokes: bool = False,
        stokes_params: np.ndarray = None,
    ):
        self._stokes_model = ObstructedStokesFlow(
            nstokes_refine, navier_stokes
        )
        self._setup_advec_diff_mesh(nadvec_diff_refine)
        self._init_time = 0
        self._final_time = final_time
        self._deltat = deltat
        self._init_condition = self._basis.project(
            FEMScalarFunctionFromCallable(lambda x: x[0] * 0)
        )
        self._kle_hyperparams = kle_hyperparams
        self._initialize_forcing()
        # initialize forcing function to be the mean of the kle
        print(self._kle)
        if stokes_params is not None:
            # fix the velocity field for all advection simulations
            self._set_velocity_field(stokes_params)
            self._set_params(np.zeros((self._kle._nterms,)))
        else:
            # should be easy just need to do
            raise NotImplementedError
        self._setup_advec_diff_mesh(nadvec_diff_refine)
        self._set_boundary_conditions()
        self._set_physics_and_solver()

    def _set_velocity_field(self, stokes_params: np.ndarray):
        self._stokes_model.set_params(stokes_params)
        sol = self._stokes_model.solve()
        vel = self._stokes_model.split_solution(sol)[0]
        element = (
            ElementQuad2()
            if self._stokes_model._use_quadmesh
            else ElementTriP2()
        )
        vel_component_basis = self._stokes_model._basis["u"].with_element(
            element
        )
        # vel_component_basis = self._stokes_model._vel_component_basis
        nodal_vels = []
        for ii in range(2):
            vel_comp = self._stokes_model.split_velocity(vel)[ii]
            # convert to nodes on advec_diff mesh
            # which can be different to stokes mesh
            quadrature_vel = self._basis.project(
                vel_component_basis.interpolator(vel_comp)
            )
            nodal_vel = self._basis.interpolate(quadrature_vel)
            nodal_vels.append(nodal_vel)
            # skfemplot(self._basis, quadrature_vel)
            # import matplotlib.pyplot as plt
            # plt.show()
        self._vel = np.stack(nodal_vels, axis=1)

    def _initialize_forcing(self):
        # Local coordinates of quadrature points on the reference element
        local_quad_points, local_quad_weights = self._basis.quadrature
        # Global coordinates of quadrature points for all elements
        quad_points_tensor = self._basis.mapping.F(local_quad_points)
        print(
            local_quad_weights.shape,
            quad_points_tensor.shape,
            (quad_points_tensor.shape[1] * quad_points_tensor.shape[2]),
        )
        self._nelements = quad_points_tensor.shape[1]
        quad_points = quad_points_tensor.reshape(
            quad_points_tensor.shape[0], -1
        )
        assert np.allclose(
            quad_points[:, : quad_points_tensor.shape[2]],
            quad_points_tensor[:, 0, :],
        )
        quad_weights = np.hstack(
            [local_quad_weights] * quad_points_tensor.shape[1]
        )
        quad_weights = None
        # import matplotlib.pyplot as plt
        self._kle = MeshKLE(
            quad_points,
            self._kle_hyperparams.lenscale,
            self._kle_hyperparams.sigma,
            0.0,
            True,
            self._kle_hyperparams.matern_nu,
            quad_weights,
            self._kle_hyperparams.nterms,
            backend=NumpyMixin,
        )
        print("KLE built")
        self._forcing = FEMScalarFunctionFromCallable(self._kle)
        self._nadvecdiff_params = self._kle._nterms

    def _setup_advec_diff_mesh(self, resolution: int):
        self._mesh = (
            self._stokes_model._domain.setup_unrefined_quad_mesh()
            .refined(resolution)
            .with_boundaries(
                self._stokes_model._domain._bndry_definitions,
                boundaries_only=True,
            )
        )
        # use linear basis so kernel matrix used to construct
        # kle is smaller
        # self._element, intorder = ElementQuad2(), 4
        self._element, intorder = ElementQuad1(), 2
        # intorder=4 is minimum order to integrate quadratic basis
        # intorder=1 is minimum order to integrate linear basis
        self._basis = Basis(self._mesh, self._element, intorder=intorder)

    def nparams(self) -> int:
        nparams = self._nadvecdiff_params
        if not hasattr(self, "_vel"):
            nparams += self._stokes_model.nparams()
        return nparams

    def _set_params(self, params: np.ndarray):
        if not hasattr(self, "_vel"):
            self._stokes_model.set_params(params[self._nadvecdiff_params :])
        self._kle(params[: self._nadvecdiff_params, None])

    def plot_forcing(self, params: np.ndarray, **kwargs):
        kle_vals = self._kle(params[:, None])
        return self._plot_kle_quantity(kle_vals)

    def plot_kle_eigenvecs(self, eigvec_indices: np.ndarray, axs, **kwargs):
        if len(axs) != eigvec_indices.shape[0]:
            raise ValueError("must provide one axes for each eigenvector")
        for ii, idx in enumerate(eigvec_indices):
            self._plot_kle_quantity(
                self._kle._eig_vecs[:, idx], ax=axs[ii], **kwargs
            )

    def _plot_kle_quantity(self, kle_vals: np.ndarray, **kwargs):
        # kle is defined on quadrature points so must convert to nodal values
        kle_vals = kle_vals.reshape((self._nelements, -1), order="C")
        # basis.interpolate: Interpolate a solution vector to quadrature points
        # basis.project: project onto basis nodes
        proj_vals = self._basis.project(kle_vals)
        return skfemplot(self._basis, proj_vals, **kwargs)

    def _set_boundary_conditions(self):
        # set no flux conditions for all but left (inlet) and right
        # (outlet) boundaries
        N_bndry_names = [
            f"obs{ii}"
            for ii in range(
                self._stokes_model._domain._obstruction_indices.shape[0]
            )
        ] + ["bottom", "top"]
        N_bndry_funs = [
            FEMScalarFunctionFromCallable(self._stokes_model._zero_bndry_fun)
            for name in N_bndry_names
        ]
        R_bndry_names = ["left", "right"]
        nominal_concentration = 1.0
        alpha = 0.1
        R_bndry_funs = [
            FEMScalarFunctionFromCallable(
                lambda x: x[0] * 0 + nominal_concentration * alpha
            ),
            FEMScalarFunctionFromCallable(
                lambda x: x[0] * 0 + nominal_concentration * alpha
            ),
        ]
        R_bndry_consts = [0.1, 0.1]
        self._bndry_conds = BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
            None,
            None,
            N_bndry_names,
            N_bndry_funs,
            R_bndry_names,
            R_bndry_funs,
            R_bndry_consts,
        )

    def _kle_values_on_quadrature_points(self, x: np.ndarray) -> np.ndarray:
        return self._kle(self._params[:, None]).reshape(
            (self._nelements, -1), order="C"
        )

    def _set_physics_and_solver(self):
        forc_fun = FEMScalarFunctionFromCallable(
            self._kle_values_on_quadrature_points
        )
        diffusivity = 1.0
        diff_op = FEMNonLinearOperatorFromCallable(
            lambda x, u: diffusivity + u * 0,
            lambda x, u: u * 0,
        )
        react_op = FEMNonLinearOperatorFromCallable(
            lambda x, u: 0 * u, lambda x, u: 0 * u
        )
        self._physics = NonLinearAdvectionDiffusionReaction(
            self._mesh,
            self._element,
            self._basis,
            self._bndry_conds,
            FEMScalarFunctionFromCallable(lambda x: x[0] * diffusivity),
            forc_fun,
            FEMVectorFunctionFromCallable(lambda x: self._vel),
            # FEMVectorFunctionFromCallable(lambda x: x * 0, swapaxes=False),
            diff_op,
            react_op,
        )
        self._solver = TransientPDE(self._physics, self._deltat, "im_beuler1")

    def solve(self) -> np.ndarray:
        if not hasattr(self, "_params"):
            raise AttributeError("must call set_params")
        sols, times = self._solver.solve(
            self._init_condition,
            self._init_time,
            self._final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2},
        )
        return sols, times

    def plot_concentration_snapshot(self, sol: np.ndarray, **kwargs):
        return skfemplot(self._basis, sol, **kwargs)

    def plot_concentration_snapshots(
        self, sols: np.ndarray, sol_indices: np.ndarray, axs, **kwargs
    ):
        if sols.ndim != 2:
            raise ValueError(
                "sols must be a 2D array containing the solution snaphsots "
                "at all times"
            )
        if len(axs) != sol_indices.shape[0]:
            raise ValueError("must provide one axes for each sol")
        for ii, idx in enumerate(sol_indices):
            self.plot_concentration_snapshot(sols[:, idx], ax=axs[ii])

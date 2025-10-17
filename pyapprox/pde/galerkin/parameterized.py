from functools import partial

import numpy as np
from skfem import Basis, MeshTri1, ElementTriP1, ElementTriP2

from pyapprox.pde.galerkin.physics import Helmholtz, BoundaryConditions
from pyapprox.pde.galerkin.solvers import SteadyStatePDE
from pyapprox.pde.galerkin.functions import (
    FEMScalarFunctionFromCallable,
    FEMNonLinearOperatorFromCallable,
)


def _2d_bndry_segment_1(m, x1, y1, y2, tol, x):
    return (
        np.isclose(x[1] - m * (x[0] - x1) - y1, 0, tol)
        & ((m * (x[0] - x1) + y1) < max(y1, y2) + tol)
        & ((m * (x[0] - x1) + y1) > min(y1, y2) - tol)
    )


def _2d_bndry_segment_2(x1, y1, y2, tol, x):
    return (
        np.isclose(x[0], x1, tol) & ((x[1] - y1) > -tol) & ((x[1] - y2) < tol)
    )


def _2d_bndry_segment_3(x1, x2, y1, tol, x):
    return (
        np.isclose(x[1], y1, tol) & ((x[0] - x1) > -tol) & ((x[0] - x2) < tol)
    )


def get_2d_bndry_segment_fun(x1, y1, x2, y2, tol=1e-10):
    """
    Define boundary segment along the line between (x1,y1) and (x2,y2)
    Assumes x1,y1 x2,y2 come in clockwise order
    """
    tol = 1e-12
    if abs(x2 - x1) > tol and abs(y2 - y1) > tol:
        m = (y2 - y1) / (x2 - x1)
        bndry_fun = partial(_2d_bndry_segment_1, m, x1, y1, y2, tol)
    elif abs(x2 - x1) < tol:
        II = np.argsort([y1, y2])
        x1, x2 = np.array([x1, x2])[II]
        y1, y2 = np.array([y1, y2])[II]
        bndry_fun = partial(_2d_bndry_segment_2, x1, y1, y2, tol)
    else:
        II = np.argsort([x1, x2])
        x1, x2 = np.array([x1, x2])[II]
        y1, y2 = np.array([y1, y2])[II]
        bndry_fun = partial(_2d_bndry_segment_3, x1, x2, y1, tol)
    return bndry_fun


def get_vertices_of_polygon(ampothem, nedges):
    assert np.issubdtype(type(nedges), np.integer)
    circumradius = ampothem / np.cos(np.pi / nedges)
    vertices = []
    for t in np.linspace(0, 2 * np.pi, nedges + 1)[:-1] + np.pi / nedges:
        vertex = [circumradius * np.cos(t), circumradius * np.sin(t)]
        vertices.append(vertex)
    vertices = np.array(vertices).T
    return vertices


def get_polygon_boundary_segments(
    ampothem, nedges, nsegments_per_edge=None, cumulative_segment_sizes=None
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
            bndry_seg = get_2d_bndry_segment_fun(p1[0], p1[1], p2[0], p2[1])
            bndry_funs.append(bndry_seg)
            p1 = p2.copy()
        x1, y1 = x2, y2
    return bndry_funs


class OctagonalBoundaries:
    def __init__(self, tol=1e-10):
        self.tol = tol
        self.ampothem = 1.5
        self.nedges = 8
        self.nsegments_per_edge = 3
        # self.cumulative_segment_sizes = [0.0625, 0.9375, 1.]
        self.cumulative_segment_sizes = [0.2, 0.8, 1.0]
        # self.nsegments_per_edge = 1
        # self.cumulative_segment_sizes = [1.]

    def boundaries_dict(self):
        bndry_funs = get_polygon_boundary_segments(
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


class OctagonalHelmholtz:
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

    def set_params(self, params: np.ndarray):
        if params.shape != (3,):
            raise ValueError(f"{params.shape=} but must be (2,)")
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
        # data = np.load("octagonal-mesh.npz")
        # coords, cells = data["coords"], data["cells"]
        coords, cells = self.create_octagonal_mesh()
        bndrys_dict = OctagonalBoundaries().boundaries_dict()
        base_mesh = MeshTri1(coords.T, cells.T).with_boundaries(
            bndrys_dict, boundaries_only=True
        )
        # draw(base_mesh, ax=plt.figure(figsize=(8, 6)).gca())
        # plt.show()
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

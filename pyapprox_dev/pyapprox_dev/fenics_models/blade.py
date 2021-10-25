#!/usr/bin/env python
from scipy import io
from pyapprox_dev.fenics_models.advection_diffusion import \
    run_steady_state_model
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import math
import dolfin as dl
import mshr


def sort_points_in_anti_clockwise_order(coords):
    coords = coords.T
    center = tuple(map(operator.truediv, reduce(
        lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    sorted_coords = sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    sorted_coords = np.array(sorted_coords)[::-1].T
    return sorted_coords, center


def numpy_to_dolfin_points(points):
    vertices = []
    for ii in range(points.shape[1]):
        point = points[:, ii]
        vertices.append(dl.Point(point[0], point[1]))
    return vertices


def generate_blade_domain():
    path = os.path.abspath(os.path.dirname(__file__))
    data = io.loadmat(os.path.join(path, 'data', 'blade_geometry.mat'))
    bedge = data['bedge']
    xy = data['xy']
    points, bndry = [], []
    for ii in range(bedge.shape[1]):
        kn1 = bedge[0, ii]-1
        kn2 = bedge[1, ii]-1
        points.append(xy[:, kn1])
        points.append(xy[:, kn2])
        bndry.append(bedge[2, ii])
        bndry.append(bedge[2, ii])
    points = np.asarray(points).T
    bndry = np.asarray(bndry)

    coords = points[:, bndry == 0]
    sorted_coords, center = sort_points_in_anti_clockwise_order(coords)
    vertices = numpy_to_dolfin_points(sorted_coords)
    airfoil_domain = mshr.Polygon(vertices)
    cooling_domains = []
    domain = airfoil_domain
    for ii in range(1, 3):
        coords = points[:, bndry == ii]
        sorted_coords, center = sort_points_in_anti_clockwise_order(coords)
        vertices = numpy_to_dolfin_points(sorted_coords)
        cooling_domains.append(mshr.Polygon(vertices))
        domain -= cooling_domains[-1]
    # last cooling domain requires special treatment
    coords = points[:, bndry == 3]
    coords = coords[:, (coords[0] >= 0.45) & (coords[0] <= 0.7)]
    sorted_coords, center = sort_points_in_anti_clockwise_order(coords)
    vertices = numpy_to_dolfin_points(sorted_coords)
    cooling_domains.append(mshr.Polygon(vertices))
    domain -= cooling_domains[-1]

    coords = points[:, bndry == 3]
    coords = coords[:, (np.absolute(coords[1]) <= 0.01) & (coords[0] >= 0.7)]
    sorted_coords, center = sort_points_in_anti_clockwise_order(coords)
    vertices = numpy_to_dolfin_points(sorted_coords)
    cooling_domains.append(mshr.Polygon(vertices))
    domain -= cooling_domains[-1]

    return domain


class CoolingPassage1(dl.SubDomain):
    def inside(self, x, on_boundary):
        cond1 = ((abs(x[1]) <= 0.045+1e-12)
                 and (x[0] >= 0.1-1e-12) and (x[0] <= 0.2+1e-12))
        cond2 = ((abs(x[1]) <= 0.035+1e-12)
                 and (x[0] <= 0.1+1e-12) and (x[0] >= 0.05-1e-12))
        cond3 = ((abs(x[1]) <= 0.02+1e-12) and
                 (x[0] <= 0.05+1e-12) and (x[0] >= 0.025-1e-12))
        return ((cond1 or cond2 or cond3) and on_boundary)


class CoolingPassage2(dl.SubDomain):
    def inside(self, x, on_boundary):
        cond1 = ((abs(x[1]) <= 0.045+1e-12) and
                 (x[0] <= 0.45+1e-12) and (x[0] >= 0.25-1e-12))
        return (cond1 and on_boundary)


class CoolingPassage3(dl.SubDomain):
    def inside(self, x, on_boundary):
        cond1 = ((abs(x[1]) <= 0.01+1e-12) and (x[0] >= 0.5))
        cond2 = ((abs(x[1]) < 0.035) and (x[0] >= 0.5) and (x[0] <= 0.7))
        cond3 = ((abs(x[1]) < 0.04+1e-12) and (x[0] >= 0.5) and (x[0] <= 0.6))
        return ((cond1 or cond2 or cond3) and on_boundary)


class Airfoil(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class AirfoilHeatTransferModel(object):
    def __init__(self, degree):
        self.degree = degree
        self.num_config_vars = 1
        self.domain = generate_blade_domain()

    def get_boundary_conditions_and_function_space(self, random_sample):
        function_space = dl.FunctionSpace(self.mesh, "Lagrange", degree)
        t_c1, t_c2, t_c3, thermal_conductivity, h_le, h_te = random_sample
        airfoil_expr = dl.Expression(
            'h_te+(h_le-h_te)*std::exp(-4*std::pow(x[0]/(chord*lhot),2))',
            degree=self.degree, h_le=h_le, h_te=h_te, lhot=0.05, chord=0.04)
        boundary_conditions = [
            ['dirichlet', Airfoil(), airfoil_expr],
            ['dirichlet', CoolingPassage1(), dl.Constant(t_c1)],
            ['dirichlet', CoolingPassage2(), dl.Constant(t_c2)],
            ['dirichlet', CoolingPassage3(), dl.Constant(t_c3)]]
        return boundary_conditions, function_space

    def initialize_random_expressions(self, random_sample):
        """
        Parameters
        ----------
        random_sample : np.ndarray (6)
            Realization of model uncertainties. The uncertainties are listed
            below in the order they appear in ``random_sample``

        h_le : float
            Leading edge heat transfer coefficient

        h_lt : float
            Tail edge heat transfer coefficient

        thermal_conductivity : float
            Blade thermal conductivity

        t_c1 : float
            First passage coolant temperature

        t_c2 : float
            Second passage coolant temperature

        t_c3 : float
            Thrid passage coolant temperature
        """
        t_c1, t_c2, t_c3, thermal_conductivity, h_le, h_te = random_sample
        kappa = dl.Constant(thermal_conductivity)
        forcing = dl.Constant(0)
        boundary_conditions, function_space = \
            self.get_boundary_conditions_and_function_space(random_sample)
        return kappa, forcing, boundary_conditions, function_space

    def get_mesh_resolution(self, resolution_level):
        resolution = 10*(2**resolution_level)
        return int(resolution)

    def solve(self, samples):
        assert samples.ndim == 2
        assert samples.shape[1] == 1
        resolution_levels = samples[-self.num_config_vars:, 0]
        self.mesh = self.get_mesh(
            self.get_mesh_resolution(resolution_levels[-1]))
        random_sample = samples[:-self.num_config_vars, 0]
        kappa, forcing, boundary_conditions, function_space =\
            self.initialize_random_expressions(random_sample)
        sol = run_steady_state_model(
            function_space, kappa, forcing, boundary_conditions)
        return sol

    def qoi_functional(self, sol):
        vol = dl.assemble(dl.Constant(1)*dl.dx(sol.function_space().mesh()))
        bulk_temperature = dl.assemble(
            sol*dl.dx(sol.function_space().mesh()))/vol
        print('Bulk (average) temperature', bulk_temperature)
        return bulk_temperature

    def __call__(self, samples):
        sol = self.solve(samples)
        vals = np.atleast_1d(self.qoi_functional(sol))
        if vals.ndim == 1:
            vals = vals[:, np.newaxis]
        return vals

    def get_mesh(self, resolution):
        mesh = mshr.generate_mesh(self.domain, resolution)
        return mesh


if __name__ == '__main__':

    degree = 1
    model = AirfoilHeatTransferModel(degree)
    sample = np.array([595, 645, 705, 29, 1025, 1000, 0])
    sol = model.solve(sample[:, np.newaxis])

    qoi = model(sample[:, np.newaxis])

    fig, axs = plt.subplots(1, 1)
    p = dl.plot(sol)
    # plt.colorbar(p)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('blade_sol.pdf')
    plt.show()


# #check boundary
# from pyapprox_dev.fenics_models.fenics_utilities import mark_boundaries, collect_dirichlet_boundaries
# boundaries = mark_boundaries(mesh,boundary_conditions)
# dirichlet_bcs = collect_dirichlet_boundaries(
#     function_space,boundary_conditions, boundaries)
# temp = dl.Function(function_space)
# temp.vector()[:]=-1
# for bc in dirichlet_bcs:
#     bc.apply(temp.vector())
# fig,axs=plt.subplots(1,1)
# p=dl.plot(temp)
# plt.colorbar(p)
# #dl.plot(mesh)
# plt.show()

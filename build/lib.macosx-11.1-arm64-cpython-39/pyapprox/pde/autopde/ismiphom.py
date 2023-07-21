"""
Benchmarks for ice-sheet models from
'Benchmark experiments for higher-order and full-Stokes ice sheet
models (ISMIP–HOM)'
https://tc.copernicus.org/articles/2/95/2008/tc-2-95-2008.pdf
"""
import numpy as np
from functools import partial


def get_ISMIPHOM_surface_expr(alpha, x):
    return (-x[0]*np.tan(alpha))[:, None]


def get_ISMIPHOM_grad_surface_expr(alpha, x):
    return np.full((x.shape[1], 1), -np.tan(alpha))


def get_ISMIPHOM_A_bottom_expr(alpha, Lx, Lz, x):
    # Lz is the mean thickness (depth)
    w = 2*np.pi/Lx
    surface_vals = get_ISMIPHOM_surface_expr(alpha, x)
    return (surface_vals-Lz +
            0.5*Lz*np.sin(w*x[0])*np.sin(w*x[1])[:, None])


def get_ISMIPHOM_A_basal_friction(x):
    return np.full((x.shape[1], 1), 1e16)


def get_ISMIPHOM_B_basal_friction(x):
    return get_ISMIPHOM_A_basal_friction(x)


def get_ISMIPHOM_B_bottom_expr(alpha, Lx, Lz, x):
    # Good for 1D velocity models
    w = 2.0*np.pi/Lx
    surface_vals = get_ISMIPHOM_surface_expr(alpha, x)
    bed_vals = (surface_vals-Lz+0.5*Lz*np.sin(w*x[0])[:, None])
    return bed_vals


def get_ISMIPHOM_C_bottom_expr(alpha, Lx, Lz, x):
    surface_vals = get_ISMIPHOM_surface_expr(alpha, x)
    return (surface_vals-Lz)


def get_ISMIPHOM_C_basal_friction(Lx, Lz, x):
    w = 2.0*np.pi/Lx
    return (Lz + Lz*np.sin(w*x[0])*np.sin(w*x[1]))[:, None]


def get_ISMIPHOM_D_bottom_expr(alpha, Lx, Lz, x):
    return get_ISMIPHOM_C_bottom_expr(alpha, Lx, Lz, x)


def get_ISMIPHOM_D_basal_friction(Lx, Lz, x):
    w = 2.0*np.pi/Lx
    return (Lz + Lz*np.sin(w*x[0]))[:, None]


def get_shallow_ice_velocity(surface_vals, grad_surface_vals, depth_vals,
                             friction_vals, z):
    A, rho = 1e-4, 910
    g, n = 9.81, 3
    gamma = 2*A*rho**n*g**n/(n+1)
    # should rho*g*depth_vals/friction_vals be
    # rho*g*(surface_vals-z)/friction_vals)
    return ((gamma*((surface_vals-z)**(n+1)-depth_vals**(n+1)) *
             np.abs(grad_surface_vals)**(n-1)-rho*g*depth_vals/friction_vals) *
            grad_surface_vals)


class ISMIPHOMBenchmark():
    def __init__(self, case, Lx):
        alpha = {"A": 0.5, "B": 0.5, "C": 0.1, "D": 0.1}[case]
        self.alpha = np.pi/180*alpha
        self.surface_fun = partial(get_ISMIPHOM_surface_expr, self.alpha)
        self.grad_surface_fun = partial(
            get_ISMIPHOM_grad_surface_expr, self.alpha)

        Lz = 1  # km
        self.friction_fun, self.bed_fun = {
            "A": (get_ISMIPHOM_A_basal_friction,
                  partial(get_ISMIPHOM_A_bottom_expr, self.alpha, Lx, Lz)),
            "B": (get_ISMIPHOM_B_basal_friction,
                  partial(get_ISMIPHOM_B_bottom_expr, self.alpha, Lx, Lz)),
            "C": (partial(get_ISMIPHOM_C_basal_friction, Lx, Lz),
                  partial(get_ISMIPHOM_C_bottom_expr, self.alpha, Lx, Lz)),
            "D": (partial(get_ISMIPHOM_D_basal_friction, Lx, Lz),
                  partial(get_ISMIPHOM_D_bottom_expr, self.alpha, Lx, Lz)),
        }[case]

    def thickness_fun(self, x):
        return self.surface_fun(x)-self.bed_fun(x)


if __name__ == "__main__":
    import torch
    from pyapprox.pde.autopde.autopde import (
        CartesianProductCollocationMesh, VectorMesh, SteadyStatePDE,
        Function, ShallowShelfVelocities)
    import matplotlib.pyplot as plt

    Lx = 80  # km
    orders = [21]

    benchmark = ISMIPHOMBenchmark("D", Lx=Lx)

    nphys_vars = len(orders)
    vel_bndry_conds = [
        [[None, "P"] for ii in range(nphys_vars*2)]
        for jj in range(nphys_vars)]
    print(vel_bndry_conds)

    nphys_vars = len(orders)
    domain_bounds = np.zeros(nphys_vars*2)
    domain_bounds[1::2] = Lx
    vel_meshes = [CartesianProductCollocationMesh(
        domain_bounds, orders, vel_bndry_conds[ii])
                  for ii in range(nphys_vars)]
    mesh = VectorMesh(vel_meshes)

    def vel_forc_fun(x):
        return np.zeros((x.shape[1], nphys_vars))

    homotopy_alpha = 12
    homotopy_val = 10**(-homotopy_alpha)
    nhomotopy_steps = 1
    init_guess = torch.zeros((mesh.nunknowns, 1), dtype=torch.double)
    # init_guess = torch.cos(
    #    torch.tensor(vel_meshes[0].mesh_pts.T/Lx*np.pi, dtype=torch.double))*0
    # print(init_guess)

    A, rho = 1e-4, 910
    solver = SteadyStatePDE(
        ShallowShelfVelocities(
            mesh, Function(vel_forc_fun),
            Function(benchmark.bed_fun),
            Function(benchmark.friction_fun),
            Function(benchmark.thickness_fun), A, rho, homotopy_val))
    for ii in range(nhomotopy_steps):
        print(solver.physics._homotopy_val)
        sol = solver.solve(
            init_guess, step_size=1, verbosity=2, maxiters=100, tol=2e-8)
        init_guess = torch.tensor(sol)
        homotopy_alpha += 0.1
        solver.physics._homotopy_val = 10**(-homotopy_alpha)

    print(sol)
    nplot_pts_1d = 100
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    vel_meshes[0].plot(
        mesh.split_quantities(sol)[0], nplot_pts_1d, ax=axs[0],
        label="SSA Velocity")
    xx = np.linspace(0, Lx, nplot_pts_1d)[None, :]
    sia_vel = get_shallow_ice_velocity(
        benchmark.surface_fun(xx),
        benchmark.grad_surface_fun(xx),
        benchmark.thickness_fun(xx),
        benchmark.friction_fun(xx),
        benchmark.surface_fun(xx))
    # axs[0].plot(xx[0, :], sia_vel, label="SIA Velocity")
    vel_meshes[0].plot(
        benchmark.bed_fun(vel_meshes[0].mesh_pts), nplot_pts_1d, ax=axs[1],
        label="Bed")
    vel_meshes[0].plot(
        benchmark.surface_fun(vel_meshes[0].mesh_pts), nplot_pts_1d, ax=axs[1],
        label="Surface")
    axs[1].plot(xx[0, :], benchmark.friction_fun(xx), label="Friction")
    axs[0].legend()
    # axs[0].set_ylim(0, 200)
    axs[1].legend()
    plt.show()

import unittest
import numpy as np
import dolfin as dl
from pyapprox_dev.fenics_models.fenics_utilities import *


class TestFenicsUtilities(unittest.TestCase):
    def test_constrained_newton_energy_solver(self):
        L, nelem = 1, 201

        mesh = dl.IntervalMesh(nelem, -L, L)
        Vh = dl.FunctionSpace(mesh, "CG", 2)

        forcing = dl.Constant(1)

        dirichlet_bcs = [dl.DirichletBC(Vh, dl.Constant(0.0), on_any_boundary)]
        bc0 = dl.DirichletBC(Vh, dl.Constant(0.0), on_any_boundary)

        uh = dl.TrialFunction(Vh)
        vh = dl.TestFunction(Vh)
        F = dl.inner((1+uh**2)*dl.grad(uh), dl.grad(vh))*dl.dx-forcing*vh*dl.dx
        F += uh*vh*dl.inner(dl.nabla_grad(uh), dl.nabla_grad(uh))*dl.dx
        u = dl.Function(Vh)
        F = dl.action(F, u)
        parameters = {"symmetric": True, "newton_solver": {
            "relative_tolerance": 1e-12, "report": True,
            "linear_solver": "cg", "preconditioner": "petsc_amg"}}
        dl.solve(F == 0, u, dirichlet_bcs, solver_parameters=parameters)
        # dl.plot(uh)
        # plt.show()

        F = 0.5*(1+uh**2)*dl.inner(dl.nabla_grad(uh),
                                   dl.nabla_grad(uh))*dl.dx - forcing*uh*dl.dx
        #F = dl.inner((1+uh**2)*dl.grad(uh),dl.grad(vh))*dl.dx-forcing*vh*dl.dx
        u_newton = dl.Function(Vh)
        F = dl.action(F, u_newton)
        constrained_newton_energy_solve(
            F, u_newton, dirichlet_bcs=dirichlet_bcs, bc0=bc0,
            linear_solver='PETScLU', opts=dict())

        F = 0.5*(1+uh**2)*dl.inner(
            dl.nabla_grad(uh), dl.nabla_grad(uh))*dl.dx - forcing*uh*dl.dx
        u_grad = dl.Function(Vh)
        F = dl.action(F, u_grad)
        grad = dl.derivative(F, u_grad)
        print(F)
        print(grad)
        parameters = {"symmetric": True, "newton_solver": {
            "relative_tolerance": 1e-12, "report": True,
            "linear_solver": "cg", "preconditioner": "petsc_amg"}}
        dl.solve(grad == 0, u_grad, dirichlet_bcs,
                 solver_parameters=parameters)
        error1 = dl.errornorm(u_grad, u_newton, mesh=mesh)
        # print(error)
        error2 = dl.errornorm(u, u_newton, mesh=mesh)
        # print(error)
        # dl.plot(u)
        # dl.plot(u_newton)
        # dl.plot(u_grad)
        # plt.show()
        assert error1 < 1e-15
        assert error2 < 1e-15

    def test_unconstrained_newton_solver(self):
        L, nelem = 1, 201

        mesh = dl.IntervalMesh(nelem, -L, L)
        Vh = dl.FunctionSpace(mesh, "CG", 2)

        forcing = dl.Constant(1)

        dirichlet_bcs = [dl.DirichletBC(Vh, dl.Constant(0.0), on_any_boundary)]
        bc0 = dl.DirichletBC(Vh, dl.Constant(0.0), on_any_boundary)

        uh = dl.TrialFunction(Vh)
        vh = dl.TestFunction(Vh)
        F = dl.inner((1+uh**2)*dl.grad(uh), dl.grad(vh))*dl.dx-forcing*vh*dl.dx
        u = dl.Function(Vh)
        F = dl.action(F, u)
        parameters = {"symmetric": True, "newton_solver": {
            # "relative_tolerance": 1e-8,
            "report": True,
            "linear_solver": "cg", "preconditioner": "petsc_amg"}}
        dl.solve(F == 0, u, dirichlet_bcs, solver_parameters=parameters)
        # dl.plot(uh)
        # plt.show()

        u_newton = dl.Function(Vh)
        F = dl.inner((1+uh**2)*dl.grad(uh), dl.grad(vh))*dl.dx-forcing*vh*dl.dx
        F = dl.action(F, u_newton)
        # Compute Jacobian
        J = dl.derivative(F, u_newton, uh)
        unconstrained_newton_solve(
            F, J, u_newton, dirichlet_bcs=dirichlet_bcs, bc0=bc0,
            # linear_solver='PETScLU',opts=dict())
            linear_solver=None, opts=dict())
        error = dl.errornorm(u, u_newton, mesh=mesh)
        assert error < 1e-15

    def test_mark_boundaries(self):
        """
        Test multiple boundaryies are marked correctly. At one time I was
        not calling boundaries.set_all(9999) which mean markers were using
        unitialized values which caused heisenbug i.e. solutions were vary 
        slightly.

        Compare marking of multiple (but homogeneous and constant) boundaries 
        and thier application to a vector with just one dirichlet boundary.

        Note: ft10_poission_extended.py shows good example of how to apply 
        boundary conditions in general way
        """

        nx, ny, degree = 31, 31, 2
        mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        bc_expr1 = dl.interpolate(dl.Constant(1.), function_space)
        boundary_conditions = \
            get_dirichlet_boundary_conditions_from_expression(
                bc_expr1, 0, 1, 0, 1)
        boundaries = mark_boundaries(mesh, boundary_conditions)
        dirichlet_bcs = collect_dirichlet_boundaries(
            function_space, boundary_conditions, boundaries)

        zeros1 = dl.Function(function_space).vector()
        num_bndrys = len(dirichlet_bcs)
        for ii in range(num_bndrys):
            dirichlet_bcs[ii].apply(zeros1)

        bc_expr2 = dl.interpolate(dl.Constant(1.), function_space)
        bc = dl.DirichletBC(function_space, bc_expr2, 'on_boundary')
        zeros2 = dl.Function(function_space).vector()
        bc.apply(zeros2)
        assert np.allclose(zeros1.get_local(), zeros2.get_local())

    def test_save_fenics_function(self):
        nx, ny, degree = 31, 31, 2
        mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        import os
        import tempfile
        fd, filename = tempfile.mkstemp()

        try:
            function1 = dl.Function(function_space)
            save_fenics_function(function1, filename)
            function2 = load_fenics_function(function_space, filename)
        finally:
            os.remove(filename)
        assert (dl.errornorm(function1, function2)) < 1e-15

        msg = 'Warning save_fenics_function does not work if calling load function again'
        #raise Exception(msg)


if __name__ == "__main__":
    fenics_utilities_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFenicsUtilities)
    unittest.TextTestRunner(verbosity=2).run(fenics_utilities_test_suite)

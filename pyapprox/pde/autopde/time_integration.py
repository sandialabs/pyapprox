import torch
import numpy as np
from functools import partial

from pyapprox.pde.autopde.util import newton_solve


def explicit_runge_kutta_update(sol, deltat, time, rhs, butcher_tableau):
    assert sol.ndim == 1
    assert callable(rhs)
    a_coef, b_coef, c_coef = butcher_tableau
    order = a_coef.shape[0]
    stage_rhs = np.empty((order, sol.shape[0]), dtype=float)
    stage_deltas = np.empty((order, sol.shape[0]), dtype=float)
    stage_rhs[0, :] = rhs(sol, time)
    stage_deltas[0, :] = 0
    new_sol = sol+deltat*b_coef[0]*stage_rhs[0, :]
    tmp = np.zeros((stage_rhs.shape[1]))
    for ii in range(1, order):
        tmp[:] = 0.
        for jj in range(ii):
            tmp += a_coef[ii, jj]*stage_rhs[jj]
        stage_deltas[ii] = deltat*tmp
        stage_rhs[ii] = rhs(
            sol+stage_deltas[ii], time+c_coef[ii]*deltat)
        new_sol += deltat*b_coef[ii]*stage_rhs[ii]
        # print(ii, stage_rhs[ii], '$')
    return new_sol, stage_deltas, stage_rhs


def butcher_tableau_explicit_forward_euler():
    c_coef = np.array([0.])
    b_coef = np.array([1.])
    a_coef = np.zeros((1, 1), dtype=float)
    return a_coef, b_coef, c_coef


def butcher_tableau_explicit_midpoint():
    c_coef = np.array([0., 0.5])
    b_coef = np.array([0., 1.])
    a_coef = np.zeros((2, 2), dtype=float)
    a_coef[1, 0] = 0.5
    return a_coef, b_coef, c_coef


def butcher_tableau_explicit_heun2():
    c_coef = np.array([0., 1.])
    b_coef = np.array([.5, .5])
    a_coef = np.zeros((2, 2), dtype=float)
    a_coef[1, 0] = 1.
    return a_coef, b_coef, c_coef


def butcher_tableau_explicit_order_3_heun():
    c_coef = np.array([0., 1./3., 2./3.])
    b_coef = np.array([1./4., 0., 3/4., 1./6.])
    a_coef = np.zeros((3, 3))
    a_coef[1, 0] = 1./3.
    a_coef[2, 1] = 2./3.
    return a_coef, b_coef, c_coef


def butcher_tableau_explicit_rk4():
    c_coef = np.array([0., 0.5, 0.5, 1.])
    b_coef = np.array([1./6., 1./3., 1./3., 1./6.])
    a_coef = np.zeros((4, 4))
    a_coef[1, 0] = 0.5
    a_coef[2, 1] = 0.5
    a_coef[3, 2] = 1.0
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_backward_euler():
    c_coef = np.array([1.])
    b_coef = np.array([1.])
    a_coef = np.ones((1, 1), dtype=float)
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_crank_nicolson():
    c_coef = np.array([0., 1.])
    b_coef = np.array([1/2, 1/2])
    a_coef = np.array([[0, 0], [1/2, 1/2]])
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_fourth_order_gauss():
    c_coef = np.array([1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6])
    b_coef = np.array([1/2, 1/2])
    a_coef = np.array([[1/4, 1/4-np.sqrt(3)/6], [1/4+np.sqrt(3)/6, 1/4]])
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_sixth_order_gauss():
    c_coef = np.array([1/2-np.sqrt(15)/10, 1/2, 1/2+np.sqrt(15)/10])
    b_coef = np.array([5/18, 4/9, 5/18])
    a_coef = np.array([[5/36, 2/9-np.sqrt(15)/15, 5/36-np.sqrt(15)/30],
                       [5/36+np.sqrt(15)/24, 2/9, 5/36-np.sqrt(15)/24],
                       [5/36+np.sqrt(15)/30, 2/9+np.sqrt(15)/15, 5/36]])
    return a_coef, b_coef, c_coef


def butcher_tableau_diag_implicit_order_2():
    # x = 1/4
    # c_coef = np.array([x, 1-x])
    # b_coef = np.array([1/2, 1/2])
    # a_coef = np.array([[x, 0], [1-2*x, x]])

    x = 1+np.sqrt(2)/2
    c_coef = np.array([x, 1])
    b_coef = np.array([1-x, x])
    a_coef = np.array([[x, 0], [1-x, x]])
    return a_coef, b_coef, c_coef


def butcher_tableau_diag_implicit_order_3():
    x = 0.4358665215
    c_coef = np.array([x, (1+x)/2, 1])
    b_coef = np.array([-3*x**2/2+4*x-1/4, 3*x**2/2-5*x+5/4, x])
    a_coef = np.array([[x, 0, 0], [(1-x)/2, x, 0],
                       [-3*x**2/2+4*x-1/4, 3*x**2/2-5*x+5/4, x]])
    return a_coef, b_coef, c_coef


def create_butcher_tableau(name, return_tensors=True):
    butcher_tableaus = {
        "ex_feuler1": butcher_tableau_explicit_forward_euler,
        "ex_mid2": butcher_tableau_explicit_midpoint,
        "ex_heun2": butcher_tableau_explicit_heun2,
        "ex_heun3": butcher_tableau_explicit_order_3_heun,
        "ex_rk4":  butcher_tableau_explicit_rk4,
        "im_beuler1": butcher_tableau_implicit_backward_euler,
        "im_crank2": butcher_tableau_implicit_crank_nicolson,
        "im_gauss4": butcher_tableau_implicit_fourth_order_gauss,
        "im_gauss6": butcher_tableau_implicit_sixth_order_gauss,
        "diag_im3": butcher_tableau_diag_implicit_order_3,
        }
    if name not in butcher_tableaus:
        raise Exception(f'tableau {name} not implemented')

    tableau = butcher_tableaus[name]()
    # the c array must always be a np.ndarray
    if return_tensors:
        return [torch.tensor(b) for b in tableau[:-1]]+[tableau[-1]]
    return tableau


def implicit_runge_kutta_stage_solution_trad(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns):
    a_coef, c_coef = butcher_tableau[0], butcher_tableau[2]
    nstages = a_coef.shape[0]
    ndof = stage_unknowns.shape[0]//nstages
    stage_rhs = []
    for ii in range(nstages):
        stage_time = time+c_coef[ii]*deltat
        stage_unknown = stage_unknowns[ii*ndof:(ii+1)*ndof]
        stage_sol = sol+stage_unknown
        srhs = rhs(stage_sol, stage_time)
        stage_rhs.append(srhs)
    stage_rhs = torch.cat(stage_rhs)
    tmp = torch.kron(a_coef, torch.eye(ndof))
    new_stage_unknowns = torch.linalg.multi_dot((tmp, deltat*stage_rhs))
    return new_stage_unknowns, stage_rhs


def implicit_runge_kutta_apply_constraints_trad(
        butcher_tableau, stage_unknowns, time, deltat, constraints,
        residual, sol):
    # The following cannot seem to accurately enforce constraints. Need to
    # fix or move on to just support wildey
    raise NotImplementedError()
    nstages = butcher_tableau[0].shape[0]
    ndof = stage_unknowns.shape[0]//nstages
    for ii in range(nstages):
        stage_time = time+butcher_tableau[2][ii]*deltat
        residual[ii*ndof:(ii+1)*ndof] = constraints(
            residual[ii*ndof:(ii+1)*ndof],
            stage_unknowns[ii*ndof:(ii+1)*ndof]+sol,
            stage_time)
    return residual


def implicit_runge_kutta_update_trad(
        sol, stage_unknowns, deltat, time, rhs, butcher_tableau):
    b_coef = butcher_tableau[1]
    nstages = stage_unknowns.shape[0]//sol.shape[0]
    stage_rhs = implicit_runge_kutta_stage_solution(
        sol, deltat, time, rhs, butcher_tableau,
        stage_unknowns)[1]
    new_sol = (sol + deltat*torch.sum(
        b_coef[:, None]*stage_rhs.reshape(
            (nstages, sol.shape[0])), dim=0)).detach()
    return new_sol


def implicit_runge_kutta_stage_solution_wildey(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns):
    a_coef, b_coef, c_coef = butcher_tableau
    nstages = a_coef.shape[0]
    ndof = stage_unknowns.shape[0]//nstages
    stage_rhs, new_stage_unknowns = [], []
    for ii in range(nstages):
        stage_time = time+c_coef[ii]*deltat
        stage_sol = sol.clone()
        for jj in range(nstages):
            stage_sol += a_coef[ii, jj]/b_coef[jj]*(
                stage_unknowns[jj*ndof:(jj+1)*ndof]-sol)
        srhs = rhs(stage_sol, stage_time)
        stage_rhs.append(srhs)
        new_stage_unknowns.append(sol+srhs*b_coef[ii]*deltat)
    stage_rhs = torch.cat(stage_rhs)
    new_stage_unknowns = torch.cat(new_stage_unknowns)
    return new_stage_unknowns, stage_rhs


def implicit_runge_kutta_update_wildey(
        sol, stage_unknowns, deltat, time, rhs, butcher_tableau):
    # the last 4 arguments are not used and only kept to keep inteface
    # between wildey and trad update functions
    nstages = stage_unknowns.shape[0]//sol.shape[0]
    new_sol = sol.clone()
    ndof = stage_unknowns.shape[0]//nstages
    for ii in range(nstages):
        new_sol += (stage_unknowns[ii*ndof:(ii+1)*ndof]-sol)
    return new_sol


def implicit_runge_kutta_apply_constraints_wildey(
        butcher_tableau, stage_unknowns, time, deltat, constraints,
        residual, sol):
    # sol arg not needed for wildey method but kept to maintain
    # consistent api with trad method
    nstages = butcher_tableau[0].shape[0]
    ndof = stage_unknowns.shape[0]//nstages
    for ii in range(nstages):
        stage_time = time+butcher_tableau[2][ii]*deltat
        residual[ii*ndof:(ii+1)*ndof] = constraints(
            residual[ii*ndof:(ii+1)*ndof],
            stage_unknowns[ii*ndof:(ii+1)*ndof], stage_time)
    return residual


# (implicit_runge_kutta_stage_solution, implicit_runge_kutta_update,
#  implicit_runge_kutta_apply_constraints) = (
#      implicit_runge_kutta_stage_solution_trad,
#      implicit_runge_kutta_update_trad,
#      implicit_runge_kutta_apply_constraints_trad
# )

(implicit_runge_kutta_stage_solution, implicit_runge_kutta_update,
 implicit_runge_kutta_apply_constraints) = (
     implicit_runge_kutta_stage_solution_wildey,
     implicit_runge_kutta_update_wildey,
     implicit_runge_kutta_apply_constraints_wildey
)

def implicit_runge_kutta_residual(sol, deltat, time, rhs,
                                  butcher_tableau, stage_unknowns, constraints):
    new_stage_unknowns = implicit_runge_kutta_stage_solution(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns)[0]
    residual = stage_unknowns-new_stage_unknowns
    if constraints is not None:
        residual = implicit_runge_kutta_apply_constraints(
            butcher_tableau, stage_unknowns, time, deltat, constraints,
            residual, sol)
    return residual


class ImplicitRungeKutta():
    def __init__(self, deltat, rhs, tableau_name="beuler1",
                 constraints_fun=None):
        self._tableau_name = tableau_name
        self._butcher_tableau = create_butcher_tableau(self._tableau_name)
        self._deltat = deltat
        self._rhs = rhs
        self._constraints_fun = constraints_fun

        self._sol = None
        self._time = None

    def _residual_fun(self, stage_unknowns):
        return implicit_runge_kutta_residual(
            self._sol, self._deltat, self._time, self._rhs,
            self._butcher_tableau, stage_unknowns,
            constraints=self._constraints_fun)

    def update(self, sol, time, init_guesses):
        self._time = time
        self._sol = sol
        init_guess = torch.cat(init_guesses)
        init_guess.requires_grad = True
        stage_unknowns = newton_solve(self._residual_fun, init_guess)
        return implicit_runge_kutta_update(
            sol, stage_unknowns, self._deltat, self._time, self._rhs,
            self._butcher_tableau)

    def integrate(self, init_sol, init_time, final_time, verbosity=0):
        sols = []
        time = init_time
        if type(init_sol) == np.ndarray:
            sol = torch.tensor(init_sol)
        else:
            sol = init_sol.clone()
        if sol.ndim == 2:
            sol = sol[:, 0]
        sols.append(sol.detach().numpy())
        while time < final_time:
            if verbosity > 0:
                print("Time", time)
            sol = self.update(
                sol, time, [sol.clone()]*self._butcher_tableau[0].shape[0])
            sols.append(sol.detach().numpy())
            time += min(self._deltat, final_time-time)
        sols = np.array(sols).T
        return sols

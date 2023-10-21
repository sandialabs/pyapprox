import torch
import numpy as np

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
    return new_sol, stage_deltas, stage_rhs


def butcher_tableau_explicit_forward_euler():
    c_coef = np.array([0.])
    b_coef = np.array([1.])
    a_coef = np.zeros((1, 1), dtype=float)
    return a_coef, b_coef, c_coef


# will not work with wildey method because some b entries are zero
# def butcher_tableau_explicit_midpoint():
#     c_coef = np.array([0., 0.5])
#     b_coef = np.array([0., 1.])
#     a_coef = np.zeros((2, 2), dtype=float)
#     a_coef[1, 0] = 0.5
#     return a_coef, b_coef, c_coef

def butcher_tableau_explicit_heun2():
    c_coef = np.array([0., 1.])
    b_coef = np.array([.5, .5])
    a_coef = np.zeros((2, 2), dtype=float)
    a_coef[1, 0] = 1.
    return a_coef, b_coef, c_coef

# will not work with wildey method because some b entries are zero
# def butcher_tableau_explicit_heun3():
#     c_coef = np.array([0., 1./3., 2./3.])
#     b_coef = np.array([1./4., 0., 3/4., 1./6.])
#     a_coef = np.zeros((3, 3))
#     a_coef[1, 0] = 1./3.
#     a_coef[2, 1] = 2./3.
#     return a_coef, b_coef, c_coef


def butcher_tableau_explicit_rk3():
    c_coef = np.array([0., 1., 1./2.])
    b_coef = np.array([1./6., 1./6., 2./3.])
    a_coef = np.zeros((3, 3))
    a_coef[1, 0] = 1
    a_coef[2, :2] = 1/4
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


def butcher_tableau_implicit_third_order_crouzeix():
    c_coef = np.array([1/2+np.sqrt(3)/6, 1/2-np.sqrt(3)/6])
    b_coef = np.array([1/2, 1/2])
    a_coef = np.array([[1/2+np.sqrt(3)/6, 0],
                       [-np.sqrt(3)/3, 1/2+np.sqrt(3)/6]])
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_fourth_order_crouzeix():
    t = 2/np.sqrt(3)*np.cos(np.pi/18)
    c_coef = np.array([(1+t)/2, 1/2, (1-t)/2])
    b_coef = np.array([1/(6*t**2), 1-1/(3*t**2), 1/(6*t**2)])
    a_coef = np.array([[(1+t)/2, 0, 0],
                       [-t/2, (1+t)/2, 0],
                       [1+t, -(1+2*t), (1+t)/2]])
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


# DO NOT WORK AS WRITTEN
# def butcher_tableau_diag_implicit_order_3():
#     x = 0.4358665215
#     c_coef = np.array([x, (1+x)/2, 1])
#     b_coef = np.array([-3*x**2/2+4*x-1/4, 3*x**2/2-5*x+5/4, x])
#     a_coef = np.array([[x, 0, 0], [(1-x)/2, x, 0],
#                        [-3*x**2/2+4*x-1/4, 3*x**2/2-5*x+5/4, x]])
#     return a_coef, b_coef, c_coef


# def butcher_tableau_diag_implicit_order_3_w_4_stages():
#     c_coef = np.array([1/2, 2/3, 1/2, 1])
#     b_coef = np.array([3/2, -3/2, 1/2, 1/2])
#     a_coef = np.array([
#         [1/2, 0, 0, 0], [1/6, 1/2, 0, 0],
#         [-1/2, 1/2, 1/2, 0], [3/2, -3/2, 1/2, 1/2]])
#     return a_coef, b_coef, c_coef


def create_butcher_tableau(name, return_tensors=True):
    butcher_tableaus = {
        "ex_feuler1": butcher_tableau_explicit_forward_euler,
        "ex_heun2": butcher_tableau_explicit_heun2,
        "ex_rk3": butcher_tableau_explicit_rk3,
        "ex_rk4":  butcher_tableau_explicit_rk4,
        "im_beuler1": butcher_tableau_implicit_backward_euler,
        "im_crank2": butcher_tableau_implicit_crank_nicolson,
        "im_gauss4": butcher_tableau_implicit_fourth_order_gauss,
        "im_gauss6": butcher_tableau_implicit_sixth_order_gauss,
        "im_crouzeix3": butcher_tableau_implicit_third_order_crouzeix,
        "im_crouzeix4": butcher_tableau_implicit_fourth_order_crouzeix,
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
    stage_rhs = implicit_runge_kutta_stage_solution_trad(
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
        # assume rhs returns jac but ignore it
        srhs = rhs(stage_sol, stage_time)[0]
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


def implicit_runge_kutta_residual(
        stage_solution_fun, stage_constraints_fun,
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns, constraints):
    new_stage_unknowns = stage_solution_fun(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns)[0]
    residual = stage_unknowns-new_stage_unknowns
    if constraints is not None:
        residual = stage_constraints_fun(
            butcher_tableau, stage_unknowns, time, deltat, constraints,
            residual, sol)
    return residual, None


def diag_runge_kutta_stage_solution(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns, ii):
    a_coef, b_coef, c_coef = butcher_tableau
    nstages = ii+1
    ndof = stage_unknowns.shape[0]//nstages
    active_stage_time = time+c_coef[ii]*deltat
    # active_sol : u(z_i)
    active_stage_sol = sol.clone()
    for jj in range(nstages):
        active_stage_sol += a_coef[ii, jj]/b_coef[jj]*(
            stage_unknowns[jj*ndof:(jj+1)*ndof]-sol)
    # shrs : k_i = f(active_stage_time, u(z_i))
    srhs, jac = rhs(active_stage_sol, active_stage_time)
    new_active_stage_unknowns = (sol+srhs*b_coef[ii]*deltat)
    return new_active_stage_unknowns, srhs, jac


def diag_runge_kutta_residual(
        ii, sol, deltat, time, rhs, butcher_tableau, stage_unknowns,
        constraints):
    nstages = ii+1  # the number of computed stages plus the active stage
    ndof = stage_unknowns.shape[0]//nstages
    out = diag_runge_kutta_stage_solution(
        sol, deltat, time, rhs, butcher_tableau, stage_unknowns, ii)
    new_active_stage_unknowns = out[0]
    residual = stage_unknowns[ii*ndof:(ii+1)*ndof]-new_active_stage_unknowns
    stage_time = time+butcher_tableau[2][ii]*deltat
    if out[2] is not None:
        stage_jac = butcher_tableau[1][ii]*deltat*out[2]
        jac = torch.eye(ndof, dtype=torch.double)-(
            butcher_tableau[0][ii, ii]/butcher_tableau[1][ii]*stage_jac)
        # jac = ((butcher_tableau[0][ii, ii]/butcher_tableau[1][ii] *
        #         butcher_tableau[1][ii]*deltat)*out[2])
        # jac.diagonal().copy_(1-torch.diagonal(jac))
    else:
        jac = None
    if constraints is not None:
        residual, jac = constraints(
            residual, jac, stage_unknowns[ii*ndof:(ii+1)*ndof], stage_time)
    return residual, jac


class ImplicitRungeKutta():
    def __init__(self, deltat, rhs, tableau_name="beuler1",
                 constraints_fun=None, auto=True):
        self._tableau_name = tableau_name
        self._butcher_tableau = create_butcher_tableau(self._tableau_name)
        self._update = self._set_update(self._butcher_tableau)
        # self._update = self._full_runge_kutta_update
        self._deltat = deltat
        self._rhs = rhs
        self._constraints_fun = constraints_fun
        self._newton_kwargs = {}
        self._auto = auto

        self._res_sol = None
        self._res_time = None
        self._res_deltat = None

        # only for diag RK methods
        self._active_stage_idx = None
        self._computed_stage_unknowns = None

    def _set_update(self, butcher_tableau):
        if np.allclose(
                torch.triu(butcher_tableau[0], diagonal=1), 0, atol=1e-15):
            return self._diag_runge_kutta_update
        return self._full_runge_kutta_update

    def _residual_fun(self, stage_unknowns):
        return implicit_runge_kutta_residual(
            implicit_runge_kutta_stage_solution_wildey,
            implicit_runge_kutta_apply_constraints_wildey,
            self._res_sol, self._res_deltat, self._res_time, self._rhs,
            self._butcher_tableau, stage_unknowns,
            constraints=self._constraints_fun)

    def _full_runge_kutta_update(self, sol, time, deltat, init_guesses):
        # Does not currently support manually computed jacobian of residual
        self._res_time = time
        # different to self._delta only at final time step if
        # final_time is not an integer multiple of self._deltat
        self._res_deltat = deltat
        self._res_sol = sol
        init_guess = torch.cat(init_guesses)
        init_guess.requires_grad = True
        stage_unknowns = newton_solve(
            self._residual_fun, init_guess).detach()
        return implicit_runge_kutta_update_wildey(
            self._res_sol, stage_unknowns, self._res_deltat, self._res_time,
            self._rhs, self._butcher_tableau)

    def _diag_residual_fun(self, active_stage_unknowns):
        stage_unknowns = torch.cat(
            self._computed_stage_unknowns+[active_stage_unknowns])
        res = diag_runge_kutta_residual(
            self._active_stage_idx,  self._res_sol, self._res_deltat,
            self._res_time, self._rhs, self._butcher_tableau, stage_unknowns,
            constraints=self._constraints_fun)
        return res

    def _diag_runge_kutta_update(self, sol, time, deltat, init_guess):
        if type(init_guess) != list and len(init_guess) == 1:
            raise ValueError(
                "init_guess must be a list of tensors for the 1st stage")
        init_guess = init_guess[0]

        self._res_time = time
        # different to self._delta only at final time step if
        # final_time is not an integer multiple of self._deltat
        self._res_deltat = deltat
        self._res_sol = sol

        self._computed_stage_unknowns = []
        nstages = self._butcher_tableau[0].shape[0]
        for ii in range(nstages):
            self._active_stage_idx = ii
            init_guess.requires_grad = self._auto
            active_stage_unknown = newton_solve(
                self._diag_residual_fun, init_guess,
                **self._newton_kwargs)
            # init_guess = active_stage_unknown.detach()
            # self._computed_stage_unknowns.append(init_guess.clone())
            self._computed_stage_unknowns.append(active_stage_unknown.detach())
        return implicit_runge_kutta_update_wildey(
            self._res_sol, torch.cat(
                self._computed_stage_unknowns), self._res_deltat,
            self._res_time, self._rhs, self._butcher_tableau)

    def update(self, sol, time, deltat, init_guess):
        return self._update(sol, time, deltat, init_guess)

    def integrate(self, init_sol, init_time, final_time, verbosity=0,
                  newton_kwargs={}, init_deltat=None):
        """
        Parameters
        ----------
        init_deltat : float
            The size of the first time step. If None then self.deltat will be
            used. This is needed for solving adjoint equations
        """
        self._newton_kwargs = newton_kwargs
        sols, times = [], []
        time = init_time
        times.append(time)
        if type(init_sol) == np.ndarray:
            sol = torch.as_tensor(init_sol)
        else:
            sol = init_sol.clone()
        if sol.ndim == 2:
            sol = sol[:, 0]
        sols.append(sol.detach())
        while time < final_time-1e-12:
            if verbosity >= 1:
                print("Time", time)
            if init_deltat is not None:
                if init_time+init_deltat > final_time:
                    raise ValueError("init_deltat is to large")
                deltat = init_deltat
                init_deltat = None
            else:
                deltat = min(self._deltat, final_time-time)
            sol = self.update(
                sol, time, deltat,
                [sol.clone()]*self._butcher_tableau[0].shape[0])
            sols.append(sol.detach())
            time += deltat
            times.append(time)
        if verbosity >= 1:
            print("Time", time)
        sols = torch.stack(sols, dim=1)
        return sols, times

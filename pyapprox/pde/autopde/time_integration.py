import torch
import numpy as np
from functools import partial


def explicit_runge_kutta_update(prev_sol, deltat, prev_time, rhs, butcher_tableau):
    assert prev_sol.ndim == 1
    assert callable(rhs)
    assert callable(butcher_tableau)
    a_coef, b_coef, c_coef = butcher_tableau()
    order = a_coef.shape[0]
    intermediate_rhs = np.empty((order, prev_sol.shape[0]), dtype=float)
    intermediate_rhs[0, :] = rhs(prev_sol, prev_time)
    new_sol = prev_sol+deltat*b_coef[0]*intermediate_rhs[0, :]
    tmp = np.zeros((intermediate_rhs.shape[1]))
    for ii in range(1, order):
        tmp[:] = 0.
        for jj in range(ii):
            tmp += a_coef[ii, jj]*intermediate_rhs[jj]
        intermediate_rhs[ii] = rhs(prev_sol+deltat*tmp, prev_time+c_coef[ii]*deltat)
        new_sol += deltat*b_coef[ii]*intermediate_rhs[ii]
    return new_sol


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


def butcher_tableau_explicit_rk4():
    c_coef = np.array([0., 0.5, 0.5, 1.])
    b_coef = np.array([1./6., 1./3., 1./3., 1./6.])
    a_coef = np.zeros((4, 4), dtype=float)
    a_coef[1, 0] = 0.5
    a_coef[2, 0] = 0.0
    a_coef[2, 1] = 0.5
    a_coef[3, 0] = 0.0
    a_coef[3, 1] = 0.0
    a_coef[3, 2] = 1.0
    return a_coef, b_coef, c_coef


def explicit_butcher_tableau(order):
    if order == 1:
        return butcher_tableau_explicit_forward_euler()
    if order == 2:
        return butcher_tableau_explicit_midpoint()
    if order == 4:
        return butcher_tableau_explicit_rk4()

    raise Exception('tableau not implemented for order specified')


def butcher_tableau_implicit_backward_euler():
    c_coef = np.array([1.])
    b_coef = np.array([1.])
    a_coef = np.zeros((1, 1), dtype=float)
    return a_coef, b_coef, c_coef


def butcher_tableau_implicit_crank_nicholson():
    c_coef = np.array([0., 1.])
    b_coef = np.array([1/2, 1/2])
    a_coef = np.array([[0, 0], [1/2, 1/2]], dtype=float)
    return a_coef, b_coef, c_coef


def implicit_butcher_tableau(order):
    if order == 1:
        return butcher_tableau_implicit_backward_euler()
    if order == 2:
        return butcher_tableau_implicit_crank_nicholson()
    raise Exception('tableau not implemented for order specified')


def newton_solve(residual_fun, initial_guess, tol=1e-8, maxiters=10,
                 verbosity=0):
    if not initial_guess.requires_grad:
        raise ValueError("initial_guess must have requires_grad=True")
    
    sol = initial_guess
    residual_norms = []
    it = 0
    while True:
        residual = residual_fun(sol)
        residual_norm = torch.linalg.norm(residual)
        residual_norms.append(residual_norm)
        if verbosity > 1:
            print("Iter", it, "rnorm", residual_norm.detach().numpy())
        if residual_norm < tol:
            exit_msg = f"Tolerance {tol} reached"
            break
        if it > maxiters:
            exit_msg = f"Max iterations {maxiters} reached"
            break
        if it > 4 and (residual_norm > residual_norms[it-5]):
            raise RuntimeError("Newton solve diverged")
        jac = torch.autograd.functional.jacobian(
            residual_fun, sol, strict=True)
        print(jac)
        sol = sol-torch.linalg.solve(jac, residual)
        it += 1

    if verbosity > 0:
        print(exit_msg)
    return sol


def implicit_runge_kuttta_stage_solution(prev_sol, deltat, prev_time, rhs,
                                         butcher_tableau, stage_deltas):
    a_coef, c_coef = butcher_tableau[0], butcher_tableau[1]
    nstages = a_coef.shape[0]
    ndof = stage_deltas.shape[0]//nstages
    tmp = torch.kron(torch.tensor(a_coef), torch.ones((ndof, ndof)))
    stage_rhs = []
    for ii in range(nstages):
        stage_time = prev_time+c_coef[ii]*deltat
        stage_delta = stage_deltas[ii*ndof:(ii+1)*ndof]
        stage_rhs.append(deltat*rhs(prev_sol+stage_delta, stage_time))
        print(ii, stage_rhs[-1])
    stage_rhs = torch.cat(stage_rhs)
    stage_deltas = torch.linalg.multi_dot((tmp, stage_rhs))
    print(stage_rhs)
    print(stage_deltas)
    return stage_deltas, stage_rhs


def implicit_runge_kuttta_residual(prev_sol, deltat, prev_time, rhs,
                                   butcher_tableau, stage_deltas):
    new_stage_deltas = implicit_runge_kuttta_stage_solution(
        prev_sol, deltat, prev_time, rhs, butcher_tableau, stage_deltas)[0]
    residual = stage_deltas-new_stage_deltas
    return residual


def implicit_runge_kuttta_update(prev_sol, deltat, prev_time, rhs,
                                 butcher_tableau, initial_guesses):
    b_coef = torch.tensor(butcher_tableau[1])
    residual_fun = partial(implicit_runge_kuttta_residual, prev_sol, deltat,
                           prev_time, rhs, butcher_tableau)
    initial_guess = torch.cat(initial_guesses)
    initial_guess.requires_grad = True
    stage_deltas = newton_solve(residual_fun, initial_guess)
    # TODO next line recomputes information that has already been computed
    stage_rhs = implicit_runge_kuttta_stage_solution(prev_sol, deltat, prev_time, rhs,
                                                     butcher_tableau, stage_deltas)[1]
    nstages = stage_deltas.shape[0]//prev_sol.shape[0]
    return (prev_sol + torch.sum(b_coef[:, None]*stage_rhs.reshape((prev_sol.shape[0]), nstages).T, dim=0).T)


if __name__ == "__main__":
    y0 = np.ones(1)
    def exact_sol(time):
        return y0*(1+time)**2

    def rhs(sol, time):
        # dy/dt = y0*2*(t+1) = y0*(t+1)**2*2/(t+1)
        return sol*2/(time+1)
    
    def exact_sol(time):
        return y0*(1+time)

    def rhs(sol, time):
        # dy/dt = y0 = y0*(t+1)/(t+1)
        return sol/(time+1)


    ntime_steps = 3
    deltat = 0.1
    time = 0
    sol = torch.tensor(y0)
    sols = [y0]
    butcher_tableau = butcher_tableau_implicit_crank_nicholson()
    for tt in range(ntime_steps):
        sol = implicit_runge_kuttta_update(
            sol, deltat, time, rhs, butcher_tableau,
            [torch.ones(sol.shape[0], dtype=torch.double)]*2)
        sols.append(sol.detach().numpy())
        time += deltat
    sols = np.array(sols)

    exact_sols = exact_sol(np.arange(ntime_steps+1)*deltat)[:, None]
    print(sols)
    print(exact_sols)
    assert np.allclose(exact_sols, sols)

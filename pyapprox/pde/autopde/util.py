import torch
import numpy as np

def newton_solve(residual_fun, init_guess, auto, tol=1e-7, maxiters=10,
                 verbosity=0, step_size=1):
    if auto and not init_guess.requires_grad:
        raise ValueError("init_guess must have requires_grad=True")

    if not init_guess.ndim == 1:
        raise ValueError("init_guess must be 1D tensor so AD can be used")

    sol = init_guess.clone()
    residual_norms = []
    it = 0
    while True:
        if auto:
            residual = residual_fun(sol)
        else:
            residual, jac = residual_fun(sol)
        residual_norm = torch.linalg.norm(residual)
        residual_norms.append(residual_norm)
        if verbosity > 1:
            print("Iter", it, "rnorm", residual_norm.detach().numpy())
        if residual_norm < tol:
            exit_msg = f"Tolerance {tol} reached"
            break
        if it >= maxiters:
            exit_msg = f"Max iterations {maxiters} reached"
            raise RuntimeError(exit_msg)
        # strict=True needed if computing adjoints and jac computation
        # needs to be part of graph
        if auto:
            jac = torch.autograd.functional.jacobian(
                residual_fun, sol, strict=True)
        sol = sol-step_size*torch.linalg.solve(jac, residual)
        # np.set_printoptions(linewidth=1000)
        # print(np.round(jac.numpy(), decimals=2))
        # print(residual.detach().numpy())
        # print(np.linalg.eigh(jac.numpy())[0])
        # print(np.linalg.cond(jac.numpy()))
        # assert False
        it += 1
    if verbosity > 0:
        print(exit_msg)
    return sol

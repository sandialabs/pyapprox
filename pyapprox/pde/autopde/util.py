import torch

def newton_solve(residual_fun, init_guess, tol=1e-7, maxiters=10,
                 verbosity=0, step_size=1):
    if not init_guess.requires_grad:
        raise ValueError("init_guess must have requires_grad=True")

    if not init_guess.ndim == 1:
        raise ValueError("init_guess must be 1D tensor so AD can be used")

    sol = init_guess.clone()
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
            raise RuntimeError(exit_msg)
        # if it > 4 and (residual_norm > residual_norms[it-5]):
        #     raise RuntimeError("Newton solve diverged")
        jac = torch.autograd.functional.jacobian(
            residual_fun, sol, strict=True)
        # import numpy as np
        # np.set_printoptions(linewidth=1000)
        # print(jac.numpy())
        # print(np.linalg.cond(jac.numpy()))
        # assert False
        sol = sol-step_size*torch.linalg.solve(jac, residual)
        it += 1
    if verbosity > 0:
        print(exit_msg)
    return sol

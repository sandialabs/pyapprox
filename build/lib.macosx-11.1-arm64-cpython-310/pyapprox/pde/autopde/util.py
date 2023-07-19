import torch


def newton_solve(residual_fun, init_guess, linear_solve=None, tol=1e-7,
                 maxiters=10, verbosity=0, step_size=1, rtol=1e-7):
    if not init_guess.ndim == 1:
        raise ValueError("init_guess must be 1D tensor so AD can be used")

    sol = init_guess.clone()
    residual_norms = []
    it = 0
    while True:
        residual, jac = residual_fun(sol)
        residual_norm = torch.linalg.norm(residual)
        residual_norms.append(residual_norm)
        if verbosity > 1:
            print("Iter", it, "rnorm", residual_norm.detach().numpy())
        if it > 0 and residual_norm <= tol+rtol*residual_norms[0]:
            # must take at least one step for cases when residual
            # is under tolerance for init_guess
            exit_msg = f"Tolerance {tol+rtol*residual_norms[0]} reached"
            break
        if it >= maxiters:
            exit_msg = f"Max iterations {maxiters} reached.\n"
            exit_msg += f"Rel residual norm is {residual_norm} "
            exit_msg += f"Needs to be {tol+rtol*residual_norms[0]}"
            raise RuntimeError(exit_msg)
        # strict=True needed if computing adjoints and jac computation
        # needs to be part of graph
        if jac is None:
            if not init_guess.requires_grad:
                raise ValueError("init_guess must have requires_grad=True")
            jac = torch.autograd.functional.jacobian(
                lambda s: residual_fun(s)[0], sol, strict=True)
        if linear_solve is None:
            sol = sol-step_size*torch.linalg.solve(jac, residual)
        else:
            sol = sol-step_size*linear_solve(jac, residual)
        # import numpy as np
        # np.set_printoptions(precision=2, suppress=True)
        # print('j', jac.detach().numpy())
        # print(residual.detach().numpy()[0])
        # print(sol.numpy()[0], 's')
        # print(np.linalg.eigh(jac.numpy())[0])
        # print(np.linalg.cond(jac.numpy()))
        # assert False
        it += 1
    if verbosity > 0:
        print(exit_msg)
    return sol

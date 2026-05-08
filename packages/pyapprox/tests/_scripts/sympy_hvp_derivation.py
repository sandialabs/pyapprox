"""
Symbolic derivation of HVP formulas for time-dependent problems.

This script uses sympy to derive and verify the Hessian-vector product
formulas for implicitly constrained optimization problems as described
in Heinkenschloss (2008) "Numerical Solution of Implicitly Constrained
Optimization Problems".

The key equations from the paper:
- Algorithm 4.1: Hessian-Times-Vector Computation
- Section 6.3: Gradient and Hessian Computation for Burgers Equation
"""

import sympy as sp
from sympy import diff


def derive_single_step_hvp():
    """
    Derive HVP for a single implicit step: c(y, u) = 0.

    From Heinkenschloss Algorithm 4.1:
    1. Solve c(y, u) = 0 for y (forward)
    2. Solve c_y^T λ = -∇_y f for λ (adjoint)
    3. Solve c_y w = c_u v (forward sensitivity)
    4. Solve c_y^T p = ∇_{yy}L·w - ∇_{yu}L·v (second adjoint)
    5. Compute HVP = c_u^T p - ∇_{uy}L·w + ∇_{uu}L·v
    """
    print("=" * 70)
    print("Single-step implicit function HVP derivation")
    print("=" * 70)

    # Symbols
    y = sp.Symbol("y")  # state
    sp.Symbol("u")  # parameter
    sp.Symbol("v")  # direction

    # Generic implicit constraint: c(y, u) = 0
    # For backward Euler: c = y - y_{n-1} - dt*f(y, u)

    # Use a specific quadratic ODE: f(y, u) = a*y + u[0]*y^2 + u[1]
    a, u0, u1, dt, y_nm1 = sp.symbols("a u_0 u_1 dt y_{n-1}", real=True)

    # ODE residual (right-hand side)
    f = a * y + u0 * y**2 + u1

    # Backward Euler residual: R = y - y_{n-1} - dt*f(y)
    R = y - y_nm1 - dt * f

    print("\n1. Backward Euler residual:")
    print(f"   R(y, u) = {R}")

    # Jacobians
    dRdy = diff(R, y)
    dRdu0 = diff(R, u0)
    dRdu1 = diff(R, u1)

    print("\n2. First derivatives:")
    print(f"   dR/dy = {dRdy}")
    print(f"   dR/du0 = {dRdu0}")
    print(f"   dR/du1 = {dRdu1}")

    # Second derivatives (Hessian of R)
    d2Rdy2 = diff(R, y, y)
    d2Rdydu0 = diff(R, y, u0)
    d2Rdydu1 = diff(R, y, u1)
    d2Rdu0dy = diff(R, u0, y)
    d2Rdu02 = diff(R, u0, u0)
    diff(R, u0, u1)
    d2Rdu12 = diff(R, u1, u1)

    print("\n3. Second derivatives:")
    print(f"   d²R/dy² = {d2Rdy2}")
    print(f"   d²R/dydu0 = {d2Rdydu0}")
    print(f"   d²R/dydu1 = {d2Rdydu1}")
    print(f"   d²R/du0dy = {d2Rdu0dy}")
    print(f"   d²R/du0² = {d2Rdu02}")
    print(f"   d²R/du1² = {d2Rdu12}")

    return dRdy, d2Rdy2, d2Rdydu0


def derive_two_step_hvp():
    """
    Derive HVP for two-step time integration.

    Forward: y_1 solves R_1 = 0, y_2 solves R_2 = 0
    Adjoint: λ_1, λ_2 solve adjoint equations backward
    """
    print("\n" + "=" * 70)
    print("Two-step time integration HVP derivation")
    print("=" * 70)

    # Symbols
    y0, y1, y2 = sp.symbols("y_0 y_1 y_2", real=True)
    p0, p1 = sp.symbols("p_0 p_1", real=True)  # parameters
    v0, v1 = sp.symbols("v_0 v_1", real=True)  # direction
    dt = sp.Symbol("dt", positive=True)
    a = sp.Symbol("a", real=True)  # linear coefficient

    # Quadratic ODE: f(y, p) = a*y + p0*y^2 + p1
    def f(y):
        return a * y + p0 * y**2 + p1

    # Backward Euler residuals
    R1 = y1 - y0 - dt * f(y1)
    R2 = y2 - y1 - dt * f(y2)

    print("\n1. Time stepping residuals:")
    print(f"   R_1 = {R1}")
    print(f"   R_2 = {R2}")

    # Forward solve (implicit):
    # From R_1 = 0: y1 = y0 + dt*(a*y1 + p0*y1^2 + p1)
    # This is quadratic in y1, solve for y1

    print("\n2. Forward sensitivity equations:")
    print("   R_1(y_1, y_0, p) = 0 implies:")
    print("   dR_1/dy_1 · w_1 + dR_1/dy_0 · w_0 + dR_1/dp · v = 0")

    dR1dy1 = diff(R1, y1)
    dR1dy0 = diff(R1, y0)
    dR1dp0 = diff(R1, p0)
    dR1dp1 = diff(R1, p1)

    print(f"\n   dR_1/dy_1 = {dR1dy1}")
    print(f"   dR_1/dy_0 = {dR1dy0}")
    print(f"   dR_1/dp_0 = {dR1dp0}")
    print(f"   dR_1/dp_1 = {dR1dp1}")

    # Similarly for R2
    dR2dy2 = diff(R2, y2)
    dR2dy1 = diff(R2, y1)
    dR2dp0 = diff(R2, p0)
    dR2dp1 = diff(R2, p1)

    print(f"\n   dR_2/dy_2 = {dR2dy2}")
    print(f"   dR_2/dy_1 = {dR2dy1}")
    print(f"   dR_2/dp_0 = {dR2dp0}")
    print(f"   dR_2/dp_1 = {dR2dp1}")

    print("\n3. Forward sensitivity solve:")
    print("   w_0 = 0 (initial condition fixed)")
    print("   w_1 = -(dR_1/dy_1)^{-1} · [dR_1/dy_0 · w_0 + dR_1/dp · v]")
    print("       = -(dR_1/dy_1)^{-1} · (dR_1/dp_0 · v_0 + dR_1/dp_1 · v_1)")
    print(f"       = -({dR1dp0} · v_0 + {dR1dp1} · v_1) / ({dR1dy1})")

    print("\n4. Adjoint equations:")
    print("   Lagrangian: L = Q(y_2, p) + λ_1^T R_1 + λ_2^T R_2")
    print("   ∂L/∂y_2 = 0: (dR_2/dy_2)^T λ_2 = -dQ/dy_2")
    print("   ∂L/∂y_1 = 0: (dR_1/dy_1)^T λ_1 = -(dR_2/dy_1)^T λ_2 - dQ/dy_1")

    print("\n5. Second adjoint equations (from Algorithm 4.1):")
    print("   Solve (dR/dy)^T p = ∇_{yy}L · w - ∇_{yu}L · v")

    # For quadratic ODE, the Hessian terms are:
    d2R1dy12 = diff(R1, y1, y1)
    d2R1dy1dp0 = diff(R1, y1, p0)
    d2R2dy22 = diff(R2, y2, y2)
    d2R2dy2dp0 = diff(R2, y2, p0)

    print(f"\n   d²R_1/dy_1² = {d2R1dy12}")
    print(f"   d²R_1/dy_1 dp_0 = {d2R1dy1dp0}")
    print(f"   d²R_2/dy_2² = {d2R2dy22}")
    print(f"   d²R_2/dy_2 dp_0 = {d2R2dy2dp0}")

    print("\n6. Second adjoint RHS at step n:")
    print("   RHS = ∇_{yy}L · w - ∇_{yu}L · v")
    print("       = (d²Q/dy² + λ^T d²R/dy²) · w - (d²Q/dy dp + λ^T d²R/dy dp) · v")
    print("\n   For endpoint functional Q = y_2:")
    print("   d²Q/dy² = 0, d²Q/dy dp = 0")
    print("   RHS = λ^T · d²R/dy² · w - λ^T · d²R/dy dp · v")
    print(f"       = λ · ({d2R2dy22}) · w - λ · ({d2R2dy2dp0}) · v_0")
    print("       = -2 dt p_0 λ w + 2 dt y λ v_0")

    return


def verify_hvp_formula():
    """
    Verify the HVP formula with a concrete numerical example.
    """
    print("\n" + "=" * 70)
    print("Numerical verification of HVP formula")
    print("=" * 70)

    import numpy as np

    # Problem setup
    a = -0.5  # stability coefficient
    dt = 0.1  # time step
    y0 = 1.0  # initial condition
    p0 = 0.1  # quadratic coefficient
    p1 = 0.2  # constant forcing
    v = np.array([1.0, 0.0])  # direction

    def f(y, p):
        return a * y + p[0] * y**2 + p[1]

    def df_dy(y, p):
        return a + 2 * p[0] * y

    def df_dp(y, p):
        return np.array([y**2, 1.0])

    def d2f_dy2(y, p):
        return 2 * p[0]

    def d2f_dydp(y, p):
        return np.array([2 * y, 0.0])

    # Forward solve: y1 = y0 + dt*f(y1, p)
    # Solve: y1 - y0 - dt*(a*y1 + p0*y1^2 + p1) = 0
    # (1 - dt*a)*y1 - dt*p0*y1^2 = y0 + dt*p1
    # Quadratic: dt*p0*y1^2 + (dt*a - 1)*y1 + y0 + dt*p1 = 0

    p = np.array([p0, p1])
    A_coef = dt * p[0]
    B_coef = dt * a - 1
    C_coef = y0 + dt * p[1]

    # Solve quadratic (take the larger root for stability)
    disc = B_coef**2 - 4 * A_coef * C_coef
    y1 = (-B_coef - np.sqrt(disc)) / (2 * A_coef) if A_coef != 0 else -C_coef / B_coef

    print("\n1. Forward solution:")
    print(f"   y_0 = {y0}")
    print(f"   y_1 = {y1:.6f}")

    # Verify residual
    R1 = y1 - y0 - dt * f(y1, p)
    print(f"   Residual R_1 = {R1:.2e}")

    # Gradient via finite difference
    eps = 1e-6

    def solve_forward(p_val):
        A = dt * p_val[0]
        B = dt * a - 1
        C = y0 + dt * p_val[1]
        if abs(A) < 1e-12:
            return -C / B
        disc = B**2 - 4 * A * C
        return (-B - np.sqrt(disc)) / (2 * A)

    # Gradient dQ/dp where Q = y_1 (endpoint)
    grad_fd = np.zeros(2)
    for i in range(2):
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        grad_fd[i] = (solve_forward(p_plus) - solve_forward(p_minus)) / (2 * eps)

    print("\n2. Gradient (FD):")
    print(f"   dQ/dp = [{grad_fd[0]:.6f}, {grad_fd[1]:.6f}]")

    # Forward sensitivity w = dy/dp · v
    # dR/dy · w + dR/dp · v = 0
    # w = -(dR/dy)^{-1} · dR/dp · v
    dRdy = 1 - dt * df_dy(y1, p)
    dRdp = -dt * df_dp(y1, p)
    w1 = -dRdp @ v / dRdy

    print("\n3. Forward sensitivity:")
    print(f"   dR/dy = {dRdy:.6f}")
    print(f"   dR/dp = [{dRdp[0]:.6f}, {dRdp[1]:.6f}]")
    print(f"   w_1 = dy/dp · v = {w1:.6f}")

    # For Q = y_1, gradient = w_1
    print(f"   Gradient from sensitivity = {w1:.6f}")
    print(f"   Gradient from FD = {grad_fd @ v:.6f}")

    # HVP: d²Q/dp² · v = d(w)/dp · v
    hvp_fd = np.zeros(2)
    for i in range(2):
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        g_plus = np.zeros(2)
        g_minus = np.zeros(2)
        for j in range(2):
            pp = p_plus.copy()
            pp[j] += eps
            pm = p_plus.copy()
            pm[j] -= eps
            g_plus[j] = (solve_forward(pp) - solve_forward(pm)) / (2 * eps)

            pp = p_minus.copy()
            pp[j] += eps
            pm = p_minus.copy()
            pm[j] -= eps
            g_minus[j] = (solve_forward(pp) - solve_forward(pm)) / (2 * eps)
        hvp_fd[i] = (g_plus @ v - g_minus @ v) / (2 * eps)

    print("\n4. HVP (FD):")
    print(f"   d²Q/dp² · v = [{hvp_fd[0]:.6f}, {hvp_fd[1]:.6f}]")

    # Now compute HVP via Algorithm 4.1
    # Step 1: y already computed
    # Step 2: λ solves (dR/dy)^T λ = -dQ/dy = -1
    lam = -1.0 / dRdy
    print("\n5. Adjoint:")
    print(f"   λ = {lam:.6f}")

    # Step 3: w already computed (forward sensitivity)
    # Step 4: p solves (dR/dy)^T s = ∇_{yy}L·w - ∇_{yu}L·v
    # L = Q + λ·R = y_1 + λ·(y_1 - y_0 - dt*f(y_1))
    # ∇_{yy}L = 0 + λ·(-dt·d²f/dy²) = -dt·λ·2p_0
    # ∇_{yu}L = 0 + λ·(-dt·d²f/dydp) = -dt·λ·[2y, 0]
    Lyy = -dt * lam * d2f_dy2(y1, p)
    Lyu = -dt * lam * d2f_dydp(y1, p)

    rhs_s = Lyy * w1 - Lyu @ v
    s = rhs_s / dRdy  # Note: (dR/dy)^T = dR/dy for scalar

    print("\n6. Second adjoint:")
    print(f"   ∇_{{yy}}L = {Lyy:.6f}")
    print(f"   ∇_{{yu}}L = [{Lyu[0]:.6f}, {Lyu[1]:.6f}]")
    print(f"   RHS = {rhs_s:.6f}")
    print(f"   s = {s:.6f}")

    # Step 5: HVP = (dR/dp)^T·s - ∇_{uy}L·w + ∇_{uu}L·v
    # For our problem: ∇_{uy}L = ∇_{yu}L^T, ∇_{uu}L = 0
    Luy = Lyu  # scalar case
    Luu = np.zeros((2, 2))  # d²R/dp² = 0

    hvp_analytical = dRdp * s - Luy * w1 + Luu @ v

    print("\n7. HVP accumulation:")
    print(f"   (dR/dp)^T · s = [{dRdp[0] * s:.6f}, {dRdp[1] * s:.6f}]")
    print(f"   ∇_{{uy}}L · w = [{Luy[0] * w1:.6f}, {Luy[1] * w1:.6f}]")
    print(f"   HVP (analytical) = [{hvp_analytical[0]:.6f}, {hvp_analytical[1]:.6f}]")
    print(f"   HVP (FD)         = [{hvp_fd[0]:.6f}, {hvp_fd[1]:.6f}]")
    print(
        f" Error = [{abs(hvp_analytical[0] - hvp_fd[0]):.2e}, {abs(hvp_analytical[1] -
        hvp_fd[1]):.2e}]"
    )

    return hvp_analytical, hvp_fd


def verify_two_step_hvp():
    """
    Verify HVP with two time steps and quadratic ODE.

    Key insight from Heinkenschloss Algorithm 4.1:
    - Step 3 defines w via: c_y·w = c_u·v (POSITIVE sign)
    - This differs from w = dy/dp·v (sensitivity) by a sign!
    - My implementation uses w = dy/dp·v (negative of Algorithm 4.1's w)

    So when applying Algorithm 4.1's formulas with my w definition:
    - My w = -w_Heinkenschloss
    - Step 4 RHS: ∇_{yy}L·w_H - ∇_{yu}L·v becomes: -∇_{yy}L·w - ∇_{yu}L·v
    - Step 5: c_u^T·p - ∇_{uy}L·w_H becomes: c_u^T·p + ∇_{uy}L·w
    """
    print("\n" + "=" * 70)
    print("Two-step numerical HVP verification")
    print("=" * 70)

    import numpy as np

    # Problem setup
    a = -0.5
    dt = 0.1
    y0 = 1.0
    p = np.array([0.1, 0.2])  # [p0, p1]
    v = np.array([1.0, 0.0])  # direction

    def f(y, p):
        return a * y + p[0] * y**2 + p[1]

    def df_dy(y, p):
        return a + 2 * p[0] * y

    def df_dp(y, p):
        return np.array([y**2, 1.0])

    def d2f_dy2(y, p):
        return 2 * p[0]

    def d2f_dydp(y, p):
        return np.array([2 * y, 0.0])

    def solve_be_step(y_prev, p):
        """Solve Backward Euler step."""
        A = dt * p[0]
        B = dt * a - 1
        C = y_prev + dt * p[1]
        if abs(A) < 1e-12:
            return -C / B
        disc = B**2 - 4 * A * C
        return (-B - np.sqrt(disc)) / (2 * A)

    def solve_forward(p_val):
        """Solve two steps forward."""
        y1 = solve_be_step(y0, p_val)
        y2 = solve_be_step(y1, p_val)
        return np.array([y0, y1, y2])

    # Forward solve
    sol = solve_forward(p)
    y1, y2 = sol[1], sol[2]
    print("\n1. Forward solution:")
    print(f"   y_0 = {y0:.6f}")
    print(f"   y_1 = {y1:.6f}")
    print(f"   y_2 = {y2:.6f}")

    # Endpoint functional Q = y_2
    eps = 1e-6

    # Gradient via FD
    grad_fd = np.zeros(2)
    for i in range(2):
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps
        grad_fd[i] = (solve_forward(p_plus)[2] - solve_forward(p_minus)[2]) / (2 * eps)

    print("\n2. Gradient (FD):")
    print(f"   dQ/dp = [{grad_fd[0]:.6f}, {grad_fd[1]:.6f}]")

    # HVP via FD
    hvp_fd = np.zeros(2)
    for i in range(2):
        p_plus = p.copy()
        p_plus[i] += eps
        p_minus = p.copy()
        p_minus[i] -= eps

        g_plus = np.zeros(2)
        g_minus = np.zeros(2)
        for j in range(2):
            pp = p_plus.copy()
            pp[j] += eps
            pm = p_plus.copy()
            pm[j] -= eps
            g_plus[j] = (solve_forward(pp)[2] - solve_forward(pm)[2]) / (2 * eps)

            pp = p_minus.copy()
            pp[j] += eps
            pm = p_minus.copy()
            pm[j] -= eps
            g_minus[j] = (solve_forward(pp)[2] - solve_forward(pm)[2]) / (2 * eps)

        hvp_fd[i] = (g_plus @ v - g_minus @ v) / (2 * eps)

    print("\n3. HVP (FD):")
    print(f"   H·v = [{hvp_fd[0]:.6f}, {hvp_fd[1]:.6f}]")

    # Now implement Algorithm 4.1/6.2
    # Step 1: Forward solve (done)
    # Step 2: Adjoint solve (backward)

    # At n=2: (dR_2/dy_2)^T λ_2 = -dQ/dy_2 = -1
    dR2dy2 = 1 - dt * df_dy(y2, p)
    dR2dy1 = -1
    dR2dp = -dt * df_dp(y2, p)

    lam2 = -1.0 / dR2dy2

    # At n=1: (dR_1/dy_1)^T λ_1 = -(dR_2/dy_1)^T λ_2 - dQ/dy_1
    # dQ/dy_1 = 0 for endpoint functional Q = y_2
    dR1dy1 = 1 - dt * df_dy(y1, p)
    dR1dy0 = -1
    dR1dp = -dt * df_dp(y1, p)

    lam1 = -dR2dy1 * lam2 / dR1dy1

    # At n=0: λ_0 = -(-1)·λ_1 / 1 (mass matrix is 1)
    lam0 = lam1  # Simplified for scalar case

    print("\n4. Adjoint solution:")
    print(f"   λ_2 = {lam2:.6f}")
    print(f"   λ_1 = {lam1:.6f}")
    print(f"   λ_0 = {lam0:.6f}")

    # Verify gradient via adjoint
    # dQ/dp = Σ_n (dR_n/dp)^T λ_n
    grad_adj = dR1dp * lam1 + dR2dp * lam2
    print("\n5. Gradient (adjoint):")
    print(f"   dQ/dp = [{grad_adj[0]:.6f}, {grad_adj[1]:.6f}]")
    print(
        f" Error = [{abs(grad_adj[0] - grad_fd[0]):.2e}, {abs(grad_adj[1] -
        grad_fd[1]):.2e}]"
    )

    # Step 3: Forward sensitivity
    # My convention: dR_1/dy_1·w_1 + dR_1/dy_0·w_0 + dR_1/dp·v = 0
    # So w = dy/dp·v (the actual sensitivity)
    w0 = 0.0

    # w_1: dR_1/dy_1·w_1 + dR_1/dy_0·w_0 + dR_1/dp·v = 0
    w1 = -(dR1dy0 * w0 + dR1dp @ v) / dR1dy1

    # w_2: dR_2/dy_2·w_2 + dR_2/dy_1·w_1 + dR_2/dp·v = 0
    w2 = -(dR2dy1 * w1 + dR2dp @ v) / dR2dy2

    print("\n6. Forward sensitivity (w = dy/dp·v):")
    print(f"   w_0 = {w0:.6f}")
    print(f"   w_1 = {w1:.6f}")
    print(f"   w_2 = {w2:.6f}")
    print(f"   Gradient from sensitivity = w_2 = {w2:.6f}")
    print(f"   Gradient from FD = {grad_fd @ v:.6f}")

    # CORRECTED Step 4: Second adjoint (backward)
    # Since my w = -w_Heinkenschloss, the RHS becomes:
    # RHS = -∇_{yy}L·w - ∇_{yu}L·v (note the NEGATIVE on first term!)

    Lyy2 = lam2 * (-dt * d2f_dy2(y2, p))
    Lyu2 = lam2 * (-dt * d2f_dydp(y2, p))

    # CORRECTED: RHS = -Lyy2*w2 - Lyu2@v (negative of Heinkenschloss)
    rhs_s2 = -Lyy2 * w2 - Lyu2 @ v
    s2 = rhs_s2 / dR2dy2

    print("\n7. Second adjoint at n=2 (CORRECTED):")
    print(f"   ∇_{{yy}}L_2 = {Lyy2:.6f}")
    print(f"   ∇_{{yu}}L_2 = [{Lyu2[0]:.6f}, {Lyu2[1]:.6f}]")
    print(f"   RHS = -∇_{{yy}}L·w - ∇_{{yu}}L·v = {rhs_s2:.6f}")
    print(f"   s_2 = {s2:.6f}")

    Lyy1 = lam1 * (-dt * d2f_dy2(y1, p))
    Lyu1 = lam1 * (-dt * d2f_dydp(y1, p))

    # CORRECTED: RHS = -offdiag*s2 - Lyy1*w1 - Lyu1@v
    rhs_s1 = -dR2dy1 * s2 - Lyy1 * w1 - Lyu1 @ v
    s1 = rhs_s1 / dR1dy1

    print("\n8. Second adjoint at n=1 (CORRECTED):")
    print(f"   ∇_{{yy}}L_1 = {Lyy1:.6f}")
    print(f"   ∇_{{yu}}L_1 = [{Lyu1[0]:.6f}, {Lyu1[1]:.6f}]")
    print(f"   RHS = {rhs_s1:.6f}")
    print(f"   s_1 = {s1:.6f}")

    # CORRECTED Step 5: HVP accumulation
    # Since my w = -w_H, the -∇_{uy}L·w_H term becomes +∇_{uy}L·w
    # HVP = Σ_n [(dR_n/dp)^T s_n + ∇_{uy}L_n·w_n + ∇_{uu}L_n·v]

    Luy1 = Lyu1
    Luy2 = Lyu2

    hvp_term1 = dR1dp * s1
    hvp_term2 = dR2dp * s2
    hvp_term3 = Luy1 * w1  # CORRECTED: positive sign
    hvp_term4 = Luy2 * w2  # CORRECTED: positive sign

    hvp_corrected = hvp_term1 + hvp_term2 + hvp_term3 + hvp_term4

    print("\n9. HVP accumulation (CORRECTED):")
    print(f"   (dR_1/dp)^T s_1 = [{hvp_term1[0]:.6f}, {hvp_term1[1]:.6f}]")
    print(f"   (dR_2/dp)^T s_2 = [{hvp_term2[0]:.6f}, {hvp_term2[1]:.6f}]")
    print(f"   +∇_{{uy}}L_1·w_1 = [{hvp_term3[0]:.6f}, {hvp_term3[1]:.6f}]")
    print(f"   +∇_{{uy}}L_2·w_2 = [{hvp_term4[0]:.6f}, {hvp_term4[1]:.6f}]")
    print(f"\n   HVP (corrected) = [{hvp_corrected[0]:.6f}, {hvp_corrected[1]:.6f}]")
    print(f"   HVP (FD)        = [{hvp_fd[0]:.6f}, {hvp_fd[1]:.6f}]")
    error = np.abs(hvp_corrected - hvp_fd)
    print(f"   Error = [{error[0]:.2e}, {error[1]:.2e}]")

    return hvp_corrected, hvp_fd


if __name__ == "__main__":
    derive_single_step_hvp()
    derive_two_step_hvp()
    print("\n\n" + "=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)
    verify_hvp_formula()
    verify_two_step_hvp()

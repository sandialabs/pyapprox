"""
Debug Forward Euler HVP by tracing through block matrix formulation.

For Forward Euler with 3 time steps (n=0,1,2):

Block matrix c_Y:
    [  I        0        0  ]   [ y0 ]   [ y0_init ]
    [ B1       A1        0  ] * [ y1 ] = [   g1    ]
    [  0       B2       A2  ]   [ y2 ]   [   g2    ]

where:
    A_n = I  (for Forward Euler, diagonal block is identity)
    B_n = -I - dt*J_{n-1}  (depends on y_{n-1})

Adjoint c_Y^T:
    [  I       B1^T      0   ]   [ λ0 ]   [ -dQ/dy0 ]
    [  0       A1^T     B2^T ] * [ λ1 ] = [ -dQ/dy1 ]
    [  0        0       A2^T ]   [ λ2 ]   [ -dQ/dy2 ]

For endpoint functional Q = y2, we have dQ/dy2 = 1, others = 0.

Second adjoint (Heinkenschloss convention c_Y w = c_p v):
    c_Y^T s = ∇_{YY}L · w - ∇_{Yp}L · v

The key question: which Hessian terms contribute to which row?

For Forward Euler:
    R1 = y1 - y0 - dt*f(y0)
    R2 = y2 - y1 - dt*f(y1)

So:
    d²R1/dy0² = -dt * d²f/dy²|_{y0}  (contributes to row 0 of ∇_{YY}L)
    d²R2/dy1² = -dt * d²f/dy²|_{y1}  (contributes to row 1 of ∇_{YY}L)

But wait - the Hessian ∇_{YY}L is summed over all residuals weighted by λ:
    [∇_{YY}L]_{y_n} = d²Q/dy_n² + Σ_m λ_m^T d²R_m/dy_n²

For explicit schemes:
    - d²R_n/dy_n² = 0 (because R_n is linear in y_n)
    - d²R_n/dy_{n-1}² ≠ 0 (because f depends on y_{n-1})

So the Hessian at y_n comes from R_{n+1}, not R_n!
"""

import numpy as np
import sympy as sp

# Define symbols
a, dt = sp.symbols('a dt', real=True)
p0, p1 = sp.symbols('p0 p1', real=True)
y0_sym = sp.Symbol('y0', real=True)
v0, v1 = sp.symbols('v0 v1', real=True)

# Define ODE RHS
def f(y, p0, p1):
    return a*y + p0*y**2 + p1

# Forward solve
y1 = y0_sym + dt * f(y0_sym, p0, p1)
y2 = y1 + dt * f(y1, p0, p1)

print("=== Forward Solve ===")
print(f"y1 = y0 + dt*f(y0)")
print(f"y2 = y1 + dt*f(y1)")

# Jacobians - df/dy = a + 2*p0*y
# J0 = df/dy at y0
# J1 = df/dy at y1
J0 = a + 2*p0*y0_sym
J1 = a + 2*p0*y1

print("\n=== State Jacobians ===")
print(f"J0 = df/dy|_{{y0}} = {J0}")
print(f"J1 = df/dy|_{{y1}} = {J1}")

# Block matrices
print("\n=== Block Matrix c_Y ===")
print("     y0        y1        y2")
print(f"R0:  I         0         0")
print(f"R1: -I-dt*J0   I         0")
print(f"R2:  0        -I-dt*J1   I")

# Adjoint solve for Q = y2
# c_Y^T λ = -∇Q
# Row 2: I * λ2 = -1  =>  λ2 = -1
# Row 1: I * λ1 + (-I-dt*J1)^T * λ2 = 0
#        λ1 = (I + dt*J1^T) * λ2 = -(1 + dt*J1)
# Row 0: I * λ0 + (-I-dt*J0)^T * λ1 = 0
#        λ0 = (I + dt*J0^T) * λ1

lam2 = sp.Integer(-1)
lam1 = -(1 + dt*J1)
lam0 = (1 + dt*J0) * lam1

print("\n=== Adjoint Solve (Q = y2) ===")
print(f"λ2 = {lam2}")
print(f"λ1 = {sp.simplify(lam1)}")
print(f"λ0 = {sp.simplify(lam0)}")

# Parameter Jacobians (dR/dp)
# R1 = y1 - y0 - dt*f(y0)
# dR1/dp0 = -dt * y0^2
# dR1/dp1 = -dt

# R2 = y2 - y1 - dt*f(y1)
# dR2/dp0 = -dt * y1^2
# dR2/dp1 = -dt

dR1dp0 = -dt * y0_sym**2
dR1dp1 = -dt
dR2dp0 = -dt * y1**2
dR2dp1 = -dt

print("\n=== Parameter Jacobians (dR/dp) ===")
print(f"dR1/dp0 = {dR1dp0}")
print(f"dR1/dp1 = {dR1dp1}")
print(f"dR2/dp0 = {dR2dp0}")
print(f"dR2/dp1 = {dR2dp1}")

# Forward sensitivity (Heinkenschloss: c_Y w = c_p v)
# Our convention: c_Y w = -dR/dp * v  (physical sensitivity)
#
# Row 0: I * w0 = dy0/dp * v = 0 (y0 fixed)
# Row 1: B1 * w0 + A1 * w1 = dR1/dp * v
#        (-I - dt*J0) * 0 + I * w1 = -(-dt*y0^2*v0 - dt*v1)
#        w1 = dt*(y0^2*v0 + v1)
# Row 2: B2 * w1 + A2 * w2 = dR2/dp * v
#        w2 = -B2 * w1 + dR2/dp * v
#           = (I + dt*J1) * w1 + dt*(y1^2*v0 + v1)

w0 = sp.Integer(0)
w1 = dt * (y0_sym**2 * v0 + v1)
w2 = (1 + dt*J1) * w1 + dt * (y1**2 * v0 + v1)

print("\n=== Forward Sensitivity (physical: w = dY/dp * v) ===")
print(f"w0 = {w0}")
print(f"w1 = {sp.simplify(w1)}")
print(f"w2 = {sp.simplify(w2)}")

# Second adjoint (Heinkenschloss: c_Y^T s = ∇_{YY}L·w_H - ∇_{Yp}L·v)
# Our convention: c_Y^T s = -∇_{YY}L·w - ∇_{Yp}L·v
#
# Lagrangian Hessians:
# ∇_{y_n y_n}L = d²Q/dy_n² + Σ_m λ_m d²R_m/dy_n²
#
# For Forward Euler:
# d²R1/dy0² = -dt * d²f/dy²|_{y0} = -dt * 2*p0  (contributes to y0 row)
# d²R1/dy1² = 0  (R1 doesn't depend on y1)
# d²R2/dy0² = 0  (R2 doesn't directly depend on y0)
# d²R2/dy1² = -dt * d²f/dy²|_{y1} = -dt * 2*p0  (contributes to y1 row)
# d²R2/dy2² = 0  (R2 is linear in y2)

# So:
# ∇_{y0 y0}L = 0 + λ1 * (-dt * 2*p0) = -λ1 * dt * 2*p0
# ∇_{y1 y1}L = 0 + λ2 * (-dt * 2*p0) = -λ2 * dt * 2*p0 = dt * 2*p0  (since λ2=-1)
# ∇_{y2 y2}L = 0

d2R1_dy0dy0 = -dt * 2 * p0
d2R2_dy1dy1 = -dt * 2 * p0

nabla_y0y0_L = lam1 * d2R1_dy0dy0  # From R1
nabla_y1y1_L = lam2 * d2R2_dy1dy1  # From R2
nabla_y2y2_L = sp.Integer(0)

print("\n=== Lagrangian State-State Hessians ===")
print(f"∇_{{y0,y0}}L = λ1 * d²R1/dy0² = {sp.simplify(nabla_y0y0_L)}")
print(f"∇_{{y1,y1}}L = λ2 * d²R2/dy1² = {sp.simplify(nabla_y1y1_L)}")
print(f"∇_{{y2,y2}}L = {nabla_y2y2_L}")

# Mixed Hessians ∇_{y_n p}L:
# d²R1/(dy0 dp0) = -dt * 2*y0
# d²R1/(dy0 dp1) = 0
# d²R2/(dy1 dp0) = -dt * 2*y1
# d²R2/(dy1 dp1) = 0

d2R1_dy0dp0 = -dt * 2 * y0_sym
d2R2_dy1dp0 = -dt * 2 * y1

nabla_y0p0_L = lam1 * d2R1_dy0dp0
nabla_y1p0_L = lam2 * d2R2_dy1dp0
nabla_y0p1_L = sp.Integer(0)
nabla_y1p1_L = sp.Integer(0)

print("\n=== Lagrangian State-Param Hessians ===")
print(f"∇_{{y0,p0}}L = λ1 * d²R1/(dy0 dp0) = {sp.simplify(nabla_y0p0_L)}")
print(f"∇_{{y1,p0}}L = λ2 * d²R2/(dy1 dp0) = {sp.simplify(nabla_y1p0_L)}")

# Second adjoint solve:
# Our convention: c_Y^T s = -∇_{YY}L·w - ∇_{Yp}L·v
#
# Row 2: A2^T * s2 = -∇_{y2,y2}L * w2 - ∇_{y2,p}L * v
#        I * s2 = -0 * w2 - 0 * v = 0
#        s2 = 0

# Row 1: A1^T * s1 + B2^T * s2 = -∇_{y1,y1}L * w1 - ∇_{y1,p0}L * v0 - ∇_{y1,p1}L * v1
#        I * s1 + (-I - dt*J1)^T * 0 = -∇_{y1,y1}L * w1 - ∇_{y1,p0}L * v0
#        s1 = -∇_{y1,y1}L * w1 - ∇_{y1,p0}L * v0
#           = -(dt * 2*p0) * w1 - (dt * 2*y1) * v0
#           = -dt * 2 * (p0 * w1 + y1 * v0)

# Row 0: similar...

s2 = sp.Integer(0)
s1 = -nabla_y1y1_L * w1 - nabla_y1p0_L * v0
s0 = (1 + dt*J1) * s1 - nabla_y0y0_L * w0 - nabla_y0p0_L * v0

print("\n=== Second Adjoint Solve ===")
print(f"s2 = {s2}")
print(f"s1 = -∇_{{y1,y1}}L * w1 - ∇_{{y1,p0}}L * v0")
print(f"   = {sp.simplify(s1)}")
print(f"s0 = (1+dt*J1)*s1 - ∇_{{y0,y0}}L * w0 - ∇_{{y0,p0}}L * v0")
print(f"   = {sp.simplify(s0)}")

# HVP accumulation:
# HVP = c_p^T s + ∇_{pY}L·w + ∇_{pp}L·v
# Our convention (w = physical): HVP = c_p^T s + ∇_{pY}L·w + ∇_{pp}L·v
#
# c_p^T s = dR1/dp^T * s1 + dR2/dp^T * s2
#
# Note: For Forward Euler, ∇_{pY}L contributes:
# λ1 * d²R1/(dp dy0) * w0 + λ2 * d²R2/(dp dy1) * w1

cp_T_s_p0 = dR1dp0 * s1 + dR2dp0 * s2
cp_T_s_p1 = dR1dp1 * s1 + dR2dp1 * s2

# ∇_{p0,y0}L = λ1 * d²R1/(dp0 dy0) = λ1 * (-dt * 2*y0)
# ∇_{p0,y1}L = λ2 * d²R2/(dp0 dy1) = λ2 * (-dt * 2*y1)
nabla_p0y0_L = lam1 * (-dt * 2 * y0_sym)
nabla_p0y1_L = lam2 * (-dt * 2 * y1)
nabla_p1y_L = sp.Integer(0)

# ∇_{pY}L·w = ∇_{p0,y0}L * w0 + ∇_{p0,y1}L * w1 + ... for p0 component
nabla_pY_w_p0 = nabla_p0y0_L * w0 + nabla_p0y1_L * w1
nabla_pY_w_p1 = sp.Integer(0)

print("\n=== HVP Terms ===")
print(f"c_p^T * s [p0] = dR1/dp0 * s1 + dR2/dp0 * s2")
print(f"              = {sp.simplify(cp_T_s_p0)}")
print(f"c_p^T * s [p1] = {sp.simplify(cp_T_s_p1)}")
print()
print(f"∇_{{pY}}L·w [p0] = ∇_{{p0,y0}}L * w0 + ∇_{{p0,y1}}L * w1")
print(f"               = {sp.simplify(nabla_pY_w_p0)}")
print(f"∇_{{pY}}L·w [p1] = {nabla_pY_w_p1}")

# ∇_{pp}L·v = 0 for this problem (no second derivative of f w.r.t. p)
nabla_pp_v_p0 = sp.Integer(0)
nabla_pp_v_p1 = sp.Integer(0)

# Total HVP
hvp_p0 = cp_T_s_p0 + nabla_pY_w_p0 + nabla_pp_v_p0
hvp_p1 = cp_T_s_p1 + nabla_pY_w_p1 + nabla_pp_v_p1

print("\n=== Total HVP ===")
print(f"HVP[p0] = c_p^T*s + ∇_{{pY}}L·w + ∇_{{pp}}L·v")
print(f"        = {sp.simplify(hvp_p0)}")
print(f"HVP[p1] = {sp.simplify(hvp_p1)}")

# Numerical evaluation
vals = {a: -0.5, dt: 0.1, y0_sym: 1.0, p0: 0.1, p1: 0.2, v0: 1.0, v1: 0.0}

print("\n" + "="*60)
print("NUMERICAL VALUES")
print("="*60)
print(f"Parameters: a={vals[a]}, dt={vals[dt]}, y0={vals[y0_sym]}, p0={vals[p0]}, p1={vals[p1]}")
print(f"Direction: v0={vals[v0]}, v1={vals[v1]}")
print()
print(f"Forward: y1={float(y1.subs(vals)):.6f}, y2={float(y2.subs(vals)):.6f}")
print(f"Adjoint: λ0={float(lam0.subs(vals)):.6f}, λ1={float(lam1.subs(vals)):.6f}, λ2={float(lam2)}")
print(f"Sensitivity: w0={float(w0)}, w1={float(w1.subs(vals)):.6f}, w2={float(w2.subs(vals)):.6f}")
print(f"Second adjoint: s0={float(s0.subs(vals)):.6f}, s1={float(s1.subs(vals)):.6f}, s2={float(s2)}")
print()
print(f"HVP components:")
print(f"  c_p^T*s [p0] = {float(cp_T_s_p0.subs(vals)):.6f}")
print(f"  c_p^T*s [p1] = {float(cp_T_s_p1.subs(vals)):.6f}")
print(f"  ∇_pY L·w [p0] = {float(nabla_pY_w_p0.subs(vals)):.6f}")
print(f"  ∇_pY L·w [p1] = {float(nabla_pY_w_p1)}")
print()
print(f"Expected HVP: [{float(hvp_p0.subs(vals)):.6f}, {float(hvp_p1.subs(vals)):.6f}]")

# Now compare with implementation
print("\n" + "="*60)
print("COMPARISON WITH IMPLEMENTATION")
print("="*60)

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.time.benchmarks.linear_ode import QuadraticODEResidual
from pyapprox.pde.time.explicit_steppers.forward_euler import ForwardEulerResidual
from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.pde.time.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP

bkd = NumpyBkd

nstates = 1
Amat = bkd.asarray(np.array([[-0.5]]))
ode_residual = QuadraticODEResidual(Amat, bkd)

time_residual = ForwardEulerResidual(ode_residual)
newton_solver = NewtonSolver(time_residual)

init_time = 0.0
final_time = 0.2
deltat = 0.1
integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

nparams = ode_residual.nparams()
functional = EndpointFunctional(
    state_idx=0, nstates=nstates, nparams=nparams, bkd=bkd
)

operator = TimeAdjointOperatorWithHVP(integrator, functional)

param_np = bkd.asarray(np.array([[0.1], [0.2]]))
init_state = bkd.asarray(np.array([1.0]))
vvec_np = bkd.asarray(np.array([[1.0], [0.0]]))

# Run the operator
operator.storage()._clear()
hvp_impl = operator.hvp(init_state, param_np, vvec_np)

# Get intermediate values
fwd_sols, times = operator._get_forward_trajectory(init_state, param_np)
adj_sols = operator._get_adjoint_trajectory(init_state, param_np)
w_sols = operator._solve_forward_sensitivity(fwd_sols, times, param_np, vvec_np)
s_sols = operator._solve_second_adjoint(fwd_sols, adj_sols, w_sols, times, param_np, vvec_np)

print(f"\nImplementation values:")
print(f"fwd_sols = {fwd_sols.flatten()}")
print(f"adj_sols = {adj_sols.flatten()}")
print(f"w_sols = {w_sols.flatten()}")
print(f"s_sols = {s_sols.flatten()}")
print(f"\nHVP from operator: {hvp_impl.flatten()}")
print(f"Expected HVP: [{float(hvp_p0.subs(vals)):.6f}, {float(hvp_p1.subs(vals)):.6f}]")

# Diagnose differences
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print(f"Expected s1 = {float(s1.subs(vals)):.6f}, Implementation s1 = {s_sols[0, 1]:.6f}")
print(f"Expected s2 = {float(s2)}, Implementation s2 = {s_sols[0, 2]:.6f}")

# Check the Hessian terms the implementation computes
print("\nChecking Hessian terms:")

# For the second adjoint at row n=1 (y1):
# - Comes from R2 (since R2 depends on y1)
# - λ2 * d²R2/dy1² * w1  (state-state Hessian)
# - λ2 * d²R2/(dy1 dp) * v  (state-param Hessian)
print("\nAt row n=1 (for y1), contribution from R2:")
time_residual.set_time(float(times[1]), 0.1, fwd_sols[:, 1])
rss_hvp_n2 = time_residual.state_state_hvp(
    fwd_sols[:, 1], fwd_sols[:, 2], adj_sols[:, 2], w_sols[:, 1]
)
rsp_hvp_n2 = time_residual.state_param_hvp(
    fwd_sols[:, 1], fwd_sols[:, 2], adj_sols[:, 2], vvec_np
)
print(f"  Using w1={w_sols[0,1]:.6f}:")
print(f"    rss_hvp (λ2 * d²R2/dy1² * w1) = {rss_hvp_n2}")
print(f"    rsp_hvp (λ2 * d²R2/(dy1 dp) * v) = {rsp_hvp_n2}")
print(f"  Expected: ∇_{{y1,y1}}L * w1 = {float(nabla_y1y1_L.subs(vals) * w1.subs(vals)):.6f}")
print(f"  Expected: ∇_{{y1,p0}}L * v0 = {float(nabla_y1p0_L.subs(vals)):.6f}")

# Now check what implementation actually computes at nn=1
print("\nWhat implementation computes at nn=1 (backward sweep):")
print(f"  w_idx = nn - 1 = 0 (uses w0={w_sols[0,0]:.6f}, should use w1!)")
time_residual.set_time(float(times[0]), 0.1, fwd_sols[:, 0])
rss_hvp_impl = time_residual.state_state_hvp(
    fwd_sols[:, 0], fwd_sols[:, 1], adj_sols[:, 1], w_sols[:, 0]
)
rsp_hvp_impl = time_residual.state_param_hvp(
    fwd_sols[:, 0], fwd_sols[:, 1], adj_sols[:, 1], vvec_np
)
print(f"  rss_hvp = {rss_hvp_impl}")
print(f"  rsp_hvp = {rsp_hvp_impl}")

print("\n" + "="*60)
print("ROOT CAUSE")
print("="*60)
print("""
The issue is the index mapping in _solve_second_adjoint.

For Forward Euler (explicit scheme):
  R1 = y1 - y0 - dt*f(y0)  -> d²R1/dy0² ≠ 0, d²R1/dy1² = 0
  R2 = y2 - y1 - dt*f(y1)  -> d²R2/dy1² ≠ 0, d²R2/dy2² = 0

The Lagrangian Hessian ∇_{y_n,y_n}L sums over residuals:
  ∇_{y1,y1}L = λ2 * d²R2/dy1²  (from R2, evaluated at y1, uses w1)

But the implementation at backward sweep step nn=1:
  - Sets time context for R1 (wrong residual!)
  - Uses w_idx = nn-1 = 0 (w0, but should be w1!)
  - Calls state_state_hvp with adj_sols[:, 1] = λ1 (but should be λ2!)

The fix: For explicit schemes, the Hessian at row n comes from R_{n+1},
not R_n. We need to use:
  - time context for step n+1
  - adj_sols[:, n+1] (λ_{n+1})
  - w_sols[:, n] (w_n, the sensitivity AT y_n)
""")

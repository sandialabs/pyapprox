"""
Debug Heun HVP by tracing through block matrix formulation.

For Heun's method (RK2) with 3 time steps (n=0,1,2):

    k1 = f(y_{n-1}, t_{n-1})
    k2 = f(y_{n-1} + Δt·k1, t_n) = f(z, t_n)  where z = y_{n-1} + Δt·k1
    y_n = y_{n-1} + (Δt/2)·(k1 + k2)

Residual: R_n = y_n - y_{n-1} - (Δt/2)·(k1 + k2) = 0

Block matrix c_Y:
    [  I        0        0  ]   [ y0 ]   [ y0_init ]
    [ B1       A1        0  ] * [ y1 ] = [   0     ]
    [  0       B2       A2  ]   [ y2 ]   [   0     ]

where for Heun:
    A_n = I  (diagonal block is identity - explicit scheme)
    B_n = -I - (Δt/2)·(J1 + J2·(I + Δt·J1))

Here J1, J2 are Jacobians of f at k1, k2 evaluation points.

Key: d²R_n/dy_{n-1}² involves chain rule through both k1 and k2 stages!
"""

import numpy as np
import sympy as sp

# Define symbols
a, dt = sp.symbols('a dt', real=True)
p0, p1 = sp.symbols('p0 p1', real=True)
y0_sym = sp.Symbol('y0', real=True)
v0, v1 = sp.symbols('v0 v1', real=True)

# Define ODE RHS: f(y) = a*y + p0*y^2 + p1
def f(y, p0, p1):
    return a*y + p0*y**2 + p1

# Jacobian df/dy = a + 2*p0*y
def J(y, p0):
    return a + 2*p0*y

# Second derivative d²f/dy² = 2*p0
d2f_dy2 = 2*p0

# Mixed derivative d²f/(dy dp0) = 2*y
def d2f_dydp0(y):
    return 2*y

# =====================================================
# Forward solve using Heun's method
# =====================================================
print("="*60)
print("FORWARD SOLVE (Heun's Method)")
print("="*60)

# Step 0 -> 1
k1_0 = f(y0_sym, p0, p1)
z0 = y0_sym + dt * k1_0  # k2 evaluation point
k2_0 = f(z0, p0, p1)
y1 = y0_sym + sp.Rational(1,2) * dt * (k1_0 + k2_0)

print(f"Step 0->1:")
print(f"  k1_0 = f(y0) = {k1_0}")
print(f"  z0 = y0 + dt*k1_0 = {z0}")
print(f"  k2_0 = f(z0) = {k2_0}")
print(f"  y1 = y0 + (dt/2)*(k1_0 + k2_0)")
print(f"     = {sp.simplify(y1)}")

# Step 1 -> 2
k1_1 = f(y1, p0, p1)
z1 = y1 + dt * k1_1
k2_1 = f(z1, p0, p1)
y2 = y1 + sp.Rational(1,2) * dt * (k1_1 + k2_1)

print(f"\nStep 1->2:")
print(f"  k1_1 = f(y1)")
print(f"  z1 = y1 + dt*k1_1")
print(f"  k2_1 = f(z1)")
print(f"  y2 = y1 + (dt/2)*(k1_1 + k2_1)")

# =====================================================
# First-order Jacobians
# =====================================================
print("\n" + "="*60)
print("FIRST-ORDER JACOBIANS")
print("="*60)

# Jacobians at each evaluation point
J1_0 = J(y0_sym, p0)  # df/dy at y0
J2_0 = J(z0, p0)      # df/dy at z0

J1_1 = J(y1, p0)      # df/dy at y1
J2_1 = J(z1, p0)      # df/dy at z1

print(f"J1_0 = df/dy|_{{y0}} = {J1_0}")
print(f"J2_0 = df/dy|_{{z0}} = {sp.simplify(J2_0)}")
print(f"J1_1 = df/dy|_{{y1}} = {sp.simplify(J1_1)}")
print(f"J2_1 = df/dy|_{{z1}} = {sp.simplify(J2_1)}")

# dR1/dy0 = -I - (dt/2)*(J1_0 + J2_0 * (I + dt*J1_0))
#         = -I - (dt/2)*J1_0 - (dt/2)*J2_0 - (dt²/2)*J2_0*J1_0
# Note: dz0/dy0 = I + dt*J1_0

dz0_dy0 = 1 + dt * J1_0
dR1_dy0 = -1 - sp.Rational(1,2) * dt * (J1_0 + J2_0 * dz0_dy0)
dR1_dy0_simplified = sp.simplify(dR1_dy0)

print(f"\ndR1/dy0 (B1) = -I - (dt/2)*(J1_0 + J2_0*(I + dt*J1_0))")
print(f"            = {dR1_dy0_simplified}")

# =====================================================
# SECOND-ORDER JACOBIANS (Hessians)
# =====================================================
print("\n" + "="*60)
print("SECOND-ORDER JACOBIANS (Hessians)")
print("="*60)

# For Heun, R1 = y1 - y0 - (dt/2)*(k1_0 + k2_0)
# We need d²R1/dy0²

# Chain rule for k1:
# dk1/dy0 = J1_0
# d²k1/dy0² = d(J1_0)/dy0 = 2*p0

# Chain rule for k2 (more complex!):
# k2 = f(z0) where z0 = y0 + dt*k1 = y0 + dt*f(y0)
# dk2/dy0 = J2_0 * dz0/dy0 = J2_0 * (1 + dt*J1_0)
#
# d²k2/dy0² = d/dy0 [J2_0 * (1 + dt*J1_0)]
#           = d(J2_0)/dy0 * (1 + dt*J1_0) + J2_0 * d(1 + dt*J1_0)/dy0
#           = (dJ2_0/dz0 * dz0/dy0) * (1 + dt*J1_0) + J2_0 * dt * d(J1_0)/dy0
#           = (2*p0 * (1 + dt*J1_0)) * (1 + dt*J1_0) + J2_0 * dt * 2*p0
#           = 2*p0 * (1 + dt*J1_0)² + 2*p0 * dt * J2_0

d2k1_dy0dy0 = d2f_dy2  # = 2*p0

# For k2: using chain rule
# d(J2_0)/dy0 = d(a + 2*p0*z0)/dy0 = 2*p0 * dz0/dy0 = 2*p0 * (1 + dt*J1_0)
dJ2_0_dy0 = d2f_dy2 * dz0_dy0
dJ1_0_dy0 = d2f_dy2

d2k2_dy0dy0 = dJ2_0_dy0 * dz0_dy0 + J2_0 * dt * dJ1_0_dy0
d2k2_dy0dy0_simplified = sp.simplify(d2k2_dy0dy0)

print(f"d²k1/dy0² = {d2k1_dy0dy0}")
print(f"d²k2/dy0² = d(J2_0)/dy0 * dz0/dy0 + J2_0 * dt * d(J1_0)/dy0")
print(f"         = {d2k2_dy0dy0_simplified}")

# d²R1/dy0² = -(dt/2)*(d²k1/dy0² + d²k2/dy0²)
d2R1_dy0dy0 = -sp.Rational(1,2) * dt * (d2k1_dy0dy0 + d2k2_dy0dy0)
d2R1_dy0dy0_simplified = sp.simplify(d2R1_dy0dy0)

print(f"\nd²R1/dy0² = -(dt/2)*(d²k1/dy0² + d²k2/dy0²)")
print(f"         = {d2R1_dy0dy0_simplified}")

# Similar for R2
# R2 = y2 - y1 - (dt/2)*(k1_1 + k2_1)
# d²R2/dy1² needs similar chain rule through z1 = y1 + dt*k1_1

dz1_dy1 = 1 + dt * J1_1
d2k1_1_dy1dy1 = d2f_dy2
dJ2_1_dy1 = d2f_dy2 * dz1_dy1
dJ1_1_dy1 = d2f_dy2

d2k2_1_dy1dy1 = dJ2_1_dy1 * dz1_dy1 + J2_1 * dt * dJ1_1_dy1
d2R2_dy1dy1 = -sp.Rational(1,2) * dt * (d2k1_1_dy1dy1 + d2k2_1_dy1dy1)
d2R2_dy1dy1_simplified = sp.simplify(d2R2_dy1dy1)

print(f"\nd²R2/dy1² = {d2R2_dy1dy1_simplified}")

# =====================================================
# MIXED HESSIANS d²R/(dy dp)
# =====================================================
print("\n" + "="*60)
print("MIXED HESSIANS d²R/(dy dp)")
print("="*60)

# d²k1/dy0 dp0 = d(J1_0)/dp0 = d(a + 2*p0*y0)/dp0 = 2*y0
d2k1_dy0dp0 = 2*y0_sym

# d²k2/dy0 dp0: chain rule through z0
# dk2/dy0 = J2_0 * dz0/dy0
# d/dp0[dk2/dy0] = d(J2_0)/dp0 * dz0/dy0 + J2_0 * d(dz0/dy0)/dp0
#
# J2_0 = a + 2*p0*z0
# dJ2_0/dp0 = 2*z0
# dz0/dy0 = 1 + dt*(a + 2*p0*y0)
# d(dz0/dy0)/dp0 = dt * 2*y0

dJ2_0_dp0 = 2*z0
d_dz0dy0_dp0 = dt * 2*y0_sym
d2k2_dy0dp0 = dJ2_0_dp0 * dz0_dy0 + J2_0 * d_dz0dy0_dp0
d2k2_dy0dp0_simplified = sp.simplify(d2k2_dy0dp0)

print(f"d²k1/(dy0 dp0) = {d2k1_dy0dp0}")
print(f"d²k2/(dy0 dp0) = {d2k2_dy0dp0_simplified}")

d2R1_dy0dp0 = -sp.Rational(1,2) * dt * (d2k1_dy0dp0 + d2k2_dy0dp0)
d2R1_dy0dp0_simplified = sp.simplify(d2R1_dy0dp0)
print(f"\nd²R1/(dy0 dp0) = {d2R1_dy0dp0_simplified}")

# Similar for R2/y1
d2k1_1_dy1dp0 = 2*y1
dJ2_1_dp0 = 2*z1
d_dz1dy1_dp0 = dt * 2*y1
d2k2_1_dy1dp0 = dJ2_1_dp0 * dz1_dy1 + J2_1 * d_dz1dy1_dp0
d2R2_dy1dp0 = -sp.Rational(1,2) * dt * (d2k1_1_dy1dp0 + d2k2_1_dy1dp0)
d2R2_dy1dp0_simplified = sp.simplify(d2R2_dy1dp0)
print(f"d²R2/(dy1 dp0) = {d2R2_dy1dp0_simplified}")

# =====================================================
# ADJOINT SOLVE
# =====================================================
print("\n" + "="*60)
print("ADJOINT SOLVE (Q = y2)")
print("="*60)

# c_Y^T λ = -∇Q
# For endpoint functional Q = y2: dQ/dy0 = 0, dQ/dy1 = 0, dQ/dy2 = 1
#
# Row 2: A2^T * λ2 = -1  =>  I * λ2 = -1  =>  λ2 = -1
# Row 1: A1^T * λ1 + B2^T * λ2 = 0
#        λ1 = -B2^T * λ2 = B2^T
#
# B2 = dR2/dy1 = -I - (dt/2)*(J1_1 + J2_1*(I + dt*J1_1))
# B2^T = B2 (scalars)

dR2_dy1 = -1 - sp.Rational(1,2) * dt * (J1_1 + J2_1 * (1 + dt * J1_1))
B2 = dR2_dy1

lam2 = sp.Integer(-1)
lam1 = -B2 * lam2  # = B2

# Row 0: A0^T * λ0 + B1^T * λ1 = 0
#        λ0 = -B1^T * λ1
B1 = dR1_dy0
lam0 = -B1 * lam1

print(f"λ2 = {lam2}")
print(f"λ1 = -B2^T * λ2 = {sp.simplify(lam1)}")
print(f"λ0 = -B1^T * λ1 = {sp.simplify(lam0)}")

# =====================================================
# FORWARD SENSITIVITY
# =====================================================
print("\n" + "="*60)
print("FORWARD SENSITIVITY (w = dY/dp * v)")
print("="*60)

# c_Y w = c_p v  (our physical convention: w = dY/dp * v)
# Row 0: I * w0 = 0  =>  w0 = 0
# Row 1: B1 * w0 + A1 * w1 = -dR1/dp * v
#        w1 = -dR1/dp * v
# Row 2: B2 * w1 + A2 * w2 = -dR2/dp * v
#        w2 = -B2 * w1 - dR2/dp * v

# dR1/dp = -(dt/2)*(dk1/dp + dk2/dp)
# dk1/dp0 = y0^2, dk1/dp1 = 1
# dk2/dp0 = z0^2 + J2_0 * dt * y0^2 (chain rule through z0)
# dk2/dp1 = 1 + J2_0 * dt

dk1_0_dp0 = y0_sym**2
dk1_0_dp1 = sp.Integer(1)
dk2_0_dp0 = z0**2 + J2_0 * dt * y0_sym**2
dk2_0_dp1 = 1 + J2_0 * dt

dR1_dp0 = -sp.Rational(1,2) * dt * (dk1_0_dp0 + dk2_0_dp0)
dR1_dp1 = -sp.Rational(1,2) * dt * (dk1_0_dp1 + dk2_0_dp1)

print(f"dR1/dp0 = {sp.simplify(dR1_dp0)}")
print(f"dR1/dp1 = {sp.simplify(dR1_dp1)}")

w0 = sp.Integer(0)
w1 = -dR1_dp0 * v0 - dR1_dp1 * v1

# For R2
dk1_1_dp0 = y1**2
dk1_1_dp1 = sp.Integer(1)
dk2_1_dp0 = z1**2 + J2_1 * dt * y1**2
dk2_1_dp1 = 1 + J2_1 * dt

dR2_dp0 = -sp.Rational(1,2) * dt * (dk1_1_dp0 + dk2_1_dp0)
dR2_dp1 = -sp.Rational(1,2) * dt * (dk1_1_dp1 + dk2_1_dp1)

w2 = -B2 * w1 - dR2_dp0 * v0 - dR2_dp1 * v1

print(f"\nw0 = {w0}")
print(f"w1 = -dR1/dp * v = {sp.simplify(w1)}")
print(f"w2 = -B2*w1 - dR2/dp*v = {sp.simplify(w2)}")

# =====================================================
# LAGRANGIAN HESSIANS
# =====================================================
print("\n" + "="*60)
print("LAGRANGIAN HESSIANS")
print("="*60)

# ∇_{y_n,y_n}L = d²Q/dy_n² + Σ_m λ_m * d²R_m/dy_n²
#
# For Heun (explicit):
# d²R1/dy0² ≠ 0, d²R1/dy1² = 0, d²R1/dy2² = 0
# d²R2/dy0² = 0, d²R2/dy1² ≠ 0, d²R2/dy2² = 0
#
# ∇_{y0,y0}L = λ1 * d²R1/dy0²
# ∇_{y1,y1}L = λ2 * d²R2/dy1²
# ∇_{y2,y2}L = 0

nabla_y0y0_L = lam1 * d2R1_dy0dy0
nabla_y1y1_L = lam2 * d2R2_dy1dy1
nabla_y2y2_L = sp.Integer(0)

print(f"∇_{{y0,y0}}L = λ1 * d²R1/dy0² = {sp.simplify(nabla_y0y0_L)}")
print(f"∇_{{y1,y1}}L = λ2 * d²R2/dy1² = {sp.simplify(nabla_y1y1_L)}")
print(f"∇_{{y2,y2}}L = {nabla_y2y2_L}")

# Mixed: ∇_{y_n,p}L = Σ_m λ_m * d²R_m/(dy_n dp)
nabla_y0p0_L = lam1 * d2R1_dy0dp0
nabla_y1p0_L = lam2 * d2R2_dy1dp0

print(f"\n∇_{{y0,p0}}L = λ1 * d²R1/(dy0 dp0) = {sp.simplify(nabla_y0p0_L)}")
print(f"∇_{{y1,p0}}L = λ2 * d²R2/(dy1 dp0) = {sp.simplify(nabla_y1p0_L)}")

# =====================================================
# SECOND ADJOINT SOLVE
# =====================================================
print("\n" + "="*60)
print("SECOND ADJOINT SOLVE")
print("="*60)

# c_Y^T s = -∇_{YY}L·w - ∇_{Yp}L·v
#
# Row 2: A2^T * s2 = -∇_{y2,y2}L * w2 - ∇_{y2,p}L * v = 0
#        s2 = 0
#
# Row 1: A1^T * s1 + B2^T * s2 = -∇_{y1,y1}L * w1 - ∇_{y1,p0}L * v0
#        s1 = -∇_{y1,y1}L * w1 - ∇_{y1,p0}L * v0
#
# Row 0: A0^T * s0 + B1^T * s1 = -∇_{y0,y0}L * w0 - ∇_{y0,p0}L * v0
#        s0 = -B1^T * s1 - ∇_{y0,y0}L * w0 - ∇_{y0,p0}L * v0

s2 = sp.Integer(0)
s1 = -nabla_y1y1_L * w1 - nabla_y1p0_L * v0
s0 = -B1 * s1 - nabla_y0y0_L * w0 - nabla_y0p0_L * v0

print(f"s2 = {s2}")
print(f"s1 = -∇_{{y1,y1}}L * w1 - ∇_{{y1,p0}}L * v0")
print(f"   = {sp.simplify(s1)}")
print(f"s0 = -B1^T * s1 - ∇_{{y0,y0}}L * w0 - ∇_{{y0,p0}}L * v0")
print(f"   = {sp.simplify(s0)}")

# =====================================================
# HVP ACCUMULATION
# =====================================================
print("\n" + "="*60)
print("HVP ACCUMULATION")
print("="*60)

# HVP = c_p^T s + ∇_{pY}L·w + ∇_{pp}L·v
#
# c_p^T s = dR1/dp^T * s1 + dR2/dp^T * s2

cp_T_s_p0 = dR1_dp0 * s1 + dR2_dp0 * s2
cp_T_s_p1 = dR1_dp1 * s1 + dR2_dp1 * s2

# ∇_{p0,Y}L·w = ∇_{p0,y0}L * w0 + ∇_{p0,y1}L * w1
nabla_p0y0_L = nabla_y0p0_L  # Symmetry
nabla_p0y1_L = nabla_y1p0_L
nabla_pY_w_p0 = nabla_p0y0_L * w0 + nabla_p0y1_L * w1

# ∇_{pp}L·v = 0 (no second derivative of f w.r.t. p)
nabla_pp_v = sp.Integer(0)

hvp_p0 = cp_T_s_p0 + nabla_pY_w_p0 + nabla_pp_v
hvp_p1 = cp_T_s_p1

print(f"c_p^T * s [p0] = {sp.simplify(cp_T_s_p0)}")
print(f"c_p^T * s [p1] = {sp.simplify(cp_T_s_p1)}")
print(f"∇_{{pY}}L·w [p0] = {sp.simplify(nabla_pY_w_p0)}")
print(f"\nHVP[p0] = {sp.simplify(hvp_p0)}")
print(f"HVP[p1] = {sp.simplify(hvp_p1)}")

# =====================================================
# NUMERICAL EVALUATION
# =====================================================
print("\n" + "="*60)
print("NUMERICAL EVALUATION")
print("="*60)

vals = {a: -0.5, dt: 0.1, y0_sym: 1.0, p0: 0.1, p1: 0.2, v0: 1.0, v1: 0.0}

print(f"Parameters: a={vals[a]}, dt={vals[dt]}, y0={vals[y0_sym]}, p0={vals[p0]}, p1={vals[p1]}")
print(f"Direction: v0={vals[v0]}, v1={vals[v1]}")
print()
print(f"Forward: y1={float(y1.subs(vals)):.8f}, y2={float(y2.subs(vals)):.8f}")
print(f"Adjoint: λ0={float(lam0.subs(vals)):.8f}, λ1={float(lam1.subs(vals)):.8f}, λ2={float(lam2)}")
print(f"Sensitivity: w0={float(w0)}, w1={float(w1.subs(vals)):.8f}, w2={float(w2.subs(vals)):.8f}")
print(f"Second adjoint: s0={float(s0.subs(vals)):.8f}, s1={float(s1.subs(vals)):.8f}, s2={float(s2)}")
print()
print(f"Expected HVP: [{float(hvp_p0.subs(vals)):.8f}, {float(hvp_p1.subs(vals)):.8f}]")

# Check intermediate Hessian values
print("\n--- Intermediate Values ---")
print(f"d²R1/dy0² = {float(d2R1_dy0dy0_simplified.subs(vals)):.8f}")
print(f"d²R2/dy1² = {float(d2R2_dy1dy1_simplified.subs(vals)):.8f}")
print(f"d²R1/(dy0 dp0) = {float(d2R1_dy0dp0_simplified.subs(vals)):.8f}")
print(f"d²R2/(dy1 dp0) = {float(d2R2_dy1dp0_simplified.subs(vals)):.8f}")
print(f"∇_{{y1,y1}}L = {float(nabla_y1y1_L.subs(vals)):.8f}")
print(f"∇_{{y1,p0}}L = {float(nabla_y1p0_L.subs(vals)):.8f}")

# =====================================================
# COMPARE WITH IMPLEMENTATION
# =====================================================
print("\n" + "="*60)
print("COMPARISON WITH IMPLEMENTATION")
print("="*60)

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.time.benchmarks.linear_ode import QuadraticODEResidual
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.pde.time.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP

bkd = NumpyBkd

nstates = 1
Amat = bkd.asarray(np.array([[-0.5]]))
ode_residual = QuadraticODEResidual(Amat, bkd)

time_residual = HeunResidual(ode_residual)
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
print(f"Expected HVP: [{float(hvp_p0.subs(vals)):.8f}, {float(hvp_p1.subs(vals)):.8f}]")

# =====================================================
# DIAGNOSIS
# =====================================================
print("\n" + "="*60)
print("DIAGNOSIS: Check Heun HVP Methods")
print("="*60)

# The key issue is whether the Heun residual's HVP methods correctly
# compute the chain rule through both k1 and k2 stages.

# Check what the Heun residual returns for state_state_hvp
print("\nChecking Heun state_state_hvp at step n=1 (for y1):")
print("  This should compute λ2 * d²R2/dy1² * w1")
print()

# Set time context for R2 (step 1->2)
time_residual.set_time(float(times[1]), 0.1, fwd_sols[:, 1])

# Expected: λ2 * d²R2/dy1² * w1
expected_ss_hvp = float(nabla_y1y1_L.subs(vals) * w1.subs(vals))
print(f"  Expected: ∇_{{y1,y1}}L * w1 = {expected_ss_hvp:.8f}")

# What implementation computes
impl_ss_hvp = time_residual._state_state_hvp(
    fwd_sols[:, 1], fwd_sols[:, 2], adj_sols[:, 2], w_sols[:, 1:2]
)
print(f"  Implementation: {float(impl_ss_hvp):.8f}")

# Also check state_param_hvp
expected_sp_hvp = float(nabla_y1p0_L.subs(vals))
print(f"\n  Expected ∇_{{y1,p0}}L = {expected_sp_hvp:.8f}")

impl_sp_hvp = time_residual._state_param_hvp(
    fwd_sols[:, 1], fwd_sols[:, 2], adj_sols[:, 2], vvec_np
)
print(f"  Implementation: {impl_sp_hvp.flatten()}")

# Check the underlying ODE HVP values
print("\n--- Underlying ODE HVP values ---")
y1_val = float(y1.subs(vals))
z1_val = y1_val + 0.1 * (vals[a] * y1_val + vals[p0] * y1_val**2 + vals[p1])

ode_residual.set_time(float(times[1]))  # t = 0.1
adj_scalar = float(adj_sols[0, 2])  # λ2 = -1
w1_scalar = float(w1.subs(vals))

# k1 contribution: λ * -(dt/2) * d²f/dy² * w1
# d²f/dy² = 2*p0 = 0.2
k1_ss = adj_scalar * (-0.5 * 0.1) * (2 * vals[p0]) * w1_scalar
print(f"k1 stage ss_hvp contribution: λ2 * -(dt/2) * 2*p0 * w1 = {k1_ss:.8f}")

# k2 contribution (simplified - what implementation does)
ode_residual.set_time(float(times[2]))  # t = 0.2
k2_ss = adj_scalar * (-0.5 * 0.1) * (2 * vals[p0]) * w1_scalar
print(f"k2 stage ss_hvp contribution (simplified): λ2 * -(dt/2) * 2*p0 * w1 = {k2_ss:.8f}")

# What it should be with proper chain rule
# d²k2/dy² = 2*p0 * (1 + dt*J1)² + 2*p0 * dt * J2
J1_val = vals[a] + 2*vals[p0]*y1_val
J2_val = vals[a] + 2*vals[p0]*z1_val
dz1dy1_val = 1 + 0.1 * J1_val
d2k2_correct = 2*vals[p0] * dz1dy1_val**2 + 2*vals[p0] * 0.1 * J2_val
k2_ss_correct = adj_scalar * (-0.5 * 0.1) * d2k2_correct * w1_scalar
print(f"\nCorrect k2 stage ss_hvp (with chain rule):")
print(f"  d²k2/dy² = 2*p0*(1+dt*J1)² + 2*p0*dt*J2 = {d2k2_correct:.8f}")
print(f"  contribution = λ2 * -(dt/2) * d²k2/dy² * w1 = {k2_ss_correct:.8f}")

print(f"\nTotal correct ss_hvp: {k1_ss + k2_ss_correct:.8f}")
print(f"Total simplified ss_hvp: {k1_ss + k2_ss:.8f}")

print("\n" + "="*60)
print("ROOT CAUSE")
print("="*60)
print("""
The Heun _state_state_hvp method is simplified and doesn't properly
compute the chain rule through the k2 stage.

Current implementation:
  k1_ss_hvp = ode.state_state_hvp(y_{n-1}, adj, w)  # uses y_{n-1}
  k2_ss_hvp = ode.state_state_hvp(z, adj, w)        # uses z, but wrong w!

The issue is:
1. k2_ss_hvp should use the chain rule: d²k2/dy² involves
   d(J2*(1+dt*J1))/dy = (2*p0*(1+dt*J1)) * (1+dt*J1) + J2*dt*2*p0

2. The Heun HVP methods pass 'wvec' directly to the ODE's state_state_hvp
   at k2_state, but this doesn't account for dz/dy = (1 + dt*J1)

The correct approach is to expand the chain rule explicitly:
  d²k2/dy² · w = d²f/dz²|_z · (dz/dy)² · w + J2 · dt · d²f/dy²|_y · w
               = 2*p0 * (1+dt*J1)² * w + J2 * dt * 2*p0 * w
""")

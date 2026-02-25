"""
SymPy derivation of HVP for 2-step Forward Euler with quadratic ODE.

This script derives the HVP analytically using SymPy and compares with
the numerical implementation step by step.

ODE: dy/dt = a*y + p0*y^2 + p1 = f(y, p)

Forward Euler residuals:
    R1 = y1 - y0 - dt*f(y0, p)
    R2 = y2 - y1 - dt*f(y1, p)

Functional: Q = y2 (endpoint)

Lagrangian: L = Q + λ1*R1 + λ2*R2
"""

import sympy as sp
import numpy as np

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
print(f"y1 = {sp.simplify(y1)}")
print(f"y2 = {sp.simplify(y2)}")

# Adjoint equations
# dL/dy2 = 1 + λ2 * 1 = 0 => λ2 = -1
# dL/dy1 = λ1*1 + λ2*(-1 - dt*df/dy1) = 0
#        => λ1 = -λ2*(1 + dt*df/dy1) = (1 + dt*(a + 2*p0*y1))

# df/dy = a + 2*p0*y
dfdy1 = a + 2*p0*y1

lam2 = sp.Integer(-1)
lam1 = -(1 + dt*dfdy1)

print("\n=== Adjoint Solve ===")
print(f"λ2 = {lam2}")
print(f"λ1 = {sp.simplify(lam1)}")

# Sensitivity (tangent linear)
# w0 = 0 (y0 doesn't depend on p)
# w1 = dt * (∂f/∂p0|_{y0} * v0 + ∂f/∂p1|_{y0} * v1) = dt*(y0^2*v0 + v1)
# w2 = (1 + dt*df/dy1)*w1 + dt*(y1^2*v0 + v1)

w0 = sp.Integer(0)
w1 = dt * (y0_sym**2 * v0 + v1)
w2 = (1 + dt*dfdy1) * w1 + dt * (y1**2 * v0 + v1)

print("\n=== Sensitivity (w = dy/dp * v) ===")
print(f"w0 = {w0}")
print(f"w1 = {sp.simplify(w1)}")
print(f"w2 = {sp.simplify(w2)}")

# Gradient: dQ/dp = Σ_n λ_n * dR_n/dp
# dR1/dp0 = -dt*y0^2, dR1/dp1 = -dt
# dR2/dp0 = -dt*y1^2, dR2/dp1 = -dt

dR1dp0 = -dt * y0_sym**2
dR1dp1 = -dt
dR2dp0 = -dt * y1**2
dR2dp1 = -dt

grad_p0 = lam1 * dR1dp0 + lam2 * dR2dp0
grad_p1 = lam1 * dR1dp1 + lam2 * dR2dp1

print("\n=== Gradient ===")
print(f"dQ/dp0 = {sp.simplify(grad_p0)}")
print(f"dQ/dp1 = {sp.simplify(grad_p1)}")

# HVP: d²Q/dp² * v = d/dp[gradient] * v
#
# Two types of terms:
# 1. (dλ/dp * v) * dR/dp  - from λ depending on p through y
# 2. λ * d²R/dpdpy * dy/dp * v - from y in dR/dp

# Term 1: dλ1/dp * dR1/dp
# λ1 = -(1 + dt*(a + 2*p0*y1))
# dλ1/dp0 = -dt*(2*y1 + 2*p0*dy1/dp0)
#         = -dt*2*(y1 + p0*w1/v0) for direction v0
# More generally, dλ1/dp * v = -dt*(2*y1*v0 + 2*p0*(dy1/dp)*v)
#                            = -dt*(2*y1*v0 + 2*p0*w1)

dy1dp = sp.Matrix([[sp.diff(y1, p0)], [sp.diff(y1, p1)]])
v_vec = sp.Matrix([[v0], [v1]])

# w1 = dy1/dp * v = (dy1/dp0 * v0 + dy1/dp1 * v1)
w1_check = (dy1dp.T @ v_vec)[0, 0]
print(f"\nVerify w1: {sp.simplify(w1_check - w1)}")  # Should be 0

dlam1_v = -dt * 2 * (y1 * v0 + p0 * w1)  # This is (dλ1/dp) * v

print("\n=== HVP Terms ===")
print(f"(dλ1/dp)·v = {sp.simplify(dlam1_v)}")

# Term 1a: (dλ1/dp·v) * dR1/dp0
term1a_p0 = dlam1_v * dR1dp0
print(f"Term 1a (n=1, comp 0): (dλ1·v) * dR1/dp0 = {sp.simplify(term1a_p0)}")

# Term 1a for p1: (dλ1/dp·v) * dR1/dp1
term1a_p1 = dlam1_v * dR1dp1
print(f"Term 1a (n=1, comp 1): (dλ1·v) * dR1/dp1 = {sp.simplify(term1a_p1)}")

# Term 1b: (dλ2/dp·v) * dR2/dp - but dλ2/dp = 0 since λ2 = -1

# Term 2: λ * d²R/dpdpy * w
# d²R1/dp0dy0 = -dt * 2*y0, d²R1/dp1dy0 = 0
# For n=1: λ1 * (-dt * 2*y0) * w0 = 0 (since w0 = 0)

# For n=2: λ2 * d²R2/dp0dy1 * w1 = (-1) * (-dt * 2*y1) * w1
term2_n2_p0 = lam2 * (-dt * 2 * y1) * w1
print(f"Term 2 (n=2, comp 0): λ2 * d²R2/dp0dy1 * w1 = {sp.simplify(term2_n2_p0)}")

# d²R2/dp1dy1 = 0
term2_n2_p1 = sp.Integer(0)
print(f"Term 2 (n=2, comp 1): λ2 * d²R2/dp1dy1 * w1 = {term2_n2_p1}")

# Total HVP
hvp_p0 = term1a_p0 + term2_n2_p0
hvp_p1 = term1a_p1 + term2_n2_p1

print("\n=== Total HVP ===")
print(f"HVP[0] = {sp.simplify(hvp_p0)}")
print(f"HVP[1] = {sp.simplify(hvp_p1)}")

# Numerical evaluation
vals = {a: -0.5, dt: 0.1, y0_sym: 1.0, p0: 0.1, p1: 0.2, v0: 1.0, v1: 0.0}

print("\n=== Numerical Values ===")
print(f"y1 = {float(y1.subs(vals)):.6f}")
print(f"y2 = {float(y2.subs(vals)):.6f}")
print(f"λ1 = {float(lam1.subs(vals)):.6f}")
print(f"λ2 = {float(lam2)}")
print(f"w1 = {float(w1.subs(vals)):.6f}")
print(f"w2 = {float(w2.subs(vals)):.6f}")
print(f"Gradient: [{float(grad_p0.subs(vals)):.6f}, {float(grad_p1.subs(vals)):.6f}]")
print(f"(dλ1/dp)·v = {float(dlam1_v.subs(vals)):.6f}")
print(f"Term 1a (p0) = {float(term1a_p0.subs(vals)):.6f}")
print(f"Term 2 (n=2, p0) = {float(term2_n2_p0.subs(vals)):.6f}")
print(f"HVP = [{float(hvp_p0.subs(vals)):.6f}, {float(hvp_p1.subs(vals)):.6f}]")

# Now compare with the implementation's accumulation
print("\n" + "="*60)
print("COMPARISON WITH IMPLEMENTATION")
print("="*60)

# The implementation computes:
# 1. s_sols (second adjoint) via backward solve
# 2. Accumulates: dR/dp^T * s + qps_hvp + rps_hvp + rpp_hvp

# Let's trace what the implementation does
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

ode_residual.set_param(param_np.flatten())
fwd_sols, times = integrator.solve(init_state)
integrator.set_functional(functional)
adj_sols = integrator.solve_adjoint(fwd_sols, times, param_np)

vvec_np = bkd.asarray(np.array([[1.0], [0.0]]))
w_sols = operator._solve_forward_sensitivity(fwd_sols, times, param_np, vvec_np)
s_sols = operator._solve_second_adjoint(fwd_sols, adj_sols, w_sols, times, param_np, vvec_np)

print("\nImplementation values:")
print(f"fwd_sols = {fwd_sols.flatten()}")
print(f"adj_sols = {adj_sols.flatten()}")
print(f"w_sols = {w_sols.flatten()}")
print(f"s_sols = {s_sols.flatten()}")

# The second adjoint s should encode (dλ/dp · v) information
# Let's check what the correct s should be

# From the HVP derivation, the key contribution is:
# (dλ1/dp · v) * dR1/dp0 = (dλ1/dp · v) * (-dt * y0^2)
#
# If we want s to satisfy: s^T * dR/dp = (dλ/dp · v)^T * dR/dp
# Then we need s to be related to (dλ/dp · v)

# But actually, the second adjoint s solves different equations.
# The issue is that the current implementation seems to be double-counting.

# Let me check the HVP computed by the operator
operator.storage()._clear()
hvp_impl = operator.hvp(init_state, param_np, vvec_np)
print(f"\nHVP from operator: {hvp_impl.flatten()}")
print(f"Expected HVP (SymPy): [{float(hvp_p0.subs(vals)):.6f}, {float(hvp_p1.subs(vals)):.6f}]")

# Trace the accumulation
print("\nAccumulation trace:")
hvp_acc = np.zeros((2, 1))

# Step 1
nn = 1
time_residual.set_time(float(times[nn-1]), deltat, fwd_sols[:, nn-1])
drdp_1 = time_residual.param_jacobian(fwd_sols[:, nn-1], fwd_sols[:, nn])
s1_contrib = drdp_1.T @ s_sols[:, nn:nn+1]
print(f"Step {nn}: dR^T * s = {s1_contrib.flatten()}")
hvp_acc += s1_contrib

w_idx = nn - 1  # For FE
rps_hvp_1 = time_residual.param_state_hvp(
    fwd_sols[:, nn-1], fwd_sols[:, nn], adj_sols[:, nn], w_sols[:, w_idx]
)
print(f"Step {nn}: rps_hvp = {rps_hvp_1}")
hvp_acc += rps_hvp_1.reshape(-1, 1)

# Step 2
nn = 2
time_residual.set_time(float(times[nn-1]), deltat, fwd_sols[:, nn-1])
drdp_2 = time_residual.param_jacobian(fwd_sols[:, nn-1], fwd_sols[:, nn])
s2_contrib = drdp_2.T @ s_sols[:, nn:nn+1]
print(f"Step {nn}: dR^T * s = {s2_contrib.flatten()}")
hvp_acc += s2_contrib

w_idx = nn - 1
rps_hvp_2 = time_residual.param_state_hvp(
    fwd_sols[:, nn-1], fwd_sols[:, nn], adj_sols[:, nn], w_sols[:, w_idx]
)
print(f"Step {nn}: rps_hvp = {rps_hvp_2}")
hvp_acc += rps_hvp_2.reshape(-1, 1)

print(f"\nAccumulated HVP: {hvp_acc.flatten()}")

# The problem is that dR^T * s at step 1 gives 0.0386, but this should be
# encoding the (dλ/dp · v) term.
#
# The expected HVP[0] = 0.0394 comes from:
# - Term 1a: (dλ1/dp · v) * dR1/dp0 = 0.0198
# - Term 2:  λ2 * d²R2/dp0dy1 * w1 = 0.0196
#
# The issue seems to be that s includes BOTH contributions somehow.

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print(f"Expected Term 1a (n=1, p0): {float(term1a_p0.subs(vals)):.6f}")
print(f"Expected Term 2 (n=2, p0): {float(term2_n2_p0.subs(vals)):.6f}")
print(f"Sum: {float(hvp_p0.subs(vals)):.6f}")
print()
print(f"Implementation gets:")
print(f"  Step 1: dR^T * s = {s1_contrib[0, 0]:.6f} (should be ~0?)")
print(f"  Step 2: dR^T * s = {s2_contrib[0, 0]:.6f}")
print(f"  Step 2: rps_hvp = {rps_hvp_2[0]:.6f}")
print(f"  Total: {hvp_acc[0, 0]:.6f}")

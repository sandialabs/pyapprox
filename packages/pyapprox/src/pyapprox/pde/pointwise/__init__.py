"""Pointwise physics: a PDE written once as functions at quadrature points.

A physics states, per field, the integrands ``f0`` and ``f1`` of its weak
residual ``R_a(u; v) = ∫ v f0_a + ∇v · f1_a dx`` as sympy expressions of
the coordinates, time, fields, their gradients, auxiliary data fields and
parameters (:mod:`~pyapprox.pde.pointwise.spec`). Every derivative a
solver or adjoint needs is generated from them by symbolic
differentiation (:mod:`~pyapprox.pde.pointwise.derivatives`), and the
expressions are compiled into kernels evaluated on batches of quadrature
points (:mod:`~pyapprox.pde.pointwise.kernel`). The discretization (mesh,
bases, quadrature, element loop, scatter, constraints, solves) belongs to
the stack that evaluates the kernel.
"""

import dolfin as dl
from pyapprox_dev.fenics_models.fenics_utilities import *
dl.set_log_level(40)
# dl.set_log_level(20)
# dl.set_log_level(16)


def get_2d_square_mesh_boundary_segments():
    bndry_obj = []
    for phys_var in [0, 1]:
        for bndry_coord in [0., 1.]:
            seg_left = 0.
            for seg_right in [0.375, 0.625, 1.]:
                bndry_obj.append(get_2d_rectangular_mesh_boundary_segment(
                    phys_var, bndry_coord, seg_left, seg_right))
                seg_left = seg_right
    return bndry_obj


def run_model(kappa, forcing, function_space, boundary_conditions=None):
    """
    Solve complex valued Helmholtz equation by solving coupled system, one
    for the real part of the solution one for the imaginary part.

    """
    mesh = function_space.mesh()
    kappa_sq = kappa**2

    if boundary_conditions == None:
        bndry_obj = dl.CompiledSubDomain("on_boundary")
        boundary_conditions = [['dirichlet', bndry_obj, [0, 0]]]

    num_bndrys = len(boundary_conditions)
    boundaries = mark_boundaries(mesh, boundary_conditions)
    dirichlet_bcs = collect_dirichlet_boundaries(
        function_space, boundary_conditions, boundaries)

    # To express integrals over the boundary parts using ds(i), we must first
    # redefine the measure ds in terms of our boundary markers:
    ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    #dx = dl.Measure('dx', domain=mesh)
    dx = dl.dx

    (pr, pi) = dl.TrialFunction(function_space)
    (vr, vi) = dl.TestFunction(function_space)

    # real part
    bilinear_form = kappa_sq*(pr*vr - pi*vi)*dx
    bilinear_form += (-dl.inner(dl.nabla_grad(pr), dl.nabla_grad(vr)) +
                      dl.inner(dl.nabla_grad(pi), dl.nabla_grad(vi)))*dx
    # imaginary part
    bilinear_form += kappa_sq*(pr*vi + pi*vr)*dx
    bilinear_form += -(dl.inner(dl.nabla_grad(pr), dl.nabla_grad(vi)) +
                       dl.inner(dl.nabla_grad(pi), dl.nabla_grad(vr)))*dx

    for ii in range(num_bndrys):
        if (boundary_conditions[ii][0] == 'robin'):
            alpha_real, alpha_imag = boundary_conditions[ii][3]
            bilinear_form -= alpha_real*(pr*vr-pi*vi)*ds(ii)
            bilinear_form -= alpha_imag*(pr*vi+pi*vr)*ds(ii)

    forcing_real, forcing_imag = forcing
    rhs = (forcing_real*vr+forcing_real*vi+forcing_imag*vr-forcing_imag*vi)*dx

    for ii in range(num_bndrys):
        if ((boundary_conditions[ii][0] == 'robin') or
                (boundary_conditions[ii][0] == 'neumann')):
            beta_real, beta_imag = boundary_conditions[ii][2]
            # real part of robin boundary conditions
            rhs += (beta_real*vr - beta_imag*vi)*ds(ii)
            # imag part of robin boundary conditions
            rhs += (beta_real*vi + beta_imag*vr)*ds(ii)

    # compute solution
    p = dl.Function(function_space)
    #solve(a == L, p)
    dl.solve(bilinear_form == rhs, p, bcs=dirichlet_bcs)

    return p

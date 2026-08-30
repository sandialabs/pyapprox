"""Operator learning end to end on the 1D Poisson equation.

Composes four subsystems that unit tests exercise separately: a
collocation PDE solve, field encoding, induced sampling, and weighted
least squares. The operator being learned is the solution operator of

    -u'' = f  on [0, 1],  u(0) = u(1) = 0

whose Green's function is known in closed form,

    G(x, y) = min(x, y) - x y

so the fitted surrogate can be checked against an analytic result that
appears nowhere in the fitting path.
"""

import math
from typing import Any, Tuple

import numpy as np
import pytest
from pyapprox.ode.mass_matrix import DiagonalMassMatrix
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.probability import UniformMarginal
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.surrogates.affine.basis.orthonormal_poly import (
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.induced import InducedSampler
from pyapprox.surrogates.affine.univariate.factory import create_bases_1d
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.operatorlearning import (
    GramProjectionEncoder,
    WeightedLeastSquaresOperatorFitter,
    bochner_error,
    sample_complexity,
)
from pyapprox.util.backends.protocols import Backend

NMODES = 5
# Clenshaw-Curtis needs 2^l + 1 points, and its nodes then coincide
# with the collocation mesh's Chebyshev-Gauss-Lobatto points.
NPTS = 33


def _poisson_solution_operator(bkd: Backend, npts: int) -> Tuple[Any, Any]:
    """Return (nodes, solution operator) for -u'' = f on [0, 1].

    With constant diffusion and no advection or reaction the residual
    is affine in the state, so the boundary-condition-applied Jacobian
    is the discrete operator and a single solve inverts it. The
    returned matrix maps nodal forcing to nodal solution values, which
    is the discrete Green's function in nodal coordinates.
    """
    mesh = TransformedMesh1D(npts, bkd, AffineTransform1D((0.0, 1.0), bkd))
    basis = ChebyshevBasis1D(mesh, bkd)
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
    boundary = [
        zero_dirichlet_bc(bkd, mesh.boundary_indices(0)),
        zero_dirichlet_bc(bkd, mesh.boundary_indices(1)),
    ]
    physics.set_boundary_conditions(boundary)

    zero = bkd.zeros((npts,))
    jacobian = physics.jacobian(zero, 0.0)
    _, jacobian = physics.apply_boundary_conditions(
        zero, jacobian, zero, 0.0
    )

    # The residual is D u'' + f, so solving it against -f gives -u'' = f.
    operator = -bkd.inv(jacobian)
    # Boundary rows of the right-hand side hold the boundary value, not
    # the forcing, so the corresponding columns must not act on f.
    np_operator = bkd.to_numpy(operator).copy()
    for boundary_id in (0, 1):
        index = int(bkd.to_numpy(mesh.boundary_indices(boundary_id))[0])
        np_operator[:, index] = 0.0
    return mesh.points()[0], bkd.asarray(np_operator)


def _sine_encoder(bkd: Backend, nodes: Any, nmodes: int) -> Any:
    """Encoder onto sine modes, orthonormal under Clenshaw-Curtis.

    Sines vanish at both ends, so they satisfy the boundary conditions
    the solution obeys, and they diagonalize the Laplacian — which is
    what makes the recovered operator interpretable.

    The inner product uses Clenshaw-Curtis, whose nodes are the
    Chebyshev-Gauss-Lobatto points the collocation mesh already uses.
    It converges spectrally for smooth integrands, so the basis is
    orthonormal to machine precision rather than to the O(h^2) a
    trapezoid rule on the same nodes would give — and a defective
    inner product would corrupt both the projection and the isometry
    on which the error measure depends.
    """
    np_nodes = bkd.to_numpy(nodes)
    modes = np.arange(1, nmodes + 1)
    raw = np.sqrt(2.0) * np.sin(np.outer(np_nodes, modes) * np.pi)

    rule = ClenshawCurtisQuadratureRule(bkd, store=True, prob_measure=True)
    quad_nodes, quad_weights = rule(np_nodes.shape[0])
    # The rule is built on [-1, 1] and ordered by its own convention;
    # match each weight to the mesh node it belongs to.
    mapped = (bkd.to_numpy(quad_nodes).ravel() + 1.0) / 2.0
    order = np.argsort(np.argsort(np_nodes))
    weights = bkd.to_numpy(quad_weights).ravel()[np.argsort(mapped)][order]

    return GramProjectionEncoder(
        bkd.asarray(raw),
        DiagonalMassMatrix(bkd.asarray(weights), bkd),
        bkd,
    )


def _linear_expansion(
    bkd: Backend, nvars: int, noutputs: int
) -> PolynomialChaosExpansion:
    """Expansion over the first-order terms, so the fit stays linear."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    basis = OrthonormalPolynomialBasis(create_bases_1d(marginals, bkd), bkd)
    basis.set_indices(
        bkd.asarray(np.eye(nvars, dtype=int), dtype=bkd.int64_dtype())
    )
    return PolynomialChaosExpansion(basis, bkd, nqoi=noutputs)


class TestPoissonOperatorLearning:
    """T11: learn the Poisson solution operator from solves."""

    def _fit(self, bkd: Backend, nsamples: int) -> Any:
        """Solve, encode, sample, fit. Returns everything downstream needs."""
        nodes, operator = _poisson_solution_operator(bkd, NPTS)
        encoder = _sine_encoder(bkd, nodes, NMODES)

        expansion = _linear_expansion(bkd, NMODES, NMODES)
        rho = IndependentJoint(
            [UniformMarginal(-1.0, 1.0, bkd) for _ in range(NMODES)], bkd
        )
        sampler = InducedSampler(
            expansion._ortho_basis(), rho, bkd, nquad=200
        )
        drawn = sampler(nsamples)

        # Each realization is a forcing field; solving gives the
        # response the surrogate must learn to reproduce.
        forcings = encoder.decode(drawn.samples)
        solutions = bkd.dot(operator, forcings)

        result = WeightedLeastSquaresOperatorFitter(
            encoder, encoder, bkd
        ).fit(expansion, forcings, solutions, weights=drawn.weights)
        return nodes, operator, encoder, result

    def test_recovers_the_solution_operator(
        self, numpy_bkd: Backend
    ) -> None:
        """The surrogate reproduces solves it was never shown.

        Error is measured on held-out forcings, in the Bochner norm the
        isometric encoder makes meaningful.
        """
        np.random.seed(0)
        nsamples = sample_complexity(NMODES, 0.5, 0.5)
        nodes, operator, encoder, result = self._fit(numpy_bkd, nsamples)

        np.random.seed(1)
        held_out = numpy_bkd.asarray(
            np.random.uniform(-1.0, 1.0, (NMODES, 50))
        )
        forcings = encoder.decode(held_out)
        truth = numpy_bkd.dot(operator, forcings)

        predicted = result.surrogate()(forcings)
        error = bochner_error(
            encoder.encode(predicted), encoder.encode(truth), numpy_bkd
        )
        assert error < 1e-8

    def test_operator_matrix_is_diagonal(self, numpy_bkd: Backend) -> None:
        """Sines diagonalize the Laplacian, so the matrix must be too.

        The fit is told nothing about this: it sees forcing and
        solution fields alone. Recovering a diagonal matrix shows it
        found the structure rather than fitting noise into off-diagonal
        couplings.
        """
        np.random.seed(0)
        nsamples = sample_complexity(NMODES, 0.5, 0.5)
        _, _, _, result = self._fit(numpy_bkd, nsamples)

        matrix = numpy_bkd.to_numpy(result.operator_matrix())
        off_diagonal = matrix - np.diag(np.diag(matrix))
        relative = np.max(np.abs(off_diagonal)) / np.max(np.abs(matrix))
        assert relative < 1e-6

    def test_eigenvalues_match_the_analytic_spectrum(
        self, numpy_bkd: Backend
    ) -> None:
        """The diagonal holds the inverse Laplacian eigenvalues.

        For -u'' = f with sine modes, mode n has eigenvalue
        1 / (pi n)^2. This value never enters the fit; it comes from
        the PDE solve alone.
        """
        np.random.seed(0)
        nsamples = sample_complexity(NMODES, 0.5, 0.5)
        _, _, _, result = self._fit(numpy_bkd, nsamples)

        # Legendre degree one is sqrt(3) x, so the fitted coefficients
        # carry that factor relative to the operator's eigenvalues.
        diagonal = np.diag(numpy_bkd.to_numpy(result.operator_matrix()))
        analytic = np.array(
            [1.0 / (math.pi * (n + 1)) ** 2 for n in range(NMODES)]
        )
        np.testing.assert_allclose(
            diagonal * np.sqrt(3.0), analytic, rtol=1e-6
        )

    @staticmethod
    def _greens_integral(nodes: Any, mode: int, nquad: int) -> Any:
        """Integrate G against a sine forcing on a uniform rule.

        G(x, y) = min(x, y) - x y has a kink on the diagonal, so a
        trapezoid rule converges at O(h^2) rather than spectrally. The
        rule is refined independently of the mesh so its error can be
        separated from the surrogate's.
        """
        quad = np.linspace(0.0, 1.0, nquad)
        greens = (
            np.minimum(nodes[:, None], quad[None, :])
            - nodes[:, None] * quad[None, :]
        )
        forcing = np.sqrt(2.0) * np.sin(mode * np.pi * quad)
        return np.trapezoid(greens * forcing[None, :], quad, axis=1)

    def test_action_matches_the_greens_function(
        self, numpy_bkd: Backend
    ) -> None:
        """The learned operator integrates against min(x, y) - x y.

        The Green's function is the analytic solution operator and
        appears nowhere in the fitting path, so agreeing with it is an
        independent check.

        The comparison is made at increasing quadrature resolution and
        the residual must fall at the rule's O(h^2) rate. A fixed
        tolerance would not distinguish quadrature error from a wrong
        operator, since a surrogate off by a small constant would sit
        under any bound loose enough to admit the quadrature.
        """
        np.random.seed(0)
        nsamples = sample_complexity(NMODES, 0.5, 0.5)
        nodes, _, encoder, result = self._fit(numpy_bkd, nsamples)
        np_nodes = numpy_bkd.to_numpy(nodes)

        # A single sine mode as forcing, whose coefficients are known.
        mode = 2
        coefs = numpy_bkd.zeros((NMODES, 1))
        coefs[mode - 1, 0] = 1.0
        forcing = encoder.decode(coefs)
        predicted = numpy_bkd.to_numpy(result.surrogate()(forcing))[:, 0]

        errors = []
        for nquad in (65, 257, 1025):
            integral = self._greens_integral(np_nodes, mode, nquad)
            errors.append(float(np.max(np.abs(predicted - integral))))

        # Sixteen times the quadrature points is sixteen times the
        # accuracy for a second-order rule; allow a factor of two.
        assert errors[1] < errors[0] / 8.0
        assert errors[2] < errors[1] / 8.0
        assert errors[2] < 1e-6

    def test_beats_a_trivial_predictor(self, numpy_bkd: Backend) -> None:
        """Guards against an error metric that would flatter anything.

        Predicting zero everywhere is the null model; the fit must be
        far better than it, or the tolerance above says nothing.
        """
        np.random.seed(0)
        nsamples = sample_complexity(NMODES, 0.5, 0.5)
        nodes, operator, encoder, result = self._fit(numpy_bkd, nsamples)

        np.random.seed(3)
        held_out = numpy_bkd.asarray(
            np.random.uniform(-1.0, 1.0, (NMODES, 20))
        )
        forcings = encoder.decode(held_out)
        truth_codes = encoder.encode(numpy_bkd.dot(operator, forcings))
        predicted_codes = encoder.encode(result.surrogate()(forcings))

        fitted = bochner_error(predicted_codes, truth_codes, numpy_bkd)
        trivial = bochner_error(
            numpy_bkd.zeros(truth_codes.shape), truth_codes, numpy_bkd
        )
        assert trivial == pytest.approx(1.0)
        assert fitted < 1e-6 * trivial

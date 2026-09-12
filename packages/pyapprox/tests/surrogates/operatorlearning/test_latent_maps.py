r"""Tests for a neural latent map.

Torch-only by construction, so these take ``torch_bkd`` rather than the
parametrized ``bkd`` fixture. What needs pinning is the protocol level it
satisfies -- the narrowest one, so the least-squares fitter refuses it --
and that the autograd graph survives its forward pass, since a severed
graph makes a fit run, change nothing and report nothing.
"""

import numpy as np
import pytest
import torch
from pyapprox.surrogates.operatorlearning import (
    IdentityFieldEncoder,
    LatentMapProtocol,
    LinearInParamsLatentMapProtocol,
    MultiIndexLatentMapProtocol,
    OperatorSurrogate,
)
from pyapprox.surrogates.operatorlearning.latent_maps import MLPLatentMap
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend


class TestProtocolLevel:
    """It satisfies the narrowest level, and that is load bearing."""

    def test_is_a_latent_map(self, torch_bkd: Backend) -> None:
        assert isinstance(
            MLPLatentMap(3, 2, [8], torch_bkd), LatentMapProtocol
        )

    def test_is_not_linear_in_its_parameters(
        self, torch_bkd: Backend
    ) -> None:
        """Which is why the closed-form fitter must refuse it.

        A network has no ``basis_matrix``, so there is no single linear
        system to solve. Asserted rather than assumed, because the
        fitter's guard is only as good as this being genuinely False.
        """
        latent_map = MLPLatentMap(3, 2, [8], torch_bkd)
        assert not isinstance(
            latent_map, LinearInParamsLatentMapProtocol
        )
        assert not isinstance(latent_map, MultiIndexLatentMapProtocol)

    def test_declares_no_derivatives_bundle(
        self, torch_bkd: Backend
    ) -> None:
        """Fitting differentiates parameters, not inputs.

        A ``Derivatives`` bundle describes derivatives with respect to
        the *inputs*, which no consumer here wants; autograd reaches the
        parameters through ``parameters()``. An absent capability is
        absent rather than present and raising.
        """
        assert not hasattr(MLPLatentMap(3, 2, [8], torch_bkd), "derivatives")


class TestShapes:
    @pytest.mark.parametrize("hidden", [[], [4], [8, 8]])
    def test_maps_codes_to_codes(
        self, torch_bkd: Backend, hidden
    ) -> None:
        """Empty hidden dims give a single affine map, the linear case."""
        latent_map = MLPLatentMap(3, 2, hidden, torch_bkd)
        codes = torch_bkd.asarray(np.random.uniform(-1.0, 1.0, (3, 11)))
        assert latent_map(codes).shape == (2, 11)

    def test_reports_its_widths(self, torch_bkd: Backend) -> None:
        latent_map = MLPLatentMap(5, 3, [8], torch_bkd)
        assert latent_map.nvars() == 5
        assert latent_map.nqoi() == 3

    def test_rejects_wrong_input_width(self, torch_bkd: Backend) -> None:
        latent_map = MLPLatentMap(3, 2, [4], torch_bkd)
        with pytest.raises(ValueError, match="input codes"):
            latent_map(torch_bkd.asarray(np.zeros((4, 6))))

    def test_rejects_non_2d(self, torch_bkd: Backend) -> None:
        latent_map = MLPLatentMap(3, 2, [4], torch_bkd)
        with pytest.raises(ValueError, match="must be 2D"):
            latent_map(torch_bkd.asarray(np.zeros((3,))))


class TestAutogradLiveness:
    r"""A severed graph is the silent failure this guards.

    Under a backend whose ops are not torch ops the graph breaks at this
    object's output, and ``loss.backward()`` then updates nothing while
    raising nothing. These assert the graph exists *and* that gradients
    reach every parameter, since the first alone would pass on a network
    whose later layers were detached.
    """

    def test_output_tracks_gradient(self, torch_bkd: Backend) -> None:
        latent_map = MLPLatentMap(3, 2, [8], torch_bkd)
        codes = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
        assert torch_bkd.tracks_gradient(latent_map(codes))

    def test_every_parameter_receives_a_gradient(
        self, torch_bkd: Backend
    ) -> None:
        latent_map = MLPLatentMap(3, 2, [8, 8], torch_bkd)
        codes = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
        latent_map(codes).sum().backward()
        parameters = list(latent_map.parameters())
        assert len(parameters) == 6
        assert all(p.grad is not None for p in parameters)

    def test_gradient_wrt_input_matches_autograd_jacobian(
        self, torch_bkd: Backend
    ) -> None:
        """The value, not merely the presence, of the gradient.

        A live graph can still be wrong -- W0.7 was exactly that. For an
        affine map the Jacobian is the weight matrix, which is known in
        closed form, so this compares against it rather than against a
        tolerance.
        """
        latent_map = MLPLatentMap(3, 2, [], torch_bkd)
        weight = next(latent_map.parameters())
        codes = torch.zeros((3, 1), dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(
            lambda z: latent_map(z)[:, 0], codes
        )[:, :, 0]
        torch_bkd.assert_allclose(jacobian, weight, rtol=1e-12)


class TestBackendPolicy:
    def test_refuses_a_non_torch_backend(self) -> None:
        """Refusing beats converting, for an object built to be fitted.

        The velocity-field precedent accepts any backend and converts at
        its boundary, which suits something a numpy ODE stepper may
        evaluate. A latent map exists to be trained, and under NumpyBkd
        that training would silently do nothing.
        """
        with pytest.raises(TypeError, match="TorchBkd"):
            MLPLatentMap(3, 2, [4], NumpyBkd())


class TestInsideASurrogate:
    def test_composes_as_the_latent_map(self, torch_bkd: Backend) -> None:
        """The point of the whole exercise.

        An operator surrogate's forward path is encode, map, decode, and
        it does not care that the map is a network -- which is what makes
        POD-DeepONet a member of the existing family rather than a new
        class.
        """
        surrogate = OperatorSurrogate(
            IdentityFieldEncoder(3, torch_bkd),
            IdentityFieldEncoder(2, torch_bkd),
            MLPLatentMap(3, 2, [8], torch_bkd),
            torch_bkd,
        )
        codes = torch_bkd.asarray(np.random.uniform(-1.0, 1.0, (3, 7)))
        assert surrogate(codes).shape == (2, 7)

    def test_width_validation_still_applies(
        self, torch_bkd: Backend
    ) -> None:
        with pytest.raises(ValueError, match="nqoi"):
            OperatorSurrogate(
                IdentityFieldEncoder(3, torch_bkd),
                IdentityFieldEncoder(5, torch_bkd),
                MLPLatentMap(3, 2, [8], torch_bkd),
                torch_bkd,
            )

    def test_indices_refuses_a_network(self, torch_bkd: Backend) -> None:
        """No index set, so affineness is a question it cannot answer."""
        surrogate = OperatorSurrogate(
            IdentityFieldEncoder(3, torch_bkd),
            IdentityFieldEncoder(2, torch_bkd),
            MLPLatentMap(3, 2, [8], torch_bkd),
            torch_bkd,
        )
        with pytest.raises(TypeError, match="MultiIndexLatentMapProtocol"):
            surrogate.indices()


class TestRejectsBadConstruction:
    @pytest.mark.parametrize("ncodes_in,ncodes_out", [(0, 2), (3, 0)])
    def test_rejects_nonpositive_widths(
        self, torch_bkd: Backend, ncodes_in: int, ncodes_out: int
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            MLPLatentMap(ncodes_in, ncodes_out, [4], torch_bkd)

    def test_rejects_unknown_activation(self, torch_bkd: Backend) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            MLPLatentMap(3, 2, [4], torch_bkd, activation="sigmoid")

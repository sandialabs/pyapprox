"""Multiplicative-additive discrepancy model for MFNet nodes.

This model combines child node outputs with scaling functions and an
additive discrepancy:

    f(x, q) = sum_j scaling_j(x) * q_j + delta(x)

where x are the spatial variables and q are child outputs.
"""

from typing import Any, Generic, List

from pyapprox.surrogates.mfnets.protocols import (
    LinearNodeModelProtocol,
    NodeModelProtocol,
    NodeModelWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class MultiplicativeAdditiveDiscrepancy(Generic[Array]):
    """Discrepancy model: f(x, q) = sum_j scaling_j(x) * q_j + delta(x).

    The input to this model is an augmented vector ``[x; q]`` where ``x``
    are the first ``delta.nvars()`` rows (spatial variables) and ``q``
    are the remaining ``nscaled_qoi`` rows (child outputs).

    Parameters
    ----------
    scaling_models : list of NodeModelProtocol[Array]
        One scaling model per output QoI. Each ``scaling_models[ii]``
        outputs ``nscaled_qoi`` values that multiply the child outputs.
    delta_model : NodeModelProtocol[Array]
        Additive discrepancy model.
    nscaled_qoi : int
        Number of child output dimensions being scaled.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        scaling_models: List[NodeModelProtocol[Array]],
        delta_model: NodeModelProtocol[Array],
        nscaled_qoi: int,
        bkd: Backend[Array],
    ) -> None:
        if len(scaling_models) != delta_model.nqoi():
            raise ValueError(
                f"Need one scaling model per QoI. "
                f"Got {len(scaling_models)} scalings, "
                f"delta has {delta_model.nqoi()} QoI."
            )
        for sm in scaling_models:
            if sm.nqoi() != nscaled_qoi:
                raise ValueError(
                    f"Each scaling model must have nqoi={nscaled_qoi}, got {sm.nqoi()}"
                )
        self._bkd = bkd
        self._scalings = scaling_models
        self._delta = delta_model
        self._nscaled_qoi = nscaled_qoi

        # Build hyp_list
        all_hyps: List[Any] = []
        for sm in self._scalings:
            all_hyps.extend(sm.hyp_list().hyperparameters())
        all_hyps.extend(self._delta.hyp_list().hyperparameters())
        self._hyp_list = HyperParameterList(all_hyps, bkd)

        # Dynamically bind optional methods
        self._setup_optional_methods()

    def _setup_optional_methods(self) -> None:
        """Bind jacobian_wrt_params / basis_matrix if sub-models support it."""
        all_have_jac = isinstance(
            self._delta, NodeModelWithParamJacobianProtocol
        ) and all(
            isinstance(s, NodeModelWithParamJacobianProtocol) for s in self._scalings
        )
        if all_have_jac:
            self.jacobian_wrt_params = self._jacobian_wrt_params

        all_linear = isinstance(self._delta, LinearNodeModelProtocol) and all(
            isinstance(s, LinearNodeModelProtocol) for s in self._scalings
        )
        if all_linear:
            self.basis_matrix = self._basis_matrix
            self.get_coefficients = self._get_coefficients
            self.set_coefficients = self._set_coefficients

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        """delta.nvars() + nscaled_qoi."""
        return self._delta.nvars() + self._nscaled_qoi

    def nqoi(self) -> int:
        return self._delta.nqoi()

    def delta_model(self) -> NodeModelProtocol[Array]:
        return self._delta

    def scaling_models(self) -> List[NodeModelProtocol[Array]]:
        return self._scalings

    def nscaled_qoi(self) -> int:
        return self._nscaled_qoi

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def __call__(self, samples: Array) -> Array:
        """Evaluate: f(x, q) = sum_j scaling_j(x) * q_j + delta(x).

        Parameters
        ----------
        samples : Array
            Augmented input. Shape: ``(nvars, nsamples)`` where
            first ``delta.nvars()`` rows are spatial variables and
            remaining ``nscaled_qoi`` rows are child outputs.

        Returns
        -------
        Array
            Output values. Shape: ``(nqoi, nsamples)``
        """
        x = samples[: self._delta.nvars()]  # (delta_nvars, nsamples)
        q = samples[self._delta.nvars() :]  # (nscaled_qoi, nsamples)

        # delta(x): (nqoi, nsamples)
        values = self._delta(x)

        # For each QoI ii: values[ii] += sum_j scaling_ii_j(x) * q[j]
        for ii, scaling in enumerate(self._scalings):
            # scaling(x): (nscaled_qoi, nsamples) — one scale per child dim
            scale = scaling(x)  # (nscaled_qoi, nsamples)
            # sum_j scale[j] * q[j] over child dims
            values[ii : ii + 1] = values[ii : ii + 1] + self._bkd.sum(
                scale * q, axis=0, keepdims=True
            )

        return values

    def _sync_from_hyp_list(self) -> None:
        """Propagate hyp_list values to sub-models."""
        for sm in self._scalings:
            if hasattr(sm, "_sync_from_hyp_list"):
                sm._sync_from_hyp_list()
        if hasattr(self._delta, "_sync_from_hyp_list"):
            self._delta._sync_from_hyp_list()

    def _jacobian_wrt_params(self, samples: Array) -> Array:
        """Jacobian of output w.r.t. active parameters.

        Parameters
        ----------
        samples : Array
            Augmented input. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Shape: ``(nsamples, nqoi, nactive_params)``
        """
        x = samples[: self._delta.nvars()]
        q = samples[self._delta.nvars() :]
        nsamples = samples.shape[1]
        nqoi = self.nqoi()
        nactive = self._hyp_list.nactive_params()

        jac = self._bkd.zeros((nsamples, nqoi, nactive))

        # Track parameter offset within active params
        param_offset = 0

        # Scaling models jacobian contributions
        for ii, scaling in enumerate(self._scalings):
            # scaling.jacobian_wrt_params(x): (nsamples, nscaled_qoi, n_scale_active)
            scale_jac = scaling.jacobian_wrt_params(x)  # type: ignore[attr-defined]
            n_scale_active = scale_jac.shape[2]

            # d f_ii / d theta_scale_ii = sum_j q[j] * d scale_ii_j / d theta
            for jj in range(self._nscaled_qoi):
                # scale_jac[:, jj, :]: (nsamples, n_scale_active)
                # q[jj, :]: (nsamples,)
                jac[:, ii, param_offset : param_offset + n_scale_active] = (
                    jac[:, ii, param_offset : param_offset + n_scale_active]
                    + scale_jac[:, jj, :] * q[jj : jj + 1, :].T  # (nsamples, 1)
                )
            param_offset += n_scale_active

        # Delta model jacobian contribution
        # delta.jacobian_wrt_params(x): (nsamples, nqoi, n_delta_active)
        delta_jac = self._delta.jacobian_wrt_params(x)  # type: ignore[attr-defined]
        n_delta_active = delta_jac.shape[2]
        jac[:, :, param_offset : param_offset + n_delta_active] = delta_jac

        return jac

    def _basis_matrix(self, samples: Array) -> Array:
        """Augmented design matrix for ALS direct solve.

        For f(x,q) = sum_j scaling_j(x)*q_j + delta(x), the design matrix
        stacks: [Phi_scale_0(x)*q_0 | ... | Phi_scale_{K-1}(x)*q_{K-1} | Phi_delta(x)]

        But this is the matrix for a SINGLE output QoI. For nqoi > 1, each
        output QoI has its own set of columns.

        Parameters
        ----------
        samples : Array
            Augmented input. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Design matrix. Shape: ``(nsamples, total_nterms)``
            where total_nterms = nqoi * (sum_j nterms_scale_j + nterms_delta)
        """
        x = samples[: self._delta.nvars()]
        q = samples[self._delta.nvars() :]

        columns = []
        for ii, scaling in enumerate(self._scalings):
            # Phi_scaling(x): (nsamples, nterms_scale)
            phi_scale = scaling.basis_matrix(x)  # type: ignore[attr-defined]
            # Multiply each column by corresponding child output
            for jj in range(self._nscaled_qoi):
                # phi_scale * q[jj]: (nsamples, nterms_scale)
                columns.append(phi_scale * q[jj : jj + 1, :].T)

        # Delta basis: (nsamples, nterms_delta)
        phi_delta = self._delta.basis_matrix(x)  # type: ignore[attr-defined]
        columns.append(phi_delta)

        return self._bkd.hstack(columns)

    def _get_coefficients(self) -> Array:
        """Get flattened coefficients from all sub-models.

        Returns
        -------
        Array
            Concatenated coefficients from scalings then delta.
        """
        parts = []
        for sm in self._scalings:
            parts.append(self._bkd.flatten(sm.get_coefficients()))  # type: ignore[attr-defined]
        parts.append(self._bkd.flatten(self._delta.get_coefficients()))  # type: ignore[attr-defined]
        return self._bkd.hstack(parts)

    def _set_coefficients(self, coef: Array) -> None:
        """Set coefficients for all sub-models from a flat array.

        Parameters
        ----------
        coef : Array
            Flattened coefficient array.
        """
        offset = 0
        for sm in self._scalings:
            n = sm.hyp_list().nparams()
            sm_coef = coef[offset : offset + n]
            sm_nterms = sm_coef.shape[0] // self._nscaled_qoi
            sm.set_coefficients(  # type: ignore[attr-defined]
                self._bkd.reshape(sm_coef, (sm_nterms, self._nscaled_qoi))
            )
            offset += n
        n_delta = self._delta.hyp_list().nparams()
        delta_coef = coef[offset : offset + n_delta]
        delta_nterms = delta_coef.shape[0] // self.nqoi()
        self._delta.set_coefficients(  # type: ignore[attr-defined]
            self._bkd.reshape(delta_coef, (delta_nterms, self.nqoi()))
        )

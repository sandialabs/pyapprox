import itertools

from pyapprox.surrogates.bases.basis import Basis, QuadratureRule
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.bases.orthopoly import (
    GaussQuadratureRule,
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.bases.basis import (
    OrthonormalPolynomialBasis,
    FixedTensorProductQuadratureRule,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.kle.kle import DataDrivenKLE
from pyapprox.variables.marginals import get_distribution_info


class MultiLinearOperatorBasis:
    def __init__(
        self,
        nin_functions,
        nout_functions,
        in_bases,
        out_bases,
        in_quad_rules,
        out_quad_rules,
    ):
        self._bkd = in_bases[0]._bkd
        self._nin_functions = nin_functions
        self._nout_functions = nout_functions
        self._in_quad_rules = in_quad_rules
        self._out_quad_rules = out_quad_rules
        self.set_bases(in_bases, out_bases)

        self._jacobian_implemented = False
        self._hessian_implemented = False

    def set_bases(self, in_bases, out_bases):
        self._check_bases(self._nin_functions, in_bases, self._in_quad_rules)
        self._check_bases(
            self._nout_functions, out_bases, self._out_quad_rules
        )
        self._in_bases = in_bases
        self._out_bases = out_bases

    def _check_quadrature_rules(self, quad_rules, nfunctions):
        if len(quad_rules) != nfunctions:
            raise ValueError(
                "A quadrature rule must be specified for each function"
            )
        for quad_rule in quad_rules:
            if not isinstance(quad_rule, QuadratureRule):
                ValueError("quad_rule must be an instance of QuadratureRule")
            if not self._bkd.bkd_equal(self._bkd, quad_rule._bkd):
                raise ValueError("backends are not consistent")

    def _check_bases(self, nfunctions, bases, quad_rules):
        if len(bases) != nfunctions:
            raise ValueError(
                "A basis must be specified for each input function"
            )
        for basis in bases:
            if not isinstance(basis, Basis):
                ValueError("basis must be an instance of Basis")
            if not self._bkd.bkd_equal(self._bkd, basis._bkd):
                raise ValueError("backends are not consistent")
        self._check_quadrature_rules(quad_rules, nfunctions)

    def _check_coef_basis(self, coef_basis):
        if not self._bkd.bkd_equal(self._bkd, coef_basis._bkd):
            raise ValueError("backends are not consistent")
        if not isinstance(coef_basis, Basis):
            raise ValueError("coef_basis must be an instance of Basis")

    def nin_functions(self):
        return self._nin_functions

    def nout_functions(self):
        return self._nout_functions

    def nterms(self):
        if not hasattr(self, "_coef_basis"):
            return 0
        return self._coef_basis.nterms() * self._bkd.prod(
            self._bkd.array(
                [basis.nterms() for basis in self._out_bases], dtype=int
            )
        )

    def set_coefficient_basis(self, coef_basis):
        self._check_coef_basis(coef_basis)
        self._coef_basis = coef_basis

    def _coef_from_fun_values(self, fun_values, bases, quad_rules):
        coefs = []
        if len(fun_values) != len(bases):
            raise ValueError("Values must be specified for each function")
        for ii in range(len(bases)):
            # the functions must be evaluated on the same number of
            # input function samples
            if (
                fun_values[ii].ndim != 2
                and fun_values[ii].shape[1] == fun_values[0].shape[1]
            ):
                raise ValueError("Function values must be a 2D array")
            quad_samples, quad_weights = quad_rules[ii]()
            basis_mat = bases[ii](quad_samples)
            coefs.append(
                self._bkd.einsum(
                    "ijk,j->ik",
                    fun_values[ii].T[..., None] * basis_mat,
                    quad_weights[:, 0],
                ).T
            )
        return self._bkd.vstack(coefs)

    def _in_coef_from_infun_values(self, infun_values):
        return self._coef_from_fun_values(
            infun_values, self._in_bases, self._in_quad_rules
        )

    def _out_coef_from_outfun_values(self, out_fun_values):
        return self._coef_from_fun_values(
            out_fun_values, self._out_bases, self._out_quad_rules
        )

    def _basis_values_from_in_coef(self, in_coef, out_samples):
        # coef_basis_mat (nin_fun_samples, ncoef_terms)
        coef_basis_mat = self._coef_basis(in_coef)
        nin_samples = in_coef.shape[1]
        # loop over output functions one at a time
        basis_mats = []
        for ii in range(self.nout_functions()):
            # outerproduct of inner and outer basis functions
            # out_basis_mat (nout_samples, nout_terms_i)
            out_basis_mat = self._out_bases[ii](out_samples[ii])
            # basis_mat (nout_samples, nin_fun_samples, nout_terms, ncoef_terms)
            nout_samples = out_basis_mat.shape[0]
            nbasis_terms = (
                self._out_bases[ii].nterms() * self._coef_basis.nterms()
            )
            # basis_mat1 = self._bkd.einsum(
            #     "ij,kl->ikjl", out_basis_mat, coef_basis_mat
            # )
            # # basis_mat (nout_samples, ninsamples, nout_terms*ncoef_terms)
            # basis_mat1 = self._bkd.reshape(
            #     basis_mat1, (nout_samples, nin_samples, nbasis_terms)
            # )
            basis_mat = self._bkd.zeros(
                (nout_samples, nin_samples, nbasis_terms)
            )
            cnt = 0
            # eisnum above can runs out of memory, so loop over outer basis
            for jj in range(out_basis_mat.shape[1]):
                basis_mat[..., cnt : cnt + self._coef_basis.nterms()] = (
                    self._bkd.einsum(
                        "i,kl->ikl", out_basis_mat[:, jj], coef_basis_mat
                    )
                )
                cnt += self._coef_basis.nterms()
            # assert self._bkd.allclose(basis_mat, basis_mat1)
            basis_mats.append(basis_mat)
        return basis_mats

    def __call__(self, infun_values, out_samples):
        """
        Parameters
        ----------
        infun_values : List [array (nquad_samples_i, nfun_samples_i)]
            The input function values at the qudrature points

        out_samples : List [array (nvars_i, nsamples_i)]
            The samples at which to query each output function
        """
        if self._coef_basis is None:
            raise ValueError("must call set_coefficient_basis")
        if len(out_samples) != self.nout_functions():
            raise ValueError(
                "samples must be specified for each output function"
            )
        in_coef = self._in_coef_from_infun_values(infun_values)
        return self._basis_values_from_in_coef(in_coef, out_samples)

    def ncoefficients_per_input_function(self):
        return [basis.nterms() for basis in self._in_bases]

    def set_coefficient_basis_indices(self, indices):
        self._coef_basis.set_indices(indices)


class TensorOrthoPolyMultiLinearOperatorBasis(MultiLinearOperatorBasis):
    def __init__(
        self,
        marginals_per_infun,
        marginals_per_outfun,
        nterms_1d_per_infun,
        nterms_1d_per_outfun,
        in_quad_rules=None,
        out_quad_rules=None,
        backend=None,
    ):
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend
        self._marginals_per_infun = marginals_per_infun
        self._marginals_per_outfun = marginals_per_outfun
        self._nterms_1d_per_infun = nterms_1d_per_infun
        self._nterms_1d_per_outfun = nterms_1d_per_outfun
        in_bases, out_bases = self.setup_bases(
            self._nterms_1d_per_infun, self._nterms_1d_per_outfun
        )
        if in_quad_rules is None:
            in_quad_rules = self.setup_function_domain_quadrature_rules(
                marginals_per_infun, nterms_1d_per_infun, self._bkd
            )
        if out_quad_rules is None:
            out_quad_rules = self.setup_function_domain_quadrature_rules(
                marginals_per_outfun, nterms_1d_per_outfun, self._bkd
            )
        super().__init__(
            len(marginals_per_infun),
            len(marginals_per_outfun),
            in_bases,
            out_bases,
            in_quad_rules,
            out_quad_rules,
        )

    def setup_bases(
            self, nterms_1d_per_infun, nterms_1d_per_outfun):
        in_bases = self._setup_function_domain_bases(
            self._marginals_per_infun, nterms_1d_per_infun
        )
        out_bases = self._setup_function_domain_bases(
            self._marginals_per_outfun, nterms_1d_per_outfun
        )
        return in_bases, out_bases

    def _setup_ortho_basis(self, marginals, nterms_1d=None):
        polys_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=self._bkd
            )
            for marginal in marginals
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        if nterms_1d is not None:
            basis.set_tensor_product_indices(nterms_1d)
        return basis

    def _setup_function_domain_bases(
        self, marginals_per_fun, nterms_1d_per_fun
    ):
        return [
            self._setup_ortho_basis(marginals, nterms_1d)
            for marginals, nterms_1d in zip(
                marginals_per_fun, nterms_1d_per_fun
            )
        ]

    @staticmethod
    def setup_function_domain_quadrature_rules(
        marginals_per_fun, nterms_1d_per_fun, bkd
    ):
        nfuns = len(marginals_per_fun)
        quad_rules = [
            FixedTensorProductQuadratureRule(
                len(marginals_per_fun[ii]),
                [
                    GaussQuadratureRule(marginal)
                    for marginal in marginals_per_fun[ii]
                ],
                2 * bkd.array(nterms_1d_per_fun[ii], dtype=int),
            )
            for ii in range(nfuns)
        ]
        return quad_rules

    def set_coefficient_basis(self, coef_marginals_per_infun):
        coef_basis = self._setup_ortho_basis(
            itertools.chain(*coef_marginals_per_infun)
        )
        super().set_coefficient_basis(coef_basis)


class MultiLinearOperatorExpansion(BasisExpansion):
    def _parse_basis(self, basis):
        if not isinstance(basis, MultiLinearOperatorBasis):
            raise ValueError("basis must be a MultiLinearOperatorBasis")
        return basis

    def _set_training_data(self, train_samples, train_values):
        self._ctrain_samples = train_samples
        self._ctrain_values = train_values

    def _fit(self, iterate):
        """Fit the expansion by finding the optimal coefficients."""
        if iterate is not None:
            raise ValueError("iterate will be ignored set to None")
        out_fun_coefs = self.basis._out_coef_from_outfun_values(
            self._ctrain_values
        )
        infun_coefs = self.basis._in_coef_from_infun_values(
            self._ctrain_samples
        )
        coef_mat = self.basis._coef_basis(infun_coefs)
        ntrain_samples = out_fun_coefs.shape[1]
        # TODO add weights to grammian and rhs construction
        grammian = coef_mat.T @ coef_mat / ntrain_samples
        # rhs1 = self._bkd.sum(
        #     self._bkd.einsum(
        #         "ij,jk->jki", out_fun_coefs, coef_mat
        #     ),
        #     axis=0
        # )/ntrain_samples
        rhs = (
            self._bkd.einsum("ij,jk->ki", out_fun_coefs, coef_mat)
            / ntrain_samples
        )
        # assert self._bkd.allclose(rhs, rhs1)
        # print("COND NO", self._bkd.cond(grammian), "NO. COEFS", grammian.shape[0])
        coef = self._solver.solve(grammian, rhs).T.flatten()[:, None]
        self.set_coefficients(coef)

    def _values_mat_vec(self, infun_values, out_samples):
        # traditional method of computing dot product of basis functions
        # and coefficients
        basis_mat = self.basis(infun_values, out_samples)
        # for now assume only one output function
        assert len(out_samples) == 1
        basis_mat = basis_mat[0]
        return [basis_mat @ self.get_coefficients()[..., 0]]

    def _values_custom(self, infun_values, out_samples):
        # Compuationally efficient method that takes adavantage of structure
        # of basis and coefficients

        # compute \psi(y),     \xi(x)      and c with sizes
        #         (Ny, P_0), (M_x, P_1)    (P_1P_0, 1)
        # The compute
        #      t = \xi(x) @ reshape(c, (P_x, P_0), order="F")
        # fortran reshape is needed because c contains the coeficients
        # of \psi_0 for all \xi then of psi_1 for all \xi and so on
        # finally compute \psi(y) @ t
        if self.basis.nout_functions() != 1:
            raise ValueError("Currently only supports 1 output function")
        ii = 0  # hack to just one funciton for now
        in_coef = self.basis._in_coef_from_infun_values(infun_values)
        coef_basis_mat = self.basis._coef_basis(in_coef)
        out_basis_mat = self.basis._out_bases[ii](out_samples[ii])
        exp_coefs = self.get_coefficients()
        # taking transpose of the reshaped matrix belwo is equivalent to
        # reorder with order="F", but the former does not require a copy
        tmp1 = (
            coef_basis_mat
            @ self._bkd.reshape(
                exp_coefs, (out_basis_mat.shape[1], coef_basis_mat.shape[1])
            ).T
        )
        vals = out_basis_mat @ tmp1.T
        return [vals]

    def __call__(self, infun_values, out_samples):
        # return self._values_mat_vec(infun_values, out_samples)
        return self._values_custom(infun_values, out_samples)


class PCABasis(Basis):
    def __init__(self, quad_rule, basis):
        super().__init__(basis._bkd)
        if not isinstance(basis, Basis):
            print(basis)
            raise ValueError("basis must be an instance of Basis")
        self._quad_rule = quad_rule
        self._basis = basis

    def set_fun_values_at_quadx(self, fun_values_at_quadx):
        self._fun_values_at_quadx = fun_values_at_quadx

    def set_nterms(self, nterms):
        if not hasattr(self, "_fun_values_at_quadx"):
            raise RuntimeError("must first call set_fun_values_at_quadx")
        self._nterms = nterms
        kle = DataDrivenKLE(
            self._fun_values_at_quadx,
            nterms=nterms,
            quad_weights=self._quad_rule()[1][:, 0],
        )
        basis_mat = self._basis(self._quad_rule()[0])
        basis_coef = self._bkd.einsum(
            "ij,ik->jk", self._quad_rule()[1] * basis_mat, kle._eig_vecs
        )
        self._bexp = BasisExpansion(self._basis, nqoi=self._nterms)
        self._bexp.set_coefficients(basis_coef)
        self._sqrt_eig_vals = kle._sqrt_eig_vals

    def nterms(self):
        if not hasattr(self, "_nterms"):
            raise RuntimeError("Must call set_nterms")
        return self._nterms

    def nvars(self):
        return self._bexp.nvars()

    def __call__(self, samples):
        if not hasattr(self, "_bexp"):
            raise RuntimeError("Must call set_nterms")
        return self._bexp(samples)[:, : self._nterms]


class PCAMultiLinearOperatorBasis(MultiLinearOperatorBasis):
    """
    Use PCA to construct orthonormal bases for the input and output functions.
    Assume that functions are defined on tensor-product domains and the
    same orthonormal polynomial can be used in each dimension of the function
    domain to approximate the PCA modes.
    """

    def __init__(
        self,
        in_representing_bases,
        out_representing_bases,
        in_quad_rules,
        out_quad_rules,
    ):
        """
        in_representing_bases : list(Basis)
            Bases used to represent the PCA modes of the input functions

        out_representing_bases : list(Basis)
            Bases used to represent the PCA modes of the output functions
        """
        self._bkd = in_quad_rules[0]._bkd
        in_bases = self._setup_pca_bases(
            in_quad_rules,
            in_representing_bases,
        )
        out_bases = self._setup_pca_bases(
            out_quad_rules,
            out_representing_bases,
        )
        super().__init__(
            len(in_quad_rules),
            len(out_quad_rules),
            in_bases,
            out_bases,
            in_quad_rules,
            out_quad_rules,
        )

    def _setup_pca_bases(self, quad_rules, bases):
        return [
            PCABasis(quad_rules[ii], bases[ii])
            for ii in range(len(quad_rules))
        ]

    def set_coefficient_basis(self, coef_marginals_per_infun):
        coef_basis = self._setup_ortho_basis(
            itertools.chain(*coef_marginals_per_infun)
        )
        super().set_coefficient_basis(coef_basis)


class PCATensorOrthoPolyMultiLinearOperatorBasis(PCAMultiLinearOperatorBasis):
    """
    Use PCA to construct orthonormal bases for the input and output functions.
    Assume that functions are defined on tensor-product domains and the
    same orthonormal polynomial can be used in each dimension of the function
    domain to approximate the PCA modes.
    """

    def __init__(
        self,
        marginals_per_infun,
        marginals_per_outfun,
        nterms_1d_per_infun,
        nterms_1d_per_outfun,
        in_quad_rules,
        out_quad_rules,
    ):
        self._bkd = in_quad_rules[0]._bkd
        inrep_bases = self._setup_function_domain_bases(
            marginals_per_infun, nterms_1d_per_infun
        )
        outrep_bases = self._setup_function_domain_bases(
            marginals_per_outfun, nterms_1d_per_outfun
        )
        super().__init__(
            inrep_bases,
            outrep_bases,
            in_quad_rules,
            out_quad_rules,
        )

    def _setup_ortho_basis(self, marginals, nterms_1d=None):
        polys_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=self._bkd
            )
            for marginal in marginals
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        if nterms_1d is not None:
            basis.set_tensor_product_indices(nterms_1d)
        return basis

    def _setup_function_domain_bases(
        self, marginals_per_fun, nterms_1d_per_fun
    ):
        return [
            self._setup_ortho_basis(marginals, nterms_1d)
            for marginals, nterms_1d in zip(
                marginals_per_fun, nterms_1d_per_fun
            )
        ]

    def compute_principal_components(
        self,
        infun_vals_at_quadx,
        outfun_vals_at_quadx,
        nin_pca_modes,
        nout_pca_modes,
        coef_base_marginals,
    ):
        nin_pca_modes = self._bkd.atleast1d(nin_pca_modes, dtype=int)
        nout_pca_modes = self._bkd.atleast1d(nout_pca_modes, dtype=int)
        if nin_pca_modes.shape[0] != self.nin_functions():
            raise ValueError(
                "must specify the number of PCA modes for each input function"
            )
        if nout_pca_modes.shape[0] != self.nout_functions():
            raise ValueError(
                "must specify the number of PCA modes for each output function"
            )

        for ii in range(self.nin_functions()):
            self._in_bases[ii].set_fun_values_at_quadx(infun_vals_at_quadx[ii])
            self._in_bases[ii].set_nterms(nin_pca_modes[ii])
        for ii in range(self.nout_functions()):
            self._out_bases[ii].set_fun_values_at_quadx(
                outfun_vals_at_quadx[ii]
            )
            self._out_bases[ii].set_nterms(nout_pca_modes[ii])
        self._setup_coefficient_basis(coef_base_marginals)

    def _setup_coefficient_basis(self, coef_base_marginals):
        ncoefs_per_infun = self.ncoefficients_per_input_function()
        if not isinstance(coef_base_marginals, list):
            raise ValueError(
                "must specifiy coef_base_marginals for each input function"
            )

        coef_marginals = []
        for ii, ncoefs in enumerate(ncoefs_per_infun):
            var_name, scales, shapes = get_distribution_info(
                coef_base_marginals[ii]
            )
            coef_marginals.append([])
            # scale disribution by sqrt of eigenvalues
            for jj in range(ncoefs):
                marginal = coef_base_marginals[ii].dist(
                    *shapes,
                    loc=scales["loc"],
                    scale=scales["scale"]
                    * self._in_bases[ii]._sqrt_eig_vals[jj],
                )
                coef_marginals[-1].append(marginal)
        self.set_coefficient_basis(coef_marginals)

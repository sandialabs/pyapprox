import itertools

from pyapprox.surrogates.bases.basis import Basis, QuadratureRule
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.bases.orthopoly import (
    GaussQuadratureRule,
    setup_univariate_orthogonal_polynomial_from_marginal
)
from pyapprox.surrogates.bases.basis import (
    OrthonormalPolynomialBasis,
    FixedTensorProductQuadratureRule,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.kle.kle import DataDrivenKLE


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
        self._check_bases(nin_functions, in_bases, in_quad_rules)
        self._check_bases(nout_functions, out_bases, out_quad_rules)
        self._nin_functions = nin_functions
        self._in_bases = in_bases
        self._in_quad_rules = in_quad_rules
        self._nout_functions = nout_functions
        self._out_bases = out_bases
        self._out_quad_rules = out_quad_rules
        self._jacobian_implemented = False
        self._hessian_implemented = False
        self._coef_basis = None

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

    def _in_coef_from_in_fun_values(self, in_fun_values):
        return self._coef_from_fun_values(
            in_fun_values, self._in_bases, self._in_quad_rules
        )

    def _out_coef_from_out_fun_values(self, out_fun_values):
        return self._coef_from_fun_values(
            out_fun_values, self._out_bases, self._out_quad_rules)

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
                basis_mat[..., cnt:cnt+self._coef_basis.nterms()] = (
                    self._bkd.einsum(
                        "i,kl->ikl", out_basis_mat[:, jj], coef_basis_mat
                    )
                )
                cnt += self._coef_basis.nterms()
            # assert self._bkd.allclose(basis_mat, basis_mat1)
            basis_mats.append(basis_mat)
        return basis_mats

    def __call__(self, in_fun_values, out_samples):
        """
        Parameters
        ----------
        in_fun_values : List [array (nquad_samples_i, nfun_samples_i)]
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
        in_coef = self._in_coef_from_in_fun_values(in_fun_values)
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
        nin_functions = len(marginals_per_infun)
        nout_functions = len(marginals_per_outfun)
        in_bases = self._setup_function_domain_bases(
            marginals_per_infun, nterms_1d_per_infun
        )
        out_bases = self._setup_function_domain_bases(
            marginals_per_outfun, nterms_1d_per_outfun
        )
        if in_quad_rules is None:
            in_quad_rules = self._setup_function_domain_quadrature_rules(
                marginals_per_infun, nterms_1d_per_infun)
        if out_quad_rules is None:
            out_quad_rules = self._setup_function_domain_quadrature_rules(
                marginals_per_outfun, nterms_1d_per_outfun
            )
        super().__init__(
            nin_functions,
            nout_functions,
            in_bases,
            out_bases,
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

    def _setup_function_domain_quadrature_rules(
            self, marginals_per_fun, nterms_1d_per_fun):
        nfuns = len(marginals_per_fun)
        quad_rules = [
            FixedTensorProductQuadratureRule(
                len(marginals_per_fun[ii]),
                [
                    GaussQuadratureRule(marginal)
                    for marginal in marginals_per_fun[ii]
                ],
                2*self._bkd.array(nterms_1d_per_fun[ii], dtype=int),
            )
            for ii in range(nfuns)
        ]
        return quad_rules

    def set_basis(self, coef_marginals_per_infun):
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
        """Fit the expansion by finding the optimal coefficients. """
        if iterate is not None:
            raise ValueError("iterate will be ignored set to None")
        out_fun_coefs = self.basis._out_coef_from_out_fun_values(
            self._ctrain_values
        )
        in_fun_coefs = self.basis._in_coef_from_in_fun_values(
            self._ctrain_samples
        )
        coef_mat = self.basis._coef_basis(in_fun_coefs)
        ntrain_samples = out_fun_coefs.shape[1]
        # TODO add weights to grammian and rhs construction
        grammian = coef_mat.T @ coef_mat / ntrain_samples
        # rhs1 = self._bkd.sum(
        #     self._bkd.einsum(
        #         "ij,jk->jki", out_fun_coefs, coef_mat
        #     ),
        #     axis=0
        # )/ntrain_samples
        rhs = self._bkd.einsum(
            "ij,jk->ki", out_fun_coefs, coef_mat
        )/ntrain_samples
        # assert self._bkd.allclose(rhs, rhs1)
        print("COND NO", self._bkd.cond(grammian), "NO. COEFS", grammian.shape[0])
        coef = self._solver.solve(grammian, rhs).T.flatten()[:, None]
        self.set_coefficients(coef)

    def __call__(self, in_fun_values, out_samples):
        basis_mat = self.basis(in_fun_values, out_samples)
        # for now assume only one output function
        assert len(out_samples) == 1
        basis_mat = basis_mat[0]
        return [basis_mat @ self.get_coefficients()[..., 0]]


class PCABasisFromExpansion(Basis):
    def __init__(self, basis_exp: BasisExpansion, sqrt_eig_vals, nterms: int):
        super().__init__(basis_exp._bkd)
        if not isinstance(basis_exp, BasisExpansion):
            raise ValueError("basis_exp must be an instance of BasisExpansion")
        self._bexp = basis_exp
        self._sqrt_eig_vals = sqrt_eig_vals
        self.set_nterms(nterms)

    def set_nterms(self, nterms):
        if nterms > self._bexp.nterms():
            raise ValueError("nterms requrested greater than allowable")
        self._nterms = nterms

    def nterms(self):
        return self._nterms

    def nvars(self):
        return self._bexp.nvars()

    def __call__(self, samples):
        return self._bexp(samples)[:, :self._nterms]


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
            infun_values_at_quadx,
            outfun_values_at_quadx,
            nin_pca_modes_max,
            nout_pca_modes_max,
            nin_pca_modes,
            nout_pca_modes,
    ):
        """
        in_representing_bases : list(Basis)
            Bases used to represent the PCA modes of the input functions

        out_representing_bases : list(Basis)
            Bases used to represent the PCA modes of the output functions

        nin_pca_modes_max : list(int)
            The max number of PCA modes to represent each input function.
            Less terms can be used by calling set_nin_pca_modes

        nout_pca_modes_max : list(int)
            The max number of PCA modes to represent each output function.
            Less terms can be used by calling set_nout_pca_modes

        nin_pca_modes : list(int)
            The number of PCA modes to represent each input function.

        nout_pca_modes : list(int)
            The number of PCA modes to represent each output function.
        """
        self._bkd = in_quad_rules[0]._bkd
        nin_pca_modes_max = self._bkd.atleast1d(nin_pca_modes_max, dtype=int)
        nout_pca_modes_max = self._bkd.atleast1d(nout_pca_modes_max, dtype=int)
        if nin_pca_modes_max.shape[0] != len(in_representing_bases):
            raise ValueError(
                "must specify the number of PCA modes for each input function"
            )
        if nout_pca_modes_max.shape[0] != len(out_representing_bases):
            raise ValueError(
                "must specify the number of PCA modes for each output function"
            )
        nin_pca_modes = self._bkd.atleast1d(nin_pca_modes, dtype=int)
        nout_pca_modes = self._bkd.atleast1d(nout_pca_modes, dtype=int)
        if nin_pca_modes.shape[0] != len(in_representing_bases):
            raise ValueError(
                "must specify the number of PCA modes for each input function"
            )
        if nout_pca_modes.shape[0] != len(out_representing_bases):
            raise ValueError(
                "must specify the number of PCA modes for each output function"
            )

        in_bases = self._get_pca_bases(
            infun_values_at_quadx,
            nin_pca_modes_max,
            nin_pca_modes,
            in_quad_rules,
            in_representing_bases
        )
        out_bases = self._get_pca_bases(
            outfun_values_at_quadx,
            nout_pca_modes_max,
            nout_pca_modes,
            out_quad_rules,
            out_representing_bases
        )
        super().__init__(
            len(in_quad_rules),
            len(out_quad_rules),
            in_bases,
            out_bases,
            in_quad_rules,
            out_quad_rules,
        )

    def _get_pca_basis(
            self, fun_values_at_quadx, npca_modes_max, npca_modes, quad_rule, basis
    ):
        kle = DataDrivenKLE(
            fun_values_at_quadx,
            nterms=npca_modes_max,
            quad_weights=quad_rule()[1][:, 0]
        )
        basis_mat = basis(quad_rule()[0])
        basis_coef = self._bkd.einsum(
            "ij,ik->jk", quad_rule()[1]*basis_mat, kle._eig_vecs
        )
        bexp = BasisExpansion(basis, nqoi=npca_modes_max)
        bexp.set_coefficients(basis_coef)
        eigvecs = PCABasisFromExpansion(bexp, kle._sqrt_eig_vals, npca_modes)
        return eigvecs

    def _get_pca_bases(
            self, fun_values_at_quadx, npca_modes_max_per_fun, npca_modes_per_fun,  quad_rules, bases
    ):
        return [
            self._get_pca_basis(
                fun_values_at_quadx[ii],
                npca_modes_max_per_fun[ii],
                npca_modes_per_fun[ii],
                quad_rules[ii],
                bases[ii],
            )
            for ii in range(len(fun_values_at_quadx))
        ]

    def set_basis(self, coef_marginals_per_infun):
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
            infun_values_at_quadx,
            outfun_values_at_quadx,
            nin_pca_modes_max,
            nout_pca_modes_max,
            nin_pca_modes,
            nout_pca_modes,
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
            infun_values_at_quadx,
            outfun_values_at_quadx,
            nin_pca_modes_max,
            nout_pca_modes_max,
            nin_pca_modes,
            nout_pca_modes,
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

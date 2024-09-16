from pyapprox.surrogates.bases.basis import (
    Basis, QuadratureRule, MultiIndexBasis
)
from pyapprox.surrogates.bases.basisexp import BasisExpansion


class MultiLinearOperatorBasis:
    def __init__(
        self,
        nin_functions,
        in_bases,
        in_quad_rules,
        nout_functions,
        out_basis_exps,
        out_quad_rules,
        coef_basis,
    ):
        self._bkd = in_bases[0]._bkd
        self._check_input_bases(nin_functions, in_bases, in_quad_rules)
        self._check_output_basis_epansions(
            nout_functions, out_basis_exps, out_quad_rules
        )
        self._check_coef_basis(coef_basis)
        self._nin_functions = nin_functions
        self._in_bases = in_bases
        self._in_quad_rules = in_quad_rules
        self._nout_functions = nout_functions
        self._out_basis_exps = out_basis_exps
        self._out_quad_rules = out_quad_rules
        self._coef_basis = coef_basis

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

    def _check_input_bases(self, nin_functions, in_bases, in_quad_rules):
        if len(in_bases) != nin_functions:
            raise ValueError(
                "A basis must be specified for each input function"
            )
        for basis in in_bases:
            if not isinstance(basis, Basis):
                ValueError("basis must be an instance of Basis")
            if not self._bkd.bkd_equal(self._bkd, basis._bkd):
                raise ValueError("backends are not consistent")
        self._check_quadrature_rules(in_quad_rules, nin_functions)

    def _check_output_basis_epansions(
        self, nout_functions, out_basis_exps, out_quad_rules
    ):
        if len(out_basis_exps) != nout_functions:
            raise ValueError(
                "One basis expansion must be specified for each output "
                "function"
            )
        for bexp in out_basis_exps:
            if not isinstance(bexp, BasisExpansion):
                ValueError(
                    "basis expansion must be an instance of BasisExpansion"
                )
            if not self._bkd.bkd_equal(self._bkd, bexp._bkd):
                raise ValueError("backends are not consistent")
        self._check_quadrature_rules(out_quad_rules, nout_functions)

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
                [bexp.basis.nterms() for bexp in self._out_basis_exps]
            )
        )

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
            out_basis_mat = self._out_basis_exps[ii].basis(out_samples[ii])
            # basis_mat (nout_samples, nin_fun_samples, nout_terms,ncoef_terms)
            basis_mat = self._bkd.einsum(
                "ij,kl->ikjl", out_basis_mat, coef_basis_mat
            )
            # basis_mat (nout_samples, ninsamples, nout_terms*ncoef_terms)
            nout_samples = out_basis_mat.shape[0]
            nbasis_terms = (
                self._out_basis_exps[ii].nterms() * self._coef_basis.nterms()
            )
            basis_mat = self._bkd.reshape(
                basis_mat, (nout_samples, nin_samples, nbasis_terms)
            )
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
        if len(out_samples) != self.nout_functions():
            raise ValueError(
                "samples must be specified for each output function"
            )
        in_coef = self._in_coef_from_in_fun_values(in_fun_values)
        print(in_coef.shape)
        return self._basis_values_from_in_coef(in_coef, out_samples)

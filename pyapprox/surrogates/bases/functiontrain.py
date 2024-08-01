import copy

# import numpy for np.nan
import numpy as np 

from pyapprox.surrogates.bases.basis import MonomialBasis
from pyapprox.surrogates.bases.basisexp import Regressor, BasisExpansion, MonomialExpansion


class FunctionTrainCore:
    def __init__(self, basisexps):
        self._bkd = basisexps[0][0]._bkd
        self._ranks = (len(basisexps), len(basisexps[0]))
        self._basisexps = basisexps
        self.hyp_list = sum(
            [bexp.hyp_list for bexps in self._basisexps for bexp in bexps]
        )
        
    def __call__(self, samples):
        values = []
        for ii in range(self._ranks[0]):
            values.append([])
            for jj in range(self._ranks[1]):
                values[-1].append(self._basisexps[ii][jj](samples))
            values[-1] = self._bkd._la_stack(values[-1], axis=0)
        return self._bkd._la_stack(values, axis=0)

    def __repr__(self):
        return "{0}(ranks={1})".format(self.__class__.__name__, self._ranks)


class HomogeneousFunctionTrainCore(FunctionTrainCore):
    def __init__(self, basisexp, ranks):
        super().__init__(
            [
                [copy.deepcopy(basisexp) for jj in range(ranks[1])]
                for ii in range(ranks[0])
            ]
        )


class FunctionTrain(Regressor):
    def __init__(self, cores, nqoi=1):
        if cores[0]._ranks[0] != 1:
            raise ValueError(
                "First rank of first core must be 1 but was {0}".format(
                    cores[0]._ranks[0]))
        if cores[-1]._ranks[1] != 1:
            raise ValueError(
                "Second rank of last core must be 1 but was {0}".format(
                    cores[-1]._ranks[1]))
        for ii in range(len(cores)-1):
            if cores[ii]._ranks[-1] != cores[ii+1]._ranks[0]:
                raise ValueError("The inner core ranks do not match")
        self._cores = cores
        self.hyp_list = sum([core.hyp_list for core in self._cores])
        self._bkd = self.hyp_list._bkd
        self._nqoi = nqoi
        self._samples = None

    def nvars(self):
        return len(self._cores)

    def nqoi(self):
        return self._nqoi

    def _core_params_eval(self, active_opt_params):
        self.hyp_list.set_active_opt_params(active_opt_params)
        return self(self._samples)

    def _core_jacobian_ad(self, samples, core_id):
        """Compute Jacobian with automatic differentiation."""
        self._samples = samples
        hyplist_bounds = self._bkd._la_copy(self.hyp_list.get_bounds())
        for ii in range(self.nvars()):
            if ii != core_id:
                self._cores[ii].hyp_list.set_bounds([np.nan, np.nan])
        jac = self._bkd._la_jacobian(
            self._core_params_eval, self.hyp_list.get_active_opt_params()
        )
        self.hyp_list.set_bounds(hyplist_bounds)
        # create list of jacobians for each qoi
        # jac will be of shape (nsamples, nqoi, ntotal_core_active_parameters)
        # so if using basis expansion with same expansion for each core and nqoi = 1
        # jac[:, 0, :] will be basis matrix B=basexp.basis_matrix(samples)
        # If nqoi = 2, jac[:, 0, :] will be [B[:, 0], 0, B[:, 1], 0, ..., B[:, P], 0]
        # where P=basisexp.nterms()
        # to get grad with respect to just coefficients of the qth qoi
        # and qoi = q use jac[:, q, q::nqoi]
        jac = [jac[:, qq, qq::self.nqoi()] for qq in range(self.nqoi())]
        return jac

    def _eval_left_cores(self, core_id, samples):
        if core_id < 1 or core_id >= self.nvars():
            raise ValueError(
                "Ensure 0 < core_id < nvars. core_id={0}".format(core_id))
        values = self._cores[0](samples[:1])
        for ii in range(1, core_id):
            c = self._cores[ii](samples[ii:ii+1])
            values = self._bkd._la_einsum(
                "ijkl, jmkl->imkl", values, self._cores[ii](samples[ii:ii+1]))
        return values

    def _eval_right_cores(self, core_id, samples):
        if core_id >= self.nvars()-1 or core_id < 0:
            raise ValueError("Ensure 0 <= core_id < nvars-1")
        values = self._cores[core_id+1](samples[core_id+1:core_id+2])
        for ii in range(core_id+2, self.nvars()):
            values = self._bkd._la_einsum(
                "ijkl, jmkl->imkl", values, self._cores[ii](samples[ii:ii+1]))
        return values

    def _core_function_jacobians(self, core_id, samples):
        core = self._cores[core_id]
        jacs = []
        for ii in range(core._ranks[0]):
            jacs.append([])
            for jj in range(core._ranks[1]):
                jacs[-1].append(core._basisexps[ii][jj].basis(samples[core_id:core_id+1]))
        return jacs

    def _core_jacobian(self, samples, core_id):
        # TODO: this function is only really used by alternating least squares
        # when used for least squares this can be improved, e.g.
        # when sweeping left to right through cores we can store
        # left core products and update it when we move to next gradient
        # this will be more efficient but may not be worth the hastle
        # as right core product will continually need to be updated
        if core_id == 0:
            return self._first_core_jacobian(samples)
        if core_id < self.nvars()-1 :
            return self._interior_core_jacobian(samples, core_id)
        return self._final_core_jacobian(samples)
        
    def _first_core_jacobian(self, samples):
        core = self._cores[0]
        fun_jacs = self._core_function_jacobians(0, samples)
        Rmat = self._eval_right_cores(0, samples)
        jacs = []
        for qq in range(self.nqoi()):
            jac = []
            for jj in range(core._ranks[1]):
                jac.append(Rmat[jj, 0, :, qq:qq+1]*fun_jacs[0][jj])
            jacs.append(self._bkd._la_hstack(jac))
        return jacs

    def _interior_core_jacobian(self, samples, core_id):
        core = self._cores[core_id]
        Lmat = self._eval_left_cores(core_id, samples)
        fun_jacs = self._core_function_jacobians(core_id, samples)
        Rmat = self._eval_right_cores(core_id, samples)
        jacs = []
        #E.g. for core fun [0, 0]
        
        # [A11, A12][G11, 0][B11] = [A12 G11, 0][B11] = [A12 G11 B11]
        #           [0,   0][B21]               [B21]
        # [A11, A12][0, G12][B11] = [0, A11 G11][B11] = [A11 G11 B12]
        #           [0,   0][B12]               [B21]
        for qq in range(self.nqoi()):
            jac = []
            for ii in range(core._ranks[0]):
                for jj in range(core._ranks[1]):
                    jac.append((Lmat[0, ii, :, qq]*Rmat[jj, 0, :, qq])[:, None]*fun_jacs[ii][jj])
            jacs.append(self._bkd._la_hstack(jac))
        return jacs

    def _final_core_jacobian(self, samples):
        core = self._cores[self.nvars()-1]
        Lmat = self._eval_left_cores(self.nvars()-1, samples)
        fun_jacs = self._core_function_jacobians(self.nvars()-1, samples)
        jacs = []
        for qq in range(self.nqoi()):
            jac = []
            for ii in range(core._ranks[0]):
                jac.append(Lmat[0, ii, :, qq:qq+1]*fun_jacs[ii][0])
            jacs.append(self._bkd._la_hstack(jac))
        return jacs

    def __call__(self, samples):
        values = self._cores[0](samples[:1])
        for ii in range(1, self.nvars()):
            values = self._bkd._la_einsum(
                "ijkl, jmkl->imkl", values, self._cores[ii](samples[ii:ii+1]))
        return values[0, 0]

    def fit(self, train_samples, train_values):
        """Fit the expansion by finding the optimal coefficients. """
        if samples.shape[1] != values.shape[0]:
            raise ValueError(
                ("Number of cols of samples {0} does not match" +
                 "number of rows of values").format(
                     samples.shape[1], values.shape[0]))
        active_active_opt_params = self._solver.solve(self, values)
        self.hyp_list.set_active_opt_params()
        
    def __repr__(self):
        return "{0}(\n\t{1}\n)".format(self.__class__.__name__, ",\n\t".join([f"{core}" for core in self._cores]))


class AdditiveFunctionTrain(FunctionTrain):
    # first core is |f_1 1|
    # middle ith cores are | 1    0|
    #                      |f_i, 1|
    # last dth core is | 1 |
    #                  |f_d|
    def __init__(self, univariate_funs : list[BasisExpansion], nqoi: int):
        self._bkd = univariate_funs[0]._bkd
        self._check_univariate_functions(univariate_funs)
        cores = self._init_cores(univariate_funs, nqoi)
        super().__init__(cores, nqoi)

    def _init_constant_fun(self, fill_value, nqoi):
        constant_basis = MonomialBasis(backend=self._bkd)
        constant_basis.set_hyperbolic_indices(1, 0, 1.)
        # set coef_bounds to [np.nan, np.nan] so these coefficients are set as
        # inactive
        fun = MonomialExpansion(
            constant_basis, coef_bounds=[np.nan, np.nan], nqoi=nqoi
        )
        fun.set_coefficients(self._bkd._la_full((1, nqoi), fill_value))
        return fun

    def _init_cores(self, univariate_funs, nqoi):
        nvars = len(univariate_funs)
        cores = [FunctionTrainCore([[univariate_funs[0], self._init_constant_fun(1., nqoi)]])]
        for ii in range(1, nvars-1):
            cores.append(
                FunctionTrainCore(
                    [
                        [self._init_constant_fun(1., nqoi), self._init_constant_fun(0., nqoi)],
                        [univariate_funs[ii], self._init_constant_fun(1., nqoi)],
                    ]
                )
            )
        cores.append(
            FunctionTrainCore([[self._init_constant_fun(1., nqoi)], [univariate_funs[nvars-1]]])
        )
        return cores

    def _check_univariate_functions(self, univariate_funs):
        nvars = len(univariate_funs)
        for ii in range(nvars):
            if not isinstance(univariate_funs[ii], BasisExpansion):
                raise ValueError("univariate function must be a BasisExpansion")
            if type(self._bkd) is not type(univariate_funs[ii]._bkd):
                raise ValueError(
                    "backends of univariate functions are not consistent"
                )

class AlternatingLeastSquaresSolver:
    def __init__(self, tol=1e-4, maxiters=10, verbosity=0):
        self._tol = tol
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._bkd = None

    def _solve_core(self, ft, samples, values, core_id):
        core = ft._cores[core_id]
        jac = ft._core_jacobian(samples, core_id)
        coefs = []
        for qq in range(ft.nqoi()):
           coefs.append(self._bkd._la_lstsq(jac[qq], values[:, qq:qq+1]))
        return self._bkd._la_hstack(coefs).flatten()

    def solve(self, ft, samples, values):
        self._bkd = ft._bkd
        if ft.hyp_list.nvars() != ft.hyp_list.nactive_vars():
            raise ValueError(
                "{0} only works if all hyperparameters are active but {1}".format(
                    self, "nvars {0} != nactive_vars {1}".format(
                        ft.hyp_list.nvars(), ft.hyp_list.nactive_vars())))
        if self._verbosity > 0:
            print(self)
            print(ft)
        it = 0
        while True:
            for ii in range(ft.nvars()):
                coefs = self._solve_core(ft, samples, values, ii)
                print(coefs)
                ft._cores[ii].hyp_list.set_active_opt_params(coefs)
            it += 1
            if it >= self._maxiters:
                if self._verbosity > 0:
                    print(f"Terminating: maxiters {self._maxiters} reached")
                break
            loss = ft._bkd._la_mean(ft._bkd._la_norm(ft(samples)-values, axis=1))
            if self._verbosity > 1:
                print("it {0}: \t loss {1}".format(it, loss))
            if loss < self._tol:
                if self._verbosity > 0:
                    print(f"Terminating: tolerance {self._tol} reached")
                break

    def __repr__(self):
        return "{0}(tol={1}, maxiters={2}, verbosity={3})".format(
            self.__class__.__name__, self._tol, self._maxiters, self._verbosity
        )

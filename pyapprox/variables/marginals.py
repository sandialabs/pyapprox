from scipy.stats._distn_infrastructure import rv_sample, rv_continuous
from scipy.stats import _continuous_distns
from scipy.stats import _discrete_distns
import numpy as np
from functools import partial


def is_continuous_variable(rv):
    """
    Is variable continuous
    """
    return bool((rv.dist.name in _continuous_distns._distn_names) or
                rv.dist.name == "continuous_rv_sample" or
                rv.dist.name == "continuous_monomial" or
                rv.dist.name == "rv_function_indpndt_vars" or
                rv.dist.name == "rv_product_indpndt_vars")


def is_bounded_continuous_variable(rv):
    """
    Is variable bounded and continuous
    """
    interval = rv.interval(1)
    return bool(is_continuous_variable(rv) and
                rv.dist.name != "continuous_rv_sample"
                and np.isfinite(interval[0]) and np.isfinite(interval[1]))


def is_bounded_discrete_variable(rv):
    """
    Is variable bounded and discrete
    """
    interval = rv.interval(1)
    return bool(((rv.dist.name in _discrete_distns._distn_names) or
                (rv.dist.name == "float_rv_discrete") or
                 (rv.dist.name == "discrete_chebyshev")) and
                np.isfinite(interval[0]) and np.isfinite(interval[1]))


def get_probability_masses(rv, tol=0):
    """
    Get the the locations and masses of a discrete random variable.

    Parameters
    ----------
    tol : float
        Fraction of total probability in (0, 1). Can be useful with
        extracting masses when numerical precision becomes a problem
    """
    # assert is_bounded_discrete_variable(rv)
    name, scales, shapes = get_distribution_info(rv)
    if (name == "float_rv_discrete" or name == "discrete_chebyshev" or
            name == "continuous_rv_sample"):
        return rv.dist.xk.copy(), rv.dist.pk.copy()
    elif name == "hypergeom":
        M, n, N = [shapes[key] for key in ["M", "n", "N"]]
        xk = np.arange(max(0, N-M+n), min(n, N)+1, dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    elif name == "binom":
        n = shapes["n"]
        xk = np.arange(0, n+1, dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    elif (name == "nbinom" or name == "geom" or name == "logser" or
          name == "poisson" or name == "planck" or name == "zipf" or
          name == "dlaplace" or name == "skellam"):
        if tol <= 0:
            raise ValueError("interval is unbounded so specify probability<1")
        lb, ub = rv.interval(1-tol)
        xk = np.arange(int(lb), int(ub), dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    elif name == "boltzmann":
        xk = np.arange(shapes["N"], dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    elif name == "randint":
        xk = np.arange(shapes["low"], shapes["high"], dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    else:
        raise ValueError(f"Variable {rv.dist.name} not supported")


def get_distribution_info(rv):
    """
    Get important information from a scipy.stats variable.

    Notes
    -----
    Shapes and scales can appear in either args of kwargs depending on how
    user initializes frozen object.
    """
    name = rv.dist.name
    shape_names = rv.dist.shapes
    if shape_names is not None:
        shape_names = [name.strip() for name in shape_names.split(",")]
        shape_values = [
            rv.args[ii] for ii in range(min(len(rv.args), len(shape_names)))]
        shape_values += [
            rv.kwds[shape_names[ii]]
            for ii in range(len(rv.args), len(shape_names))]
        shapes = dict(zip(shape_names, shape_values))
    else:
        shapes = dict()

    scale_values = [rv.args[ii] for ii in range(len(shapes), len(rv.args))]
    scale_values += [rv.kwds[key] for key in rv.kwds if key not in shapes]
    if len(scale_values) == 0:
        # if is_bounded_discrete_variable(rv):
        #     lb, ub = rv.interval(1)
        #     if rv.pmf(lb) == 0:
        #         # scipy has precision issues which cause interval to return
        #         # wrong value
        #         lb = rv.ppf(1e-15)
        #     if rv.pmf(ub) == 0:
        #         # scipy has precision issues which cause interval to return
        #         # wrong value
        #         ub = rv.ppf(1-1e-15)
        #     scale_values = [lb, ub-lb]
        #     print(scale_values)
        # else:
        scale_values = [0, 1]
    elif len(scale_values) == 1 and len(rv.args) > len(shapes):
        scale_values += [1.]
    elif len(scale_values) == 1 and "scale" not in rv.kwds:
        scale_values += [1.]
    elif len(scale_values) == 1 and "loc" not in rv.kwds:
        scale_values = [0]+scale_values
    scale_names = ["loc", "scale"]
    scales = dict(zip(scale_names, [np.atleast_1d(s) for s in scale_values]))

    if type(rv.dist) == float_rv_discrete:
        xk = rv.dist.xk.copy()
        shapes = {"xk": xk, "pk": rv.dist.pk}

    return name, scales, shapes


def scipy_raw_pdf(pdf, loc, scale, shapes, x):
    """
    Get the raw pdf of a scipy.stats variable.

    Evaluating this function avoids error checking which can
    slow evaluation significantly. Use with caution
    """
    return pdf((x - loc)/scale, **shapes)/scale


def get_pdf(rv, log=False):
    """
    Return a version of rv.pdf that does not use all the error checking.
    Use with caution. Does speed up calculation significantly though
    """
    name, scales, shapes = get_distribution_info(rv)

    if name == "ncf":
        raise ValueError("scipy implementation prevents generic wraping")

    if not log:
        pdf = partial(
            scipy_raw_pdf, rv.dist._pdf, scales['loc'], scales['scale'],
            shapes)
        return pdf
    else:
        return partial(
            scipy_raw_pdf, rv.dist._logpdf, scales['loc'], scales['scale'],
            shapes)


def transform_scale_parameters(var):
    """
    Transform scale parameters so that when any bounded variable is transformed
    to the canonical domain [-1, 1]
    """
    if (is_bounded_continuous_variable(var)):
        a, b = var.interval(1)
        loc = (a+b)/2
        scale = b-loc
        return loc, scale

    if is_bounded_discrete_variable(var):
        # var.interval(1) can return incorrect bounds
        xk, pk = get_probability_masses(var)
        a, b = xk.min(), xk.max()
        loc = (a+b)/2
        scale = b-loc
        return loc, scale

    scale_dict = get_distribution_info(var)[1]
    # copy is essential here because code below modifies scale
    loc, scale = scale_dict["loc"].copy(), scale_dict["scale"].copy()
    return loc, scale


def variables_equivalent(rv1, rv2):
    """
    Determine if 2 scipy variables are equivalent

    Let
    a = beta(1,1,-1,2)
    b = beta(a=1,b=1,loc=-1,scale=2)

    then a==b will return False because .args and .kwds are different
    """
    name1, scales1, shapes1 = get_distribution_info(rv1)
    name2, scales2, shapes2 = get_distribution_info(rv2)
    # print(scales1, shapes1, scales2, shapes2)
    if name1 != name2:
        return False
    if scales1 != scales2:
        return False
    return variable_shapes_equivalent(rv1, rv2)


def get_unique_variables(variables):
    """
    Get the unique 1D variables from a list of variables.
    """
    nvars = len(variables)
    unique_variables = [variables[0]]
    unique_var_indices = [[0]]
    for ii in range(1, nvars):
        found = False
        for jj in range(len(unique_variables)):
            if variables_equivalent(variables[ii], unique_variables[jj]):
                unique_var_indices[jj].append(ii)
                found = True
                break
        if not found:
            unique_variables.append(variables[ii])
            unique_var_indices.append([ii])
    return unique_variables, unique_var_indices


def variable_shapes_equivalent(rv1, rv2):
    """
    Are the variable shape parameters the same.
    """
    name1, __, shapes1 = get_distribution_info(rv1)
    name2, __, shapes2 = get_distribution_info(rv2)
    if name1 != name2:
        return False
    # if name1 == "float_rv_discrete" or name1 == "discrete_chebyshev":
    if "xk" in shapes1:
        # xk and pk shapes are list so != comparison will not work
        not_equiv = np.any(shapes1["xk"] != shapes2["xk"]) or np.any(
            shapes1["pk"] != shapes2["pk"])
        return not not_equiv
    else:
        return shapes1 == shapes2
    return True


class float_rv_discrete(rv_sample):
    """Discrete distribution defined on locations represented as floats.

    rv_discrete in scipy only allows for integer locations.

    Currently we only guarantee that overloaded functions and cdf, ppf and
    moment work and are tested
    """

    def __init__(self, a=0, b=np.inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, seed=None, extradoc=None):
        # extradoc must appear in __init__ above even though it is not
        # used for backwards capability.
        super(float_rv_discrete, self).__init__(
            a, b, name, badvalue, moment_tol, values, inc, longname, shapes,
            seed=seed)
        self.xk = self.xk.astype(dtype=float)

    def __new__(cls, *args, **kwds):
        return super(float_rv_discrete, cls).__new__(cls)

    def _rvs(self):
        samples = np.random.choice(self.xk, size=self._size, p=self.pk)
        return samples

    def rvs(self, *args, **kwds):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None or int or ``np.random.RandomState`` instance,
            optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        rndm = kwds.pop("random_state", None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = np.logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc*np.ones(size, "d")

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            from scipy._lib._util import check_random_state
            self._random_state = check_random_state(rndm)

        # `size` should just be an argument to _rvs(), but for, um,
        # historical reasons, it is made an attribute that is read
        # by _rvs().
        self._size = size
        vals = self._rvs(*args)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # JDJAKEM: commenting this scipy code out allows for non integer
        # locations
        # # Cast to int if discrete
        # if discrete:
        #     if size == ():
        #         vals = int(vals)
        #     else:
        #         vals = vals.astype(int)

        return vals

    def pmf(self, x):
        x = np.atleast_1d(x)
        vals = np.zeros(x.shape[0])
        for jj in range(x.shape[0]):
            for ii in range(self.xk.shape[0]):
                if np.allclose(
                        self.xk[ii], x[jj], atol=np.finfo(float).eps*2):
                    vals[jj] = self.pk[ii]
                    break
        return vals


class rv_function_indpndt_vars_gen(rv_continuous):
    """
    Custom variable representing the function of random variables.

    Used to create 1D polynomial chaos basis orthogonal to the PDFs
    of the scalar function of the 1D variables.
    """
    def _argcheck(self, fun, init_variables, quad_rules):
        return True

    def _pdf(self, x, fun, init_variables, quad_rules):
        raise NotImplementedError("Expression for PDF not known")


rv_function_indpndt_vars = rv_function_indpndt_vars_gen(
    shapes="fun, initial_variables, quad_rules",
    name="rv_function_indpndt_vars")


class rv_product_indpndt_vars_gen(rv_continuous):
    """
    Custom variable representing the product of random variables.

    Used to create 1D polynomial chaos basis orthogonal to the PDFs
    of the scalar product of the 1D variables.
    """
    def _argcheck(self, funs, init_variables, quad_rules):
        return True

    def _pdf(self, x, funs, init_variables, quad_rules):
        raise NotImplementedError("Expression for PDF not known")


rv_product_indpndt_vars = rv_product_indpndt_vars_gen(
    shapes="funs, initial_variables, quad_rules",
    name="rv_product_indpndt_vars")


def get_truncated_range(var, unbounded_alpha=0.99, bounded_alpha=1.0):
    """
    Get the truncated range of a 1D variable

    Parameters
    ----------
    var : `scipy.stats.dist`
        A 1D variable

    unbounded_alpha : float
        fraction in (0, 1) of probability captured by ranges for unbounded
        random variables

    bounded_alpha : float
        fraction in (0, 1) of probability captured by ranges for bounded
        random variables. bounded_alpha < 1 is useful when variable is
        bounded but is used in a copula

    Returns
    -------
    range : iterable
        The finite (possibly truncated) range of the random variable
    """
    if (is_bounded_continuous_variable(var) or
            is_bounded_discrete_variable(var)):
        return var.interval(bounded_alpha)

    return var.interval(unbounded_alpha)

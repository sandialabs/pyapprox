import unittest
from functools import partial
import numpy as np

from pyapprox.optimization.cvar_regression import (
    smooth_conditional_value_at_risk_gradient,
    smooth_conditional_value_at_risk,
    smooth_max_function,
    smooth_max_function_first_derivative,
    smooth_max_function_second_derivative,
    smooth_conditional_value_at_risk_composition,
)

from pyapprox.optimization.risk import ValueAtRisk


# TODO: update these functions and those in cvar_regression module to use
# new api, then use models check_apply_jacobian to replace custom check
# gradients here


def _check_gradients(fun, zz, direction, plot, disp, rel, fd_eps):
    function_val, directional_derivative = fun(zz, direction)
    if isinstance(function_val, np.ndarray):
        function_val = function_val.squeeze()

    if fd_eps is None:
        fd_eps = np.logspace(-13, 0, 14)[::-1]
    errors = []
    row_format = "{:<12} {:<25} {:<25} {:<25}"
    if disp:
        if rel:
            print(
                row_format.format(
                    "Eps", "norm(jv)", "norm(jv_fd)", "Rel. Errors"
                )
            )
        else:
            print(
                row_format.format(
                    "Eps", "norm(jv)", "norm(jv_fd)", "Abs. Errors"
                )
            )
    row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
    for ii in range(fd_eps.shape[0]):
        zz_perturbed = zz.copy() + fd_eps[ii] * direction
        # perturbed_function_val = fun(zz_perturbed)
        # add jac=False so that exact gradient is not always computed
        perturbed_function_val = fun(zz_perturbed, direction=None)
        if isinstance(perturbed_function_val, np.ndarray):
            perturbed_function_val = perturbed_function_val.squeeze()
        # print(inspect.getfullargspec(fun).args)
        # print(perturbed_function_val, function_val, fd_eps[ii])
        fd_directional_derivative = (
            perturbed_function_val - function_val
        ) / fd_eps[ii]
        # print(fd_directional_derivative)
        errors.append(
            np.linalg.norm(
                fd_directional_derivative.reshape(directional_derivative.shape)
                - directional_derivative
            )
        )
        if rel:
            errors[-1] /= np.linalg.norm(directional_derivative)

        if disp:
            print(
                row_format.format(
                    fd_eps[ii],
                    np.linalg.norm(directional_derivative),
                    np.linalg.norm(fd_directional_derivative),
                    errors[ii],
                )
            )

    if plot:
        plt.loglog(fd_eps, errors, "o-")
        plt.ylabel(r"$\lvert\nabla_\epsilon f\cdot p-\nabla f\cdot p\rvert$")
        plt.xlabel(r"$\epsilon$")
        plt.show()

    return np.asarray(errors)


def _wrap_function_with_gradient(fun, return_grad):
    if (
        (return_grad is not None)
        and not callable(return_grad)
        and (return_grad != "return_gradp")
        and (return_grad != True)
    ):
        raise ValueError("return_grad must be callable, 'jacp', or None")

    if callable(return_grad):

        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x)
            return fun(x), return_grad(x).dot(direction)

        return fun_wrapper

    if return_grad == True and has_kwarg(fun, "return_grad"):
        # this is PyApprox's preferred convention
        def fun_wrapper(x, direction=None):
            if direction is None:
                val = fun(x, return_grad=False)
                return val
            vals, grad = fun(x, return_grad=True)
            return vals, grad.dot(direction)

        return fun_wrapper

    if return_grad == True:

        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x)[0]
            vals, grad = fun(x)
            return vals, grad.dot(direction)

        return fun_wrapper

    if return_grad == "jacp":
        assert has_kwarg(fun, "return_grad")

        # this is PyApprox's other preferred convention
        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x, return_grad=False)
            val, grad = fun(x, return_grad=True)
            return fun(x), grad.dot(direction)

        return fun_wrapper
    return fun


def check_gradients(
    fun, jac, zz, plot=False, disp=True, rel=True, direction=None, fd_eps=None
):
    """
    Compare a user specified jacobian with the jacobian computed with finite
    difference with multiple step sizes.

    Parameters
    ----------
    fun : callable

        A function with one of the following signatures

        ``fun(z) -> (vals)``

        or

        ``fun(z, jac) -> (vals, grad)``

        or

        ``fun(z, direction) -> (vals, directional_grad)``

        where ``z`` is a 2D np.ndarray with shape (nvars, 1) and the
        first output is a 2D np.ndarray with shape (nqoi, 1) and the second
        output is a gradient with shape (nqoi, nvars).
        jac is a flag that specifies if the function returns only
        the funciton value (False) or the function value and gradient (True)

    jac : callable or string
        If jac="jacp" then provided the jacobian of ``fun`` with signature

        ``jac(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, 1) and the
        output is a 2D np.ndarray with shape (nqoi, nvars).
        This assumes that fun
        only returns a value (not gradient) and has signature

        ``fun(z) -> np.ndarray``


    zz : np.ndarray (nvars, 1)
        A sample of ``z`` at which to compute the gradient

    plot : boolean
        Plot the errors as a function of the finite difference step size

    disp : boolean
        True - print the errors
        False - do not print

    rel : boolean
        True - compute the relative error in the directional derivative,
        i.e. the absolute error divided by the directional derivative using
        ``jac``.
        False - compute the absolute error in the directional derivative

    direction : np.ndarray (nvars, 1)
        Direction to which Jacobian is applied. Default is None in which
        case random direction is chosen.

    fd_eps : np.ndarray (nstep_sizes)
        The finite difference step sizes used to compute the gradient.
        If None then fd_eps=np.logspace(-13, 0, 14)[::-1]

    Returns
    -------
    errors : np.ndarray (14, nqoi)
        The errors in the directional derivative of ``fun`` at 14 different
        values of finite difference tolerance for each quantity of interest
    """
    assert zz.ndim == 2
    assert zz.shape[1] == 1

    fun_wrapper = _wrap_function_with_gradient(fun, jac)

    if direction is None:
        direction = np.random.normal(0, 1, (zz.shape[0], 1))
        direction /= np.linalg.norm(direction)
    assert direction.ndim == 2 and direction.shape[1] == 1

    return _check_gradients(
        fun_wrapper, zz, direction, plot, disp, rel, fd_eps
    )


from pyapprox.util.sys_utilities import has_kwarg


class TestCVARRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def help_check_smooth_max_function_gradients(self, smoother_type, eps):
        x = np.array([0.01])
        errors = check_gradients(
            partial(smooth_max_function, smoother_type, eps),
            partial(smooth_max_function_first_derivative, smoother_type, eps),
            x[:, np.newaxis],
        )

        errors = check_gradients(
            partial(smooth_max_function_first_derivative, smoother_type, eps),
            partial(smooth_max_function_second_derivative, smoother_type, eps),
            x[:, np.newaxis],
        )
        assert errors.min() < 1e-6

    def test_smooth_max_function_gradients(self):
        smoother_type, eps = 0, 1e-1
        self.help_check_smooth_max_function_gradients(smoother_type, eps)

        smoother_type, eps = 1, 1e-1
        self.help_check_smooth_max_function_gradients(smoother_type, eps)

    def help_check_smooth_conditional_value_at_risk(
        self, smoother_type, eps, alpha
    ):
        samples = np.linspace(-1, 1, 11)
        VaR = ValueAtRisk(alpha)
        VaR.set_samples(samples[None, :])
        t = VaR()[0]
        # t = value_at_risk(samples, alpha)[0]
        x0 = np.hstack((samples, t))[:, None]
        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk(
                smoother_type, eps, alpha, xx
            ),
            lambda xx: smooth_conditional_value_at_risk_gradient(
                smoother_type, eps, alpha, xx
            ),
            x0,
        )
        assert errors.min() < 1e-6

        weights = np.random.uniform(1, 2, samples.shape[0])
        weights /= weights.sum()
        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk(
                smoother_type, eps, alpha, xx, weights
            ),
            lambda xx: smooth_conditional_value_at_risk_gradient(
                smoother_type, eps, alpha, xx, weights
            ),
            x0,
        )
        assert errors.min() < 1e-6

    def test_smooth_conditional_value_at_risk_gradient(self):
        smoother_type, eps, alpha = 0, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk(
            smoother_type, eps, alpha
        )

        smoother_type, eps, alpha = 1, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk(
            smoother_type, eps, alpha
        )

    def help_check_smooth_conditional_value_at_risk_composition_gradient(
        self, smoother_type, eps, alpha, nsamples, nvars
    ):
        samples = np.arange(nsamples * nvars).reshape(nvars, nsamples)
        t = 0.1
        x0 = np.array([2, 3, t])[:, np.newaxis]

        def fun(x):
            return (np.sum((x * samples) ** 2, axis=0).T)[:, np.newaxis]

        def jac(x):
            return 2 * (x * samples**2).T

        errors = check_gradients(fun, jac, x0[:2], disp=False)
        assert errors.min() < 1e-6

        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk_composition(
                smoother_type, eps, alpha, fun, jac, xx
            ),
            True,
            x0,
        )
        assert errors.min() < 1e-7

    def test_smooth_conditional_value_at_risk_composition_gradient(self):
        nsamples, nvars = 4, 2
        smoother_type, eps, alpha = 0, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk_composition_gradient(
            smoother_type, eps, alpha, nsamples, nvars
        )

        nsamples, nvars = 10, 2
        smoother_type, eps, alpha = 1, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk_composition_gradient(
            smoother_type, eps, alpha, nsamples, nvars
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

import sys
import os
import inspect
import importlib
import numpy as np


def trace_error_with_msg(msg, e: Exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(msg)
    print(f'Failed with error: {e}')
    details = f"""
    Error type: {exc_type}
    file/location: {fname} | {exc_tb.tb_lineno}
    """
    print(details)


def hash_array(array, decimals=None):
    r"""
    Hash an array for dictionary or set based lookup

    Parameters
    ----------
    array : np.ndarray
       The integer array to hash

    Returns
    -------
    key : integer
       The hash value of the array
    """
    # assert array.ndim==1
    # array = np.ascontiguousarray(array)
    # array.flags.writeable = False
    # return hash(array.data)
    if decimals is not None:
        array = np.around(array, decimals)
    # return hash(array.tostring())
    return hash(array.tobytes())


def package_available(name):
    pkg_available = True
    try:
        mod = importlib.import_module(name)
    except (ModuleNotFoundError, ImportError):
        pkg_available = False

    return pkg_available


def get_num_args(function):
    """
    Return the number of arguments of a function.
    If function is a member function of a class the self argument is not
    counted.

    Parameters
    ----------
    function : callable
        The Python callable to be interrogated

    Return
    ------
    num_args : integer
        The number of arguments to the function including
        args, varargs, keywords
    """
    args = inspect.getfullargspec(function)
    num_args = 0
    if args[0] is not None:
        num_args += len(args[0])
        if 'self' in args[0]:
            num_args -= 1
    if args[1] is not None:
        num_args += len(args[1])
    if args[2] is not None:
        num_args += len(args[2])
    # do not count defaults of keywords conatined in args[3]
    # if args[3] is not None:
    #    num_args += len(args[3])
    return num_args



# Keyword-only arguments are not the same as normal keyword arguments.
# Keyword-only arguments are arguments that come after *args and
# before **kwargs in a function call. e.g.
# def func(arg, *args, kwonly, **kwargs):


def has_kwarg(fun, name):
    info = inspect.getfullargspec(fun)
    if name in info.args:
        return True
    if info.varkw is not None and name in info.varkw:
        return True
    if info.kwonlydefaults is not None and name in info.kwonlydefaults:
        return True
    return False

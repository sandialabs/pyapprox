import numpy as np
import time
from pyapprox.models.wrappers import evaluate_1darray_function_on_2d_array

def fun_0(sample):
    assert sample.ndim == 1
    return np.sum(sample**2)


def pyapprox_fun_0(samples):
    values = evaluate_1darray_function_on_2d_array(fun_0, samples)
    return values


def fun_pause_1(sample):
    assert sample.ndim == 1
    time.sleep(np.random.uniform(0, .05))
    return np.sum(sample**2)


def pyapprox_fun_1(samples):
    return evaluate_1darray_function_on_2d_array(fun_pause_1, samples)


def fun_pause_2(sample):
    time.sleep(np.random.uniform(.05, .1))
    return np.sum(sample**2)


def pyapprox_fun_2(samples):
    return evaluate_1darray_function_on_2d_array(fun_pause_2, samples)

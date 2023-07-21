try:
    from numba import njit
    from numba.extending import get_cython_function_address
    import ctypes

    _PTR = ctypes.POINTER
    _dble = ctypes.c_double
    _ptr_dble = _PTR(_dble)

    addr = get_cython_function_address(
        "scipy.special.cython_special", "gammaln")
    functype = ctypes.CFUNCTYPE(_dble, _dble)
    gammaln_float64 = functype(addr)
except ImportError:
    # msg = "Could not import numba. Reverting to pure python implementations"
    # print(msg)
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    gammaln_float64 = None

import sys, os


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
    #assert array.ndim==1
    #array = np.ascontiguousarray(array)
    #array.flags.writeable = False
    # return hash(array.data)
    if decimals is not None:
        array = np.around(array, decimals)
    # return hash(array.tostring())
    return hash(array.tobytes())
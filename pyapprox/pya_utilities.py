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

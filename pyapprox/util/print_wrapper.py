# The following can be used to append location of print statement so
# that errant print statements can be found and removed

import builtins
from inspect import getframeinfo, stack


original_print = print


def print_wrap(*args, **kwargs):
    caller = getframeinfo(stack()[1][0])
    original_print("Filename:", caller.filename, "Lineno:", caller.lineno,
                   "function:", caller.function, "::", *args, **kwargs)


builtins.print = print_wrap

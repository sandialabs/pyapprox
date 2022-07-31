"""
PyApprox : Sotware for model and data analysis

Find way to avoid importing anything from matplotlib, scipy and other big
pacakges. It increases overhead before function is run
"""
import sys as _sys
from pyapprox.util.sys_utilities import (
    package_available as _package_available
)

name = "pyapprox"

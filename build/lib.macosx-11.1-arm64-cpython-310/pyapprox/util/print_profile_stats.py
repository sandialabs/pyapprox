# python -m cProfile -o profile.out <file.py>
import pstats
import sys
import getopt
import subprocess


if __name__ == '__main__':

    opts, args = getopt.getopt(
        sys.argv[1:], "p:s:n:h", ["pyfile=", "statfile=", "nstats="])

    run = True
    stats_filename = 'profile.out'
    py_filename = None
    view_html = False
    num_funcs = 30
    for opt, arg in opts:
        if opt in ('--pyfile', '-p'):
            py_filename = arg
        if opt in ('--statfile', '-s'):
            stats_filename = arg
        if opt == '-h':
            view_html = True
        if opt in ('-n', '--nstats'):
            num_funcs = int(arg)

    if py_filename is not None:
        shell_command = 'python -m cProfile -o %s %s' % (
            stats_filename, py_filename)
        out = subprocess.call(shell_command, shell=True)

    p = pstats.Stats(stats_filename)
    p.strip_dirs().sort_stats('cumulative').print_stats(num_funcs)
    p.strip_dirs().sort_stats('time').print_stats(num_funcs)

    if view_html:
        # pip install cprofilev
        shell_command = 'cprofilev -f %s' % stats_filename
        out = subprocess.call(shell_command, shell=True)

# gprof2dot -f pstats profile.out | dot -Tpng -o output.png && eog output.png

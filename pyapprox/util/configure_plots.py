import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['text.usetex'] = True # use latex for all text handling
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf' # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
#print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'

# NOTES
# using plt.plot(visible=False ) will allow linestyle/marker to be included
# in legend but line will not plotted


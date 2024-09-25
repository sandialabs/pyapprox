import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16
# set font to default font used by most latex document classes
plt.rcParams["font.family"] = "Computer Modern"
plt.rcParams["lines.linewidth"] = 3
# use latex for all text handling (slows down ploting)
plt.rcParams["text.usetex"] = True
# remove all extra white space
plt.rcParams["savefig.bbox"] = "tight"
# plot with transparent background
plt.rcParams['savefig.transparent'] = True
# gives best resolution plots
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 16
# print plt.rcParams.keys()
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}"
)

# NOTES
# using plt.plot(visible=False) will allow linestyle/marker to be included
# in legend but line will not plotted

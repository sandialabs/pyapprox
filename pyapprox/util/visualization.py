import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def convert_plot_to_tikz(tikz_file, tikz_dir, show=False,
                         remove_all_files=True):
    """
    Create a standalone PDF from a tikz enviornment
    """
    import matplotlib2tikz as pylab2tikz

    pylab2tikz.save(tikz_file, show_info=False)

    with open(tikz_file, 'r') as f:
        tikz_string = f.read()

    # create a latex wrapper for the tikz
    wrapper = r'''\documentclass{standalone}
    \usepackage[utf8]{inputenc}
    \usepackage{pgfplots}
    \usepackage{amsmath,amssymb,amsfonts}
    \usepgfplotslibrary{groupplots}
    \pgfplotsset{compat=newest}
    \begin{document}
    %s
    \end{document}''' % (tikz_string)

    os.remove(tikz_file)
    compile_tikz_figure(wrapper, tikz_file, tikz_dir, show, remove_all_files)


def show_pdf(pdf_file):
    """
    Call PDF viewer to display a PDF.

    Parameters
    ----------
    pdf_file : string
        The name of the PDF.
    """
    try:
        # check_call will raise an error if the command is not found
        subprocess.check_call('okular %s' % pdf_file, shell=True)
    except RuntimeError:
        try:
            # if using aliases the bashrc or bash_profile is not loaded
            # by subprocess.call unless the following modification is made.
            # skim is primarily used with os x
            subprocess.check_call(
                ['/bin/bash', '-i', '-c', 'skim %s' % pdf_file])
            # skim has to be manually quit, not just close specific file
            # before the code will proceed from here
        except RuntimeError:
            print('No pdf reader could be found. plot cannot be shown')


def compile_tikz_figure(file_string, filename_out, tikz_dir, show=False,
                        remove_all_files=False):
    r"""
    Compile a latex file using pdflatex.

    Parameters
    ----------
    file_string : string
        The latex commands including headers and everything inside
        \begin{document} and \end{document}

    filename_out : string
        The latex filename produced. must end with ".tex"

    tikz_dir : string
        Directory to store the generated files. Directory must already exist

    remove_all_files : boolean
        True - remove all generated files including .tex and .pdf files.
               This is useful when show=True
        False - all generated files will be removed except .tex. and .pdf
    """

    # write the new tex file to a specified directory
    cur_dir = os.path.abspath(os.path.curdir)
    os.chdir(tikz_dir)
    with open(filename_out, 'w') as f:
        f.write(file_string)

    # compile tex
    subprocess.check_output(
        ['pdflatex', '--interaction=nonstopmode', filename_out],
        stderr=subprocess.STDOUT)

    pdf_file = filename_out[:-4] + '.pdf'
    # remove the latex generated files except the pdf and tex file

    # show plot with a pdf reader
    if show:
        show_pdf(pdf_file)

    import glob
    filenames = glob.glob(filename_out[:-4]+'*')
    if not remove_all_files:
        filenames.remove(pdf_file)
        filenames.remove(filename_out)

    for filename in filenames:
        os.remove(filename)

    os.chdir(cur_dir)


def plot_3d_indices_latex(
        indices, filename_out, coeff=None, show=True, tikz_dir='.',
        tol=1e-5, remove_all_files=True, label_indices=False):
    """
    Plot a set of three dimensional multivariate indices using latex.

    Parameters
    ----------
    indices : np.ndarray (num_vars,num_indices)
        The indices to plot

    filename_out : string
        The latex filename produced. must end with ".tex"

    tikz_dir : string
        Directory to store the generated files. Directory must already exist

    remove_all_files : boolean
        True - remove all generated files including .tex and .pdf files.
               This is useful when show=True
        False - all generated files will be removed except .tex. and .pdf files

    show : boolean
        Call PDF viewer to display generated PDF

    tol : float
        Plot only indices with coefficients whose magnitude exceeds this
        threshold. If coeff is None this argument is ignored

    coeff : np.ndarray(num_indices,1)
        The coefficients associated with each index

    label_indices : boolean
        Print the entries of each index on each 3d index cube.
    """

    assert filename_out[-4:] == '.tex'
    if coeff is not None:
        assert coeff.shape[0] == indices.shape[1]

    # load in a template file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    template_filename = os.path.join(dir_path, '3d-indices.tex')
    string = ''
    with open(template_filename, 'r') as f:
        lines = f.read()

    # determine the maximum degree and use this to set limits of plot axes
    max_degree = 0
    for i in range(indices.shape[1]):
        # +1 to have write coordinates for latex file
        index = indices[:, i]+1
        max_degree = max(max_degree, index.max())

    # for clear plotting need to order indices
    II = np.lexsort((indices[2, :], indices[1, :], indices[0, :]))
    indices = indices[:, II]

    # determine all non-zero indices
    for i in range(indices.shape[1]):
        if coeff is None or abs(coeff[i, 0]) > tol:
            # +1 to have write coordinates for latex file
            index = indices[:, i]  # +1
            # Last three arguments to latex function \cube are color
            # percentages. I could not get latex to do the math I needed here.
            if label_indices:
                tmp = r'\cube{%d}{%d}{%d}{(%d,%d,%d)}{%d}{%d}{%d}' % (
                    index[0], index[1], index[2], index[0], index[1], index[2],
                    float(max_degree-index[2])/max_degree*80,
                    float(max_degree-index[0])/max_degree*80,
                    float(max_degree-index[1])/max_degree*80)
            else:
                tmp = r'\cube{%d}{%d}{%d}{}{%d}{%d}{%d}' % (
                    index[0], index[1], index[2],
                    64, 64, 64)
                # float(max_degree-index[2])/max_degree*80,
                # float(max_degree-index[0])/max_degree*80,
                # float(max_degree-index[1])/max_degree*80)
            string += tmp

    # modify the template file to plot the cubes. All template file does
    # before modification is plot the axes with a default size
    file_string = lines.replace(
        r'\end{tikzpicture}', string+'\n'+r'\end{tikzpicture}')
    file_string = file_string.replace(
        r'\sdef\myscale{6}', r'\def\myscale{%d}' % max_degree)

    compile_tikz_figure(file_string, filename_out,
                        tikz_dir, show, remove_all_files)


def get_meshgrid_samples(plot_limits, num_pts_1d, logspace=False):
    """
    Generate meshgrid samples.
    """
    if isinstance(num_pts_1d, int):
        num_pts_1d = [num_pts_1d]*2
    if not logspace:
        x = np.linspace(plot_limits[0], plot_limits[1], num_pts_1d[0])
        y = np.linspace(plot_limits[2], plot_limits[3], num_pts_1d[1])
    else:
        x = np.logspace(
            np.log10(plot_limits[0]), np.log10(plot_limits[1]), num_pts_1d[0])
        y = np.logspace(
            np.log10(plot_limits[2]), np.log10(plot_limits[3]), num_pts_1d[1])

    # X, Y =  meshgrid (x,y) same as
    # X = cartesian_product([x, y])[0].reshape((ny, nx)
    # Y = cartesian_product([x, y])[1].reshape((ny, nx)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack((X.reshape((1, X.shape[0]*X.shape[1])),
                     Y.reshape((1, Y.shape[0]*Y.shape[1]))))
    return X, Y, pts


def get_meshgrid_function_data(function, plot_limits, num_pts_1d, qoi=0,
                               logspace=False):
    """
    Generate data from a function in the format needed for plotting.
    Samples are generated between specified lower and upper bounds
    and the function is evaluated on the tensor product of the 1d samples.

    Parameters
    ----------
    function : callable function
        The function must accept an np.ndarray of size (2, num_pts_1d**2)
        and return a np.ndarray of size (num_pts_1d,num_qoi)

    plot_limits : np.ndarray or list
        The lower and upper bounds used to sample the function
        [lb1,ub1,lb1,lb2]

    num_pts_1d : integer or iterable
        The number of samples in each dimension. The function is evaluated
        on the tensor product of the 1d samples

    qoi : integer
        function returns a np.ndarray of size (num_pts_1d,num_qoi) qoi
        specifies which column of the array to access.

    Returns
    -------
    X : np.ndarray of size (num_pts_1d,num_pts_1d)
        The 1st coordinate of the samples

    Y : np.ndarray of size (num_pts_1d,num_pts_1d)
        The 2nd coordinate of the samples

    Z : np.ndarray of size (num_pts_1d,num_pts_1d)
        The function values at each sample
    """
    X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d, logspace)
    Z = function(pts)
    if (Z.ndim == 2):
        Z = Z[:, qoi]
    Z = np.reshape(Z, (X.shape[0], X.shape[1]))
    return X, Y, Z


def plot_contours(X, Y, Z, ax, num_contour_levels=10, offset=0,
                  cmap="coolwarm", zorder=None):
    """
    Plot the contours of a two-dimensional function using matplotlib.contourf.

    Parameters
    ----------
    num_contour_levels : boolean (default=10)
       Plot the contours of the function to plot beneath the 3d surface.

    offset : boolean (default=0)
       Plot the contours offset from z=0 plane

    X,Y,Z: see documentation of Return of function get_meshgrid_function_data

    Returns
    -------
    cset : a QuadContourSet object.
       Handle useful for modifying contours after this function has been called
    """
    # Plot contours of functions underneath surface
    if num_contour_levels > 0:
        cset = ax.contourf(
            X, Y, Z, zdir='z', offset=offset,
            levels=np.linspace(Z.min(), Z.max(), num_contour_levels),
            cmap=cmap, zorder=zorder)

    return cset


def create_3d_axis():
    """
    Create a 3D axis for plotting surfaces
    """
    # try:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    # except TypeError:
    #    # Add following import to avoid error Unknown projection '3d'
    #    # when using ax = fig.gca(projection='3d')
    #    # from mpl_toolkits.mplot3d import Axes3D
    #    # fig = plt.figure()
    #    # ax = Axes3D(fig)
    ax = plt.subplots(subplot_kw={"projection": "3d"})[1]
    return ax


def _turn_off_3d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def plot_surface(X, Y, Z, ax, samples=None, limit_state=None,
                 num_contour_levels=0, plot_axes=True, cmap="coolwarm",
                 axis_labels=None, angle=None, alpha=1., zorder=None,
                 antialiased=False):
    """
    Plot the three-dimensional surface of a two-dimensional function using
    matplotlib.plot_surface

    Parameters
    ----------
    samples : np.ndarray (num_vars+1 x num_samples)
        The x,y,z coordinates of samples to plot

    limit_state : callable_function
        Function must take samples np.ndarray (num_vars x num_samples)
        and values np.ndarray (num_samples), and return the indices
        a set of indices which correspond to samples within the limit state
        and the valuethat should be assigned to each of those samples , e.g.
        e.g. I, default_value = limit_state(samples,values)

    plot_axes : boolean (default=True)
       Plot the 3 coordinate axes and ticks

    angle : integer (default=None)
       The angle at which the plot is viewed. If None matplotlib will
       use its default angle

    axis_labels : list of strings size (3)
       Labels of the x, y, and z axes

    alpha : double
       Transperancy of plot

    zorder : the priority of the plot on the axes. A zorder=2 places plot
    above all plots with zorder=1

    X,Y,Z: see documentation of Return of function get_meshgrid_function_data

    """

    # Define transperancy of plot
    if samples is not None:
        ax.scatter3D(samples[0, :], samples[1, :], samples[2, :], marker='o',
                     s=100, color='k')

    if limit_state is not None:
        pts = np.vstack((X.reshape((1, X.shape[0]*X.shape[1])),
                         Y.reshape((1, Y.shape[0]*Y.shape[1]))))
        vals = Z.flatten()
        I, default_value = limit_state(pts, vals)
        # print(I, default_value)
        vals[I] = default_value
        Z_tmp = vals.reshape((X.shape[0], X.shape[1]))
    else:
        Z_tmp = Z

    # Plot surface
    ax.plot_surface(X, Y, Z_tmp, cmap=cmap, antialiased=antialiased,
                    cstride=1, rstride=1, linewidth=0., alpha=alpha,
                    zorder=zorder)

    if not plot_axes:
        _turn_off_3d_axes(ax)

    if (axis_labels is not None):
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])

    # Set the angle of the plot (how it is viewed)
    if (angle is not None):
        ax.view_init(ax.elev, angle)


def remove_nonfinite_meshgrid_data(Z, val):
    """
    Replace nonfinite values obtained on a meshgrid.
    """
    shape = Z.shape
    Z = Z.flatten()
    II = np.isfinite(Z)
    mask = np.ones(Z.shape[0], dtype=bool)
    mask[II] = False
    Z[mask] = val
    Z = np.reshape(Z, shape)
    return Z


def plot_2d_indices(indices, coeffs=None, other_indices=None, ax=None):
    """
    Plot a set of 2D multi-indices.
    """
    assert indices.ndim == 2
    assert indices.shape[0] == 2
    if coeffs is not None:
        assert indices.shape[1] == coeffs.shape[0]

    if other_indices is not None and type(other_indices) == np.ndarray:
        # code below assumes other indices is a list of other indices
        # e.g. that might correspond to n different index sets
        other_indices = [other_indices]

    if ax is None:
        ax = plt

    xMax = np.max(indices[0, :])
    yMax = np.max(indices[1, :])
    for ii in range(indices.shape[1]):
        box = np.array([[indices[0, ii]-1, indices[1, ii]-1],
                        [indices[0, ii], indices[1, ii]-1],
                        [indices[0, ii], indices[1, ii]],
                        [indices[0, ii]-1, indices[1, ii]],
                        [indices[0, ii]-1, indices[1, ii]-1]]).T + 0.5
        ax.plot(box[0, :], box[1, :], '-k', lw=1)
        ax.fill(box[0, :], box[1, :], color='gray', alpha=0.5, edgecolor='k')
        if coeffs is not None:
            ax.text(indices[0, ii], indices[1, ii], '$%1.0f$' % coeffs[ii],
                    fontsize=mpl.rcParams['xtick.labelsize'],
                    verticalalignment='center', horizontalalignment='center')

    colors = ['r', 'b']
    alphas = [0.5, 0.2]
    if (other_indices is not None):
        for jj in range(len(other_indices)):
            for ii in range(other_indices[jj].shape[1]):
                box = np.array([[other_indices[jj][0, ii]-1, other_indices[jj][1, ii]-1],
                                [other_indices[jj][0, ii],
                                    other_indices[jj][1, ii]-1],
                                [other_indices[jj][0, ii],
                                    other_indices[jj][1, ii]],
                                [other_indices[jj][0, ii]-1,
                                    other_indices[jj][1, ii]],
                                [other_indices[jj][0, ii]-1, other_indices[jj][1, ii]-1]]).T + 0.5
                ax.plot(box[0, :], box[1, :], '-', c=colors[jj], lw=1)
                ax.fill(box[0, :], box[1, :], color=colors[jj],
                        alpha=alphas[jj], edgecolor=colors[jj])
            if other_indices[jj].shape[1] > 0:
                xMaxNew = np.max(other_indices[jj][0, :])
                xMax = max(xMax, xMaxNew)
                yMaxNew = np.max(other_indices[jj][1, :])
                yMax = max(yMax, yMaxNew)

    lim = max(xMax, yMax)
    if ax == plt:
        ax.xticks(np.arange(0, lim + 1))
        ax.yticks(np.arange(0, lim + 1))
        ax.xlim(-0.5, lim + 1)
        ax.ylim(-0.5, lim + 1)
    else:
        ax.set_xticks(np.arange(0, lim + 1))
        ax.set_yticks(np.arange(0, lim + 1))
        ax.set_xlim(-0.5, lim + 1)
        ax.set_ylim(-0.5, lim + 1)

    # set maximum number of ticks so labels do not overlap
    ax.locator_params(nbins=10)


def plot_2d_polygon(vertices, ax=plt):
    """
    Plot a polygon.
    """
    convhull = ConvexHull(vertices.T)
    plt.plot(vertices[0, :], vertices[1, :], 'ko')
    for simplex in convhull.simplices:
        ax.plot(vertices[0, simplex], vertices[1, simplex], 'k-')


def plot_multiple_2d_gaussian_slices(means, variances, texfilename,
                                     reference_gaussian_data=None,
                                     domain=None, show=True, num_samples=100):
    """
    Plot 2D slices of a multivariate Gaussian variable

    If there are wiggles in plot near zero of Gaussians then increase num_samples
    """
    assert len(means) == len(variances)
    if domain is None:
        domain = [np.inf, -np.inf]
        from scipy.stats import norm
        for i in range(len(variances)):
            lb, ub = norm.ppf([0.001, 0.999], scale=np.sqrt(variances[i]))
            domain[0] = min(means[i]+lb, domain[0])
            domain[1] = max(means[i]+ub, domain[1])

        lb, ub = norm.ppf([0.001, 0.999], scale=np.sqrt(
            reference_gaussian_data[1]))
        domain[0] = min(reference_gaussian_data[0]+lb, domain[0])
        domain[1] = max(reference_gaussian_data[0]+ub, domain[1])
        print(domain)

    file_string = r'''
\documentclass[border=10pt]{standalone}
\usepackage{tikz}\usepackage{pgfplots}
\pgfplotsset{width=7cm}
\usepackage{pgfplotstable}
\begin{document}
\begin{tikzpicture}[
declare function={normal(\m,\s)=1/(2*\s*sqrt(pi))*exp(-(x-\m)^2/(2*\s^2));}
]'''
    file_string += r'''
\begin{axis}[
samples=%d,domain=%1.3f:%1.3f,samples y=0,tick style={draw=none},yticklabels={},zmin=0,area plot/.style={fill opacity=0.75,draw=blue!80!black,fill=blue!70,mark=none,smooth},plot box ratio = 1 3 1,view={20}{30},axis background/.style={ shade,top color=gray,bottom color=white}
]''' % (num_samples, domain[0], domain[1])

    num_gaussians = len(means)
    if reference_gaussian_data is not None:
        assert len(reference_gaussian_data) == 2
        num_gaussians += 1
    # going to 3 makes y-axis 3 times longer than others
    ylocations = np.linspace(0., 1., num_gaussians)

    # plot backwards so that first gaussian is plotted on top of
    # all others
    idx = 1
    if reference_gaussian_data is not None:
        file_string += r'\addplot3 [area plot,draw=red!80!black,fill=red!70,mark=none,smooth] (x,%1.3f,{normal(%1.3f,%1.3f)}) -- (axis cs:%1.3f,%1.3f,0);' % (
            ylocations[-1], reference_gaussian_data[0], np.sqrt(reference_gaussian_data[1]), domain[0], ylocations[-1])+'\n'
        idx += 1

    for i in range(num_gaussians-idx, -1, -1):
        file_string += r'\addplot3 [area plot] (x,%1.3f,{normal(%1.3f,%1.3f)}) -- (axis cs:%1.3f,%1.3f,0);' % (
            ylocations[i], means[i], np.sqrt(variances[i]), domain[0], ylocations[i])+'\n'

    file_string += r'''
\end{axis}\end{tikzpicture}
\end{document}'''

    with open(texfilename, "w") as text_file:
        text_file.write(file_string)

    tex_out = subprocess.check_output(
        # use pdflatex for now until travis features a more modern lualatex
        ['pdflatex', '--interaction=nonstopmode', texfilename],
        stderr=subprocess.STDOUT)

    if show:
        subprocess.call(["okular", texfilename[:-4]+'.pdf'])


def plot_voxels(data, ax, color=[0, 0, 1, 0.5]):
    """Plot voxels of `data`.
        Args:
            data - n-dimensional np.array
            color - [r,g,b,alpha]
    """
    # ax.patch.set_alpha(0.5)  # Set semi-opacity
    colors = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))
    colors[np.where(data)] = color
    ax.voxels(data, facecolors=colors, edgecolor='gray')


def plot_3d_indices(indices, ax, other_indices=None):
    """
    Plot a set of 3D multi-indices.
    """
    shape = indices.max(axis=1)+1
    if other_indices is not None:
        shape = np.maximum(shape, other_indices.max(axis=1))+1

    filled = np.zeros(shape, dtype=int)
    for nn in range(indices.shape[1]):
        ii, jj, kk = indices[:, nn]
        filled[ii, jj, kk] = 1

    plot_voxels(filled, ax, color=[1, 1, 1, .9])

    if other_indices is not None:
        filled = np.zeros(shape, dtype=int)
        for nn in range(other_indices.shape[1]):
            ii, jj, kk = other_indices[:, nn]
            filled[ii, jj, kk] = 1
        plot_voxels(filled, ax, color=[1, 0, 0, 0.5])

    angle = 45
    # angle=60
    ax.view_init(30, angle)
    ax.set_axis_off()


def pya_colorbar(mappable, ax_in=None):
    """
    Create custom colorbar.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if ax_in is None:
        last_axes = plt.gca()
        ax = mappable.axes
    else:
        ax = ax_in
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if ax_in is None:
        plt.sca(last_axes)
    return cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    ranges = im.get_extent()
    xlocs = np.linspace(ranges[0], ranges[1], data.shape[1]+1)[:-1]
    xlocs += (xlocs[1]-xlocs[0])/2
    ylocs = np.linspace(ranges[2], ranges[3], data.shape[0]+1)[:-1]
    ylocs += (ylocs[1]-ylocs[0])/2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(
                xlocs[j], ylocs[i], valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_2d_samples(samples, ax=None, **kwargs):
    """
    Plot 2D samples

    samples : np.ndarray (2, nsamples)
        The coordinates of the samples to plot

    ax : :class:`pyplot.axes`
        Use the axes object if provided
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(samples[0, :], samples[1, :], linestyle='None', **kwargs)
    return ax


linestyle_tuples = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}


def mathrm_label(label):
    label = r"$\mathrm{%s}$" % label.replace(" ", r"\;")
    label = label.replace("-", r"}$-$\mathrm{")
    return label


def mathrm_labels(labels):
    return [mathrm_label(label) for label in labels]

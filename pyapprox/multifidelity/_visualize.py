import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def _hp(G, root, width=1., vert_gap=0.2, vert_loc=0,
        xcenter=0.5, pos=None, parent=None):
    '''
    see hierarchy_pos docstring for most arguments

    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch. - only affects it if non-directed

    '''

    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width/len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hp(
                G, child, width=dx, vert_gap=vert_gap,
                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                pos=pos, parent=root)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    '''
    Motivated by Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with
           other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        msg = 'cannot use hierarchy_pos on a graph that is not a tree'
        raise TypeError(msg)

    return _hp(G, root, width, vert_gap, vert_loc, xcenter)


def _plot_model_recursion(recursion_index, ax):
    nmodels = len(recursion_index)+1
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(nmodels))
    for ii, jj in enumerate(recursion_index):
        graph.add_edge(ii+1, jj)
    pos = _hierarchy_pos(graph, 0, vert_gap=0.1, width=0.1)
    nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=[2000],
            font_size=24)


def _plot_partition(ii, jj, ax, color, text):
    box = np.array(
        [[ii, jj], [ii+1, jj], [ii+1, jj+1], [ii, jj+1], [ii, jj]]).T
    ax.plot(*box, color='k')
    ax.fill(*box, color=color)
    if text is not None:
        ax.text(*(box[:, 0]+0.5), text,
                verticalalignment='center', horizontalalignment='center')


def _plot_allocation_matrix(allocation_mat, npartition_samples, ax,
                            set_symbol=None):
    if set_symbol is None:
        set_symbol = r"\mathcal{Z}"
    nmodels, nacv_subsets = allocation_mat.shape
    cycle = iter(plt.cm.rainbow(np.linspace(0, 1, nmodels)))
    colors = [c for c in cycle]
    # loop over subsets
    for ii in range(nmodels):
        # loop over partitions in subset
        for jj in range(1, nacv_subsets):
            if allocation_mat[ii, jj] == 1.:
                if npartition_samples is not None:
                    text = "$%d$" % npartition_samples[ii]
                else:
                    text = None
                _plot_partition(jj, ii, ax, colors[ii], text)
    xticks = np.arange(1, nacv_subsets)+0.5
    ax.set_xticks(xticks)
    labels = [r"$%s_{%d}^*$" % (set_symbol, ii//2) if ii % 2 == 0 else
              r"$%s_{%d}$" % (set_symbol, ii//2)
              for ii in range(1, nacv_subsets)]
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(nmodels)+0.5)
    # number of samples are rounded cost est_rounded cost,
    # but target cost is not rounded
    ax.set_yticklabels(
        [r'$\mathcal{P}_{%d}$' % ii for ii in range(nmodels)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)


def _autolabel(ax, rects, model_labels):
    # Attach a text label in each bar in *rects*
    for rect, label in zip(rects, model_labels):
        try:
            rect = rect[0]
        except TypeError:
            pass
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width()/2,
                        rect.get_y() + rect.get_height()/2),
                    xytext=(0, -10),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

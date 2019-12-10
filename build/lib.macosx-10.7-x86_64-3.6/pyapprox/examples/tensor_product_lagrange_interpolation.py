
from pyapprox.visualization import get_meshgrid_function_data,\
    create_3d_axis,mpl,plt,plot_surface
from pyapprox.configure_plots import *
from pyapprox.barycentric_interpolation import *
from pyapprox.utilities import cartesian_product

def plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax=None):
    abscissa, tmp = clenshaw_curtis_pts_wts_1D(level)
    abscissa_1d = [abscissa,abscissa]
    barycentric_weights_1d = [compute_barycentric_weights_1d(abscissa_1d[0]),
                              compute_barycentric_weights_1d(abscissa_1d[1])]
    training_samples = cartesian_product(abscissa_1d,1)
    fn_vals=np.zeros((training_samples.shape[1],1))
    idx = jj*abscissa_1d[1].shape[0]+ii
    fn_vals[idx]=1.
    f = lambda samples: multivariate_barycentric_lagrange_interpolation(
        samples,abscissa_1d,barycentric_weights_1d,fn_vals,np.array([0,1]))

    plot_limits = [-1,1,-1,1]; num_pts_1d=101
    X,Y,Z = get_meshgrid_function_data(f, plot_limits, num_pts_1d)
    if ax is None:
        ax = create_3d_axis()
    cmap = mpl.cm.coolwarm

    plot_surface(X,Y,Z,ax,axis_labels=None,limit_state=None,
                 alpha=0.3,cmap=mpl.cm.coolwarm,zorder=3,plot_axes=False)
    num_contour_levels=30
    offset=-(Z.max()-Z.min())/2
    cmap = mpl.cm.gray
    cset = ax.contourf(
        X, Y, Z, zdir='z', offset=offset,
        levels=np.linspace(Z.min(),Z.max(),num_contour_levels),
        cmap=cmap, zorder=-1)
    ax.plot(training_samples[0,:],training_samples[1,:],
            offset*np.ones(training_samples.shape[1]),'o',zorder=100,color='b')
    
    x=np.linspace(-1,1,100)
    y=training_samples[1,idx]*np.ones((x.shape[0]))
    z = f(np.vstack((x[np.newaxis,:],y[np.newaxis,:])))[:,0]
    ax.plot(x,Y.max()*np.ones((x.shape[0])),z,'-r')
    ax.plot(abscissa_1d[0],Y.max()*np.ones(
        (abscissa_1d[0].shape[0])),np.zeros(abscissa_1d[0].shape[0]),'or')
    
    y=np.linspace(-1,1,100)
    x=training_samples[0,idx]*np.ones((y.shape[0]))
    z = f(np.vstack((x[np.newaxis,:],y[np.newaxis,:])))[:,0]
    ax.plot(X.min()*np.ones((x.shape[0])),y,z,'-r')
    ax.plot(X.min()*np.ones(
        (abscissa_1d[1].shape[0])),abscissa_1d[1],
            np.zeros(abscissa_1d[1].shape[0]),'or')
   

plot_limits = [0,1,0,1]
num_pts_1d = 30
from pyapprox.configure_plots import *
from pyapprox.visualization import plot_surface, get_meshgrid_function_data

fig = plt.figure(figsize=(2*8,6))
ax=fig.add_subplot(1,2,1,projection='3d')
X,Y,Z = get_meshgrid_function_data(model, plot_limits, num_pts_1d)
plot_surface(X,Y,Z,ax)

ax=fig.add_subplot(1,2,2,projection='3d')
error = lambda x: np.absolute(model(x)-poly(x))
X,Y,Z = get_meshgrid_function_data(error, plot_limits, num_pts_1d)
plot_surface(X,Y,Z,ax)
offset = -(Z.max()-Z.min())/2
ax.plot(train_samples[0,:],train_samples[1,:],
#offset*np.ones(train_samples.shape[1]),'o',zorder=100,color='b')
error(train_samples)[:,0],'o',zorder=100,color='k')
ax.view_init(80, 45)
plt.show()
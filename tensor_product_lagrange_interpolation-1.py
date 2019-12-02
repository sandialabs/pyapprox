from pyapprox.examples.tensor_product_lagrange_interpolation import *
fig = plt.figure(figsize=(2*8,6))
ax=fig.add_subplot(1,2,1,projection='3d')
level = 2; ii=1; jj=1
plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)

ax=fig.add_subplot(1,2,2,projection='3d')
level = 2; ii=1; jj=3
plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)
plt.show()
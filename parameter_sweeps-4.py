from pyapprox.density import plot_gaussian_contours
f, ax = plt.subplots(1,1)
ax=plot_gaussian_contours(mean,np.linalg.cholesky(covariance),ax=ax)[1]
ax.plot(samples[0,:],samples[1,:],'o')
ax.plot(samples[0,[0,num_samples_per_sweep-1]],
        samples[1,[0,num_samples_per_sweep-1]],'sr')
if num_sweeps>1:
    ax.plot(samples[0,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
            samples[1,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
            'sr')
plt.show()
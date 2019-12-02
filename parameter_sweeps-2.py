f, ax = plt.subplots(1,1)
ax.plot(samples[0,:],samples[1,:],'o')
# plot the samples at the begining and end of each parameter sweep
for ii in range(num_sweeps):
    ax.plot(
    samples[0,[ii*num_samples_per_sweep,(ii+1)*num_samples_per_sweep-1]],
    samples[1,[ii*num_samples_per_sweep,(ii+1)*num_samples_per_sweep-1]],
    'sr')
plt.show()
from typing import List

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array


def print_statistics(
    samples: Array,
    values: Array = None,
    sample_labels: List = None,
    value_labels: List = None,
):
    """
    Print statistics about a set of samples and associated values

    Parameters
    ----------
    samples : np.ndarray (nvars,nsamples)
        Random samples

    values : np.ndarray (nsamples,nqoi)
       Function values at samples

    Examples
    --------
    >>> nvars = 2
    >>> np.random.seed(1)
    >>> samples = np.random.normal(0, 1, (nvars, 100))
    >>> values = np.array([np.sum(samples**2, axis=0), 2*np.sum(samples**2, axis=0)]).T
    >>> print_statistics(samples, values)
               z0         z1         y0         y1
    count  100.000000 100.000000 100.000000 100.000000
    mean     0.060583   0.152795   1.679132   3.358265
    std      0.885156   0.931995   1.705877   3.411753
    min     -2.301539  -2.434838   0.031229   0.062458
    max      2.185575   2.528326   9.575905  19.151810
    """
    nvars, nsamples = samples.shape
    if values is None:
        values = np.empty((nsamples, 0))
    if values.ndim == 1:
        values = values[:, np.newaxis]
    nqoi = values.shape[1]
    assert nsamples == values.shape[0]
    if sample_labels is None:
        sample_labels = ["z%d" % ii for ii in range(nvars)]
    if value_labels is None:
        value_labels = ["y%d" % ii for ii in range(nqoi)]
    data = [(label, s) for s, label in zip(samples, sample_labels)]
    data += [(label, s) for s, label in zip(values.T, value_labels)]

    # data = [(label, s) for s, label in zip(samples, sample_labels)]
    # data += [(label, s) for s, label in zip(values.T, value_labels)]
    # data = dict(data)
    # df = DataFrame(index=np.arange(nsamples), data=data)
    # print(df.describe())

    str_format = " ".join(["{:<6}"] + ["{:^10}"] * (len(data)))
    print(str_format.format(*([" "] + [dat[0] for dat in data])))
    stat_funs = [
        lambda x: x.shape[0],
        lambda x: x.mean(),
        lambda x: x.std(),
        lambda x: x.min(),
        lambda x: x.max(),
    ]
    stat_labels = ["count", "mean", "std", "min", "max"]
    str_format = " ".join(["{:<6}"] + ["{:10.6f}"] * (len(data)))
    for stat_fun, stat_label in zip(stat_funs, stat_labels):
        print(
            str_format.format(
                *([stat_label] + [stat_fun(dat[1]) for dat in data])
            )
        )


def rejection_sampling(
    target_density: callable,
    proposal_density: callable,
    generate_proposal_samples: callable,
    envelope_factor: float,
    nvars: int,
    nsamples: int,
    verbose: bool = False,
    batch_size: int = None,
) -> Array:
    """
    Obtain samples from a density f(x) using samples from a proposal
    distribution g(x).

    Parameters
    ----------
    target_density : callable vals = target_density(samples)
        The target density f(x)

    proposal_density : callable vals = proposal_density(samples)
        The proposal density g(x)

    generate_proposal_samples : callable samples = generate_samples(nsamples)
        Generate samples from the proposal density

    envelope_factor : double
        Factor M that satifies f(x)<=Mg(x). Set M such that inequality is as
        close to equality as possible

    nvars : integer
        The number of variables

    nsamples : integer
        The number of samples required

    verbose : boolean
        Flag specifying whether to print diagnostic information

    batch_size : integer
        The number of evaluations of each density to be performed in a batch.
        Almost always we should set batch_size=nsamples

    Returns
    -------
    samples : np.ndarray (nvars, nsamples)
        Independent samples from the target distribution
    """
    if batch_size is None:
        batch_size = nsamples

    cntr = 0
    nproposal_samples = 0
    samples = np.empty((nvars, nsamples), dtype=float)
    while cntr < nsamples:
        proposal_samples = generate_proposal_samples(batch_size)
        target_density_vals = target_density(proposal_samples)
        proposal_density_vals = proposal_density(proposal_samples)
        assert target_density_vals.shape[0] == batch_size
        assert proposal_density_vals.shape[0] == batch_size
        urand = np.random.uniform(0.0, 1.0, (batch_size))

        # ensure envelop_factor is large enough
        if np.any(
            target_density_vals > (envelope_factor * proposal_density_vals)
        ):
            idx = np.argmax(
                target_density_vals / (envelope_factor * proposal_density_vals)
            )
            msg = "proposal_density*envelop factor does not bound target "
            msg += "density: %f,%f" % (
                target_density_vals[idx],
                (envelope_factor * proposal_density_vals)[idx],
            )
            raise ValueError(msg)

        idx = np.where(
            urand
            < target_density_vals / (envelope_factor * proposal_density_vals)
        )[0]

        nbatch_samples_accepted = min(idx.shape[0], nsamples - cntr)
        idx = idx[:nbatch_samples_accepted]
        samples[:, cntr : cntr + nbatch_samples_accepted] = proposal_samples[
            :, idx
        ]
        cntr += nbatch_samples_accepted
        nproposal_samples += batch_size

    if verbose:
        print(("num accepted", nsamples))
        print(("num rejected", nproposal_samples - nsamples))
        print(("inverse envelope factor", 1 / envelope_factor))
        print(
            (
                "acceptance probability",
                float(nsamples) / float(nproposal_samples),
            )
        )
    return samples


def discrete_sampling(N: int, probs: Array, states: Array = None) -> Array:
    r"""
    discrete_sampling -- samples iid from a discrete probability measure

    x = discrete_sampling(N, prob, states)

    Generates N iid samples from a random variable X whose probability mass
    function is

    prob(X = states[j]) = prob[j],    1 <= j <= length(prob).

    If states is not given, the states are gives by 1 <= state <= length(prob)
    """

    p = probs.squeeze() / np.sum(probs)

    bins = (
        np.digitize(
            np.random.uniform(0.0, 1.0, (N, 1)), np.hstack((0, np.cumsum(p)))
        )
        - 1
    )

    if states is None:
        x = bins
    else:
        assert states.shape[0] == probs.shape[0]
        x = states[bins]

    return x.squeeze()

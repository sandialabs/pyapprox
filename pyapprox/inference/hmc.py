"""
Hamiltonian Monte Carlo. Created by Cosmin Safta
"""
import numpy as np
from math import log, exp, sqrt


def logsumexp(a, b):
    """
    Adapted from scipy.special.logsumexp
    """
    if a > b:
        return a + log(1 + exp(b-a))
    else:
        return b + log(1 + exp(a-b))


def hmc(q, p, fun, fun_args, opts, random, return_diagnostics=True):
    #TODO remove return diagonostics and always return
    #TODO make sure rej_free_energy is always defined
    """
    Hamiltonian Monte Carlo

    Radford Neal "MCMC using Hamiltonian dynamics"
    (https://arxiv.org/pdf/1206.1901.pdf)
    """
    num_samples = opts.get('num_samples', 1)  # number of samples
    num_burn = opts.get('num_burn', 0)    # number of "burn-in" samples
    num_steps = opts.get('num_steps', 1)   # number of steps
    epsilon = opts.get('epsilon', 0.2)   # step size
    # window size (Sec. 5.4 in the reference)
    window = opts.get('window', 1)
    # for adjusting momenta betweek steps
    persistence = opts.get('persistence', False)
    ofreq = opts.get('ofreq', 10)       # output to screen
    sfreq = opts.get('sfreq', num_samples)  # output to file
    verb = opts.get('verb', 0)        # verbosity

    assert num_steps >= 1
    assert num_samples >= 1
    assert num_burn >= 0
    assert window >= 0

    window = min(window, num_steps)

    if persistence:
        decay = opts.get('decay', 0.9)
        assert decay >= 0.0 and decay <= 1.0
        decaysq = sqrt(1.0-decay**2)

    num_params = q.shape[0]
    num_acc = 0        #  number of accepted samples
    num_rej = 0        # number of rejected samples
    num_evl = 0        # number of function evaluations
    window_offset = 0  # window offset initialised to zero
    k = -num_burn      # burn samples are omitte

    samples = np.zeros((num_samples, num_params))
    logps = np.zeros(num_samples)
    if return_diagnostics:
        diagn_pos = np.zeros((num_samples, num_params))
        diagn_mom = np.zeros((num_samples, num_params))
        diagn_acc = np.zeros(num_samples)

    # Evaluate starting energy.
    q = q.copy()
    logp, gradlp = fun(np.asarray(q), *fun_args)
    if len(gradlp) != num_params:
        msg = 'fun(q, *fun_args) must return (logp, gradlogp)'
        msg += "\nlen(gradlp)={0} != num_params={1}".format(
            len(gradlp), num_params)
        raise ValueError(msg)

    num_evl = num_evl + 1
    Enrg = -logp

    while k < num_samples:  # keep samples from k >= 0

        # starting position q and momenta p
        q_old = q.copy()
        p_old = p.copy()

        # recalculate Hamiltonian
        Enrg_old = Enrg
        H_old = Enrg + 0.5 * np.sum(p**2)

        # window offset in [0,window]
        if window > 1:
            window_offset = int(window * random.rand())

        have_rej = 0
        have_acc = 0
        n = window_offset
        direction = -1  # assumes that windowing is used

        while (direction == -1) or (n != num_steps):

            # if windowing is not used or we have already taken
            # window_offset steps backwards...
            if direction == -1 and n == 0:
                # Restore, next state should be original start state.
                if window_offset > 0:
                    q = q_old.copy()
                    p = p_old.copy()
                    n = window_offset

                # set direction for forward steps
                Enrg = Enrg_old
                H = H_old
                direction = 1
                stps = direction
            else:
                # check if state not in the accept and/or reject window.
                if (n * direction + 1 < window) or (n > (num_steps-window)):
                    # state in the accept and/or reject window.
                    stps = direction
                else:
                    stps = num_steps - 2 * (window - 1)

                # leapfrog: 1st half-step
                logp, gradlp = fun(np.asarray(q), *fun_args)
                # print(q, '\n', p, '\n', gradlp, "L1")
                num_evl += 1
                p = p + direction * 0.5 * epsilon * gradlp
                q = q + direction * epsilon * p

                # leapfrog: (stps-1)
                for m in range(abs(stps)-1):
                    logp, gradlp = fun(np.asarray(q), *fun_args)
                    # print(q, '\n', p, '\n', gradlp, "L2")
                    num_evl = num_evl + 1
                    p = p + direction * epsilon * gradlp
                    q = q + direction * epsilon * p

                # leapfrog: final half-step
                logp, gradlp = fun(np.asarray(q), *fun_args)
                # print(q, '\n', p, '\n', gradlp, "L3")
                num_evl += 1
                Enrg = -logp
                p = p + direction * 0.5 * epsilon * gradlp
                H = Enrg + 0.5 * np.sum(p**2)
                n = n + stps


            if (window != num_steps) and (n < window):
                # state in reject window
                if not have_rej:
                    rej_free_energy = H
                else:
                    rej_free_energy = -logsumexp(-rej_free_energy, -H)

                if not have_rej or random.rand() < exp(rej_free_energy-H):
                    Enrg_rej = Enrg
                    q_rej = q.copy()
                    p_rej = p.copy()
                    have_rej = 1

            if n > (num_steps - window):
                # state in the accept window.
                if not have_acc:
                    acc_free_energy = H
                else:
                    acc_free_energy = -logsumexp(-acc_free_energy, -H)

                if not have_acc or random.rand() < exp(acc_free_energy-H):
                    Enrg_acc = Enrg
                    q_acc = q.copy()
                    p_acc = p.copy()
                    have_acc = 1

        # acceptance threshold.
        a = exp(rej_free_energy - acc_free_energy)

        if return_diagnostics and k >= 0:
            diagn_pos[k] = q_acc.copy()
            diagn_mom[k] = p_acc.copy()
            diagn_acc[k] = a

        # retrieve new state from the appropriate window.
        if a > random.rand():
            # accept
            Enrg = Enrg_acc
            q = q_acc.copy()
            p = (-1)*p_acc.copy()
            num_acc = num_acc + 1
        else:
            # reject
            if k >= 0:
                # record rejections after burn-in
                num_rej = num_rej + 1

            Enrg = Enrg_rej
            q = q_rej.copy()
            p = p_rej.copy()

        if k >= 0:
            # store after burn-in
            samples[k] = q.copy()
            logps[k] = -Enrg

        if k > 0 and k % sfreq == 0:

            if verb > 0:
                print("Saving samples to file at step {}".format(k))
            np.save("samples.npy", samples[:k])
            np.savetxt("logprob.dat", logps[:k])
            np.save("hmc_qp.npy", q, p)

        # Set momenta for next iteration
        if persistence:
            # Reverse and adjust momenta by a small random amount
            p = -decay * p + decaysq * random.randn(num_params)
        else:
            # Replace all momenta
            p = random.randn(num_params)

        k += 1
        if verb > 0 and k % ofreq == 0:
            if k < 0:
                print("{0:7d} iterations burnin".format(k))
            else:
                print("{0:7d} iterations".format(k))

    if return_diagnostics:
        return_dic = {
            'samples': samples, 'num_rej': num_rej, 'num_acc': num_acc,
            'num_evl': num_evl,
            'diagn': {'pos': diagn_pos, 'mom': diagn_mom, 'acc':
                      diagn_acc}}
    else:
        raise ValueError

    return return_dic

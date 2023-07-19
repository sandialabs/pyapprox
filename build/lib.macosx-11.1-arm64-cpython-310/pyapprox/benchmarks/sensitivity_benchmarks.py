import numpy as np

from pyapprox.util.utilities import evaluate_quadratic_form


def variance_linear_combination_of_indendent_variables(coef, variances):
    assert coef.shape[0] == variances.shape[0]
    return np.sum(coef**2*variances)


def get_oakley_function_data():
    r"""
    Get the data :math:`a_1,a_2,a_3` and :math:`M` of the Oakley function

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    Returns
    -------
    a1 : np.ndarray (15)
       The vector :math:`a_1` of the Oakley function

    a2 : np.ndarray (15)
       The vector :math:`a_2` of the Oakley function

    a3 : np.ndarray (15)
       The vector :math:`a_3` of the Oakley function

    M : np.ndarray (15,15)
       The non-symmetric matrix :math:`M` of the Oakley function

    Examples
    --------

    >>> from pyapprox.benchmarks.sensitivity_benchmarks import get_oakley_function_data
    >>> a1,a2,a3,M=get_oakley_function_data()
    >>> print(a1)
    [0.0118 0.0456 0.2297 0.0393 0.1177 0.3865 0.3897 0.6061 0.6159 0.4005
     1.0741 1.1474 0.788  1.1242 1.1982]
    >>> print(a2)
    [0.4341 0.0887 0.0512 0.3233 0.1489 1.036  0.9892 0.9672 0.8977 0.8083
     1.8426 2.4712 2.3946 2.0045 2.2621]
    >>> print(a3)
    [0.1044 0.2057 0.0774 0.273  0.1253 0.7526 0.857  1.0331 0.8388 0.797
     2.2145 2.0382 2.4004 2.0541 1.9845]
    >>> print(M)
    [[-0.02248289 -0.18501666  0.13418263  0.36867264  0.17172785  0.13651143
      -0.44034404 -0.08142285  0.71321025 -0.44361072  0.50383394 -0.02410146
      -0.04593968  0.21666181  0.05588742]
     [ 0.2565963   0.05379229  0.25800381  0.23795905 -0.59125756 -0.08162708
      -0.28749073  0.41581639  0.49752241  0.08389317 -0.11056683  0.03322235
      -0.13979497 -0.03102056 -0.22318721]
     [-0.05599981  0.19542252  0.09552901 -0.2862653  -0.14441303  0.22369356
       0.14527412  0.28998481  0.2310501  -0.31929879 -0.29039128 -0.20956898
       0.43139047  0.02442915  0.04490441]
     [ 0.66448103  0.43069872  0.29924645 -0.16202441 -0.31479544 -0.39026802
       0.17679822  0.05795266  0.17230342  0.13466011 -0.3527524   0.25146896
      -0.01881053  0.36482392 -0.32504618]
     [-0.121278    0.12463327  0.10656519  0.0465623  -0.21678617  0.19492172
      -0.06552113  0.02440467 -0.09682886  0.19366196  0.33354757  0.31295994
      -0.08361546 -0.25342082  0.37325717]
     [-0.2837623  -0.32820154 -0.10496068 -0.22073452 -0.13708154 -0.14426375
      -0.11503319  0.22424151 -0.03039502 -0.51505615  0.01725498  0.03895712
       0.36069184  0.30902452  0.05003019]
     [-0.07787589  0.00374566  0.88685604 -0.26590028 -0.07932536 -0.04273492
      -0.18653782 -0.35604718 -0.17497421  0.08869996  0.40025886 -0.05597969
       0.13724479  0.21485613 -0.0112658 ]
     [-0.09229473  0.59209563  0.03133829 -0.03308086 -0.24308858 -0.09979855
       0.03446019  0.09511981 -0.3380162   0.006386   -0.61207299  0.08132542
       0.88683114  0.14254905  0.14776204]
     [-0.13189434  0.52878496  0.12652391  0.04511362  0.58373514  0.37291503
       0.11395325 -0.29479222 -0.57014085  0.46291592 -0.09405018  0.13959097
      -0.38607402 -0.4489706  -0.14602419]
     [ 0.05810766 -0.32289338  0.09313916  0.07242723 -0.56919401  0.52554237
       0.23656926 -0.01178202  0.0718206   0.07827729 -0.13355752  0.22722721
       0.14369455 -0.45198935 -0.55574794]
     [ 0.66145875  0.34633299  0.14098019  0.51882591 -0.28019898 -0.1603226
      -0.06841334 -0.20428242  0.06967217  0.23112577 -0.04436858 -0.16455425
       0.21620977  0.00427021 -0.08739901]
     [ 0.31599556 -0.02755186  0.13434254  0.13497371  0.05400568 -0.17374789
       0.17525393  0.06025893 -0.17914162 -0.31056619 -0.25358691  0.02584754
      -0.43006001 -0.62266361 -0.03399688]
     [-0.29038151  0.03410127  0.03490341 -0.12121764  0.02603071 -0.33546274
      -0.41424111  0.05324838 -0.27099455 -0.0262513   0.41024137  0.26636349
       0.15582891 -0.18666254  0.01989583]
     [-0.24388652 -0.44098852  0.01261883  0.24945112  0.07110189  0.24623792
       0.17484502  0.00852868  0.2514707  -0.14659862 -0.08462515  0.36931333
      -0.29955293  0.1104436  -0.75690139]
     [ 0.04149432 -0.25980564  0.46402128 -0.36112127 -0.94980789 -0.16504063
       0.00309433  0.05279294  0.22523648  0.38390366  0.45562427 -0.18631744
       0.0082334   0.16670803  0.16045688]]
    """
    a1 = np.array([0.0118, 0.0456, 0.2297, 0.0393, 0.1177, 0.3865, 0.3897, 0.6061, 0.6159, 0.4005,
                   1.0741, 1.1474, 0.7880, 1.1242, 1.1982])
    a2 = np.array([0.4341, 0.0887, 0.0512, 0.3233, 0.1489, 1.0360, 0.9892, 0.9672, 0.8977, 0.8083,
                   1.8426, 2.4712, 2.3946, 2.0045, 2.2621])
    a3 = np.array([0.1044, 0.2057, 0.0774, 0.2730, 0.1253, 0.7526, 0.8570, 1.0331, 0.8388, 0.7970,
                   2.2145, 2.0382, 2.4004, 2.0541, 1.9845])
    M = np.array([[-2.2482886e-002, -1.8501666e-001, 1.3418263e-001, 3.6867264e-001, 1.7172785e-001, 1.3651143e-001, -4.4034404e-001, -8.1422854e-002, 7.1321025e-001, -4.4361072e-001, 5.0383394e-001, -2.4101458e-002, -4.5939684e-002, 2.1666181e-001, 5.5887417e-002],
                  [2.5659630e-001, 5.3792287e-002, 2.5800381e-001, 2.3795905e-001, -5.9125756e-001, -8.1627077e-002, -2.8749073e-001, 4.1581639e-001,
                      4.9752241e-001, 8.3893165e-002, -1.1056683e-001, 3.3222351e-002, -1.3979497e-001, -3.1020556e-002, -2.2318721e-001],
                  [-5.5999811e-002, 1.9542252e-001, 9.5529005e-002, -2.8626530e-001, -1.4441303e-001, 2.2369356e-001, 1.4527412e-001, 2.8998481e-001,
                      2.3105010e-001, -3.1929879e-001, -2.9039128e-001, -2.0956898e-001, 4.3139047e-001, 2.4429152e-002, 4.4904409e-002],
                  [6.6448103e-001, 4.3069872e-001, 2.9924645e-001, -1.6202441e-001, -3.1479544e-001, -3.9026802e-001, 1.7679822e-001, 5.7952663e-002,
                   1.7230342e-001, 1.3466011e-001, -3.5275240e-001, 2.5146896e-001, -1.8810529e-002, 3.6482392e-001, -3.2504618e-001],
                  [-1.2127800e-001, 1.2463327e-001, 1.0656519e-001, 4.6562296e-002, -2.1678617e-001, 1.9492172e-001, -6.5521126e-002,
                   2.4404669e-002, -9.6828860e-002, 1.9366196e-001, 3.3354757e-001, 3.1295994e-001, -8.3615456e-002, -2.5342082e-001, 3.7325717e-001],
                  [-2.8376230e-001, -3.2820154e-001, -1.0496068e-001, -2.2073452e-001, -1.3708154e-001, -1.4426375e-001, -1.1503319e-001,
                   2.2424151e-001, -3.0395022e-002, -5.1505615e-001, 1.7254978e-002, 3.8957118e-002, 3.6069184e-001, 3.0902452e-001, 5.0030193e-002],
                  [-7.7875893e-002, 3.7456560e-003, 8.8685604e-001, -2.6590028e-001, -7.9325357e-002, -4.2734919e-002, -1.8653782e-001, -
                   3.5604718e-001, -1.7497421e-001, 8.8699956e-002, 4.0025886e-001, -5.5979693e-002, 1.3724479e-001, 2.1485613e-001, -1.1265799e-002],
                  [-9.2294730e-002, 5.9209563e-001, 3.1338285e-002, -3.3080861e-002, -2.4308858e-001, -9.9798547e-002, 3.4460195e-002,
                   9.5119813e-002, -3.3801620e-001, 6.3860024e-003, -6.1207299e-001, 8.1325416e-002, 8.8683114e-001, 1.4254905e-001, 1.4776204e-001],
                  [-1.3189434e-001, 5.2878496e-001, 1.2652391e-001, 4.5113625e-002, 5.8373514e-001, 3.7291503e-001, 1.1395325e-001, -2.9479222e-001, -
                   5.7014085e-001, 4.6291592e-001, -9.4050179e-002, 1.3959097e-001, -3.8607402e-001, -4.4897060e-001, -1.4602419e-001],
                  [5.8107658e-002, -3.2289338e-001, 9.3139162e-002, 7.2427234e-002, -5.6919401e-001, 5.2554237e-001, 2.3656926e-001, -1.1782016e-002,
                   7.1820601e-002, 7.8277291e-002, -1.3355752e-001, 2.2722721e-001, 1.4369455e-001, -4.5198935e-001, -5.5574794e-001],
                  [6.6145875e-001, 3.4633299e-001, 1.4098019e-001, 5.1882591e-001, -2.8019898e-001, -1.6032260e-001, -6.8413337e-002, -
                   2.0428242e-001, 6.9672173e-002, 2.3112577e-001, -4.4368579e-002, -1.6455425e-001, 2.1620977e-001, 4.2702105e-003, -8.7399014e-002],
                  [3.1599556e-001, -2.7551859e-002, 1.3434254e-001, 1.3497371e-001, 5.4005680e-002, -1.7374789e-001, 1.7525393e-001, 6.0258929e-002, -
                   1.7914162e-001, -3.1056619e-001, -2.5358691e-001, 2.5847535e-002, -4.3006001e-001, -6.2266361e-001, -3.3996882e-002],
                  [-2.9038151e-001, 3.4101270e-002, 3.4903413e-002, -1.2121764e-001, 2.6030714e-002, -3.3546274e-001, -4.1424111e-001,
                   5.3248380e-002, -2.7099455e-001, -2.6251302e-002, 4.1024137e-001, 2.6636349e-001, 1.5582891e-001, -1.8666254e-001, 1.9895831e-002],
                  [-2.4388652e-001, -4.4098852e-001, 1.2618825e-002, 2.4945112e-001, 7.1101888e-002, 2.4623792e-001, 1.7484502e-001, 8.5286769e-003,
                   2.5147070e-001, -1.4659862e-001, -8.4625150e-002, 3.6931333e-001, -2.9955293e-001, 1.1044360e-001, -7.5690139e-001],
                  [4.1494323e-002, -2.5980564e-001, 4.6402128e-001, -3.6112127e-001, -9.4980789e-001, -1.6504063e-001, 3.0943325e-003, 5.2792942e-002, 2.2523648e-001, 3.8390366e-001, 4.5562427e-001, -1.8631744e-001, 8.2333995e-003, 1.6670803e-001, 1.6045688e-001]])
    return a1, a2, a3, M


def oakley_function(samples):
    a1, a2, a3, M = get_oakley_function_data()
    term1, term2 = a1.T.dot(samples), a2.T.dot(np.sin(samples))
    term3, term4 = a3.T.dot(
        np.cos(samples)), evaluate_quadratic_form(M, samples)
    vals = term1+term2+term3+term4
    return vals[:, np.newaxis]


def oakley_function_statistics():
    e = np.exp(1)
    a1, a2, a3, M = get_oakley_function_data()
    nvars = M.shape[0]

    term1_mean, term2_mean = 0, 0
    term3_mean, term4_mean = np.sum(a3/np.sqrt(e)), np.trace(M)
    mean = term1_mean+term2_mean+term3_mean+term4_mean

    term1_var = variance_linear_combination_of_indendent_variables(
        a1, np.ones(a1.shape[0]))
    variances_1d = np.ones(a2.shape[0])*(0.5*(1-1/e**2))
    term2_var = variance_linear_combination_of_indendent_variables(
        a2, variances_1d)
    variances_1d = np.ones(a3.shape[0])*(0.5*(1+1/e**2)-1.0/e)
    term3_var = variance_linear_combination_of_indendent_variables(
        a3, variances_1d)
    A = 0.5*(M.T+M)  # needed because M is not symmetric
    term4_var = 2*np.trace(A.dot(A))

    cov_xsinx = 1/np.sqrt(e)
    covar13, covar14, covar23, covar24 = 0, 0, 0, 0
    covar12 = np.sum(a1*a2*cov_xsinx)
    covar34 = np.sum(-1/np.sqrt(e)*a3*np.diag(M))

    variance = term1_var+term2_var+term3_var+term4_var
    variance += 2*(covar12+covar13+covar14+covar23+covar24+covar34)
    main_effects = np.empty((nvars, 1))
    for ii in range(nvars):
        var1 = a1[ii]**2
        var2 = a2[ii]**2*(0.5*(1-1/e**2))
        var3 = a3[ii]**2*(0.5*(1+1/e**2)-1.0/e)
        var4 = 2*M[ii, ii]**2
        cov12 = cov_xsinx*a1[ii]*a2[ii]
        cov34 = -1/np.sqrt(e)*a3[ii]*M[ii, ii]
        main_effects[ii] = var1+var2+var3+var4+2*cov12+2*cov34

    return mean, variance, main_effects/variance


def ishigami_function(samples, a=7, b=0.1):
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    vals = np.sin(samples[0, :])+a*np.sin(samples[1, :])**2 +\
        b*samples[2, :]**4*np.sin(samples[0, :])
    return vals[:, np.newaxis]


def ishigami_function_jacobian(samples, a=7, b=0.1):
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    assert samples.shape[1] == 1
    nvars = 3
    assert samples.shape[0] == nvars
    jac = np.empty((1, nvars))
    jac[0, 0] = np.cos(samples[0, :]) + b * \
        samples[2, :]**4*np.cos(samples[0, :])
    jac[0, 1] = 2*a*np.sin(samples[1, :])*np.cos(samples[1, :])
    jac[0, 2] = 4*b*samples[2, :]**3*np.sin(samples[0, :])
    return jac


def ishigami_function_hessian(samples, a=7, b=0.1):
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    assert samples.shape[1] == 1
    nvars = 3
    assert samples.shape[0] == nvars
    hess = np.empty((nvars, nvars))
    hess[0, 0] = -np.sin(samples[0, :]) - b * \
        samples[2, :]**4*np.sin(samples[0, :])
    hess[1, 1] = 2*a*(np.cos(samples[1, :])**2-np.sin(samples[1, :])**2)
    hess[2, 2] = 12*b*samples[2, :]**2*np.sin(samples[0, :])
    hess[0, 1], hess[1, 0] = 0, 0
    hess[0, 2] = 4*b*samples[2, :]**3*np.cos(samples[0, :])
    hess[2, 0] = hess[0, 2]
    hess[1, 2], hess[2, 1] = 0, 0
    return hess


def get_ishigami_funciton_statistics(a=7, b=0.1):
    """
    p_i(X_i) ~ U[-pi,pi]
    """
    mean = a/2
    variance = a**2/8+b*np.pi**4/5+b**2*np.pi**8/18+0.5
    D_1 = b*np.pi**4/5+b**2*np.pi**8/50+0.5
    D_2, D_3, D_12, D_13 = a**2/8, 0, 0, b**2*np.pi**8/18-b**2*np.pi**8/50
    D_23, D_123 = 0, 0
    main_effects = np.array([D_1, D_2, D_3])/variance
    # the following two ways of calulating the total effects are equivalent
    total_effects1 = np.array(
        [D_1+D_12+D_13+D_123, D_2+D_12+D_23+D_123, D_3+D_13+D_23+D_123])/variance
    total_effects = 1 - \
        np.array([D_2+D_3+D_23, D_1+D_3+D_13, D_1+D_2+D_12])/variance
    assert np.allclose(total_effects1, total_effects)
    sobol_indices = np.array([D_1, D_2, D_3, D_12, D_13, D_23, D_123])/variance
    sobol_interaction_indices = [[0], [1], [
        2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    return mean, variance, main_effects[:, np.newaxis], total_effects[:, np.newaxis], sobol_indices[:, np.newaxis], sobol_interaction_indices


def sobol_g_function(coefficients, samples):
    """
    The coefficients control the sensitivity of each variable. Specifically
    they limit the range of the outputs, i.e.
    1-1/(1+a_i) <= (abs(4*x-2)+a_i)/(a_i+1) <= 1-1/(1+a_i)
    """
    nvars, nsamples = samples.shape
    assert coefficients.shape[0] == nvars
    print(samples.min(axis=1), samples.max(axis=1))
    vals = np.prod((np.absolute(4*samples-2)+coefficients[:, np.newaxis]) /
                   (1+coefficients[:, np.newaxis]), axis=0)[:, np.newaxis]
    assert vals.shape[0] == nsamples
    return vals


def get_sobol_g_function_statistics(a, interaction_terms=None):
    """
    See article: Variance based sensitivity analysis of model output. 
    Design and estimator for the total sensitivity index
    """
    nvars = a.shape[0]
    mean = 1
    unnormalized_main_effects = 1/(3*(1+a)**2)
    variance = np.prod(unnormalized_main_effects+1)-1
    main_effects = unnormalized_main_effects/variance
    total_effects = np.tile(np.prod(unnormalized_main_effects+1), (nvars))
    total_effects *= unnormalized_main_effects/(unnormalized_main_effects+1)
    total_effects /= variance
    if interaction_terms is None:
        return mean, variance, main_effects, total_effects

    sobol_indices = np.array([
        unnormalized_main_effects[index].prod()/variance
        for index in interaction_terms])
    return mean, variance, main_effects[:, np.newaxis], total_effects[:, np.newaxis], sobol_indices[:, np.newaxis]


def morris_function(samples):
    assert samples.shape[0] == 20
    beta0 = np.random.randn()
    beta_first_order = np.empty(20)
    beta_first_order[:10] = 20
    beta_first_order[10:] = np.random.normal(0, 1, 10)
    beta_second_order = np.empty((20, 20))
    beta_second_order[:6, :6] = -15
    beta_second_order[6:, 6:] = np.random.normal(0, 1, (14, 14))
    #beta_third_order = np.zeros((20,20,20))
    # beta_third_order[:5,:5,:5]=-10
    beta_third_order = -10
    #beta_forth_order = np.zeros((20,20,20,20))
    # beta_forth_order[:4,:4,:4,:4]=5
    beta_forth_order = 5
    ww = 2*(samples-0.5)
    I = [3, 5, 7]
    ww[I] = 2 * (1.1 * samples[I]/(samples[I]+0.1)-0.5)

    values = beta0
    values += np.sum(beta_first_order[:, np.newaxis]*ww, axis=0)

    for jj in range(20):
        for ii in range(jj):
            values += beta_second_order[ii, jj]*ww[ii]*ww[jj]

    for kk in range(5):
        for jj in range(kk):
            for ii in range(jj):
                values += beta_third_order*ww[ii]*ww[jj]*ww[kk]

    for ll in range(4):
        for kk in range(ll):
            for jj in range(kk):
                for ii in range(jj):
                    values += beta_forth_order*ww[ii]*ww[jj]*ww[kk]*ww[ll]
    return values[:, np.newaxis]

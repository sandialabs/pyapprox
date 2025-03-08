import os
import numpy as np
from scipy.linalg import eigh as generalized_eigevalue_decomp

from pyapprox.util.randomized_svd import randomized_svd
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.variables.joint import MultivariateGaussian


class DenseMatrixLaplacePosteriorApproximation:
    def __init__(
        self,
        matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_cov: Array,
        vec: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        r"""
        Compute the mean and covariance of the Laplace posterior of a
        linear (or linearized) model with a Gaussian prior and noise model.

        Given some data d and a linear forward model, A(x) = Ax+b,
        and a Gaussian likelihood and a Gaussian prior, the resulting posterior
        is always Gaussian.

        Parameters
        ----------
        matrix : Array (num_qoi, nvars)
            The matrix reprsenting the linear forward model.

        prior_mean : Array (nvars, 1)
            The mean of the Gaussian prior

        prior_covariance: Array (nvars, nvars)
            The covarianceof the Gaussian prior

        noise_covariancev : Array (num_qoi, num_qoi)
            The covariance of the observational noise

        obs : Array (num_qoi, 1)
            The observations

        vec : Array (num_qoi, 1)
            The deterministic shift of the linear model
        """
        self._bkd = backend
        self._nobs, self._nvars = matrix.shape
        self._matrix = matrix
        if prior_mean.shape != (self.nvars(), 1):
            raise ValueError("prior_mean has the wrong shape")
        self._prior_mean = prior_mean
        if prior_cov.shape != (self.nvars(), self.nvars()):
            raise ValueError("prior_cov has the wrong shape")
        self._prior_cov = prior_cov
        if noise_cov.shape != (self.nobs(), self.nobs()):
            raise ValueError("noise_cov has the wrong shape")
        self._noise_cov = noise_cov
        if vec is None:
            vec = self._bkd.zeros((self._nobs, 1))
        if vec.shape != (self.nobs(), 1):
            raise ValueError("vec has the wrong shape")
        self._vec = vec

        self._noise_cov_inv = self._bkd.inv(self._noise_cov)
        self._prior_hessian = self._bkd.inv(self._prior_cov)
        model = DenseMatrixLinearModel(
            self._matrix, self._vec, backend=self._bkd
        )
        self._loglike = ModelBasedGaussianLogLikelihood(model, self._noise_cov)
        self._prior = MultivariateGaussian(
            self._prior_mean, self._prior_cov, self._bkd
        )

    def _set_observations(self, obs: Array):
        if obs.shape != (self.nobs(), 1):
            raise ValueError("obs has the wrong shape")
        self._obs = obs

    def nvars(self) -> int:
        return self._nvars

    def nobs(self) -> int:
        return self._nobs

    def compute(self, obs: Array):
        self._set_observations(obs)
        misfit_hessian = self._matrix.T @ self._noise_cov_inv @ self._matrix
        self._posterior_cov = self._bkd.inv(
            misfit_hessian + self._prior_hessian
        )
        residual = self._obs - self._matrix @ self._prior_mean - self._vec
        temp = self._matrix.T @ (self._noise_cov_inv @ residual)
        self._posterior_mean = self._prior_mean + self._posterior_cov @ temp
        self._compute_evidence()
        self._compute_expected_posterior_statistics()
        self._compute_expected_kl_divergence()

    def _compute_evidence(self) -> Array:
        """
        References
        ----------
        Ryan, K. (2003). Estimating Expected Information Gains for Experimental
        Designs with Application to the Random Fatigue-Limit Model. Journal of
        Computational and Graphical Statistics, 12(3), 585-603.
        http://www.jstor.org/stable/1391040

        Friel, N. and Wyse, J. (2012), Estimating the evidence – a review.
        Statistica Neerlandica, 66: 288-308.
        https://doi.org/10.1111/j.1467-9574.2011.00515.x
        """
        self._loglike.set_observations(self._obs)
        lval = self._bkd.exp(self._loglike(self._posterior_mean))[:, 0]
        prior_val = self._prior.pdf(self._posterior_mean)
        assert lval.ndim == 1
        assert prior_val.ndim == 2
        self._evidence = (
            (2 * np.pi) ** (self._nvars / 2)
            * self._bkd.sqrt(self._bkd.det(self.posterior_covariance()))
            * lval[0]
            * prior_val[0, 0]
        )

    def posterior_mean(self) -> Array:
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_mean

    def posterior_covariance(self) -> Array:
        if not hasattr(self, "_posterior_mean"):
            raise RuntimeError("must first call compute()")
        return self._posterior_cov

    def evidence(self) -> Array:
        return self._evidence

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def posterior_variable(self) -> MultivariateGaussian:
        return MultivariateGaussian(
            self.posterior_mean(),
            self.posterior_covariance(),
            backend=self._bkd,
        )

    def _compute_expected_posterior_statistics(self):
        """
        Compute the mean and variance of the posterior mean with respect to
        uncertainty in the observation data. The posterior mean is a
        Gaussian variable
        """
        Rmat = np.linalg.multi_dot(
            (self.posterior_covariance(), self._matrix.T, self._noise_cov_inv)
        )
        ROmat = Rmat.dot(self._matrix)
        self._nu_vec = (ROmat @ self._prior_mean) + self._bkd.multidot(
            (
                self.posterior_covariance(),
                self._prior_hessian,
                self._prior_mean,
            )
        )
        self._Cmat = self._bkd.multidot(
            (ROmat, self._prior_cov, ROmat.T)
        ) + np.linalg.multi_dot((Rmat, self._noise_cov, Rmat.T))

    def _compute_expected_kl_divergence(self):
        """
        Compute the expected KL divergence between a Gaussian posterior
        and prior, where average is taken with respect to the data
        """
        kl_div = (
            self._bkd.trace(self._prior_hessian @ self.posterior_covariance())
            - self.nvars()
        )
        kl_div += self._bkd.log(
            self._bkd.det(self._prior_cov)
            / self._bkd.det(self.posterior_covariance())
        )
        kl_div += self._bkd.trace(self._prior_hessian @ self._Cmat)
        xi = self._prior_mean - self._nu_vec
        kl_div += self._bkd.multidot((xi.T, self._prior_hessian, xi))
        kl_div *= 0.5
        self._kl_div = kl_div[0, 0]

    def expected_kl_divergence(self) -> float:
        return self._kl_div


class GaussianPushForward:
    def __init__(
        self,
        matrix: Array,
        mean: Array,
        cov: Array,
        vec: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        r"""
        Find the mean and covariance of a gaussian distribution when it
        is push forward through a linear model. A linear transformation
        applied to a Gaussian is still a Gaussian.

        Original Gaussian with mean x and covariance \Sigma
        z~N(x,\Sigma)

        Transformation with b is a constant vector, e.g has no variance
        y = Az + b

        Distribution of resulting gaussian
        y~N(Ax+b,A\Sigma A^T)
        """
        self._bkd = backend
        self._nqoi, self._nvars = matrix.shape
        self._mat = matrix
        if mean.shape != (self.nvars(), 1):
            raise ValueError("mean has the wrong shape")
        self._mean = mean
        if cov.shape != (self.nvars(), self.nvars()):
            raise ValueError("cov has the wrong shape")
        self._cov = cov
        if vec is None:
            vec = self._bkd.zeros((self.nqoi(), 1))
        if vec.shape != (self.nqoi(), 1):
            raise ValueError("vec has the wrong shape")
        self._vec = vec
        self._compute()

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def _compute(self) -> Array:
        self._pushforward_mean = self._mat @ self._mean + self._vec
        self._pushforward_cov = self._mat @ self._cov @ self._mat.T

    def mean(self) -> Array:
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_mean

    def covariance(self) -> Array:
        if not hasattr(self, "_pushforward_mean"):
            raise RuntimeError("must first call compute()")
        return self._pushforward_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> MultivariateGaussian:
        return MultivariateGaussian(self.mean(), self.covariance())


class PriorConditionedHessianMatVecOperator(object):
    r"""
    Compute the action of prior conditioned misfit Hessian on a vector.

    E.g. for a arbitrary vector w, the Cholesky factor L of the prior
    and the misfit Hessian H compute
        L*H*L'*w
    """

    def __init__(
        self, prior_covariance_sqrt_operator, misfit_hessian_operator
    ):
        self.prior_covariance_sqrt_operator = prior_covariance_sqrt_operator
        self.misfit_hessian_operator = misfit_hessian_operator

    def apply(self, vectors, transpose=None):
        r"""
        Compute L'*H*L*w.

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        transpose : boolean (default=True)
            The prior-conditioned Hessian is Symmetric so transpose does
            not matter. But randomized svd  assumes operator has a function
            apply(x, transpose)

        Returns
        -------
        z : (num_dims,num_vectors) matrix
            The matrix vector products: L'*H*L*w
        """
        x = self.prior_covariance_sqrt_operator.apply(vectors, transpose=False)
        assert (
            x.shape[1] == vectors.shape[1]
        ), "prior_covariance_sqrt_operator is returning incorrect values"
        y = self.misfit_hessian_operator.apply(x)
        assert (
            y.shape[1] == x.shape[1]
        ), "misfit_hessian_operator is returning incorrect values"
        z = self.prior_covariance_sqrt_operator.apply(y, transpose=True)
        return z

    def num_rows(self):
        return self.prior_covariance_sqrt_operator.num_vars()

    def num_cols(self):
        return self.prior_covariance_sqrt_operator.num_vars()


class LaplaceSqrtMatVecOperator(object):
    r"""
    Compute the action of the sqrt of the covariance of a Laplace
    approximation of the posterior on a vector.

    E.g. for a arbtirary vector w, the Cholesky factor L of the prior,
    the low rank eigenvalues e_r and eigenvectors V_r of the misfit hessian,
    and the misfit Hessian H compute
        L*(V*D*V'+I)*w
    where D = diag(np.sqrt(1./(e_r+1.))-1)
    """

    def __init__(
        self,
        prior_covariance_sqrt_operator,
        e_r=None,
        V_r=None,
        M=None,
        filename=None,
    ):
        r"""
        Parameters
        ----------
        e_r : (rank,1) vector
            The r largest eigenvalues of the prior conditioned misfit hessian

        V_r : (num_dims,rank) matrix
            The eigenvectors corresponding to the r-largest eigenvalues

        M : (num_dims) vector (default=None)
            Weights defineing a weighted inner product

        filename : string
            The name of the file that contains data to initialize object.
            e_r, V_r, and M will be ignored.
        """
        self.prior_covariance_sqrt_operator = prior_covariance_sqrt_operator
        if filename is not None:
            assert V_r is None and e_r is None and M is None
            self.load(filename)
        else:
            assert V_r is not None and e_r is not None
            self.V_r = V_r
            self.M = M
            self.set_eigenvalues(e_r)

    def num_vars(self):
        return self.prior_covariance_sqrt_operator.num_vars()

    def set_eigenvalues(self, e_r):
        self.diagonal = np.sqrt(1.0 / (e_r + 1.0)) - 1
        self.e_r = e_r

    def save(self, filename):
        if self.M is not None:
            np.savez(filename, e_r=self.e_r, V_r=self.V_r, M=self.M)
        else:
            # savez cannot save python None
            np.savez(filename, e_r=self.e_r, V_r=self.V_r)

    def load(self, filename):
        if not os.path.exists(filename):
            raise Exception("file %s does not exist" % filename)
        data = np.load(filename)
        self.V_r = data["V_r"]
        if "M" in list(data.keys()):
            self.M = data["M"]
        else:
            self.M = None
        self.set_eigenvalues(data["e_r"])

    def apply_mass_weighted_eigvec_adjoint(self, vectors):
        r"""
        Apply the mass weighted adjoint of the eigenvectors V_r to a set
        of vectors w. I.e. compute
           x = V_r^T*M*w

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        Returns
        -------
        x : (rank,num_vectors) matrix
            The matrix vector products: V'M*w
        """
        if self.M is not None:
            print((self.M, "a", type(self.M)))
            assert self.M.ndim == 1 and self.M.shape[0] == vectors.shape[0]
            for i in range(vectors.shape[0]):
                vectors[i, :] *= self.M[i]
        # else: M is the identity so do nothing
        return np.dot(self.V_r.T, vectors)

    def apply(self, vectors, transpose=False):
        r"""
        Compute L*(V*D*V'+I)*w = L(V*D*V'*w+w)

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        transpose : boolean
            True - apply L.T
            False - apply L

        Returns
        -------
        z : (num_dims,num_vectors) matrix
            The matrix vector products: L*(V*D*V'+I)*w
        """
        if transpose:
            vectors = self.prior_covariance_sqrt_operator.apply(
                vectors, transpose=True
            )
        # x = V'*vectors
        x = self.apply_mass_weighted_eigvec_adjoint(vectors)
        # y = D*x
        y = x * self.diagonal[:, np.newaxis]
        # z = V*y
        z = np.dot(self.V_r, y)
        z += vectors
        if not transpose:
            z = self.prior_covariance_sqrt_operator.apply(z, transpose=False)
        return z

    def __call__(self, vectors, transpose=False):
        return self.apply(vectors, transpose)


def get_laplace_covariance_sqrt_operator(
    prior_covariance_sqrt_operator,
    misfit_hessian_operator,
    svd_opts,
    weights=None,
    min_singular_value=0.1,
):
    r"""
    Get the operator representing the action of the cholesky factorization
    of the Laplace posterior approximation on a vector.

    Parameters
    ----------
    prior_covariance_sqrt_operator : Matrix vector multiplication operator
        The operator representing the action of the sqrt of the prior
        covariance on a vector. This is often but not always the cholesky
        factorization oft the prior covariance.

    misfit_hessian_operator : Matrix vector multiplication operator
        The operator representing the action of the misfit hessian on a vector

    svd_opts : dictionary
       The options to the SVD algorithm. See documentation of randomized_svd().

    weights : (num_dims) vector (default=None)
        Weights defineing a weighted inner product

    min_singular_value : double (default=0.1)
       The minimum singular value to retain in SVD. Note
       This can be different from the entry 'min_singular_value' in svd_opts

    Returns
    -------
    covariance_sqrt_operator :  Matrix vector multiplication operator
        The action of the sqrt of the covariance on a vector
    """
    e_r, V_r = get_low_rank_prior_conditioned_misfit_hessian(
        prior_covariance_sqrt_operator,
        misfit_hessian_operator,
        svd_opts,
        min_singular_value,
    )

    operator = LaplaceSqrtMatVecOperator(
        prior_covariance_sqrt_operator, e_r, V_r, weights
    )

    return operator


def get_low_rank_prior_conditioned_misfit_hessian(
    prior_covariance_sqrt_operator,
    misfit_hessian_operator,
    svd_opts,
    min_singular_value=0.1,
):
    r"""
    Get the low rank approximation of the prior conditioned misfit hessian
    using only matrix vector multiplication operators.

    Parameters
    ----------
    prior_covariance_sqrt_operator : Matrix vector multiplication operator
        The operator representing the action of the sqrt of the prior
        covariance on a vector. This is often but not always the cholesky
        factorization oft the prior covariance.

    misfit_hessian_operator : Matrix vector multiplication operator
        The operator representing the action of the misfit hessian on a vector

    min_singular_value : double (default=0.1)
       The minimum singular value to retain in SVD. Note
       This can be different from the entry 'min_singular_value' in svd_opts

    svd_opts : dictionary
       The options to the SVD algorithm. See documentation of randomized_svd().


    Returns
    -------
    e_r : (rank,1) vector
        The r largest eigenvalues of the prior conditioned misfit hessian
    V_r : (num_dims,rank) matrix
        The eigenvectors corresponding to the r-largest eigenvalues
    """

    operator = PriorConditionedHessianMatVecOperator(
        prior_covariance_sqrt_operator, misfit_hessian_operator
    )

    svd_opts["single_pass"] = True
    U, S, V = randomized_svd(operator, svd_opts)
    I = np.where(S >= min_singular_value)[0]
    e_r = S[I]
    V_r = U[:, I]
    return e_r, V_r


def get_pointwise_laplace_variance(prior, laplace_covariance_sqrt):
    prior_pointwise_variance = prior.pointwise_variance()
    return get_pointwise_laplace_variance_using_prior_variance(
        prior, laplace_covariance_sqrt, prior_pointwise_variance
    )


def get_pointwise_laplace_variance_using_prior_variance(
    prior, laplace_covariance_sqrt, prior_pointwise_variance
):
    # compute L*V_r
    tmp1 = prior.apply_covariance_sqrt(laplace_covariance_sqrt.V_r, False)
    # compute D*(L*V_r)**2
    tmp2 = laplace_covariance_sqrt.e_r / (1.0 + laplace_covariance_sqrt.e_r)
    tmp3 = np.sum(tmp1**2 * tmp2, axis=1)
    return prior_pointwise_variance - tmp3, prior_pointwise_variance


def generate_and_save_laplace_posterior(
    prior,
    misfit_model,
    num_singular_values,
    svd_history_filename="svd-history.npz",
    Lpost_op_filename="laplace_sqrt_operator.npz",
    num_extra_svd_samples=10,
    fd_eps=2 * np.sqrt(np.finfo(float).eps),
):

    if os.path.exists(svd_history_filename):
        raise Exception(
            "File %s already exists. Exiting so as not to overwrite"
            % svd_history_filename
        )
    if os.path.exists(Lpost_op_filename):
        raise Exception(
            "File %s already exists. Exiting so as not to overwrite"
            % Lpost_op_filename
        )

    sample = misfit_model.map_point()
    misfit_hessian_operator = MisfitHessianVecOperator(
        misfit_model, sample, fd_eps=fd_eps
    )
    standard_svd_opts = {
        "num_singular_values": num_singular_values,
        "num_extra_samples": num_extra_svd_samples,
    }
    svd_opts = {
        "single_pass": True,
        "standard_opts": standard_svd_opts,
        "history_filename": svd_history_filename,
    }
    L_post_op = get_laplace_covariance_sqrt_operator(
        prior.sqrt_covariance_operator,
        misfit_hessian_operator,
        svd_opts,
        weights=None,
        min_singular_value=0.0,
    )

    L_post_op.save(Lpost_op_filename)
    return L_post_op


def generate_and_save_pointwise_variance(
    prior,
    L_post_op,
    prior_variance_filename="prior_pointwise-variance.npz",
    posterior_variance_filename="posterior_pointwise-variance.npz",
):
    if not os.path.exists(prior_variance_filename):
        posterior_pointwise_variance, prior_pointwise_variance = (
            get_pointwise_laplace_variance(prior, L_post_op)
        )
        np.savez(
            prior_variance_filename,
            prior_pointwise_variance=prior_pointwise_variance,
        )
        np.savez(
            posterior_variance_filename,
            posterior_pointwise_variance=posterior_pointwise_variance,
        )
    else:
        print(
            ("File %s already exists. Loading data" % prior_variance_filename)
        )
        prior_pointwise_variance = np.load(prior_variance_filename)[
            "prior_pointwise_variance"
        ]
        if not os.path.exists(posterior_variance_filename):
            posterior_pointwise_variance, prior_pointwise_variance = (
                get_pointwise_laplace_variance_using_prior_variance(
                    prior, L_post_op, prior_pointwise_variance
                )
            )
            np.savez(
                posterior_variance_filename,
                posterior_pointwise_variance=posterior_pointwise_variance,
            )
        else:
            posterior_pointwise_variance = np.load(
                posterior_variance_filename
            )["posterior_pointwise_variance"]
    return prior_pointwise_variance, posterior_pointwise_variance


class DenseMatrixLaplaceApproximationForPrediction:
    def __init__(
        self,
        obs_matrix: Array,
        pred_matrix: Array,
        prior_mean: Array,
        prior_cov: Array,
        obs_noise_cov: Array,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self._obs_matrix = obs_matrix
        self._pred_matrix = pred_matrix
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._obs_noise_cov = obs_noise_cov

    def compute(self, obs: Array):
        # step 1
        OP = self._pred_matrix @ self._prior_cov
        # step 2
        C = OP @ self._obs_matrix.T
        # step 3
        Pz = OP @ self._pred_matrix.T
        # step 4
        Pz_inv = self._bkd.inv(Pz)
        # step 5
        A = C.T @ Pz_inv @ C
        # step 6
        data_cov = (
            self._obs_matrix @ self._prior_cov @ self._obs_matrix.T
            + self._obs_noise_cov
        )
        # step 7
        # print 'TODO replace generalized_eigevalue_decomp by my
        # subspace iteration'
        evals, evecs = generalized_eigevalue_decomp(A, data_cov)
        evecs = evecs[:, ::-1]
        evals = evals[::-1]
        rank = min(self._pred_matrix.shape[0], self._obs_matrix.shape[0])
        evecs = evecs[:, :rank]
        evals = evals[:rank]
        # step 8
        ppf_cov_evecs = C @ evecs

        residual = obs - self._obs_matrix @ self._prior_mean
        self._opt_pf_cov = Pz - ppf_cov_evecs @ ppf_cov_evecs.T
        self._opt_pf_mean = (ppf_cov_evecs @ (evecs.T @ residual)) + (
            self._pred_matrix @ self._prior_mean
        )

    def mean(self) -> Array:
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_mean

    def covariance(self) -> Array:
        if not hasattr(self, "_opt_pf_mean"):
            raise RuntimeError("must first call compute()")
        return self._opt_pf_cov

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def pushfowward_variable(self) -> MultivariateGaussian:
        return MultivariateGaussian(self.mean(), self.covariance())

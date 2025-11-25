


class IndependentMarginalsVariable:
    def __init__(
        self,
        univariate_marginals: Sequence[RandomVariableProtocol],
    ):
        self._validate_univariate_marginals(univariate_marginals)
        self._bkd = univariate_marginals[0]._bkd
        self._univariate_marginals = univariate_marginals

    def nvars(self) -> int:
        return len(self._univariate_marginals)

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability distribution function.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Values in the domain of the random variable X

        Returns
        -------
        values : np.ndarray (nsamples, 1)
            The values of the PDF at x
        """
        self._check_samples(samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.pdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.prod(marginal_vals, axis=0)[:, None]

    def logpdf(self, samples: Array, log: bool = False) -> Array:
        self._check_samples(samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.logpdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.sum(marginal_vals, axis=0)[:, None]

    def __repr__(self) -> str:
        pass

    def rvs(self, nsamples: int) -> Array:
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        nsamples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (nvars, nsamples)
            Independent samples from the target distribution
        """
        marginal_samples = [
            marginal.rvs(nsamples) for marginal in self.marginals()
        ]
        return self._bkd.stack(marginal_samples, axis=0)

    def pdf_jacobian(self, samples: Array) -> Array:
        pdf_vals = self._bkd.stack(
            [
                marginal.pdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.hstack(
            [
                marginal.pdf_jacobian(samples[ii])
                * self._bkd.prod(pdf_vals[:ii])
                * self._bkd.prod(pdf_vals[ii + 1 :])
                for ii, marginal in enumerate(self.marginals())
            ]
        )

    def logpdf_jacobian(self, samples: Array) -> Array:
        return self._bkd.hstack(
            [
                marginal.logpdf_jacobian(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ]
        )

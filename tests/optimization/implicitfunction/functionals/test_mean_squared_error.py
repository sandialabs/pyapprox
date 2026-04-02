import pytest

from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)


class TestMSEFunctional:
    """
    Base test class for MSEFunctional.
    """

    def _make_obs(self, bkd):
        nstates = 3
        nparams = 2
        obs = bkd.reshape(bkd.array([1.0, 2.0, 3.0]), (3, 1))
        return nstates, nparams, obs

    def test_initialization(self, bkd) -> None:
        """
        Test the initialization of MSEFunctional.
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        assert func.nstates() == nstates
        assert func.nparams() == nparams
        assert func.nqoi() == 1  # MSE always returns scalar
        assert func.nunique_params() == 0
        assert func.bkd() is not None

    def test_set_observations_valid(self, bkd) -> None:
        """
        Test setting observations with valid shape.
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)
        # No exception should be raised

    def test_set_observations_invalid_shape(self, bkd) -> None:
        """
        Test that set_observations raises ValueError for invalid shape.
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        bad_obs = bkd.array([1.0, 2.0])  # Wrong shape
        with pytest.raises(ValueError):
            func.set_observations(bad_obs)

    def test_functional_evaluation(self, bkd) -> None:
        """
        Test the functional evaluation (__call__).

        MSE = sum((obs - state)^2) / 2
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.reshape(bkd.array([2.0, 3.0, 4.0]), (3, 1))
        param = bkd.zeros((2, 1))  # Not used in MSE

        result = func(state, param)

        # Expected: sum((obs - state)^2) / 2
        # = sum(([1, 2, 3] - [2, 3, 4])^2) / 2
        # = sum([-1, -1, -1]^2) / 2 = sum([1, 1, 1]) / 2 = 3 / 2 = 1.5
        expected = bkd.reshape(bkd.array([1.5]), (1, 1))
        bkd.assert_allclose(result, expected)

    def test_state_jacobian(self, bkd) -> None:
        """
        Test the state Jacobian computation.

        For MSE: dJ/dstate = (state - obs)^T
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.reshape(bkd.array([2.0, 3.0, 5.0]), (3, 1))
        param = bkd.zeros((2, 1))

        jac = func.state_jacobian(state, param)

        # Expected: (state - obs)^T = ([2, 3, 5] - [1, 2, 3])^T = [1, 1, 2]^T
        expected = bkd.reshape(bkd.array([[1.0, 1.0, 2.0]]), (1, 3))
        bkd.assert_allclose(jac, expected)

    def test_param_jacobian(self, bkd) -> None:
        """
        Test the parameter Jacobian computation.

        For MSE with respect to parameters: should be zeros
        (MSE doesn't depend on parameters).
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.zeros((3, 1))
        param = bkd.zeros((2, 1))

        jac = func.param_jacobian(state, param)

        # Expected: zeros (MSE doesn't depend on params)
        expected = bkd.zeros((1, nparams))
        bkd.assert_allclose(jac, expected)

    def test_state_state_hvp(self, bkd) -> None:
        """
        Test state-state Hessian-vector product.

        For MSE: d^2J/dstate^2 = I (identity), so HVP returns the input vector.
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.zeros((3, 1))
        param = bkd.zeros((2, 1))
        vvec = bkd.reshape(bkd.array([1.0, 2.0, 3.0]), (3, 1))

        hvp = func.state_state_hvp(state, param, vvec)

        # Expected: identity matrix times vector = vector itself
        bkd.assert_allclose(hvp, vvec)

    def test_param_param_hvp(self, bkd) -> None:
        """
        Test parameter-parameter Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.zeros((3, 1))
        param = bkd.zeros((2, 1))
        vvec = bkd.ones((2, 1))

        hvp = func.param_param_hvp(state, param, vvec)

        # Expected: zeros
        expected = bkd.zeros((nparams, 1))
        bkd.assert_allclose(hvp, expected)

    def test_param_state_hvp(self, bkd) -> None:
        """
        Test parameter-state Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.zeros((3, 1))
        param = bkd.zeros((2, 1))
        vvec = bkd.ones((3, 1))

        hvp = func.param_state_hvp(state, param, vvec)

        # Expected: zeros
        expected = bkd.zeros((nparams, 1))
        bkd.assert_allclose(hvp, expected)

    def test_state_param_hvp(self, bkd) -> None:
        """
        Test state-parameter Hessian-vector product.

        For MSE: should be zeros (MSE doesn't depend on parameters).
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        func.set_observations(obs)

        state = bkd.zeros((3, 1))
        param = bkd.zeros((2, 1))
        vvec = bkd.ones((2, 1))

        hvp = func.state_param_hvp(state, param, vvec)

        # Expected: zeros
        expected = bkd.zeros((nstates, 1))
        bkd.assert_allclose(hvp, expected)

    def test_repr(self, bkd) -> None:
        """
        Test the string representation of MSEFunctional.
        """
        nstates, nparams, obs = self._make_obs(bkd)
        func = MSEFunctional(nstates, nparams, bkd)
        repr_str = repr(func)
        assert "MSEFunctional" in repr_str
        assert f"nstates={nstates}" in repr_str
        assert f"nparams={nparams}" in repr_str

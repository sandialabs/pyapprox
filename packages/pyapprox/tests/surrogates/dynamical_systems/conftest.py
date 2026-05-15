"""Shared fixtures for dynamical systems tests."""

import numpy as np
import pytest


@pytest.fixture()
def van_der_pol_data(bkd):
    """Generate Van der Pol derivative-matching data (mu=1.0).

    Returns states (2, nsamples) and derivatives (2, nsamples) from
    the Van der Pol equations:
        x1_dot = x2
        x2_dot = mu*(1 - x1^2)*x2 - x1
    """
    mu = 1.0
    np.random.seed(42)
    nsamples = 200
    x1 = np.random.uniform(-2.0, 2.0, nsamples)
    x2 = np.random.uniform(-2.0, 2.0, nsamples)

    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1

    states = bkd.array(np.stack([x1, x2], axis=0))
    derivatives = bkd.array(np.stack([dx1, dx2], axis=0))
    return states, derivatives, mu


@pytest.fixture()
def lotka_volterra_data(bkd):
    """Generate Lotka-Volterra derivative-matching data.

    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y

    Parameters: alpha=1.5, beta=1.0, delta=1.0, gamma=3.0
    """
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    np.random.seed(43)
    nsamples = 200
    x = np.random.uniform(0.5, 5.0, nsamples)
    y = np.random.uniform(0.5, 5.0, nsamples)

    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    states = bkd.array(np.stack([x, y], axis=0))
    derivatives = bkd.array(np.stack([dx, dy], axis=0))
    params = {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma}
    return states, derivatives, params

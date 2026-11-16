r"""Closed-form responses of 1D diffusion subdomains.

On :math:`[a, b]`, :math:`L = b - a`, with constant source :math:`f`,

.. math::

    -(k(u)\, u')' = f, \qquad u(a) = \alpha, \quad u(b) = \beta,

with conductivity :math:`k(u) = \sum_j c_j u^j`. The Kirchhoff transform
:math:`K(u) = \int_0^u k = \sum_j c_j u^{j+1}/(j+1)` turns the equation
into :math:`-K(u)'' = f`, so :math:`K(u)` is quadratic in :math:`x` and
the outward fluxes :math:`k(u)\, \partial u/\partial n` at the ends are

.. math::

    r_a = -\frac{K(\beta) - K(\alpha)}{L} - \frac{f L}{2}, \qquad
    r_b = \frac{K(\beta) - K(\alpha)}{L} - \frac{f L}{2},

with no interior solve, even for nonlinear :math:`k`. Each end is a port
(its value is port data) or carries a fixed Dirichlet value.

**Exact interface values of a chain.** Subdomains chained over
:math:`[0, 1]`, with :math:`u(0) = g_0`, :math:`u(1) = g_1` and possibly
a different conductivity :math:`k_i` on each piece, have interface values
known independently of any skeleton solve. The flux
:math:`q = -k(u)\, u'` satisfies :math:`q' = f`, so

.. math::

    q(x) = q_0 + f x

across the whole domain, continuous across interfaces (which is the
transmission condition); only :math:`q_0` is unknown, and the end values
fix it.

1. One constant :math:`k`: :math:`u(x) = g_0 + (g_1 - g_0) x +
   f x (1 - x) / (2k)`, and :math:`\lambda_j = u(x_j)` at each interface
   :math:`x_j`; for :math:`g_0 = g_1 = 0`, :math:`u(1/2) = f / (8k)`.
2. One nonlinear :math:`k(u)`: :math:`K(u(x))` is that quadratic with
   end values :math:`K(g_0)` and :math:`K(g_1)`, and
   :math:`\lambda_j = K^{-1}(K(u(x_j)))`, one scalar monotone root
   (:math:`K` is increasing when :math:`k > 0`).
3. A constant :math:`k_i` per piece: there :math:`u' = -(q_0 + f x)/k_i`,
   so :math:`u(x_j) = g_0 - \sum_i \int (q_0 + f x)/k_i \, dx` over the
   pieces left of :math:`x_j` is linear in :math:`q_0`; :math:`u(1) = g_1`
   gives :math:`q_0` in closed form, then every :math:`\lambda_j`.
4. A nonlinear :math:`k_i(u)` per piece: there :math:`K_i(u)' = -q(x)`;
   marching from :math:`x = 0` and inverting each :math:`K_i` at its
   piece's end carries :math:`u` across, making :math:`u(1)` a monotone
   function of :math:`q_0`: one scalar root, then every
   :math:`\lambda_j`.

These references use only :math:`q' = f`, never the responses, port
transfers or skeleton residual, so they check that machinery
independently. With constant :math:`k_i` the skeleton system is linear
and symmetric positive definite (each subdomain adds :math:`+k_i/L_i`
to its ports' Steklov–Poincaré operator), so Newton converges in one
step.
"""

from typing import Generic, Optional, Sequence

from pyapprox.pde.coupling.responses import ResponseDerivatives, TimeWindow
from pyapprox.util.backends.protocols import Array, Backend


class KirchhoffResponse1D(Generic[Array]):
    r"""A 1D diffusion subdomain with polynomial conductivity.

    Satisfies
    :class:`~pyapprox.pde.coupling.responses.LocalResponseProtocol`.
    Steady, without parameters or quantities of interest. Port rows are
    the port ends, the left end first.

    Parameters
    ----------
    length
        The subdomain length :math:`L > 0`.
    source
        The constant source :math:`f`.
    conductivity
        The coefficients :math:`c_0, c_1, \dots` of :math:`k(u)`.
    left, right
        The Dirichlet value at each end, or ``None`` for a port.
    bkd
        Computational backend.
    """

    def __init__(
        self,
        length: float,
        source: float,
        conductivity: Sequence[float],
        left: Optional[float],
        right: Optional[float],
        bkd: Backend[Array],
    ) -> None:
        if length <= 0.0 or len(conductivity) == 0:
            raise ValueError(
                f"length must be positive and conductivity nonempty, got "
                f"{length} and {tuple(conductivity)}"
            )
        if left is not None and right is not None:
            raise ValueError("at least one end must be a port")
        self._length, self._source = length, source
        self._coefs = tuple(conductivity)
        self._ends = (left, right)
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nport(self) -> int:
        return sum(end is None for end in self._ends)

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def kirchhoff(self, u: Array) -> Array:
        """:math:`K(u)`, entrywise."""
        return sum(
            (c * u ** (j + 1) / (j + 1) for j, c in enumerate(self._coefs)),
            self._bkd.zeros(u.shape),
        )

    def conductivity(self, u: Array) -> Array:
        """:math:`k(u) = K'(u)`, entrywise."""
        return sum(
            (c * u**j for j, c in enumerate(self._coefs)), self._bkd.zeros(u.shape)
        )

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> "KirchhoffPoint1D[Array]":
        if tuple(port_data.shape) != (self.nport(), 1):
            raise ValueError(
                f"port_data must have shape ({self.nport()}, 1), got "
                f"{tuple(port_data.shape)}"
            )
        if tuple(params.shape) != (0, 1):
            raise ValueError(
                f"params must have shape (0, 1), got {tuple(params.shape)}"
            )
        if not window.is_steady():
            raise ValueError("KirchhoffResponse1D is steady: the window must be too")
        return KirchhoffPoint1D(self, port_data)

    def _end_values(self, port_data: Array) -> tuple[Array, Array]:
        values, row = [], 0
        for end in self._ends:
            if end is None:
                values.append(port_data[row : row + 1, :])
                row += 1
            else:
                values.append(self._bkd.asarray([[end]]))
        return values[0], values[1]

    def _port_rows(self, at_a: Array, at_b: Array) -> Array:
        """Stack the end rows that are ports."""
        rows = [r for r, end in zip((at_a, at_b), self._ends) if end is None]
        return self._bkd.vstack(rows)

    def _flux(self, port_data: Array) -> Array:
        alpha, beta = self._end_values(port_data)
        slope = (self.kirchhoff(beta) - self.kirchhoff(alpha)) / self._length
        half_source = self._source * self._length / 2.0
        return self._port_rows(-slope - half_source, slope - half_source)

    def _jvp(self, port_data: Array, directions: Array) -> Array:
        if directions.ndim != 2 or directions.shape[0] != self.nport():
            raise ValueError(
                f"directions must have shape ({self.nport()}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        alpha, beta = self._end_values(port_data)
        ncols = directions.shape[1]
        zeros = self._bkd.zeros((1, ncols))
        d_alpha, d_beta, row = zeros, zeros, 0
        if self._ends[0] is None:
            d_alpha, row = directions[0:1, :], 1
        if self._ends[1] is None:
            d_beta = directions[row : row + 1, :]
        d_slope = (
            self.conductivity(beta) * d_beta - self.conductivity(alpha) * d_alpha
        ) / self._length
        return self._port_rows(-d_slope, d_slope)


class KirchhoffPoint1D(Generic[Array]):
    """A :class:`KirchhoffResponse1D` at one set of port data."""

    def __init__(self, response: KirchhoffResponse1D[Array], port_data: Array) -> None:
        self._response = response
        self._port_data = port_data
        self._flux = response._flux(port_data)
        self._derivs = ResponseDerivatives.tangent(self.jvp_trace)

    def flux(self) -> Array:
        return self._flux

    def qoi(self) -> Array:
        return self._response.bkd().zeros((0, 1))

    def jvp_trace(self, directions: Array) -> Array:
        r""":math:`(\partial r/\partial \lambda)` ``directions``, for a
        block of shape (nport, ncols)."""
        return self._response._jvp(self._port_data, directions)

    def derivatives(self) -> ResponseDerivatives[Array]:
        return self._derivs

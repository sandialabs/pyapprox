r"""
Backwards propagation for neural networks
=========================================

Backwards propagation for neural networks is typically derived using two
different notational conventions

Numerator convention
--------------------
The gradient of scalar :math:`y` and matrix :math:`\mat{X}^{s\times t}` using
the numerator layout has the shape of :math:`\mat{X}^\top`, i.e.

.. math::

    \dydx{y}{\mat{X}}=\begin{bmatrix}\
    \dydx{y}{X_{11}} & \cdots &\dydx{y}{X_{s1}}\\
    \vdots & \ddots & \vdots\\
    \dydx{y}{X_{t1}} & \cdots &\dydx{y}{X_{st}
    }\end{bmatrix}\in\reals^{t\times s}

The gradient of a vector :math:`\mat{y}\in\reals^s` with respect to a vector
:math:`\mat{x}\in\reals^t` is

.. math:: \dydx{\mat{y}}{\mat{x}}\in\reals^{s\times t}

Chain Rule
Using numerator convention

.. math:: \dydx{f\circ g\circ h(x)}{x}=\dydx{f}{g}\dydx{g}{h}\dydx{h}{x}

This is not true for the denominator convention (see below)

Denominator convention
----------------------
The gradient of scalar :math:`y` and matrix :math:`\mat{X}^{s\times t}` using
the numerator layout has the shape of :math:`\mat{X}`, i.e.

.. math::

    \dydx{y}{\mat{X}}=\begin{bmatrix}
    \dydx{y}{X_{11}} & \cdots &\dydx{y}{X_{1t}}\\
    \vdots & \ddots & \vdots\\
    \dydx{y}{X_{s1}} & \cdots &\dydx{y}{X_{st}}
    \end{bmatrix}\in\reals^{t\times s}

The gradient of a vector :math:`\mat{y}\in\reals^s` with respect to a vector
:math:`\mat{x}\in\reals^t` is

.. math:: \dydx{\mat{y}}{\mat{x}}\in\reals^{t\times s}

Chain Rule
Using denominator convention

.. math:: \dydx{f\circ g\circ h(x)}{x}=(\dydx{h}{x}\dydx{g}{h}\dydx{f}{g})


Identities
----------
Gradient of :math:`u=Wy` with respect to :math:`y`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let :math:`u=Wy`, :math:`W\in\reals^{N\times M}`, :math:`y\in\reals^M` and use
numerator convention

.. math:: \dydx{u}{y}=W

Let :math:`u=yW`

.. math:: \dydx{u}{y}=W^\top

Proof

.. math::
  y_n&=\sum_{m=1}^M W_{nm}u_m\\
  (\dydx{u}{y})_{ij}&=\dydx{u_i}{y_j}=\dydx{}{y_j}\sum_{m=1}^M W_{im}y_m=
  \sum_{m=1}^M W_{im}\dydx{y_m}{y_j}=W_{ij}

Similar Proof for :math:`u=yW`

Gradient of :math:`u=Wy` with respect to :math:`W`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let :math:`u=Wy`, :math:`u\in\reals^N`, :math:`W\in\reals^{N\times M}`,
:math:`y\in\reals^M` and use numerator convention

.. math::
    :name: eq:identity-dWudW

    \dydx{\mathcal{L}}{W}&=\dydx{\mathcal{L}}{u}\dydx{u}{W}\\
    &=y\dydx{\mathcal{L}}{u}\\


Proof. We want to avoid computing :math:`\dydx{u}{W}\in\reals^{N\times N
\times M}`. First note

.. math::

  \dydx{\mathcal{L}}{W}=
  \begin{bmatrix}
    \dydx{\mathcal{L}}{W_{11}} &  \cdots &\dydx{\mathcal{L}}{W_{1M}}\\
    \vdots & \ddots & \\ \vdots
    \dydx{\mathcal{L}}{W_{N1}} &  \cdots &\dydx{\mathcal{L}}{W_{NM}}
  \end{bmatrix}

.. math::
  u_n&=\sum_{m=1}^M W_{nm}y_m\\
  \dydx{u_n}{W_{ij}}&=\sum_{m=1}^M y_m\dydx{W_{nm}}{W_{ij}},\quad
  \dydx{W_{nm}}{W_{ij}}=\begin{cases}1, & n=i \text{ and } m=j\\
    0, &\text{otherwise}\end{cases}

Thus

.. math::

  \dydx{u}{W_{ij}}&=\begin{cases}y_j, & n=i\\0, &\text{otherwise}\end{cases}\\
  &=[0, \ldots, 0, y_j, 0, \ldots, 0]^\top

Where :math:`i`-th element is only non-zero entry.

.. math::

  \dydx{\mathcal{L}}{W_{ij}}=\dydx{\mathcal{L}}{u}\dydx{u}{W_{ij}}=
  \delta\dydx{u}{W_{ij}}=\delta_iy_j

Where we defined :math:`\delta=\dydx{\mathcal{L}}{u}\in\reals^{1\times N}`
(numerator format) and :math:`\delta=\dydx{\mathcal{L}}{u}\in\reals^{N\times
1}` (denominator format).

The choice of how to iterate over :math:`i,j` is arbitrary. Either the
numerator of denominator format can be used.

Using the numerator layout that corresponds to the layout used by Jacobians, we
have

.. math::

    \left(\dydx{\mathcal{L}}{W}\right)_{ij}=\dydx{\mathcal{L}}{W_{ji}}=y \delta
      \in\reals^{M\times N}

Or using demoninator layout :math:`\tilde{\delta}=\dydx{\mathcal{L}}{u}\in
\reals^{N\times 1}`

.. math::

    \left(\dydx{\mathcal{L}}{W}\right)_{ij}=\dydx{\mathcal{L}}{W_{ij}}=
    \hat{\delta} y^\top=\delta^\top y^\top \in\reals^{N\times M}


Forward propagation (numerator convention)
------------------------------------------
Forward pass (let :math:`\V{1}_S^\top = [1, 1, \ldots, 1]\in\reals^{1
\times S}`)

.. math::
  y_0&=x & x\in \reals^{N_0\times S}\\
  u_1 &= W_1y_0+b_1\V{1}_S^\top & u_1\in \reals^{N_1\times S}, W_1\in\reals^{
    N_1\times N_0}\\
  y_1 &= \sigma(u_1) & y_1\in \reals^{N_1\times S}\\
  u_2 &= W_2y_1+b_2\V{1}_S^\top & u_2\in \reals^{N_2\times S}, W_2\in\reals^{
    N_2\times N_1}\\
  y_2 &= u_2 & y_2\in \reals^{N_2\times S}\\
  l&=\mathcal{L}(y_2)=(2S)^{-1}\sum_{s=1}^{S}(y_2^{(s)}-d^{(s)})^\top(y_2^{(s)}
    -d^{(s)}) & l\in\reals

Could also use :math:`W_{l}y_{l-1}` when just considering one sample, but to
vectorize it is easier to use :math:`y_{l-1}W_{l}`.

Note the l2 loss can also be written as

.. math:: l=(2S)^{-1}\text{Trace}\left[(y_2-d){(y_2-d)}^\top\right]


Backward propagation (numerator convention)
-------------------------------------------

.. math:: \dydx{\mathcal{L}}{y_2} = S^{-1}(y_2-d)^\top\in\reals^{S\times N_2}

When no activation funcation applied to final layer

.. math::
  \delta_2=\dydx{\mathcal{L}}{u_2}=\dydx{\mathcal{L}}{y_2}\in\reals^{S
  \times N_2}

.. math::

  \dydx{\mathcal{L}}{W_2}&=\dydx{\mathcal{L}}{y_2}\dydx{y_2}{u_2}\dydx{u_2}{W_2}\\
  &=\delta_2 \dydx{u_2}{W_2}\\
  &=y_1 \delta_2 \in \reals^{N_1\times N_2}


where we used :ref:`Equation (1) <eq:identity-dWudW>`.

If an activation function is used on the final output then :math:`\delta_2=
\dydx{\mathcal{L}}{u_2}` but :math:`\delta_2\neq \dydx{\mathcal{L}}{y_2}`.

.. math::

  \dydx{\mathcal{L}}{b_2}&=\dydx{\mathcal{L}}{y_2}\dydx{y_2}{u_2}\dydx{u_2}{b_2}\\
  &=\delta_2 \dydx{y_2}{b_2} \\
  &=\V{1}_S^\top \delta_2 \in\reals^{1\times N_2}

where again we used :ref:`Equation (1) <eq:identity-dWudW>` while setting
:math:`W=b` and :math:`u=\V{1}_S^\top`.

.. math::

  \delta_1 = \dydx{\mathcal{L}}{u_1} &= \dydx{\mathcal{L}}{u_2}\dydx{u_2}{y_1}
    \dydx{y_1}{u_1}\\
  &= \left(\delta_2 W_1 \right)\circ [\sigma^\prime(u_1)]^\top \in\reals^{S
    \times N_1}

The transpose results from using the numerator convention.

Using the arguments applied to the final layer we have for the last hidden
layer

.. math::

  \dydx{\mathcal{L}}{W_1}&=\dydx{\mathcal{L}}{u_1}\dydx{u_1}{W_1}\\
  &=\delta_1\dydx{u_1}{W_1}\\
  &=y_0\delta_1\in\reals^{N_0\times N_1}


.. math::

  \dydx{\mathcal{L}}{b_1}&=\dydx{\mathcal{L}}{u_1}\dydx{u_1}{b_1}\\
  &=\delta_1\dydx{u_1}{b_1}\\
  &=\V{1}_S^\top\delta_1



"""

# Gaussian Process Hessian-Vector Product Implementation

## Mathematical Foundation

### Problem Statement

For a Gaussian process posterior mean:

$$\mu_*(x) = m(x) + k(x, X) \alpha$$

where $\alpha = [K + \sigma^2 I]^{-1}(y - m(X))$, compute the Hessian-vector product:

$$\text{HVP}(x, V) = H_x[\mu_*(x)] \cdot V$$

where $H_x$ is the Hessian matrix $\nabla_x^2 \mu_*(x) \in \mathbb{R}^{d \times d}$ and $V \in \mathbb{R}^d$ is a direction vector.

### Kernel Hessian for Radial Kernels

For radial kernels $k(x, x') = \varphi(r)$ where $r = \|x - x'\|_\ell = \sqrt{\sum_{j=1}^d \frac{(x_j - x'_j)^2}{\ell_j^2}}$, the chain rule gives:

$$\frac{\partial k}{\partial x_j} = \varphi'(r) \frac{\partial r}{\partial x_j}$$

$$\frac{\partial^2 k}{\partial x_j \partial x_k} = \varphi''(r) \frac{\partial r}{\partial x_j} \frac{\partial r}{\partial x_k} + \varphi'(r) \frac{\partial^2 r}{\partial x_j \partial x_k}$$

### Geometry Derivatives (Anisotropic Length Scales)

The scaled distance derivatives are:

$$\frac{\partial r}{\partial x_j} = \frac{\Delta x_j}{r \ell_j^2}, \quad \Delta x_j = x_j - x'_j$$

$$\frac{\partial^2 r}{\partial x_j \partial x_k} = -\frac{\Delta x_j \Delta x_k}{r^3 \ell_j^2 \ell_k^2} + \frac{\delta_{jk}}{r \ell_j^2}$$

### Hessian-Vector Product Formula

The HVP $H[k(x, x')] \cdot V$ for a single pair $(x, x')$ is:

$$H \cdot V = \varphi''(r) \left(\nabla_x r\right) \left(\nabla_x r \cdot V\right) + \varphi'(r) \left(\nabla_x^2 r \cdot V\right)$$

where:
- $\nabla_x r = \left[\frac{\partial r}{\partial x_1}, \ldots, \frac{\partial r}{\partial x_d}\right]^T \in \mathbb{R}^d$
- $\nabla_x r \cdot V = \sum_{j=1}^d \frac{\partial r}{\partial x_j} V_j \in \mathbb{R}$ (scalar)
- $\nabla_x^2 r \cdot V = \left[\sum_{k=1}^d \frac{\partial^2 r}{\partial x_j \partial x_k} V_k\right]_{j=1}^d \in \mathbb{R}^d$

**Term 1** (outer product term):
$$\text{Term}_1 = \varphi''(r) \frac{\Delta x_j}{r \ell_j^2} \sum_{k=1}^d \frac{\Delta x_k}{r \ell_k^2} V_k$$

**Term 2** (Hessian contraction):
$$\text{Term}_2 = \varphi'(r) \left[-\frac{\Delta x_j}{r^3 \ell_j^2} \sum_{k=1}^d \frac{\Delta x_k}{\ell_k^2} V_k + \frac{V_j}{r \ell_j^2}\right]$$

### Radial Derivatives for Matern Kernels

The implementation uses analytical derivatives of the radial function $\varphi(r)$:

**RBF (ν = ∞)**: $\varphi(r) = e^{-r^2/2}$
$$\varphi'(r) = -r e^{-r^2/2}, \quad \varphi''(r) = (r^2 - 1) e^{-r^2/2}$$

**Matern 3/2 (ν = 3/2)**: $\varphi(r) = (1 + \sqrt{3}r) e^{-\sqrt{3}r}$
$$\varphi'(r) = -3r e^{-\sqrt{3}r}, \quad \varphi''(r) = 3(\sqrt{3}r - 1) e^{-\sqrt{3}r}$$

**Matern 5/2 (ν = 5/2)**: $\varphi(r) = (1 + \sqrt{5}r + \frac{5r^2}{3}) e^{-\sqrt{5}r}$
$$\varphi'(r) = \frac{5r}{3}(-\sqrt{5}r - 1) e^{-\sqrt{5}r}, \quad \varphi''(r) = \frac{5}{3}(5r^2 - \sqrt{5}r - 1) e^{-\sqrt{5}r}$$

### GP Mean HVP

For the full GP posterior mean:

$$\nabla_x^2 \mu_*(x) = \sum_{i=1}^n \alpha_i \nabla_x^2 k(x, x_i)$$

The HVP is:

$$H_x[\mu_*(x)] \cdot V = \sum_{i=1}^n \alpha_i \left(H_x[k(x, x_i)] \cdot V\right)$$

## Vectorization Strategy

### Key Insight: Batch Over Training Points, Not Dimensions

For $n$ training points and $d$ dimensions, computing HVP for all training points simultaneously:

**Input shapes**:
- Query point: $x_* \in \mathbb{R}^d$ → represented as `(d, 1)`
- Training points: $X \in \mathbb{R}^{d \times n}$ → `(d, n)`
- Direction: $V \in \mathbb{R}^d$ → `(d,)` (1D array)

**Intermediate shapes** (vectorized over $n$ training points):
- Differences: $\Delta X = x_* - X$ → `(d, n)`
- Scaled distances: $r$ → `(n,)` one per training point
- Radial derivatives: $\varphi'(r), \varphi''(r)$ → `(n,)` each
- Geometry derivatives: $\nabla r$ → `(d, n)` vectorized over $n$

### Vectorized HVP Computation

**Step 1: Compute scaled differences and distances**
```python
diffs = x_star[:, None] - X_train  # (d, n)
diffs_scaled = diffs / lenscale[:, None]  # (d, n)
r = sqrt(sum(diffs_scaled^2, axis=0))  # (n,)
```

**Step 2: Get radial derivatives**
```python
phi_prime, phi_double_prime = radial_derivatives(r)  # (n,), (n,)
```

**Step 3: Compute geometry derivatives**
```python
dr_dx = diffs / (r[None, :] * lenscale[:, None]**2)  # (d, n)
```

**Step 4: Contract with direction vector**
```python
dr_dot_V = einsum('ki,k->i', dr_dx, V)  # (n,) — vectorized dot product
```
This computes $\nabla r \cdot V$ for all $n$ training points simultaneously.

**Step 5: Compute Term 1** (outer product term)
```python
term1 = dr_dx * (phi_double_prime[None, :] * dr_dot_V[None, :])  # (d, n)
```
Broadcasting: `(d, n) * ((n,) * (n,))` → `(d, n)`

**Step 6: Compute Term 2** (Hessian contraction)
```python
diffs_over_ell2 = diffs / lenscale[:, None]**2  # (d, n)
diffs_scaled_dot_V = einsum('ki,k->i', diffs_over_ell2, V)  # (n,)

d2r_dot_V = (-diffs_over_ell2 * (r_inv**3 * diffs_scaled_dot_V)[None, :]
             + V[:, None] * r_inv[None, :] / lenscale[:, None]**2)  # (d, n)

term2 = d2r_dot_V * phi_prime[None, :]  # (d, n)
```

**Step 7: Combine and transpose**
```python
hvp_2d = term1 + term2  # (d, n)
hvp = transpose(hvp_2d[:, None, :], (1, 2, 0))  # (1, n, d)
```

Output shape `(1, n, d)` represents: 1 query point × n training points × d dimensions

### GP-Level Contraction

At the GP level, contract with dual coefficients $\alpha$:

```python
kernel_hvp = kernel.hvp_wrt_x1(x_star[:, None], X_train, V)  # (1, n, d)
gp_hvp = einsum('iqj,q->j', kernel_hvp, alpha)  # (d,)
```

The einsum `'iqj,q->j'` performs:
$$\sum_{i=1}^1 \sum_{q=1}^n \text{kernel\_hvp}[i,q,j] \cdot \alpha[q] = \sum_{q=1}^n H[k(x_*, x_q)] \cdot V \cdot \alpha_q$$

## Performance Analysis

**Complexity**: $O(nd)$ where $n$ is number of training points, $d$ is dimension

**Memory**:
- Intermediate arrays: `(d, n)` for differences, derivatives
- Output: `(1, n, d)` kernel HVP
- No $d \times d$ Hessian matrices stored

**Vectorization benefits**:
- Single pass over all $n$ training points
- No Python loops over training points
- Efficient einsum contractions
- All operations broadcast-compatible

**Comparison to naive approach**:
- Naive: Loop over $n$ points → $n$ calls, each $O(d^2)$ → $O(nd^2)$ total
- Vectorized: Single call → $O(nd)$, but with better constant factors from vectorization

The key optimization is recognizing that for GP inference, we always have:
- **One query point** ($n_1 = 1$)
- **Many training points** ($n_2 = n \gg 1$)
- **Relatively few dimensions** ($d \ll n$ typically)

Therefore, vectorizing over $n$ training points is the critical optimization.

## Implementation Files

- `matern.py`: `MaternKernel.hvp_wrt_x1()` with `radial_derivatives()`
- `composition.py`: `ProductKernel.hvp_wrt_x1()` and `SumKernel.hvp_wrt_x1()`
- `iid_gaussian_noise.py`: `IIDGaussianNoiseKernel.hvp_wrt_x1()` (returns zeros)
- `scalings.py`: `PolynomialScaling.hvp_wrt_x1()` for spatially-varying noise
- `exact.py`: `ExactGaussianProcess.hvp()` orchestrates kernel HVP + contraction

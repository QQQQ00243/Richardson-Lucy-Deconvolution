# README

Python implementation of shift variant one-dimensional Richardson-Lucy reconstruction method.

## Derivation

For a set of objects at different positions, the set of number is $x=[x_1, x_2, \cdots, x_n]$, our observation is $m=[m_1, m_2, \cdots, m_n]$. However, objects may spread into neighboring positions and our observation may be blurred by this transition, i.e. observed objects at certain position are actually consisted by objects at the same position and other objects from other positions. For example, for a optical system, $x_i$ is the number of emitted photons of light source at position $i$ and $m_i$ the number of received photons by detector at position $i$. Our observation, however, is inevitably blurred, with an ideal point source not appearing as a point but being spread out into what is known as the point spread function.

We assume that for each object, the shift between true and observed position follows a categorical distribution. When marginalizing on objects from position $j$, the observation at $i$ becomes binomial distribution 
$$P(m_i|x_j, h_{i-j}^{(j)}) = \mathcal{B}(x_j, h_{i-j}^{(j)}),$$
where $h_{i-j}^{(j)}$ is the probability of shift being $(i-j)$ for objects from position $j$.

By Le Cam's Theorem, the observation could be approximated by Poisson distribution for large number of objects at every position. For observation at position $i$, the mean $E_i$ is the sum of products of number of objects and corresponding shift probability at every position 
$$E_i = \sum_{j=1}^Kh_{i-j}^{(j)}x_j.(1)$$
The transition could be written as
$$E = Hx,(2)$$
where $H$ is the transition matrix with $[h_{ij}]=[h_{i-j}^{(j)}]$.

The probability of observation $m$ is given by
```math
P(m|E) = \prod_i^KPois(E_i) = \prod_i^K\frac{E_i^{m_i}e^{-E_i}}{m_i!}. (3)
```
It is convenient to work with $ln(P)$ when analysing its maximum by taking derivative
$$ln(P(m|E)) = \sum_i^K\left[(m_ilnE_i - E_i)-ln(m_i!)\right]$$
We want to reconstruct ground truth $x$ from observation $m$ by iteration. The estimator of ground truth at $(k+1)$ step is

```math
\hat{x}^{(k+1)} = \hat{x}^{(k)} + \lambda\frac{\partial{\ ln(P(m|E))}}{\partial{x}}\bigg|_{\hat{x}^{(k)}}.(4)
```
For the $j$-th element of the gradient, we have
```math
\frac{\partial\ ln(P(m|E))}{\partial x_j} 
= \frac{\partial}{\partial x_j}\sum_i^K\left[(m_ilnE_i - E_i)-ln(m_i!)\right] 
= \sum_i^K\left[m_i\frac{\partial}{\partial x_j}lnE_i - \frac{\partial}{\partial x_j}E_i\right] 
= \sum_i^K\frac{\partial E_i}{\partial x_j}\left[\frac{m_i}{E_i} - 1\right].
```

By $Eq.(1)$ we have
```math
\frac{\partial\ ln(P(m|E))}{\partial x_j} 
= \sum_i^K H_{ij} \left[\frac{m_i}{E_i} - 1\right] = \sum_i^K H_{ji}^T \left[\frac{m_i}{E_i} - 1\right].
```
The gradient in $Eq. (4)$ written as
```math
\frac{\partial\ ln(P(m|E))}{\partial x} = H^T \left(\frac{m}{E} - 1\right). (5)
```
We propose the following arbitrary and key step
$$\lambda = \hat{x}^{(k)}, (6)$$
where the division is element-wise. The step size at position $i$ of $(j+1)$ step is proportional to the estimated ground truth of last step divided by the probability the observation at this position not being blurred, e.g., the probability of the photon from the light source being received by the detector at the same position. With $Eq. (5)$ and $Eq.(6)$ we have

```math
\hat{x}^{(k+1)} = \hat{x}^{(k)} + \lambda\frac{\partial\ ln(P(m|E))}{\partial x}\bigg|_{\hat{x}^{(k)}} 
= \hat{x}^{(k)} \otimes \frac{H^Tm}{Hx^{(k)}},
```
where the division is element-wise.

## Example

For a set of five independent light sources, set of number of photons emitted from these light sources are
$$x = (1000, 2000, 1500, 1800, 1100).$$
However, our observation is blurred, which could be described by the matrix $H=[H_{ij}]$ with $H_{ij}$ the probability that photon from light source at position $i$ being observed by detector at position $j$
$$H = 
\left( \begin{array}{cc}
0.8 &0.1  &0.1 &0    &0 \\
0.1 &0.8  &0.1 &0    &0  \\
0   &0.05 &0.9 &0.05 &0   \\
0   &0    &0.1 &0.85 &0.05 \\
0   &0    &0   &0.9  &0.1
\end{array} \right).$$
 Then the distribution of observation follow Poisson distribution with mean
$$m = Hx.$$
We sample one observation from this Poisson distribution with `numpy.random` module

```python
import numpy as np
from numpy.random import poisson

x = np.array([1000, 2000, 1500, 1800, 1100])
H = np.array([
    [0.8, 0.1, 0.1, 0, 0],
    [0.1, 0.8, 0.1, 0, 0],
    [0, 0.05, 0.9, 0.05, 0],
    [0, 0, 0.1, 0.85, 0.05],
    [0, 0, 0, 0.9, 0.1]
])
m = np.dot(H, x)
o = poisson(m)
```

with observation

```python
>>> o
>>> array([1200, 1788, 1558, 1741, 1747])
```

Then we reconstruct ground truth $m$ by call `R_L_deconvolve` from `Richardson_Lucy_Deconvolution`, notice a sufficiently large number of iterations is required to reach convergence

```python
x_hat = R_L_deconvolve(o, H, 1000)
```

```python
>>> x_hat
>>> array([1070.48030331, 1910.48030342, 1525.67726553, 1787.328989, 1384.05147652])
```

Compared to ground truth $x = (1000, 2000, 1500, 1800, 1100)$, we notice that artefact appears at the edge of array.  








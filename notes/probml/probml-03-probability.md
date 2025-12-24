# Probability: Advanced Topics

This chapter covers more advanced probability concepts including covariance, correlation, mixture models, and Markov chains — essential tools for understanding many machine learning algorithms.

## Covariance and Correlation

### Covariance

Covariance measures how two random variables **vary together**:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

**Interpretation**:
- **Positive covariance**: When X is above its mean, Y tends to be above its mean
- **Negative covariance**: When X is above its mean, Y tends to be below its mean
- **Zero covariance**: No linear relationship

**Alternative formula**:
$$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

**Properties**:
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, Y) = a \cdot \text{Cov}(X, Y)$

### Correlation

Correlation is a **scaled** version of covariance, always between -1 and 1:

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)} \cdot \sqrt{\text{Var}(Y)}}$$

**Interpretation**:
- $\rho = 1$: Perfect positive linear relationship
- $\rho = -1$: Perfect negative linear relationship
- $\rho = 0$: No linear relationship (but could have non-linear relationship!)

### Independence vs. Uncorrelation

**Key distinction**:
- **Independent** ⟹ **Uncorrelated** (always true)
- **Uncorrelated** ⟸ **NOT** ⟹ **Independent** (converse is false!)

**Example**: Let X ~ Uniform(-1, 1) and Y = X². Then:
- $\text{Cov}(X, Y) = 0$ (by symmetry)
- But X and Y are clearly dependent (Y is completely determined by X!)

### Correlation ≠ Causation

A correlation between X and Y could be due to:
1. X causes Y
2. Y causes X
3. A third variable Z causes both (confounding)
4. Pure coincidence (spurious correlation)

**Simpson's Paradox**: A trend that appears in groups of data can **disappear or reverse** when groups are combined. Always be cautious about aggregated data!

---

## Mixture Models

Mixture models represent complex distributions as **weighted combinations** of simpler distributions.

### Definition

$$p(y | \theta) = \sum_{k=1}^K \pi_k \cdot p_k(y)$$

Where:
- $\pi_k$ are **mixing proportions** (weights): $\sum_k \pi_k = 1$, all $\pi_k \geq 0$
- $p_k(y)$ are **component distributions**

### Generative Process

To sample from a mixture:
1. Sample component index: $k \sim \text{Categorical}(\pi_1, ..., \pi_K)$
2. Sample from chosen component: $y \sim p_k(y)$

### Gaussian Mixture Models (GMMs)

The most common mixture model uses Gaussian components:

$$p(y) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(y | \mu_k, \Sigma_k)$$

**Applications**:
- **Clustering**: Soft assignment of points to clusters
- **Density estimation**: Model complex, multi-modal distributions
- **Outlier detection**: Points with low probability under all components

### K-Means as a Special Case

K-Means clustering is a limiting case of GMM with:
- Uniform mixing proportions: $\pi_k = 1/K$
- Spherical Gaussians: $\Sigma_k = \sigma^2 I$ (same for all components)
- Hard assignments (as $\sigma \to 0$)

---

## Markov Chains

Markov chains model **sequences** where each state depends only on the previous state.

### Chain Rule for Sequences

For a sequence $(x_1, x_2, x_3, ...)$:
$$p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2)$$

This is exact but requires modeling complex conditional dependencies.

### The Markov Assumption

**First-order Markov property**: The future depends only on the present, not the past.
$$p(x_t | x_1, x_2, ..., x_{t-1}) = p(x_t | x_{t-1})$$

This simplifies the chain rule to:
$$p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_2)$$

### State Transition Matrix

The conditional distribution $p(x_t | x_{t-1})$ defines a **transition matrix** T where:
$$T_{ij} = p(x_t = j | x_{t-1} = i)$$

**Properties**:
- Rows sum to 1 (each row is a valid probability distribution)
- $T^n$ gives n-step transition probabilities

### Higher-Order Markov Models

**Second-order Markov**: $p(x_t | x_{t-1}, x_{t-2})$
- Used in bigram language models

**n-th order Markov**: $p(x_t | x_{t-1}, ..., x_{t-n})$
- Used in n-gram language models

**Trade-off**: Higher order captures more context but requires more parameters.

### Applications in ML

- **Language modeling**: Predicting the next word
- **Hidden Markov Models**: Markov chains with hidden states
- **Reinforcement learning**: MDP (Markov Decision Process)
- **MCMC**: Markov Chain Monte Carlo for sampling

---

## The Multivariate Gaussian

The multivariate Gaussian (MVN) is crucial for modeling correlated continuous variables.

### Definition

For a d-dimensional random vector $\mathbf{x}$:

$$\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

Where:
- $\boldsymbol{\mu}$: Mean vector (d × 1)
- $\boldsymbol{\Sigma}$: Covariance matrix (d × d, symmetric positive definite)

### Key Properties

**Marginals**: If $(X, Y) \sim \mathcal{N}$, then $X \sim \mathcal{N}$ and $Y \sim \mathcal{N}$

**Conditionals**: $X | Y \sim \mathcal{N}$ (also Gaussian!)

**Linear transformations**: If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:
$$A\mathbf{x} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$$

### Geometry of the Covariance Matrix

The covariance matrix determines the shape of the Gaussian:
- **Diagonal Σ**: Ellipse aligned with axes
- **Spherical Σ = σ²I**: Circle/sphere
- **General Σ**: Rotated ellipse

The eigenvectors of Σ give the principal axes; eigenvalues give the variance along each axis.

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **Covariance** | Measures linear co-variation; unscaled |
| **Correlation** | Scaled covariance between -1 and 1 |
| **Independence** | Implies zero correlation, but not vice versa |
| **Mixture Models** | Complex distributions as weighted sums of simple ones |
| **GMM** | Gaussian components; soft clustering |
| **Markov Chains** | Future depends only on present, not past |
| **Transition Matrix** | Encodes all transition probabilities |
| **Multivariate Gaussian** | Generalizes Gaussian to multiple correlated variables |

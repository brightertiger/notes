# Kernel Methods

Kernel methods are a powerful family of techniques that enable us to work in high-dimensional (even infinite-dimensional!) feature spaces without explicitly computing the transformations. This chapter covers density estimation, classification using kernels, and radial basis functions.

## The Big Picture

Many real-world patterns are non-linear. Kernel methods handle this by:
1. **Implicitly mapping** data to a higher-dimensional space where patterns become linear
2. **Using local information** — nearby points matter more than distant ones
3. **Avoiding the curse of dimensionality** through the "kernel trick"

---

## Kernel Density Estimation

Before classifying data, we often need to estimate the underlying probability distribution. Kernel density estimation (KDE) is a non-parametric way to do this.

### The Problem

Given a random sample $[x_1, x_2, ..., x_N]$ drawn from an unknown distribution $f_X(x)$, how do we estimate what $f_X$ looks like?

### A First Attempt: Histograms

The simplest approach is a histogram: divide the space into bins and count how many points fall in each bin. But histograms are bumpy and depend heavily on bin placement.

### Parzen (Kernel) Density Estimation

A smoother approach: for any point $x_0$, estimate the density by looking at how many training points are nearby:

$$\hat{f}_X(x_0) = \frac{1}{N\lambda} \times \#\{x_i \in \text{neighborhood of } x_0\}$$

Where λ is the **bandwidth** — the width of the neighborhood.

**Problem**: This is still bumpy at the neighborhood boundaries.

### Gaussian Kernels for Smooth Estimates

Instead of hard boundaries, use a **smooth kernel** that gives more weight to closer points:

$$\hat{f}_X(x_0) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_i, x_0)$$

Where the Gaussian kernel is:

$$K_\lambda(x_i, x_0) = \frac{1}{\lambda\sqrt{2\pi}}\exp\left(-\frac{\|x_i - x_0\|^2}{2\lambda^2}\right)$$

**Intuition**: We're placing a small "bump" (Gaussian) at each data point, then adding them all up. The result is a smooth density estimate.

### The Effect of Bandwidth

- **Small λ**: Peaks around each data point, very bumpy (overfitting)
- **Large λ**: Very smooth, may miss important features (underfitting)
- **Just right λ**: Captures the true density shape

Choosing bandwidth is similar to choosing model complexity — cross-validation helps!

### Mathematical View: Convolution

The kernel density estimate can be viewed as a **convolution** of the empirical distribution (spikes at each data point) with a Gaussian kernel:

$$\hat{f}_X(x_0) = \frac{1}{N}\sum_{i=1}^N \phi_\lambda(x_0 - x_i)$$

This "smears out" the point masses into a smooth function.

---

## Kernel Density Classification

Now we can use density estimation for classification!

### Bayes' Theorem for Classification

For class j:

$$P(G=j | X=x_0) \propto \hat{\pi}_j \times \hat{f}_j(x_0)$$

Where:
- $\hat{\pi}_j$ = estimated prior probability (proportion of class j in training data)
- $\hat{f}_j(x_0)$ = kernel density estimate for class j, evaluated at $x_0$

**Algorithm**:
1. Estimate the density separately for each class
2. Multiply by the class prior
3. Classify to the class with highest product

### A Subtlety: Where Density Matters

Learning separate class densities everywhere can be misleading. Consider:
- In dense regions (many training points), estimates are reliable
- In sparse regions (few training points), estimates are noisy

**Key insight**: The density estimates only matter near the **decision boundary**. Far from the boundary, one class dominates anyway.

---

## Naive Bayes Classifier

### When Dimensions Are High

In high dimensions, kernel density estimation struggles — you need exponentially more data to fill the space (curse of dimensionality).

**Naive Bayes** makes a strong simplifying assumption: given the class, features are **conditionally independent**.

$$f_j(x) = \prod_{p=1}^P f_{jp}(x_p)$$

### Breaking Down the Independence Assumption

Instead of estimating one P-dimensional density (hard), we estimate P one-dimensional densities (easy!).

**For continuous features**: Use univariate Gaussian or kernel density estimates for each feature.

**For categorical features**: Simply count proportions.

### The Log-Odds Decomposition

$$\log\frac{P(G=k|X)}{P(G=l|X)} = \log\frac{\pi_k}{\pi_l} + \sum_{p=1}^P \log\frac{f_{kp}(x_p)}{f_{lp}(x_p)}$$

Each feature contributes additively to the log-odds — simple and interpretable!

### Why "Naive" Works

The independence assumption is almost always wrong. Yet Naive Bayes often performs well because:
1. **We only need rankings**: For classification, we just need to rank classes correctly, not get exact probabilities
2. **Errors may cancel**: Overestimating some terms and underestimating others can balance out
3. **Robustness**: Fewer parameters means less overfitting

**Best applications**: Text classification (spam filtering, sentiment analysis), where features (words) are high-dimensional.

---

## Radial Basis Functions (RBF)

Radial basis functions offer another approach: explicitly construct basis functions centered at various points, then fit a linear model in this new feature space.

### The Idea of Basis Functions

Instead of modeling $f(x) = \beta^T x$ (linear in original features), use:

$$f(x) = \sum_{j=1}^M \beta_j h_j(x)$$

Where $h_j(x)$ are **basis functions** — transformations of the original features.

### Why Higher Dimensions Help

Data that isn't linearly separable in the original space may become linearly separable in a higher-dimensional feature space.

**Classic example**: XOR problem. Points at (0,0) and (1,1) are class 1; points at (0,1) and (1,0) are class 2. No line separates them in 2D, but adding the feature $x_1 \cdot x_2$ makes separation easy!

### Gaussian RBFs

RBF uses Gaussian kernels as basis functions:

$$f(x) = \sum_{j=1}^M \beta_j K_{\lambda_j}(\xi_j, x)$$

Where:
- $\xi_j$ = center of the j-th basis function (a point in feature space)
- $\lambda_j$ = width of the j-th kernel
- $\beta_j$ = coefficient (learned by regression)

More explicitly:

$$f(x) = \sum_{j=1}^M \beta_j \exp\left(-\frac{\|x - \xi_j\|^2}{2\lambda_j^2}\right)$$

### Connection to Infinite Dimensions

The Gaussian kernel has a remarkable property: its Taylor series expansion involves polynomials of all degrees!

$$\exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + ...$$

So using Gaussian RBFs is like using polynomial features of infinite degree — but without the computational explosion.

### Fitting RBF Models

**Full optimization**: Learn $\beta$, $\xi$, and $\lambda$ by minimizing squared error.
- Non-linear in $\xi$ and $\lambda$
- Requires gradient descent or other iterative methods
- Risk of local minima

**Simpler approach**: Fix $\xi$ and $\lambda$, only learn $\beta$.
- Choose centers using unsupervised methods (e.g., k-means on training data)
- Use a constant bandwidth based on average distances
- Then it's just linear regression!

### Potential Pitfalls

If the basis function centers don't cover the input space well, there can be "holes" — regions where no kernel has significant weight, leading to poor predictions.

---

## Gaussian Mixture Models

Mixture models extend RBFs to probabilistic density estimation.

### The Model

$$f(x) = \sum_{j=1}^M \alpha_j \phi(x; \mu_j, \Sigma_j)$$

Where:
- $\phi$ = Gaussian density function
- $\alpha_j$ = mixing proportion (how much weight to give component j)
- $\sum_j \alpha_j = 1$
- $\mu_j$, $\Sigma_j$ = mean and covariance of component j

### Interpretation

The data is generated by:
1. Randomly selecting a component j with probability $\alpha_j$
2. Drawing a point from the Gaussian $N(\mu_j, \Sigma_j)$

Each component represents a "cluster" or subpopulation in the data.

### Connection to RBF

If we restrict covariances to be spherical ($\Sigma_j = \sigma^2 I$), mixture models reduce to radial basis expansions!

### Fitting with Maximum Likelihood

Parameters are fit by maximizing the likelihood of the data. This is typically done using the **EM algorithm** (covered in Model Selection chapter).

### Classification with Mixtures

For classification:
1. Fit a separate mixture model for each class
2. Apply Bayes' theorem: $P(G=j|x) \propto \hat{\pi}_j \times \hat{f}_j(x)$

This is more flexible than single-Gaussian LDA — each class can have multiple modes.

---

## Summary: When to Use What

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Kernel Density Estimation** | Low dimensions, flexible shape | Non-parametric, intuitive | Curse of dimensionality |
| **Naive Bayes** | High dimensions, text/categorical | Fast, scales well | Wrong independence assumption |
| **RBF Networks** | Smooth non-linear functions | Flexible, local | Need to choose centers/widths |
| **Mixture Models** | Multi-modal distributions | Probabilistic, interpretable | Need to choose number of components |

### Key Takeaways

1. **Kernels enable locality**: Nearby points matter more than distant ones
2. **Bandwidth/width controls bias-variance**: Too small = overfit, too large = underfit
3. **High dimensions are hard**: Naive Bayes and the kernel trick help
4. **Generative vs. Discriminative**: Kernel density classification is generative (models P(X|G)); logistic regression is discriminative (models P(G|X))

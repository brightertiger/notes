# Linear Regression

Linear regression is the foundation of supervised learning for continuous outputs. Understanding it deeply gives insight into more complex models.

## The Big Picture

**The model**:
$$p(y | x, \theta) = \mathcal{N}(y | w^T x + b, \sigma^2)$$

**Translation**: Given features x, the output y is normally distributed around a linear prediction, with some noise σ².

---

## Types of Linear Regression

| Type | Description |
|------|-------------|
| **Simple** | One input feature |
| **Multiple** | Many input features |
| **Multivariate** | Multiple output variables |
| **Polynomial** | Non-linear by adding $x^2, x^3$, etc. as features |

**Key insight**: "Linear" refers to linearity in **parameters**, not features. Polynomial regression is still "linear regression"!

---

## Least Squares Estimation

### The Objective

Minimize the Negative Log-Likelihood:
$$\text{NLL}(w, \sigma^2) = \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \hat{y}_i)^2 + \frac{N}{2}\log(2\pi\sigma^2)$$

The first term is the **Residual Sum of Squares (RSS)**.

### The Normal Equations

Setting $\nabla_w \text{RSS} = 0$:
$$X^T X w = X^T y$$

**Solution**:
$$\hat{w} = (X^T X)^{-1} X^T y$$

**Why "normal"?** The residual vector $(y - Xw)$ is orthogonal (normal) to the column space of X.

### Geometric Interpretation

$\hat{y} = X\hat{w}$ is the **projection** of y onto the column space of X. We find the closest point in the subspace spanned by the features.

### Practical Computation

Direct matrix inversion can be numerically unstable. Better approaches:
1. **SVD**: $X = U \Sigma V^T$, then $\hat{w} = V \Sigma^{-1} U^T y$
2. **QR decomposition**: More stable for ill-conditioned problems

### Simple Linear Regression

For one feature:
$$\hat{w} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
$$\hat{b} = \bar{y} - \hat{w}\bar{x}$$

**Intuition**: Slope is ratio of covariance to variance. Intercept ensures line passes through $(\bar{x}, \bar{y})$.

### Estimating the Noise Variance

$$\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{\text{RSS}}{N}$$

**Note**: This is biased! Unbiased version divides by (N - p - 1).

---

## Goodness of Fit

### Residual Analysis

Check assumptions by plotting residuals:
- Should be normally distributed
- Should have zero mean
- Should be homoscedastic (constant variance)
- Should be independent

### Coefficient of Determination (R²)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Where:
- **TSS** (Total Sum of Squares): Variance of y
- **RSS** (Residual Sum of Squares): Unexplained variance

**Interpretation**:
- R² = 1: Perfect fit
- R² = 0: Model no better than predicting the mean
- R² < 0: Model is worse than predicting the mean (possible with regularization)

### RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$$

In same units as y — more interpretable than MSE.

---

## Ridge Regression (L2 Regularization)

### The Problem with OLS

MLE can overfit when:
- Features are correlated (multicollinearity)
- Number of features exceeds samples (p > N)
- $(X^T X)$ is ill-conditioned

### The Ridge Solution

Add L2 penalty on weights:
$$L(w) = \text{RSS} + \lambda \|w\|^2$$

**Closed-form solution**:
$$\hat{w}^{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$$

**Effect**: Adding λI to the diagonal makes the matrix invertible!

### Bayesian Interpretation

Ridge = MAP estimation with Gaussian prior:
$$p(w) = \mathcal{N}(0, \lambda^{-1} \sigma^2 I)$$

### Connection to PCA

Ridge regression shrinks coefficients more in directions of low variance (small eigenvalues of $X^TX$).

**Intuition**: Directions with little data support get regularized more heavily.

---

## Robust Regression

### The Outlier Problem

OLS is sensitive to outliers (squared error heavily penalizes large residuals).

### Solutions

1. **Student-t distribution**: Heavy tails don't penalize outliers as much
   - Fit via EM algorithm
   
2. **Laplace distribution**: Corresponds to L1 loss (MAE)
   - More robust than Gaussian

3. **Huber Loss**: Best of both worlds
   - L2 for small errors (smooth optimization)
   - L1 for large errors (robustness)

4. **RANSAC**: Iteratively identify and exclude outliers

---

## Lasso Regression (L1 Regularization)

### The L1 Penalty

$$L(w) = \text{RSS} + \lambda \|w\|_1 = \text{RSS} + \lambda \sum_j |w_j|$$

### Sparsity!

Unlike Ridge, Lasso can set coefficients exactly to zero.

**Why?** Consider the Lagrangian view:
- L2 constraint: $\|w\|^2 \leq B$ (sphere)
- L1 constraint: $\|w\|_1 \leq B$ (diamond)

The diamond has corners on the axes. The optimal solution often hits a corner, making some weights zero.

### Regularization Path

As λ decreases from ∞ to 0:
- Weights "enter" the model one by one
- Order of entry indicates relative importance
- Use cross-validation to select optimal λ

### Bayesian Interpretation

Lasso = MAP with Laplace prior:
$$p(w) \propto \exp(-\lambda |w|)$$

---

## Elastic Net

### Combining L1 and L2

$$L(w) = \text{RSS} + \lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$$

### Advantages

- **Sparsity** from L1
- **Grouping effect** from L2: Correlated features tend to get similar coefficients
- More stable than pure Lasso

---

## Optimization: Coordinate Descent

### The Algorithm

For Lasso and Elastic Net:
1. Initialize all weights
2. For each coordinate j:
   - Fix all other weights
   - Optimize w_j (has closed-form solution!)
3. Repeat until convergence

**Why it works**: Each subproblem is easy, and cycling through converges to the global optimum for convex problems.

---

## Summary

| Method | Penalty | Sparsity | Computation | Best For |
|--------|---------|----------|-------------|----------|
| **OLS** | None | No | Closed-form | Well-conditioned problems |
| **Ridge** | L2 | No | Closed-form | Multicollinearity |
| **Lasso** | L1 | Yes | Iterative | Feature selection |
| **Elastic Net** | L1 + L2 | Yes | Iterative | Correlated features |

### Practical Tips

1. **Always visualize residuals** to check assumptions
2. **Standardize features** before regularization
3. **Use cross-validation** to choose λ
4. **Start simple** (OLS), add complexity as needed
5. **Lasso for interpretability** (sparse models)
6. **Ridge for prediction** (usually slightly better than Lasso)

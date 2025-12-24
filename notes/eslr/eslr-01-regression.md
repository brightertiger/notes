# Linear Regression

Linear regression is one of the most fundamental tools in statistics and machine learning. It models the relationship between a continuous response variable and one or more predictor variables. Despite its simplicity, it forms the foundation for understanding more complex methods.

## The Big Picture

Imagine you want to predict house prices based on features like square footage, number of bedrooms, and location. Linear regression assumes that the price is approximately a weighted sum of these features, plus some random noise.

**Key Insight**: Linear regression finds the "best" weights (coefficients) that minimize the difference between predicted and actual values.

---

## Model Formulation

### The Linear Model

We assume the response variable Y relates to predictors X through:

$$y = X\beta + \epsilon$$

Where:
- **y**: Vector of observed outcomes (e.g., house prices)
- **X**: Matrix of predictor values (each row is one observation, each column is one feature)
- **β**: Coefficients we want to estimate (the "weights")
- **ε**: Random error term, assumed to follow $\epsilon \sim N(0, \sigma^2I)$

The assumption of Gaussian errors is important — it means errors are symmetric around zero, with most errors being small and large errors being rare.

### Linear Function Approximation

We're approximating the conditional expectation:

$$E(Y|X) = f(X) = \beta_0 + \sum_{j=1}^p \beta_j x_j$$

**Interpretation of coefficients**: $\beta_j$ represents the expected change in Y for a one-unit increase in $x_j$, *holding all other predictors constant*. This "holding constant" interpretation is crucial and often misunderstood!

**Note**: The model is "linear" in the *parameters* (β), not necessarily in the predictors. You can include $x^2$, $\log(x)$, or interactions — the model is still "linear" because coefficients enter linearly.

---

## Finding the Best Coefficients

### Least Squares: The Objective

We want coefficients that make our predictions as close as possible to the actual values. We measure "closeness" using the Residual Sum of Squares (RSS):

$$RSS = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} (y_i - f(x_i))^2 = (y - X\beta)^T(y - X\beta)$$

**Why squared errors?** Squaring penalizes large errors more heavily, gives a smooth (differentiable) objective function, and leads to elegant closed-form solutions.

### Deriving the Solution

To minimize RSS, we take the derivative with respect to β and set it to zero:

$$\frac{\partial RSS}{\partial \beta} = -2X^T(y - X\beta) = 0$$

Solving for β gives us the famous **Normal Equations**:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

This is the Ordinary Least Squares (OLS) estimator.

### Predicted Values and the Hat Matrix

The fitted values are:

$$\hat{y} = X\hat{\beta} = X(X^TX)^{-1}X^Ty = Hy$$

Where **H** is called the **Hat Matrix** or projection matrix:
- H "puts the hat" on y (transforms y into $\hat{y}$)
- The diagonal elements $H_{ii}$ are called **leverage values**
- High leverage points have outsized influence on the fit
- Leverage values range from 1/N to 1, with average p/N

---

## Understanding Uncertainty

### Sampling Distribution of β

If we collected new data and re-estimated β, we'd get slightly different values. The sampling distribution tells us about this variability:

$$\hat{\beta} \sim N(\beta, (X^TX)^{-1}\sigma^2)$$

**What this means**: Our estimated coefficients follow a normal distribution centered on the true coefficients (unbiased!), with a variance that depends on:
- The noise level σ² (more noise = more uncertainty)
- The structure of the predictors $(X^TX)^{-1}$

### Estimating the Noise Level

Since σ² is unknown, we estimate it from the data:

$$\hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 = \frac{RSS}{N-p-1}$$

**Why N-p-1?** We lose one degree of freedom for each parameter we estimate (p slopes + 1 intercept), leaving N-p-1 degrees of freedom.

---

## Testing Statistical Significance

### Is a Coefficient Different from Zero?

For each coefficient, we can test whether it's significantly different from zero using a t-test:

$$Z_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}}$$

Where $v_j$ is the j-th diagonal element of $(X^TX)^{-1}$.

**Under the null hypothesis** $H_0: \beta_j = 0$, this statistic follows a t-distribution with N-p-1 degrees of freedom.

**Practical interpretation**: A large |Z| (typically > 2) suggests the coefficient is statistically significant, meaning we have evidence that the predictor matters for predicting Y.

### Testing Groups of Coefficients

Sometimes we want to test whether a group of coefficients (e.g., all levels of a categorical variable) are jointly zero. We use the F-test:

$$F = \frac{(RSS_0 - RSS_1)/(p_1 - p_0)}{RSS_1/(N - p_1 - 1)}$$

Where:
- $RSS_0$ = RSS from the restricted model (without the group)
- $RSS_1$ = RSS from the full model (with the group)
- $p_0$, $p_1$ = number of parameters in each model

Under the null hypothesis, $F \sim F_{p_1-p_0, N-p_1-1}$.

---

## Gauss-Markov Theorem

This is one of the most important theoretical results in linear regression:

**Statement**: Among all *linear, unbiased* estimators, OLS has the *smallest variance*.

This is why OLS is called **BLUE** (Best Linear Unbiased Estimator).

### Expected Prediction Error

When predicting a new observation:

$$E[(Y_0 - \hat{Y}_0)^2] = \sigma^2 + \text{MSE}(\hat{f}(X_0))$$

The prediction error has two components:
1. **Irreducible error** (σ²): randomness we can't eliminate
2. **Estimation error** (MSE): uncertainty from estimating f

### Assumptions Required

The Gauss-Markov theorem relies on these assumptions:
1. **Linearity**: The true relationship is linear
2. **Independence**: Errors are independent across observations
3. **Homoscedasticity**: Error variance is constant (not changing with X)
4. **No perfect multicollinearity**: Predictors aren't perfectly correlated

**Important**: Gauss-Markov says OLS is best among unbiased estimators. But sometimes a *biased* estimator with lower variance gives better predictions (see Shrinkage Methods below).

---

## Subset Selection

When you have many predictors, some may be irrelevant. Including them adds noise and hurts interpretability. Subset selection methods help identify the most important predictors.

### Best Subset Selection

Idea: For each subset size k, find the k variables that minimize RSS. Choose the best k using cross-validation or information criteria.

**Problem**: With p predictors, there are $2^p$ possible subsets — computationally infeasible for large p.

### Forward Selection

Start with no predictors and iteratively add the one that most improves the fit:

1. Start with intercept only
2. For each remaining predictor, compute RSS if added
3. Add the predictor that reduces RSS the most
4. Repeat until stopping criterion (e.g., no significant improvement)

**Pros**: Computationally efficient (O(p²) instead of O(2^p))
**Cons**: May miss the optimal subset (greedy algorithm)

**Technical note**: QR decomposition or successive orthogonalization can efficiently compute which variable to add next.

### Backward Selection

Start with all predictors and iteratively remove the least important:

1. Start with all p predictors
2. Compute the t-statistic for each coefficient
3. Remove the predictor with smallest |t| (least significant)
4. Repeat until stopping criterion

**Pros**: Considers all variables initially, captures interactions
**Cons**: Requires N > p (can't start with all variables if you have more variables than observations)

### Hybrid Stepwise Selection

At each step, consider both adding and removing variables. Use criteria like AIC to guide decisions. This explores the model space more thoroughly than pure forward or backward selection.

### Forward Stagewise Selection

A more gradual approach:
1. Start with all coefficients at zero
2. Find the variable most correlated with the current residuals
3. Add a *small* amount of that variable's coefficient (don't fully optimize)
4. Repeat

This is similar to gradient descent in function space and connects to boosting methods discussed later.

---

## Shrinkage Methods

Subset selection is "all or nothing" — a variable is either in or out. Shrinkage methods take a softer approach: they keep all variables but *shrink* coefficients toward zero.

**Key insight**: Accepting a small amount of bias in exchange for reduced variance often improves prediction accuracy.

### Ridge Regression

Ridge regression adds a penalty on the size of coefficients:

$$\hat{\beta}^{\text{ridge}} = \arg\min_\beta \left[ (y - X\beta)^T(y - X\beta) + \lambda\sum_{j=1}^p\beta_j^2 \right]$$

Equivalently, it's a constrained optimization:

$$\hat{\beta}^{\text{ridge}} = \arg\min_\beta (y - X\beta)^T(y - X\beta) \quad \text{subject to} \quad \sum\beta_j^2 \le t$$

Here:
- **λ** (or equivalently, **t**) controls the strength of the penalty
- λ = 0 gives ordinary OLS
- λ → ∞ shrinks all coefficients toward zero

**Closed-form solution**:

$$\hat{\beta}^{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$$

**Why does Ridge help with correlated predictors?**

When predictors are correlated, $(X^TX)$ is nearly singular (not full rank). This makes OLS coefficients unstable — small changes in data cause huge changes in coefficients. Adding λI to the diagonal "regularizes" the matrix, making inversion stable.

**Geometric interpretation using eigenvalue decomposition**:

$X^TX = UDU^T$ where D is diagonal with eigenvalues $d_1, ..., d_p$

Ridge coefficients become: $\hat{\beta}^{\text{ridge}} = \sum_{j=1}^p \frac{d_j}{d_j + \lambda} u_j^T y \cdot u_j$

The term $\frac{d_j}{d_j + \lambda}$ shrinks coefficients most in directions with small eigenvalues (high collinearity).

**Important**: Ridge regression is NOT scale invariant — you must standardize predictors before applying it. Don't penalize the intercept.

### Lasso Regression

Lasso uses an L1 penalty instead of L2:

$$\hat{\beta}^{\text{lasso}} = \arg\min_\beta \left[ (y - X\beta)^T(y - X\beta) + \lambda\sum_{j=1}^p|\beta_j| \right]$$

**Key difference from Ridge**: Lasso can shrink coefficients exactly to zero, performing automatic variable selection!

**Why does L1 give sparsity?**

Geometrically:
- Ridge constraint region: a disk/sphere ($\beta_1^2 + \beta_2^2 \le t$)
- Lasso constraint region: a diamond/rhombus ($|\beta_1| + |\beta_2| \le t$)

The diamond has corners on the axes. The optimal solution often hits a corner, setting some coefficients exactly to zero.

**Bayesian interpretation**:
- Lasso = MAP estimation with Laplace prior: $p(\beta) \propto e^{-\alpha|\beta|}$
- Ridge = MAP estimation with Gaussian prior: $p(\beta) \propto e^{-\alpha\beta^2/2}$

The Laplace prior has heavier tails and more mass at zero, encouraging sparsity.

**Computation**: Unlike Ridge, Lasso requires iterative optimization (coordinate descent is popular).

### Elastic Net

Elastic Net combines L1 and L2 penalties:

$$\text{Penalty} = \lambda\left[\alpha\sum|\beta_j| + (1-\alpha)\sum\beta_j^2\right]$$

**Benefits**:
- Variable selection like Lasso (from L1)
- Stability with correlated predictors like Ridge (from L2)
- Better handles groups of correlated predictors (selects groups together)

---

## Partial Least Squares (PLS)

When predictors are highly correlated, both OLS and Ridge can struggle. PLS offers an alternative approach by constructing new features that capture both high variance AND high correlation with the response.

### Comparing to Principal Component Regression (PCR)

- **PCR**: First does PCA on X (ignoring Y), then regresses Y on the top principal components
- **PLS**: Finds directions in X that have both high variance AND high correlation with Y (supervised)

### PLS Algorithm

1. Standardize X and y (zero mean, unit variance)
2. For m = 1, 2, ..., M components:
   - Compute weight vector: $w_m \propto X_{m-1}^T y$ (direction most correlated with response)
   - Create score: $z_m = X_{m-1} w_m$
   - Regress y on $z_m$ to get coefficient $\hat{\phi}_m$
   - Regress each column of $X_{m-1}$ on $z_m$ to get loadings $\hat{p}_m$
   - Orthogonalize: $X_m = X_{m-1} - z_m \hat{p}_m^T$ (remove what's explained)
3. Final prediction: $\hat{y} = \bar{y} + \sum_{m=1}^M \hat{\phi}_m z_m$

**When to use PLS**: High-dimensional data with many correlated predictors (common in chemometrics, genomics).

---

## Summary: Choosing a Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| OLS | Low-dimensional, well-behaved data | Unbiased, interpretable | Poor with collinearity or p > N |
| Ridge | Collinear predictors | Stable, works when p > N | Keeps all variables |
| Lasso | Sparse signals | Automatic variable selection | Can be unstable with correlated predictors |
| Elastic Net | Correlated groups of predictors | Best of both worlds | Extra tuning parameter |
| PLS | High-dimensional, many correlated predictors | Dimension reduction + prediction | Less interpretable |

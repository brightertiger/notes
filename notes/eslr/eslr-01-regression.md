# Regression

## Derivation

-   $y = X\beta + \epsilon$
    -   $\epsilon \sim N(0, \sigma^2I)$
-   Linear Function Approximation
    -   $E(Y|X) = f(X) = \beta_0 + \sum_{j=1}^p \beta_j x_j$
    -   Model is linear in parameters
    -   Coefficient $\beta_j$ represents the expected change in response for a one-unit change in $x_j$, holding other predictors constant
-   Minimize Residual Sum of Squares
    -   $RSS = \sum (y_i - f(x_i))^2 = (y - X\beta)^T(y - X\beta)$
-   Optimal value of beta:
    -   $\frac{\partial RSS}{\partial \beta} = 0$
    -   $\hat \beta = (X^T X)^{-1}(X^Ty)$
    -   $\hat y = X \hat \beta = (X(X^T X)^{-1}X^T)y = H y$
        -   H is the projection or Hat matrix
        -   $H_{ii}$ are leverage values indicating influence of each observation

## Sampling Distribution of $\beta$

-   Deviations around the conditional mean are Gaussian
-   $Var(\hat \beta) = (X^T X)^{-1} \sigma^2$
-   Estimate of Sigma can be done by looking at sample variance
-   $\hat \sigma^2 = \frac{1}{N-p-1} \sum (y_i - \hat y_i)^2$
-   $\hat \beta \sim N(\beta, (X^T X)^{-1} \sigma^2)$

## Statistical Significance

-   $Z_i = \frac{\hat\beta_i}{SE_i} = \frac{\hat\beta_i}{\hat \sigma \sqrt{v_i}}$
    -   $v_i$ is the $i$-th diagonal element of $(X^T X)^{-1}$
    -   Under null hypothesis $H_0: \beta_i = 0$, $Z_i \sim t_{N-p-1}$
-   Testing significance for a group of parameters
    -   Say categorical variables with all k variables
    -   $F = \frac{(RSS_0 - RSS_1)/(p_1-p_0)}{RSS_1 / (N - p_1-1)}$
    -   $RSS_0$ is from restricted model, $RSS_1$ from full model
    -   $p_0$ and $p_1$ are the number of parameters in each model
    -   Under $H_0$, $F \sim F_{p_1-p_0, N-p_1-1}$

## Gauss-Markov Theorem

-   Among all unbiased estimators, the least square estimates have lowest variance
-   $E[(Y_0 - \hat Y_0)^2] = \sigma^2 + MSE(\hat f(X_0))$
-   Assumptions required:
    -   Linearity of the true relationship
    -   Independence of errors
    -   Homoscedasticity (constant error variance)
    -   No perfect multicollinearity

## Subset Selection

-   Select only a few variables for better interpretability
-   Best subset selection of size K is the one that yields minimum RSS
-   Forward Selection
    -   Sequentially add one variable that most improves the fit
    -   QR decomposition / successive orthogonalization to look at correlation
    -   Computationally efficient but may miss optimal subset
-   Backward Selection
    -   Sequentially delete the variable that has least impact on the fit
    -   Z Score
    -   Requires starting with all variables (can't be used when N < p)
-   Hybrid Stepwise Selection
    -   Consider both forward and backward moves at each step
    -   AIC for weighting the choices
    -   Better exploration of the model space
-   Forward Stagewise Selection
    -   Add the variable most correlated with current residual
    -   Don't re-adjust the coefficients of the existing variables
    -   Similar to gradient descent in function space

## Shrinkage Methods

-   Shrinkage methods result in biased estimators but a large reduction in variance
-   More continuous and don't suffer from high variability
-   Ridge Regression
    -   Impose a penalty on the size of the coefficients
    -   $\hat \beta^{\text{ridge}} = \arg \min (y - X \beta)^T(y - X \beta) + \lambda \sum \beta^2$
    -   $\hat \beta^{\text{ridge}} = \arg \min (y - X \beta)^T(y - X \beta) \; \text{subject to} \sum \beta^2 \le t$
    -   t is the budget
    -   In case of correlated variables, coefficients are poorly determined
    -   A large positive coefficient of a variable is canceled by a large negative coefficient of the correlated variable
    -   Solution not invariant to scaling. Standardize the inputs and don't impose penalty on intercept
    -   $\hat \beta^{\text{ridge}} = (X^T X + \lambda I)^{-1}(X^Ty)$
    -   In case of correlated predictors, the original $(X^T X)$ wasn't full rank. But by adding noise to diagonal elements, the matrix can now be inverted.
    -   Eigenvalue decomposition: $X^TX = U D U^T$
    -   Ridge coefficients: $\hat{\beta}^{ridge} = \sum_{j=1}^p \frac{d_j}{d_j + \lambda} u_j^T y \cdot u_j$
    -   As $\lambda$ increases, coefficients shrink toward zero but not exactly zero
    -   In case of orthonormal inputs (PCA), the ridge coefficients are scaled versions of the original least-square estimates.
    -   $\lambda$ controls the degrees of freedom. A large value results in effectively dropping the variables.
-   Lasso Regression
    -   $\hat \beta^{\text{lasso}} = \arg \min (y - X \beta)^T(y - X \beta) + \lambda \sum |\beta|$
    -   Non-linear optimization
    -   A heavy restriction on budget makes some coefficients exactly zero
    -   Continuous subset selection
    -   Comparison between Ridge and Lasso
        -   Ridge represents a disk $\beta_1^2 + \beta_2^2 <= t$
        -   Lasso represents a rhombus $|\beta_1| + |\beta_2| <= t$
        -   At optimal value, the estimated parameters can be exactly zero (corner solutions)
        -   Bayesian MAP estimates with different priors
            -   Lasso has Laplace Prior ($p(\beta) \propto e^{-\alpha|\beta|}$)
            -   Ridge has Gaussian Prior ($p(\beta) \propto e^{-\alpha\beta^2/2}$)
    -   Elastic Net
        -   $\lambda \sum \alpha \beta^2 + (1 - \alpha) |\beta|$
        -   Variable selection like Lasso
        -   Shrinking coefficients like Ridge
        -   Better handles groups of correlated predictors

## Partial Least Squares

-   Alternative approach to PCA to deal with correlated features
-   Supervised transformation
-   Principal component regression seeks directions that have high variance
-   Partial Least Square seeks direction with high variance and high correlation with response
-   Derive new features by linear combination of raw variables re-weighted by the correlation
-   Algorithm:
    1. Standardize X and y
    2. For m = 1,2,...M:
       - Compute weight vector $w_m \propto X^T_{m-1}y$
       - Create score vector $z_m = X_{m-1}w_m$
       - Regress y on $z_m$ to get coefficient $\hat{\phi}_m$
       - Regress each column of $X_{m-1}$ on $z_m$ to get loadings $\hat{p}_m$
       - Orthogonalize: $X_m = X_{m-1} - z_m\hat{p}_m^T$
    3. Final prediction: $\hat{y} = \bar{y} + \sum_{m=1}^M \hat{\phi}_m z_m$ 
# Regression

Regression is the task of predicting a continuous outcome from input features. It's one of the oldest and most fundamental tools in statistics and machine learning, dating back to Gauss and Legendre in the early 1800s.

## Bi-variate Regression

**The Goal**: Fit a straight line to data—understand how the response variable changes with one explanatory variable.

**The Model**:
$$E(y|x) = \hat{y} = a + bx$$

Where:
- $a$ = intercept (predicted $y$ when $x = 0$)
- $b$ = slope (change in $y$ for unit change in $x$)

**Fitting the Line** (Ordinary Least Squares):

Minimize the Sum of Squared Errors (SSE):
$$\text{SSE} = \sum_i (y_i - \hat{y}_i)^2$$

**Solutions**:
- Slope: $b = \frac{S_{xy}}{S_{xx}} = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sum(x - \bar{x})^2}$
- Intercept: $a = \bar{y} - b\bar{x}$

**Why Squared Errors?**
- Penalizes large errors more than small ones
- Mathematically convenient (differentiable)
- Leads to closed-form solutions
- Has nice statistical properties (BLUE under certain assumptions)

**Important Concepts**:

**Outliers and Influential Points**:
- **Outlier**: Point far from the rest of the data
- **Influential point**: Point that significantly affects the slope
- A point can be an outlier without being influential (if $x$ is near $\bar{x}$)
- A point can be influential without being an outlier (high leverage)

**Residual Variance** (estimate of error variance):
$$s = \sqrt{\frac{\text{SSE}}{n - p}}$$

Where $p$ = number of parameters (2 for simple regression).

We divide by $(n-p)$ not $n$ because we "use up" degrees of freedom estimating parameters.

**Homoscedasticity**: Assumption that variance is constant across all $x$ values.

**Correlation**:
$$r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2} \cdot \sqrt{\sum(y - \bar{y})^2}}$$

Properties:
- Ranges from -1 to +1
- Measures strength of *linear* association
- Relationship to slope: $r = \frac{s_x}{s_y} \cdot b$
- For standardized variables: $r = b$

**Regression Toward the Mean**:
- Since $|r| \leq 1$, a 1 SD increase in $x$ predicts less than 1 SD increase in $y$
- Extreme values tend to be followed by less extreme values
- This is why the technique is called "regression"!

**R-Squared** (Coefficient of Determination):
$$R^2 = \frac{\text{TSS} - \text{SSE}}{\text{TSS}} = 1 - \frac{\text{SSE}}{\text{TSS}}$$

Where:
- TSS = Total Sum of Squares = $\sum(y - \bar{y})^2$ (variance in $y$)
- SSE = Sum of Squared Errors = $\sum(y - \hat{y})^2$ (unexplained variance)

**Interpretation**: Proportion of variance in $y$ explained by $x$.

For simple regression: $R^2 = r^2$ (squared correlation)

**Statistical Significance** (Is the slope different from zero?):
$$t = \frac{b}{\text{SE}(b)} = \frac{b}{s / \sqrt{S_{xx}}}$$

This follows a t-distribution with $(n-2)$ degrees of freedom.

Equivalently, the F-test: $F = t^2 = \frac{R^2 / 1}{(1-R^2)/(n-2)}$

## Multivariate Regression

**The Model**:
$$E(y|x) = \hat{y} = a + b_1 x_1 + b_2 x_2 + ... + b_p x_p$$

**Interpreting Coefficients**:
- $b_1$ is the effect of $x_1$ on $y$ **holding all other variables constant**
- This is called the **partial regression coefficient**
- Very different from simple regression coefficient!

**Why Controlling Matters** (Types of Relationships):

| Relationship | What Happens |
|--------------|--------------|
| **Confounding** | Third variable causes both $x$ and $y$; controlling reveals true (weaker) relationship |
| **Mediation** | Third variable transmits effect from $x$ to $y$; controlling removes indirect effect |
| **Suppression** | Third variable masks relationship; controlling reveals hidden relationship |

**Partial Regression Plots**:
Visualize the true relationship between $x_1$ and $y$ after removing the effect of other variables:

1. Regress $y$ on all variables except $x_1$ → get residuals $e_y$
2. Regress $x_1$ on all other $x$ variables → get residuals $e_{x_1}$
3. Plot $e_y$ vs $e_{x_1}$

The slope of this plot equals $b_1$ from the full model.

**Statistical Tests**:

**F-test** (Are any predictors significant?):
$$F = \frac{R^2 / (p-1)}{(1-R^2)/(n-p)}$$

Tests whether the model explains more variance than expected by chance.

**t-test** (Is a specific predictor significant?):
$$t = \frac{b_j}{\text{SE}(b_j)}$$

Tests whether $b_j$ is significantly different from zero.

**Comparing Nested Models**:
- Complete model: All variables
- Reduced model: Some variables dropped
$$F = \frac{(\text{SSE}_r - \text{SSE}_c) / (df_c - df_r)}{\text{SSE}_c / df_c}$$

**ANOVA Table** (Partitioning Variance):

| Source | Sum of Squares | df | Mean Square |
|--------|----------------|-----|-------------|
| Regression | $\sum(\hat{y} - \bar{y})^2$ | $p-1$ | SSR/(p-1) |
| Error | $\sum(y - \hat{y})^2$ | $n-p$ | SSE/(n-p) |
| Total | $\sum(y - \bar{y})^2$ | $n-1$ | — |

$F = \text{MSR} / \text{MSE}$

**Bonferroni Correction**: When testing multiple coefficients, divide significance level by number of tests to control overall Type I error.

## Logistic Regression

**The Problem**: Linear regression for binary outcomes predicts values outside [0,1].

**The Solution**: Model the probability using a sigmoid (S-shaped) curve.

**The Model**:
$$P(y=1|x) = \frac{e^{\alpha + \beta x}}{1 + e^{\alpha + \beta x}} = \frac{1}{1 + e^{-(\alpha + \beta x)}}$$

Equivalently, the **log-odds** (logit) is linear:
$$\log\left(\frac{P(y=1)}{1 - P(y=1)}\right) = \alpha + \beta x$$

**Why Log-Odds?**
- Odds can range from 0 to ∞
- Log-odds can range from -∞ to +∞
- Makes sense to model with a linear function

**Interpreting Coefficients**:
- $\beta$ = change in log-odds for unit increase in $x$
- $e^\beta$ = **odds ratio** for unit increase in $x$
- If $\beta = 0.5$, then $e^{0.5} \approx 1.65$: the odds multiply by 1.65 for each unit of $x$

**Propensity Scores** (Causal Inference):

When comparing treatment groups, selection bias can confound results.

**Propensity score** = $P(\text{treatment} | \text{covariates})$

Use logistic regression to estimate propensity, then:
1. Match treated/control units with similar propensity
2. Weight by inverse propensity
3. Stratify by propensity quintiles

This "balances" groups on observed covariates.

**Model Comparison**:

**Likelihood Ratio Test**:
$$\chi^2 = -2(\log L_{\text{reduced}} - \log L_{\text{full}})$$

Follows chi-squared distribution with $df$ = difference in number of parameters.

**Wald Test**: $(b_j / \text{SE}(b_j))^2$ follows chi-squared(1).

**Ordinal Logistic Regression** (ordered categories):
- Model cumulative probabilities: $P(y \leq j)$
- Same slopes across all cutpoints (proportional odds assumption)

**Multinomial Logistic Regression** (unordered categories):
- One-vs-Rest: Separate model for each class
- One-vs-One: Model for each pair of classes

## Regression Diagnostics

Good regression analysis requires checking assumptions.

**Residual Analysis**:
- **Residuals vs. Fitted**: Should show no pattern (random scatter around 0)
- **Q-Q Plot**: Residuals should follow the diagonal (normality check)
- **Scale-Location**: Spread should be constant (homoscedasticity check)
- **Residuals vs. Leverage**: Identifies influential outliers

**Multicollinearity** (correlated predictors):
- Makes coefficient estimates unstable
- Inflates standard errors

**Variance Inflation Factor (VIF)**:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is from regressing $x_j$ on all other predictors.

Interpretation:
- VIF = 1: No correlation with other predictors
- VIF = 5: Moderate multicollinearity
- VIF > 10: Serious problem—consider removing variables or using regularization

**Influential Observations**:

| Measure | What It Detects |
|---------|-----------------|
| **Leverage** (hat values) | Points far from $\bar{x}$ that *could* influence fit |
| **Residual** | Points far from fitted line |
| **Cook's Distance** | Combined influence on all predictions |
| **DFBETA** | Effect on individual coefficient estimates |

High leverage + large residual = influential point.

**Model Selection Criteria**:
- **AIC** = $2k - 2\ln(L)$: Penalizes complexity (lower is better)
- **BIC** = $k\ln(n) - 2\ln(L)$: Stronger penalty, favors simpler models
- **Adjusted $R^2$**: $R^2$ penalized for number of predictors

Where $k$ = number of parameters, $L$ = likelihood, $n$ = sample size.

## Advanced Regression Techniques

**Ridge Regression** (L2 regularization):
$$\min_\beta ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2$$

Properties:
- Shrinks coefficients toward zero (but never exactly zero)
- Reduces variance at cost of some bias
- Excellent for multicollinearity
- All predictors kept in model

**Lasso Regression** (L1 regularization):
$$\min_\beta ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$$

Properties:
- Shrinks some coefficients exactly to zero
- Performs automatic **feature selection**
- Great for high-dimensional, sparse problems
- Selects only one from a group of correlated predictors

**Elastic Net** (L1 + L2):
$$\min_\beta ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda_1||\boldsymbol{\beta}||_1 + \lambda_2||\boldsymbol{\beta}||^2$$

Properties:
- Combines benefits of Ridge and Lasso
- Can select groups of correlated features
- Two hyperparameters to tune

**Choosing $\lambda$**: Use cross-validation to find the value that minimizes prediction error on held-out data.

**Quantile Regression**:
- Standard regression models the **mean**: $E(y|x)$
- Quantile regression models **quantiles**: e.g., median, 10th percentile
- Robust to outliers
- Shows how *distribution* of $y$ changes with $x$

**When to Use**:
- When relationship differs across the distribution (e.g., effect on high vs. low income)
- When outliers are a concern
- When you care about specific quantiles (e.g., 95th percentile for risk)

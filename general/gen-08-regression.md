# Regression

## Bi-variate Regression

-   Fit a straight line to data
    -   How the conditional mean of response variable changes with change in explanatory variables
    -   Minimize SSE: $\sum (y - \hat y)^2$
    -   $E(y|x) = \hat y = a + bx$
    -   $a = \bar y - b \bar x$
    -   $b = S_{xy} / S_{xx} = \sum (x-\bar x)(y - \bar y) / \sum (x - \bar x)^2$
-   Outlier: If a point lies far off from the rest of the data
-   Influential: If a point causes large change in slope of the fitted line
-   Variance
    -   Assume constant variance (homoscedasticity)
    -   $s = \sqrt{SSE \over (n-p)}$
-   Correlation
    -   Strength of linear association
    -   $r = \frac{\sum{(x - \bar x)(y - \bar y)}}{\sqrt{(x - \bar x)^2} \sqrt{(y - \bar y)^2}}$
    -   $r = (s_x / s_y) b$
        -   In case of standardized variables, r = b
        -   Regression towards mean
            -   $|r| <= 1$
            -   Line passes thourgh $(\bar x, \bar y)$
            -   A unit increase in x leads to r increase in y
-   R-Square
    -   Coefficient of determination
    -   $R^2 = (TSS - SSE) / TSS$
    -   $SSE = \sum(y - \hat y)^2$
    -   $TSS = \sum(y - \bar y)^2$
    -   Squared correlation coefficient
-   Statistical Significance
    -   $t = b / se$
    -   $se = s / \sqrt{S_{xx}}$
    -   $s = \sqrt{SSE / (n-2)}$
    -   $t^2 = F = \frac{r^2 / (2-1)}{1-r^2 / n - 2}$

## Multivariate Regression

-   Types of relationships
    -   Spurious: Both variables jointly affected by a third variable
    -   Mediator: An intervening third variable indirectly affects the two variables
    -   Suppressor: Association only exists after controlling for a third variable
-   Extending regression to multiple explanatory variables
    -   $E(y|x) = \hat y = a + b_1 x_1 + b_2 x_2$
    -   b1 is the relationship between y and x1 after controlling for all other variables (x2)
    -   b1 is the partial regression coefficient
    -   It represents first order partial correlation
    -   $r_{yx_1.x_2}= (R^2 - r^2_{yx_2}) / (1 - r^2_{yx_2})$
-   Partial Regression Plots
    -   True association between x1 and y after controlling for x2
    -   Regress y on x2: $\hat y = a' + b'x_2$
    -   Regress x1 on x2: $\hat x_1 = a'' + b'' x_2$
    -   Plot the residuals from first regression against the second.
-   Statistical Significance
    -   Collective Influence
        -   F Test
        -   $F = \frac{R^2 / p-1}{(1 - R^2) / (n-p)}$
    -   Individual Influence
        -   t test
        -   $t = \beta / se$
        -   $s = \sqrt{SSE / n-p}$
    -   Comparing Two Models
        -   Complete Model: With all the variables
        -   Reduced Model: Dropping some of the variables
        -   $F = \frac{(SSE_r - SSE_c)/(df_c - df_r)}{SSE_c / df_c}$
-   ANOVA
    -   Total $SST = \sum (y - \bar y)^2, \; df = n-1$
    -   Regression $SSR = \sum (\hat y - \bar y)^2, \; df = p-1$
    -   Error $SSE = \sum (y - \hat y)^2, \; df = n-p$
    -   $F = MSR / MSE = (SSR / df) / (SSE / df)$
-   Bonferroni Correction
    -   Multiple comparisons
    -   Significance Level // \# of comparisons

## Logistic Regression

-   S-shaped curve to model binary response variable
-   Models underlying probability as CDF of logistic distribution
    -   $P(y=1) = \frac{\exp(x \beta)}{1 + \exp(x \beta)}$
-   Log-odds ratio modeled as linear function of features
    -   $\log{\frac{P(y=1)}{1- P(y=1)}} = \alpha + \beta x$
    -   Logit link function
-   Interpreting Logistic Regression Model
    -   The coefficients denote odds ratio
    -   $\exp \beta$ is the odds ratio wrt of x
-   Propensity Scores
    -   Adjust for selection bias in comparing two groups
        -   Smokers and treatment impact on Death. Treatment is more prevalant in smokers\
    -   Control for confounding variables when computing ATE (average treatment effect)
    -   Propensity is the probability of being in a particular group for a given setting of explanatory variable
    -   Can be computed using conditional probabilities but difficult to estimate in high-dimension
    -   Use logistic regression to estimate how propensity depends on explanatory variables
    -   Use propensity score to do pair matching (weighted random sampling using proensity scores)
-   Likelihood Ratio Test
    -   Compare two models
    -   Probability of observed data as a function of parameters
    -   Likelihood Ratio: $-2 (\log l_0 - \log l_1)$ follows chi-squared distribution
-   Wald Statistic
    -   Square of Z-stat: $\beta / se$
-   Ordinal Response (ordered categories):
    -   Setup the problem as one of cumulative logits
    -   $P(y \le 2) = P(y=1) + P(y=2)$
-   Nominal Response (unordered categories):
    -   One-vs-Rest setup: One model per class
    -   One-vs-One setup: One model per pair of classes 

## Regression Diagnostics

-   Residual Analysis:
    -   Residual plots: Check for patterns in residuals vs. fitted values
    -   Q-Q plots: Assess normality of residuals
    -   Scale-location plots: Check homoscedasticity assumption
    -   Residuals vs. leverage: Identify influential observations
-   Multicollinearity:
    -   Variance Inflation Factor (VIF): $VIF_j = \frac{1}{1-R_j^2}$
    -   $R_j^2$ is the R-squared from regressing the jth predictor on all other predictors
    -   VIF > 10 indicates problematic multicollinearity
    -   Remedies: Remove variables, principal components regression, ridge regression
-   Influential Observations:
    -   Cook's distance: Measures effect of deleting observations
    -   DFBETA: Change in coefficient estimates when observation is removed
    -   Leverage (hat values): Potential to influence the fit
    -   High leverage + high residual = influential point
-   Model Selection:
    -   Akaike Information Criterion (AIC): $AIC = 2k - 2\ln(L)$
    -   Bayesian Information Criterion (BIC): $BIC = k\ln(n) - 2\ln(L)$
    -   Adjusted R-squared: $R_{adj}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$
    -   Where k is number of parameters, L is likelihood, n is sample size, p is number of predictors

## Advanced Regression Techniques

-   Ridge Regression:
    -   Adds L2 penalty to objective: $\min_\beta ||y - X\beta||^2 + \lambda||\beta||^2$
    -   Shrinks coefficients towards zero but doesn't eliminate variables
    -   Particularly useful for multicollinearity
-   Lasso Regression:
    -   Adds L1 penalty to objective: $\min_\beta ||y - X\beta||^2 + \lambda||\beta||_1$
    -   Can shrink coefficients exactly to zero (feature selection)
    -   Works well for high-dimensional sparse data
-   Elastic Net:
    -   Combines L1 and L2 penalties: $\min_\beta ||y - X\beta||^2 + \lambda_1||\beta||_1 + \lambda_2||\beta||^2$
    -   Balances feature selection and coefficient shrinkage
-   Quantile Regression:
    -   Models conditional quantiles instead of conditional mean
    -   Robust to outliers and heteroscedasticity
    -   Provides more complete picture of relationship between variables 
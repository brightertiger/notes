# Model Selection

## Maximum Likelihood

-   Maximum Likelihood Inference
    -   Parametric Model
        -   Random variable $z_i \sim g_\theta(z)$
        -   Unknown Parameters $\theta = (\mu, \sigma^2)$
    -   Likelihood Function
        -   $L(\theta, Z) = \prod g_\theta(z_i)$
        -   Probability of observed data under the model $g_\theta$
        -   Usually work with log-likelihood: $\ell(\theta, Z) = \sum \log g_\theta(z_i)$
    -   Maximize the Likelihood function
        -   Select the parameters $\theta$ such that the probability of observed data is maximized under the model
        -   For many distributions, has closed-form solution
    -   Score Function $\frac{\partial L}{\partial \theta}$
        -   Vector of partial derivatives of log-likelihood
        -   At MLE: $S(\hat{\theta}) = 0$
    -   Information Matrix $I(\theta) = -E\left[\frac{\partial^2 \log L}{\partial \theta^2}\right]$
        -   Expected curvature of log-likelihood
        -   Measures information data contains about parameters
    -   Fisher Information $i(\theta) = I(\theta)_{\hat \theta}$
        -   Evaluated at the MLE
-   Sampling Distribution of MLE has limiting normal distribution
    -   $\hat\theta \sim N(\theta, I(\theta)^{-1})$
    -   Asymptotic result (as N → ∞)
    -   Allows construction of confidence intervals and hypothesis tests
-   OLS estimates are equivalent to MLE estimates for Linear Regression with Gaussian errors
    -   $\text{Var}(\hat \beta) = \sigma^2 / S_{xx}$
    -   $\text{Var}(\hat y_i) = \sigma^2 X_i^T (X^TX)^{-1} X_i$
    -   For non-Gaussian errors, OLS still gives unbiased estimates but may not be efficient

## Bootstrap

-   Bootstrap assesses uncertainty by sampling from training data
    -   Estimate different models using bootstrap datasets
    -   Calculate the variance of estimates for ith observation from these models
    -   Provides empirical sampling distribution when theoretical one is unavailable
-   Non-Parametric Bootstrap
    -   Uses raw data for sampling, model free
    -   Makes minimal assumptions about data distribution
    -   Approaches:
        -   Case resampling: Sample observations with replacement
        -   Residual resampling: Resample residuals and add to fitted values
-   Parametric Bootstrap
    -   Simulate new target variable by adding gaussian noise to predicted values from model
    -   Predictions estimated from this sampling will follow Gaussian distribution
    -   Assumes error distribution is correctly specified
-   Computational alternative to MLE
    -   No formulae are available
    -   Especially useful for complex models or statistics
-   Bootstrap mean is equivalent to posterior average in Bayesian inference
    -   Under certain conditions, has Bayesian interpretation
-   Bagging averages predictions over collection of bootstrap samples
    -   Reduces variance of estimates
    -   Bagging often decreases mean-squared error
    -   Most effective for high-variance, low-bias models (like decision trees)
-   Bootstrap confidence intervals
    -   Percentile method: Use empirical quantiles from bootstrap distribution
    -   BCa method: Bias-corrected and accelerated, adjusts for bias and skewness

## Bayesian Methods

-   Assume a prior distribution over unknown parameters
    -   $P(\theta)$
    -   Encodes initial beliefs about parameters before seeing data
    -   Types:
        -   Informative priors: strong beliefs about parameters
        -   Non-informative priors: minimal assumptions (e.g., uniform)
        -   Conjugate priors: result in posterior of same family as prior
-   Sampling Distribution of data given the parameters
    -   $P(Z | \theta)$
    -   Likelihood function from frequentist approach
-   Posterior Distribution
    -   Updated knowledge of parameters after seeing the data
    -   $P(\theta | Z) \propto P(Z | \theta) \times P(\theta)$
    -   Full distribution rather than point estimate
    -   Permits probabilistic statements about parameters
-   Predictive Distribution
    -   Predicting values of new unseen observations
    -   $P(z | Z) = \int P(z | \theta) P(\theta | Z) d\theta$
    -   Integrates over all possible parameter values, weighted by posterior
    -   Accounts for parameter uncertainty unlike plug-in estimates
-   MAP Estimate
    -   Maximum a Posterior, point estimate of unknown parameters
    -   Select the parameters that maximize posterior density function
    -   $\hat \theta = \arg \max P(\theta | Z)$
    -   Compromise between MLE and fully Bayesian approach
-   MAP differs from frequentist approaches (like MLE) in its use of prior distribution
    -   Prior Distribution acts as regularization
    -   MAP for linear regression with Gaussian priors yields Ridge Regression
    -   MAP for linear regression with Laplace priors yields Lasso Regression
-   Hierarchical Bayesian models
    -   Place priors on hyperparameters
    -   Allows borrowing strength across groups
    -   Naturally handles multilevel/grouped data

## EM Algorithm

-   Simplifies difficult MLE problems involving latent variables
-   Applications:
    -   Missing data
    -   Mixture models
    -   Latent variable models
    -   Hidden Markov models
-   Bimodal Data Distribution
    -   $Y_1 \sim N(\mu_1, \sigma^2_1)$
    -   $Y_2 \sim N(\mu_2, \sigma^2_2)$
    -   $Y = \Delta Y_1 + (1 - \Delta) Y_2$
        -   $\Delta \in \{0,1\}$
        -   $P(\Delta = 1) = \pi$
    -   Density function of Y
        -   $g_Y(y) = (1 - \pi) \phi_1(y) + \pi \phi_2(y)$
-   Direct maximization of likelihood difficult
    -   Sum operation inside log
    -   $\log L(\theta) = \sum_{i=1}^N \log[(1-\pi)\phi_1(y_i) + \pi\phi_2(y_i)]$
-   Responsibility
    -   $\Delta_i$ is latent for a given observation
    -   $\gamma_i(\theta) = P(\Delta_i = 1 | y_i, \theta)$
    -   Soft Assignments
    -   Posterior probability of component membership
-   EM Algorithm
    -   Take Initial Guesses for parameters
        -   Sample Mean, Sample Variances, Proportion
        -   Can use K-means or random initialization
    -   Expectation Step: Compute the responsibility
        -   $\hat \gamma_i = \frac{\hat \pi \phi_2(y_i)}{(1 - \hat \pi) \phi_1(y_i) + \hat \pi \phi_2(y_i)}$
        -   Calculate expected value of log-likelihood with respect to latent variables
    -   Maximization Step: Compute the weighted means and variances, and mixing probability
        -   $\mu_1 = \frac{\sum (1 - \hat \gamma_i) y_i}{\sum (1 - \hat \gamma_i)}$
        -   $\mu_2 = \frac{\sum \hat \gamma_i y_i}{\sum \hat \gamma_i}$
        -   $\hat \pi = \frac{\sum \gamma_i}{N}$
        -   Maximize the expected log-likelihood from E-step
    -   Iterate until convergence
    -   Properties:
        -   Monotonic likelihood increase
        -   Convergence to local maximum guaranteed
        -   Multiple restarts may be needed to find global maximum

## MCMC

-   Given a set of random variables $U_1, U_2, U_3...$
    -   Sampling from joint distribution is difficult
    -   Sampling from conditional distribution is easy
    -   For example bayesian inference
        -   Joint distribution $P(Z, \theta)$
        -   Conditional Distribution $P(Z | \theta)$
-   Gibbs Sampling
    -   Take Some initial values of RVs $U^0_k$
    -   Draw from conditional Distribution
        -   $P(U_1 | U_2^{(t)}, U_3^{(t)},..., U_K^{(t)})$
        -   $P(U_2 | U_1^{(t+1)}, U_3^{(t)},..., U_K^{(t)})$
        -   And so on, updating each variable in turn
    -   Continue until the joint distribution doesn't change
    -   Markov Chain whose stationary distribution is the true joint distribution
    -   Markov Chain Monte Carlo
-   Metropolis-Hastings Algorithm
    -   More general MCMC approach than Gibbs sampling
    -   Steps:
        1. Generate proposal $\theta^* \sim q(\theta^*|\theta^{(t)})$
        2. Calculate acceptance ratio $r = \min\left(1, \frac{p(\theta^*|Z)q(\theta^{(t)}|\theta^*)}{p(\theta^{(t)}|Z)q(\theta^*|\theta^{(t)})}\right)$
        3. Accept proposal with probability r
    -   Special cases include random walk and independent proposals
-   Practical considerations
    -   Burn-in period: discard initial samples
    -   Thinning: use every kth sample to reduce autocorrelation
    -   Convergence diagnostics: trace plots, Gelman-Rubin statistic
-   Gibbs Sampling is related to EM algorithm
    -   Generate $\Delta_i \in {0,1}$ using $p(\Delta_i = 1) = \gamma_i (\theta)$
    -   Calculate the means and variances
        -   $\mu_1 = \frac{\sum (1 - \Delta_i) y_i}{\sum (1 - \Delta_i)}$
        -   $\mu_2 = \frac{\sum \Delta_i y_i}{\sum \Delta_i}$
    -   Keep repeating until the joint distribution doesn't change
    -   EM finds mode of posterior, MCMC explores full posterior distribution 
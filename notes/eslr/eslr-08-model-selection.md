# Model Inference and Averaging

This chapter covers the statistical foundations of model fitting and methods for combining multiple models. We'll explore Maximum Likelihood Estimation, Bayesian methods, the EM algorithm, and Markov Chain Monte Carlo (MCMC) — tools that underpin much of modern machine learning.

## The Big Picture

So far we've focused on finding the "best" model. But we haven't asked:
- **How confident are we** in our parameter estimates?
- **What if the data came from a mixture** of underlying processes?
- **Can we combine multiple models** for better predictions?

This chapter provides the statistical machinery to answer these questions.

---

## Maximum Likelihood Estimation (MLE)

MLE is the most widely used approach for fitting statistical models. The intuition is simple: choose parameters that make the observed data most likely.

### The Setup

We have:
- Data: $Z = \{z_1, z_2, ..., z_N\}$
- Parametric model: $z_i \sim g_\theta(z)$
- Unknown parameters: $\theta$ (could be $\mu$, $\sigma^2$, etc.)

### The Likelihood Function

The likelihood is the probability of observing our data, viewed as a function of the parameters:

$$L(\theta; Z) = \prod_{i=1}^N g_\theta(z_i)$$

**Key insight**: The likelihood treats the data as fixed and varies the parameters. It answers: "For different values of θ, how probable was it to see exactly this data?"

### Log-Likelihood

Products are numerically unstable, so we usually work with the log-likelihood:

$$\ell(\theta; Z) = \sum_{i=1}^N \log g_\theta(z_i)$$

Taking logs turns products into sums — much easier to optimize!

### Finding the MLE

The maximum likelihood estimate $\hat{\theta}$ maximizes $L(\theta)$ (equivalently, $\ell(\theta)$):

$$\hat{\theta} = \arg\max_\theta \ell(\theta; Z)$$

For many distributions (Normal, Binomial, etc.), we can solve this analytically by:
1. Taking the derivative (the **score function**): $S(\theta) = \frac{\partial \ell}{\partial \theta}$
2. Setting it to zero: $S(\hat{\theta}) = 0$
3. Solving for $\hat{\theta}$

### The Information Matrix

How curved is the log-likelihood around the maximum? This tells us how "sharp" or "flat" the peak is — sharper = more confident in our estimate.

The **Information Matrix** captures this curvature:

$$I(\theta) = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]$$

**Fisher Information** is this evaluated at the MLE: $i(\theta) = I(\theta)|_{\hat{\theta}}$

### Sampling Distribution of the MLE

Under regularity conditions, as N → ∞:

$$\hat{\theta} \sim N(\theta, I(\theta)^{-1})$$

**What this means**:
- The MLE is asymptotically unbiased (centered on truth)
- Its variance is the inverse of the information matrix
- We can construct confidence intervals and hypothesis tests!

### Example: Linear Regression

For linear regression with Gaussian errors, OLS gives the MLE:

$$\text{Var}(\hat{\beta}) = \sigma^2(X^TX)^{-1}$$
$$\text{Var}(\hat{y}_i) = \sigma^2 X_i^T(X^TX)^{-1}X_i$$

For non-Gaussian errors, OLS is still unbiased but may not be the most efficient estimator.

---

## Bootstrap

The bootstrap is a powerful resampling technique that provides uncertainty estimates when theoretical formulas don't exist (or are too complex).

### The Core Idea

Pretend your training data IS the population. Resample from it with replacement to simulate "new datasets." The variation across these resampled datasets estimates uncertainty.

### Non-Parametric Bootstrap

1. Sample N observations **with replacement** from your data
2. Fit your model to this bootstrap sample
3. Repeat B times (typically B = 100-1000)
4. The distribution of estimates approximates the sampling distribution

**Common approaches**:
- **Case resampling**: Sample entire (X, Y) pairs with replacement
- **Residual resampling**: Fit model, resample residuals, add to fitted values

### Parametric Bootstrap

1. Fit model to original data
2. Simulate new data from the fitted model (add Gaussian noise to predictions)
3. Refit model to simulated data
4. Repeat and analyze distribution

This assumes you know the correct error distribution (often Gaussian).

### Why Bootstrap Works

Under certain conditions, the bootstrap distribution of $\hat{\theta}^* - \hat{\theta}$ (difference between bootstrap and original estimate) approximates the true sampling distribution of $\hat{\theta} - \theta$.

### Bootstrap Confidence Intervals

**Percentile Method**: Use the 2.5th and 97.5th percentiles of bootstrap estimates as a 95% CI.

**BCa Method** (Bias-Corrected and Accelerated): Adjusts for bias and skewness — more accurate but more complex.

### Connection to Bayesian Inference

There's a remarkable connection: under certain priors, the bootstrap distribution approximates the Bayesian posterior distribution!

### Bagging: Bootstrap for Prediction

**Bagging** (Bootstrap Aggregating) averages predictions across bootstrap samples:

1. Generate B bootstrap samples
2. Fit a model to each
3. Average predictions (regression) or vote (classification)

**Why it helps**: Reduces variance, especially for unstable models like decision trees.

---

## Bayesian Methods

Bayesian inference takes a fundamentally different view: parameters are random variables with their own probability distributions.

### Prior, Likelihood, Posterior

**Prior** $P(\theta)$
- What we believe about parameters BEFORE seeing data
- Encodes prior knowledge or assumptions

**Likelihood** $P(Z|\theta)$
- Probability of the data given parameters
- Same as in MLE

**Posterior** $P(\theta|Z)$
- Updated beliefs AFTER seeing data
- This is what we want!

**Bayes' Theorem**:
$$P(\theta|Z) \propto P(Z|\theta) \times P(\theta)$$

"Posterior is proportional to Likelihood times Prior"

### Types of Priors

**Informative Priors**: Strong beliefs based on domain knowledge
- Example: "The coefficient is probably between 0 and 1"

**Non-informative Priors**: Minimal assumptions
- Example: Uniform distribution over all possible values
- Let the data speak!

**Conjugate Priors**: Mathematical convenience — the posterior has the same form as the prior
- Example: Gaussian prior + Gaussian likelihood → Gaussian posterior

### The Posterior Distribution

Unlike MLE (which gives a single point estimate), Bayesian inference gives a **full distribution** over parameters. This lets us:
- Quantify uncertainty directly
- Make probabilistic statements ("there's a 95% probability that β is between 0.3 and 0.7")
- Incorporate prior knowledge naturally

### Predictive Distribution

To predict a new observation, we don't just plug in a single parameter value. We integrate over all possible values:

$$P(z_{\text{new}}|Z) = \int P(z_{\text{new}}|\theta) P(\theta|Z) d\theta$$

This **accounts for parameter uncertainty** — predictions are more honest about what we don't know.

### MAP Estimation

**Maximum A Posteriori (MAP)** is a compromise: find the single most probable parameter value under the posterior.

$$\hat{\theta}^{MAP} = \arg\max_\theta P(\theta|Z) = \arg\max_\theta [P(Z|\theta) \cdot P(\theta)]$$

Or equivalently:
$$\hat{\theta}^{MAP} = \arg\max_\theta [\log P(Z|\theta) + \log P(\theta)]$$

**Key insight**: MAP = MLE + regularization penalty from the prior!

- **Gaussian prior** → L2 penalty → Ridge Regression
- **Laplace prior** → L1 penalty → Lasso Regression

### Hierarchical Bayesian Models

Sometimes we have grouped data (e.g., students within schools). **Hierarchical models** place priors on hyperparameters, allowing "borrowing strength" across groups.

Example: Estimating school-level effects
- Each school has its own mean μ_k
- But the μ_k's come from a common distribution N(μ, τ²)
- We estimate μ and τ² from all schools together

This naturally handles the bias-variance tradeoff across groups!

---

## The EM Algorithm

The **Expectation-Maximization (EM)** algorithm is an elegant solution for maximum likelihood when data is "incomplete" — either literally missing or involving latent (hidden) variables.

### Motivating Example: Gaussian Mixture Models

Suppose your data comes from a mixture of two Gaussians:
- Component 1: $N(\mu_1, \sigma^2_1)$ with probability $\pi$
- Component 2: $N(\mu_2, \sigma^2_2)$ with probability $1-\pi$

The density is:
$$g(y) = (1-\pi)\phi_1(y) + \pi\phi_2(y)$$

### The Problem with Direct MLE

The log-likelihood is:
$$\ell(\theta) = \sum_{i=1}^N \log[(1-\pi)\phi_1(y_i) + \pi\phi_2(y_i)]$$

The sum is INSIDE the log — no nice closed-form solution!

### The Latent Variable Perspective

Imagine we knew which component each observation came from (a latent indicator $\Delta_i \in \{0,1\}$). Then MLE would be easy!

**The EM insight**: We don't know $\Delta_i$, but we can compute its expected value given current parameter estimates.

### The Algorithm

**1. Initialize**: Start with guesses for all parameters (e.g., sample means and variances)

**2. E-Step (Expectation)**: Compute "soft" assignments — the probability each point belongs to each component:

$$\gamma_i = P(\Delta_i = 1|y_i, \theta) = \frac{\hat{\pi}\phi_2(y_i)}{(1-\hat{\pi})\phi_1(y_i) + \hat{\pi}\phi_2(y_i)}$$

This is called the **responsibility** — how responsible is component 2 for observation i?

**3. M-Step (Maximization)**: Update parameters using weighted averages:

$$\hat{\mu}_1 = \frac{\sum(1-\gamma_i)y_i}{\sum(1-\gamma_i)}$$
$$\hat{\mu}_2 = \frac{\sum\gamma_i y_i}{\sum\gamma_i}$$
$$\hat{\pi} = \frac{\sum\gamma_i}{N}$$

**4. Repeat** until convergence (parameters stop changing).

### Key Properties of EM

1. **Monotonic**: Each iteration increases the likelihood (never goes down)
2. **Converges**: Always reaches a fixed point
3. **Local optima**: May not find the global maximum — try multiple initializations!
4. **Slow near convergence**: Can take many iterations to converge precisely

### Applications of EM

- **Mixture models**: Clustering with soft assignments
- **Missing data**: Impute missing values, then estimate
- **Hidden Markov Models**: Speech recognition, genomics
- **Factor analysis**: Latent variable models

---

## Markov Chain Monte Carlo (MCMC)

When posteriors are too complex for closed-form solutions, MCMC provides a way to **sample** from them numerically.

### The Challenge

In Bayesian inference, we need:
$$P(\theta|Z) = \frac{P(Z|\theta)P(\theta)}{P(Z)}$$

But the denominator $P(Z) = \int P(Z|\theta)P(\theta)d\theta$ is often intractable!

### The MCMC Idea

Instead of computing the posterior exactly, generate **samples** from it. With enough samples, we can estimate anything we want (means, quantiles, etc.).

### Gibbs Sampling

When you have multiple parameters and can sample from **conditional distributions** (one parameter at a time, given all others):

1. Initialize all parameters: $\theta^{(0)} = (\theta_1^{(0)}, \theta_2^{(0)}, ..., \theta_K^{(0)})$
2. For each iteration t:
   - Sample $\theta_1^{(t+1)} \sim P(\theta_1|\theta_2^{(t)}, \theta_3^{(t)}, ..., \theta_K^{(t)}, Z)$
   - Sample $\theta_2^{(t+1)} \sim P(\theta_2|\theta_1^{(t+1)}, \theta_3^{(t)}, ..., \theta_K^{(t)}, Z)$
   - ...and so on for all parameters
3. After a burn-in period, keep the samples

**Key property**: The sequence of samples forms a Markov chain whose stationary distribution is the true joint posterior!

### Metropolis-Hastings Algorithm

A more general MCMC approach:

1. **Propose** a new value: $\theta^* \sim q(\theta^*|\theta^{(t)})$
2. **Compute** acceptance ratio:
   $$r = \min\left(1, \frac{P(\theta^*|Z) \cdot q(\theta^{(t)}|\theta^*)}{P(\theta^{(t)}|Z) \cdot q(\theta^*|\theta^{(t)})}\right)$$
3. **Accept** $\theta^*$ with probability r; otherwise keep $\theta^{(t)}$

**Special cases**:
- Random walk proposals: $q(\theta^*|\theta) = N(\theta, \sigma^2)$
- Independent proposals: $q(\theta^*|\theta) = g(\theta^*)$

### Practical Considerations

**Burn-in**: Discard early samples (they depend on initialization)

**Thinning**: Keep every k-th sample to reduce autocorrelation

**Convergence diagnostics**:
- Trace plots: Should look like "noise" around a stable value
- Gelman-Rubin statistic: Compare multiple chains

### EM vs. MCMC

| Aspect | EM | MCMC |
|--------|----|----- |
| Output | Point estimate (mode) | Samples from full posterior |
| Speed | Usually faster | Can be slow |
| Uncertainty | Limited | Full posterior available |
| Local optima | Can get stuck | Explores full space |

---

## Summary: Choosing an Inference Method

| Method | When to Use | What You Get |
|--------|-------------|--------------|
| **MLE** | Large samples, well-specified model | Point estimate + asymptotic inference |
| **Bootstrap** | Unknown distribution, complex statistics | Empirical confidence intervals |
| **Bayesian + MCMC** | Prior knowledge, need full uncertainty | Full posterior distribution |
| **EM** | Latent variables, mixture models | MLE for incomplete data |
| **MAP** | Want regularization with Bayesian interpretation | Regularized point estimate |

### Key Takeaways

1. **MLE is the workhorse**: Simple, interpretable, asymptotically optimal
2. **Bootstrap is versatile**: Works when theory fails
3. **Bayesian methods quantify uncertainty**: But require prior choices
4. **EM handles hidden structure**: Essential for clustering and missing data
5. **MCMC explores complex posteriors**: Powerful but computationally intensive

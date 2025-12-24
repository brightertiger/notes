# Statistics

Statistics is the science of learning from data. This chapter covers the key concepts for estimating model parameters and quantifying uncertainty in those estimates.

## The Big Picture

**Inference**: Quantifying uncertainty about unknown quantities using finite data samples.

Two major paradigms:
- **Frequentist**: Parameters are fixed; uncertainty comes from random sampling
- **Bayesian**: Parameters are random variables with prior distributions

---

## Maximum Likelihood Estimation (MLE)

The most common approach to parameter estimation: choose parameters that make the observed data most probable.

### The Setup

Given:
- Data: $D = \{x_1, x_2, ..., x_N\}$ (assumed i.i.d.)
- Parametric model: $p(x | \theta)$

### The Likelihood Function

$$L(\theta; D) = p(D | \theta) = \prod_{i=1}^N p(x_i | \theta)$$

**Key insight**: We treat the data as fixed and vary θ. For which θ was this data most likely?

### Log-Likelihood

Products are numerically unstable. Convert to sums using logs:

$$\ell(\theta; D) = \log L(\theta; D) = \sum_{i=1}^N \log p(x_i | \theta)$$

### The MLE Estimate

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta; D) = \arg\min_\theta -\ell(\theta; D)$$

Equivalently: minimize the **Negative Log-Likelihood (NLL)**.

### Why MLE Works

**Theoretical justifications**:
1. **Bayesian view**: MLE is MAP estimate with uniform (uninformative) prior
2. **Information-theoretic view**: MLE minimizes KL divergence between model and empirical distribution

### Sufficient Statistics

A **sufficient statistic** summarizes all information in the data relevant to estimating θ.

**Example (Bernoulli)**: For $N$ coin flips, the sufficient statistics are:
- $N_1$ = number of heads
- $N_0$ = number of tails

You don't need to know the order of the flips!

---

## MLE Examples

### Bernoulli Distribution

Model: $p(y | \theta) = \theta^y (1-\theta)^{1-y}$

NLL:
$$\text{NLL}(\theta) = -[N_1 \log\theta + N_0 \log(1-\theta)]$$

Setting derivative to zero:
$$\hat{\theta}_{MLE} = \frac{N_1}{N_0 + N_1} = \frac{\text{# heads}}{\text{# flips}}$$

**Intuitive result**: Estimate probability as observed frequency.

### Gaussian Distribution

Model: $p(y | \mu, \sigma^2) = \mathcal{N}(y | \mu, \sigma^2)$

MLE estimates:
$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^N y_i \quad \text{(sample mean)}$$
$$\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{\mu})^2 \quad \text{(sample variance)}$$

### Linear Regression

Model: $p(y | x, w, \sigma^2) = \mathcal{N}(y | w^T x + b, \sigma^2)$

NLL is proportional to:
$$\text{NLL} \propto \sum_{i=1}^N (y_i - w^T x_i - b)^2$$

**Key insight**: MLE for Gaussian regression = minimize squared error!

---

## Empirical Risk Minimization

ERM generalizes MLE beyond log-loss to any loss function.

### Definition

$$\hat{\theta}_{ERM} = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \ell(y_i, f(x_i; \theta))$$

**Common loss functions**:
- Log-loss: gives MLE
- Squared loss: regression
- 0-1 loss: classification accuracy
- Hinge loss: SVMs

### Surrogate Losses

The 0-1 loss is non-differentiable. We use smooth **surrogate losses** that are easier to optimize:
- Log-loss (cross-entropy)
- Hinge loss
- Exponential loss

---

## Online Learning

When data arrives sequentially, we can't afford to retrain from scratch each time.

### Recursive Updates

Many statistics can be updated incrementally:

**Running mean**:
$$\mu_t = \mu_{t-1} + \frac{1}{t}(x_t - \mu_{t-1})$$

**Exponentially Weighted Moving Average (EWMA)**:
$$\mu_t = \beta \mu_{t-1} + (1-\beta) x_t$$

---

## Regularization

MLE can overfit — the estimated model fits training data perfectly but fails on new data.

### The Problem

- Empirical distribution ≠ true distribution
- MLE finds parameters optimal for the empirical distribution
- May not generalize well

### The Solution: Add a Penalty

$$\hat{\theta} = \arg\min_\theta \left[\text{NLL}(\theta) + \lambda R(\theta)\right]$$

Where $R(\theta)$ penalizes complex models.

### MAP Estimation

From the Bayesian view, regularization corresponds to adding a prior:

$$\hat{\theta}_{MAP} = \arg\max_\theta [p(D | \theta) \cdot p(\theta)]$$

Taking logs:
$$\hat{\theta}_{MAP} = \arg\min_\theta [-\log p(D|\theta) - \log p(\theta)]$$

**Examples**:
- Gaussian prior → L2 regularization (Ridge)
- Laplace prior → L1 regularization (Lasso)

### Choosing Regularization Strength

The regularization parameter λ controls the bias-variance trade-off.

**Methods to choose λ**:
- **Validation set**: Test on held-out data
- **Cross-validation**: For small datasets
- **One Standard Error Rule**: Choose simplest model within one SE of best

### Early Stopping

Another form of regularization: stop training before the model overfits.
- Monitor validation error
- Stop when it starts increasing

---

## Bayesian Statistics

The Bayesian approach treats parameters as random variables.

### The Bayesian Recipe

1. **Prior**: $p(\theta)$ — initial beliefs before seeing data
2. **Likelihood**: $p(D | \theta)$ — probability of data given parameters
3. **Posterior**: $p(\theta | D) \propto p(D | \theta) \cdot p(\theta)$ — updated beliefs

### Posterior Predictive Distribution

To predict new data, **integrate over parameter uncertainty**:

$$p(y_{new} | x_{new}, D) = \int p(y_{new} | x_{new}, \theta) \cdot p(\theta | D) d\theta$$

**Compare to plug-in prediction**: $p(y_{new} | x_{new}, \hat{\theta})$

The Bayesian approach properly accounts for uncertainty in θ!

### Conjugate Priors

When the prior and posterior have the same functional form:
- **Bernoulli-Beta**: Prior on coin bias
- **Gaussian-Gaussian**: Prior on Gaussian mean
- **Poisson-Gamma**: Prior on Poisson rate

Makes computation tractable.

### MAP vs. Full Bayesian

| Aspect | MAP | Full Bayesian |
|--------|-----|---------------|
| Output | Point estimate | Full distribution |
| Computation | Optimization | Integration |
| Uncertainty | Not captured | Fully captured |
| Regularization | Equivalent to adding prior | Built-in |

---

## Frequentist Statistics

In the frequentist view:
- Parameters θ are fixed (unknown) constants
- Data D is random (sampled from true distribution)
- Uncertainty comes from randomness in sampling

### Sampling Distribution

If we repeated the experiment many times, our estimate $\hat{\theta}$ would vary. The **sampling distribution** describes this variation.

### Bootstrap

When the true sampling distribution is unknown, approximate it by resampling:

1. Draw N samples **with replacement** from your data
2. Compute the statistic of interest
3. Repeat many times
4. The distribution of statistics approximates the sampling distribution

**Key fact**: Each bootstrap sample contains ~63.2% of unique original observations:
$$P(\text{included}) = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - \frac{1}{e} \approx 0.632$$

### Confidence Intervals

A 95% confidence interval means: if we repeated the experiment many times, 95% of the computed intervals would contain the true parameter.

**Note**: This is NOT the same as "95% probability that θ is in this interval"!

---

## Bias-Variance Trade-off

### Bias

How far off is our estimator on average?

$$\text{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta^*$$

**Unbiased**: $\mathbb{E}[\hat{\theta}] = \theta^*$

**Example**: Sample variance $\frac{1}{N}\sum(x_i - \bar{x})^2$ is biased! The unbiased version divides by N-1.

### Variance

How much does our estimate fluctuate across different datasets?

$$\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$$

### Mean Squared Error

Combines both:
$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta^*)^2] = \text{bias}^2 + \text{variance}$$

**Key insight**: Sometimes it's worth accepting bias if it substantially reduces variance!

This is exactly what regularization does.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| **MLE** | Choose θ that maximizes probability of observed data |
| **NLL** | Negative log-likelihood; what we minimize |
| **Sufficient Statistics** | Compress data without losing information about θ |
| **ERM** | Generalization of MLE to any loss function |
| **Regularization** | Penalty on complexity to prevent overfitting |
| **MAP** | MLE + prior = regularized MLE |
| **Bayesian** | Full distribution over θ, not just point estimate |
| **Posterior Predictive** | Integrate predictions over parameter uncertainty |
| **Bootstrap** | Approximate sampling distribution by resampling |
| **Bias-Variance** | MSE = bias² + variance; trade-off is fundamental |

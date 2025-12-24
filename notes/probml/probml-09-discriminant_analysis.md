# Discriminant Analysis

Discriminant analysis covers two fundamental approaches to classification: generative and discriminative models. Understanding their differences helps you choose the right approach for your problem.

## Generative vs. Discriminative Models

### Discriminative Models

Model the **posterior probability** directly:
$$p(y | x, \theta)$$

**Approach**: Learn the decision boundary between classes.

**Examples**: Logistic regression, SVMs, neural networks

**Advantages**:
- Often more accurate when we have enough data
- Make fewer assumptions
- More robust to model misspecification

### Generative Models

Model the **joint distribution** via class-conditional densities and priors:
$$p(y = c | x, \theta) \propto p(x | y = c, \theta) \times p(y = c)$$

**Components**:
- $p(x | y = c, \theta)$: Class-conditional density — how does data look for class c?
- $p(y = c)$: Prior probability — how common is class c?

**Examples**: Naive Bayes, LDA, QDA, Gaussian mixture models

**Advantages**:
- Can generate new samples
- Handle missing data naturally
- Work well with small datasets
- Can incorporate prior knowledge

---

## Gaussian Discriminant Analysis

Assume class-conditional densities are multivariate Gaussian:
$$p(x | y = c, \theta) = \mathcal{N}(x | \mu_c, \Sigma_c)$$

### Quadratic Discriminant Analysis (QDA)

Each class has its own mean **and** covariance:
$$p(x | y = c) = \mathcal{N}(\mu_c, \Sigma_c)$$

**Log posterior** (up to constant):
$$\log p(y = c | x) = \log \pi_c - \frac{1}{2}\log|\Sigma_c| - \frac{1}{2}(x - \mu_c)^T\Sigma_c^{-1}(x - \mu_c)$$

The decision boundary is **quadratic** in x (hence the name).

### Linear Discriminant Analysis (LDA)

**Key assumption**: All classes share the same covariance matrix.
$$\Sigma_c = \Sigma \quad \text{for all } c$$

This simplifies the log posterior to:
$$\log p(y = c | x) = \log \pi_c + x^T\Sigma^{-1}\mu_c - \frac{1}{2}\mu_c^T\Sigma^{-1}\mu_c$$

The decision boundary is **linear** in x!

**Connection to logistic regression**: LDA can be written in the same form, but makes stronger (Gaussian) assumptions.

### Fitting GDA

**Parameter estimation** (usually via MLE):
- $\hat{\pi}_c = N_c / N$ (class proportions)
- $\hat{\mu}_c = \frac{1}{N_c}\sum_{i: y_i = c} x_i$ (class means)
- $\hat{\Sigma}_c = \frac{1}{N_c}\sum_{i: y_i = c} (x_i - \hat{\mu}_c)(x_i - \hat{\mu}_c)^T$ (class covariances)

For LDA, pool covariances:
$$\hat{\Sigma} = \frac{1}{N}\sum_c \sum_{i: y_i = c} (x_i - \hat{\mu}_c)(x_i - \hat{\mu}_c)^T$$

### LDA vs QDA Trade-off

| Aspect | LDA | QDA |
|--------|-----|-----|
| Parameters | $O(Cd + d^2)$ | $O(Cd + Cd^2)$ |
| Flexibility | Lower | Higher |
| Variance | Lower | Higher |
| Best when | Classes have similar shapes | Classes have different shapes |

**Regularization**: When covariance estimates are unstable, shrink towards LDA:
$$\hat{\Sigma}_c(\alpha) = \alpha\hat{\Sigma}_c + (1-\alpha)\hat{\Sigma}$$

### Nearest Centroid Classifier

Classification simplifies to: assign x to the class with nearest mean (using Mahalanobis distance with Σ⁻¹).

---

## Naive Bayes Classifiers

### The Naive Independence Assumption

Assume features are **conditionally independent** given the class:
$$p(x | y = c) = \prod_{d=1}^D p(x_d | y = c)$$

**Why "naive"?** This assumption is almost never true in practice!

### The Posterior

$$p(y = c | x, \theta) \propto \pi_c \prod_{d=1}^D p(x_d | y = c, \theta_{dc})$$

Each feature contributes independently to the log-posterior.

### Feature Distributions

Choose distribution based on feature type:
- **Binary features**: Bernoulli distribution
- **Categorical features**: Categorical distribution
- **Continuous features**: Gaussian distribution

### Why Naive Bayes Works

Despite the wrong assumption:
1. **Few parameters**: Very sample-efficient
2. **Rankings often correct**: We only need relative, not absolute probabilities
3. **Errors cancel**: Overestimates and underestimates may balance out

### When to Use Naive Bayes

- **Text classification**: High-dimensional, sparse features
- **Small datasets**: Fewer parameters = less overfitting
- **Fast prediction needed**: Inference is very efficient
- **As a baseline**: Simple and hard to beat in some domains

---

## Comparing Approaches

### Generative Advantages

1. **Handle missing data**: Can marginalize out missing features
2. **Semi-supervised learning**: Can use unlabeled data (via EM)
3. **Prior knowledge**: Natural way to incorporate domain knowledge
4. **Sample generation**: Can create synthetic examples

### Discriminative Advantages

1. **Direct objective**: Optimize what we care about
2. **Fewer assumptions**: More robust to model misspecification
3. **Often more accurate**: With enough data
4. **Flexible**: Can model complex decision boundaries

### When to Use Which

| Situation | Recommendation |
|-----------|----------------|
| Small dataset | Generative (LDA, NB) |
| Large dataset | Discriminative (logistic, NN) |
| Missing features | Generative |
| Need probabilities | Either (both can be calibrated) |
| Need interpretability | LDA or logistic regression |
| High dimensions + sparse | Naive Bayes |

---

## Summary

| Model | Assumptions | Decision Boundary | Parameters |
|-------|-------------|-------------------|------------|
| **QDA** | Gaussian per class | Quadratic | $O(Cd^2)$ |
| **LDA** | Gaussian, shared Σ | Linear | $O(Cd + d^2)$ |
| **Naive Bayes** | Conditional independence | Can be nonlinear | $O(Cd)$ |
| **Logistic Regression** | Linear log-odds | Linear | $O(Cd)$ |

**Key insight**: The choice between generative and discriminative is about the bias-variance trade-off. Generative models make stronger assumptions (more bias) but need less data (less variance).

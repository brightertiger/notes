# Probability Foundations

Probability is the mathematical language of uncertainty. In machine learning, it provides the foundation for reasoning about noisy data, model uncertainty, and making predictions. This chapter covers the essential probability concepts you'll use throughout ML.

## Two Views of Probability

### Frequentist View
Probability is the **long-run relative frequency** of an event in repeated experiments.

**Example**: The probability of heads is 0.5 because if you flip a coin many times, about half will be heads.

**Limitation**: What about events that can't be repeated? (e.g., "probability it rains tomorrow")

### Bayesian View
Probability is a **quantification of subjective uncertainty** or degree of belief.

**Example**: "I'm 70% confident it will rain tomorrow" represents my current belief given available evidence.

**Advantage**: Can express uncertainty about any proposition, including model parameters.

### Two Types of Uncertainty

**Epistemic Uncertainty (Model Uncertainty)**
- Uncertainty due to lack of knowledge
- Can be reduced with more data
- Example: Uncertainty about which model is correct

**Aleatoric Uncertainty (Data Uncertainty)**
- Uncertainty due to inherent randomness
- Cannot be reduced even with infinite data
- Example: Outcome of a fair coin flip

---

## Basic Probability Rules

### Events and Probabilities

An **event** A is some state of the world that either holds or doesn't.

**Probability axioms**:
- $0 \leq P(A) \leq 1$ (probabilities are between 0 and 1)
- $P(A) + P(\bar{A}) = 1$ (something happens or it doesn't)
- $P(\Omega) = 1$ (something must happen)

### Joint Probability

The probability that **both** A and B occur:
$$P(A, B) = P(A \cap B)$$

**Special case — Independence**: If A and B are independent:
$$P(A, B) = P(A) \cdot P(B)$$

**Inclusion-Exclusion Principle**:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

### Conditional Probability

The probability of B **given that** A has occurred:
$$P(B | A) = \frac{P(A \cap B)}{P(A)} \quad \text{where } P(A) > 0$$

**Intuition**: We restrict our attention to the "world where A happened" and ask how likely B is in that world.

**Example**: P(has cancer | positive test) ≠ P(positive test | has cancer)

This is a common source of confusion called the **base rate fallacy**!

---

## Random Variables

A **random variable** is a function that maps outcomes from a sample space to real numbers. This allows mathematical manipulation of random events.

### Discrete Random Variables
Take values from a countable set (integers, categories).
- Example: Number of customers, dice roll, coin flip

### Continuous Random Variables
Take values from an uncountable set (real numbers, intervals).
- Example: Height, temperature, time

---

## Probability Distributions

### Probability Mass Function (PMF)

For discrete random variables, the PMF gives the probability of each value:
$$p(x) = P(X = x)$$

**Properties**:
- $0 \leq p(x) \leq 1$ for all x
- $\sum_x p(x) = 1$ (probabilities sum to 1)

### Probability Density Function (PDF)

For continuous random variables, the PDF describes relative likelihood:
$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

**Important**: For continuous variables, $P(X = x) = 0$ for any specific x!

**Properties**:
- $f(x) \geq 0$ (but can be greater than 1!)
- $\int_{-\infty}^{\infty} f(x) dx = 1$

### Cumulative Distribution Function (CDF)

The probability that X is less than or equal to x:
$$F_X(x) = P(X \leq x)$$

**Properties**:
- Monotonically non-decreasing
- $\lim_{x \to -\infty} F_X(x) = 0$
- $\lim_{x \to \infty} F_X(x) = 1$
- $P(a \leq X \leq b) = F_X(b) - F_X(a)$

**Relationship**: PDF is the derivative of CDF.

### Inverse CDF (Quantile Function)

Given a probability, find the value:
$$F^{-1}(q) = \inf\{x : F(x) \geq q\}$$

**Common quantiles**:
- $F^{-1}(0.5)$ = median
- $F^{-1}(0.25)$, $F^{-1}(0.75)$ = lower and upper quartiles

---

## Working with Multiple Variables

### Marginal Distribution

Given joint distribution $p(X, Y)$, the marginal distribution of X:
$$p(X = x) = \sum_y p(X = x, Y = y)$$

**Intuition**: "Sum out" the variable you don't care about.

### Conditional Distribution

$$p(Y = y | X = x) = \frac{p(X = x, Y = y)}{p(X = x)}$$

### Product Rule

$$p(X, Y) = p(Y | X) \cdot p(X) = p(X | Y) \cdot p(Y)$$

### Chain Rule

For multiple variables:
$$p(X_1, X_2, X_3) = p(X_1) \cdot p(X_2 | X_1) \cdot p(X_3 | X_1, X_2)$$

### Independence

X and Y are **independent** if:
$$X \perp Y \Leftrightarrow p(X, Y) = p(X) \cdot p(Y)$$

**Equivalently**: Knowing X tells you nothing about Y.

### Conditional Independence

X and Y are **conditionally independent** given Z if:
$$X \perp Y | Z \Leftrightarrow p(X, Y | Z) = p(X | Z) \cdot p(Y | Z)$$

**Example**: Given the weather, whether I carry an umbrella is independent of whether you carry one. But marginally (without knowing the weather), they're correlated!

---

## Summary Statistics

### Expected Value (Mean)

The "center" of a distribution — the average value weighted by probability:
$$\mathbb{E}[X] = \sum_x x \cdot p(x) \quad \text{or} \quad \int_{-\infty}^{\infty} x \cdot f(x) dx$$

**Linearity of Expectation** (extremely useful!):
$$\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$$
$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \quad \text{(always, even if not independent!)}$$

### Variance

How spread out the distribution is:
$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Properties**:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$

### Mode

The most probable value:
$$\text{mode} = \arg\max_x p(x)$$

---

## Laws of Iterated Expectations

### Law of Total Expectation
$$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X | Y]]$$

**Intuition**: The overall average equals the average of conditional averages.

### Law of Total Variance
$$\text{Var}(X) = \mathbb{E}[\text{Var}(X | Y)] + \text{Var}(\mathbb{E}[X | Y])$$

**Interpretation**:
- First term: Average variance within groups
- Second term: Variance between group means

---

## Bayes' Rule

The foundation of Bayesian inference:
$$P(H | Y) = \frac{P(Y | H) \cdot P(H)}{P(Y)}$$

**Components**:
- $P(H)$: **Prior** — belief before seeing data
- $P(Y | H)$: **Likelihood** — probability of data given hypothesis
- $P(H | Y)$: **Posterior** — updated belief after seeing data
- $P(Y)$: **Evidence** — normalizing constant

**The Bayesian Recipe**:
$$\text{Posterior} \propto \text{Prior} \times \text{Likelihood}$$

---

## Common Distributions

### Bernoulli Distribution

Models a single binary outcome:
$$Y \sim \text{Ber}(\theta)$$
$$p(Y = y) = \theta^y (1 - \theta)^{1-y} \quad \text{for } y \in \{0, 1\}$$

- Mean: $\theta$
- Variance: $\theta(1 - \theta)$

### Binomial Distribution

Models number of successes in N independent Bernoulli trials:
$$p(k | N, \theta) = \binom{N}{k} \theta^k (1 - \theta)^{N-k}$$

- Mean: $N\theta$
- Variance: $N\theta(1 - \theta)$

### Logistic Function

The **sigmoid** function maps any real number to (0, 1):
$$\sigma(a) = \frac{1}{1 + e^{-a}}$$

**Properties**:
- $\sigma(-a) = 1 - \sigma(a)$
- $\sigma'(a) = \sigma(a)(1 - \sigma(a))$

**Log-odds (logit)**: The inverse transformation:
$$a = \log\frac{p}{1-p}$$

**Usage**: Binary classification — map raw scores to probabilities:
$$p(y = 1 | x, \theta) = \sigma(f(x, \theta))$$

### Categorical Distribution

Generalizes Bernoulli to multiple categories:
$$\text{Cat}(y | \theta) = \prod_{c=1}^C \theta_c^{I(y=c)}$$

**Constraints**:
- $0 \leq \theta_c \leq 1$
- $\sum_c \theta_c = 1$

### Softmax Function

Maps raw logits to valid categorical probabilities:
$$\text{softmax}(a)_c = \frac{e^{a_c}}{\sum_{j=1}^C e^{a_j}}$$

**Properties**:
- Output sums to 1
- Each output is in (0, 1)
- Invariant to adding constant to all inputs

**Temperature Scaling**: Divide by temperature T before softmax:
- T → 0: "Winner takes all" (approaches argmax)
- T → ∞: Approaches uniform distribution

### Log-Sum-Exp Trick

For numerical stability when computing softmax:
$$\log \sum_c e^{a_c} = m + \log \sum_c e^{a_c - m}$$

where $m = \max_c a_c$. This prevents overflow!

### Gaussian (Normal) Distribution

The workhorse of statistics:
$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Properties**:
- Mean, median, and mode all equal μ
- Symmetric around the mean
- 68-95-99.7 rule: ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ

**Why Gaussian is so common**:
1. Maximum entropy distribution for given mean and variance
2. Central Limit Theorem: Sum of many independent RVs → Gaussian
3. Mathematical convenience (conjugate to itself)

### Student's t-Distribution

More robust alternative to Gaussian (heavier tails):
- PDF decays polynomially, not exponentially
- Parameter ν (degrees of freedom) controls tail weight
- As ν → ∞, approaches Gaussian
- Better for handling outliers

---

## Transformations of Random Variables

### Discrete Case

If Y = f(X), the PMF of Y:
$$p_Y(y) = \sum_{x : f(x) = y} p_X(x)$$

### Continuous Case (Change of Variables)

If Y = f(X) where f is monotonic and differentiable:
$$p_Y(y) = p_X(f^{-1}(y)) \cdot \left|\frac{d f^{-1}(y)}{dy}\right|$$

The absolute value of the derivative (Jacobian in multivariate case) accounts for how the transformation stretches or compresses probability.

### Convolution

For Y = X₁ + X₂ (sum of independent RVs):
$$p_Y(y) = \int p_{X_1}(x_1) \cdot p_{X_2}(y - x_1) dx_1$$

**Key result**: Sum of Gaussians is Gaussian!

---

## Central Limit Theorem

One of the most important theorems in statistics:

If $X_1, X_2, ..., X_N$ are i.i.d. with mean μ and variance σ², then as N → ∞:
$$\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{N}\right)$$

**Implications**:
- Sample means are approximately Gaussian (for large N)
- Justifies using Gaussian assumptions in many settings
- Variance of sample mean decreases as 1/N

---

## Monte Carlo Approximation

When exact computation is intractable, **sample**!

To estimate $\mathbb{E}[f(X)]$ where $X \sim p(x)$:
1. Draw samples $x_1, x_2, ..., x_N \sim p(x)$
2. Approximate: $\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i)$

This is unbiased and converges by the Law of Large Numbers.

---

## Summary

| Concept | Key Formula | Intuition |
|---------|-------------|-----------|
| **Conditional Probability** | $P(B\|A) = P(A,B)/P(A)$ | Probability in restricted world |
| **Bayes' Rule** | $P(H\|Y) \propto P(Y\|H)P(H)$ | Update beliefs with evidence |
| **Expected Value** | $\mathbb{E}[X] = \sum x \cdot p(x)$ | Probability-weighted average |
| **Variance** | $\text{Var}(X) = \mathbb{E}[(X-\mu)^2]$ | Spread of distribution |
| **Sigmoid** | $\sigma(a) = 1/(1+e^{-a})$ | Map reals to (0,1) |
| **Softmax** | $e^{a_c}/\sum e^{a_j}$ | Map vector to probabilities |
| **CLT** | $\bar{X} \to \mathcal{N}(\mu, \sigma^2/N)$ | Sample means are Gaussian |

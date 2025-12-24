# Decision Theory

Decision theory provides a formal framework for making optimal choices under uncertainty. It bridges the gap between probabilistic predictions and concrete actions.

## The Big Picture

Having a good model isn't enough — you need to **make decisions**. Decision theory tells us how to choose actions that minimize expected loss (or maximize expected utility).

**Key question**: Given uncertainty about the world and different costs for different errors, what should we do?

---

## Risk Attitudes

### Risk Neutrality

A risk-neutral agent values expected outcomes:
- $50 for sure = 50% chance of $100

### Risk Aversion

A risk-averse agent prefers certainty:
- Would take $45 for sure over 50% chance of $100
- Most people are risk-averse for gains

### Risk Seeking

A risk-seeking agent prefers uncertainty:
- Would reject $55 for sure to keep 50% chance of $100
- Gamblers exhibit this behavior

---

## Classification Decision Rules

### Zero-One Loss

The simplest loss: you're either right or wrong.

$$\ell_{01}(y, \hat{y}) = I\{y \neq \hat{y}\} = \begin{cases} 0 & \text{if } y = \hat{y} \\ 1 & \text{if } y \neq \hat{y} \end{cases}$$

**Optimal policy**: Predict the most probable class!

$$\pi^*(x) = \arg\max_c P(Y = c | X = x)$$

**Derivation**: The risk (expected loss) for predicting $\hat{y}$:
$$R(\hat{y} | x) = P(Y \neq \hat{y} | x) = 1 - P(Y = \hat{y} | x)$$

Minimizing risk = maximizing the probability of being correct.

### Cost-Sensitive Classification

Different errors have different consequences!

**Medical diagnosis example**:
- **False Negative** (miss cancer): Potentially fatal
- **False Positive** (false alarm): Unnecessary tests, anxiety

We assign different costs:
- $\ell_{01}$: Cost of predicting 0 when truth is 1 (false negative)
- $\ell_{10}$: Cost of predicting 1 when truth is 0 (false positive)

**Optimal decision rule**: Predict class 1 if:
$$P(Y=0|x) \cdot \ell_{01} < P(Y=1|x) \cdot \ell_{10}$$

**Effect**: Higher cost of false negatives → lower threshold for predicting positive.

### The Confusion Matrix

For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

**Key metrics**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Sensitivity (Recall, TPR)** | TP / (TP + FN) | Of actual positives, how many did we catch? |
| **Specificity (TNR)** | TN / (TN + FP) | Of actual negatives, how many did we correctly identify? |
| **Precision (PPV)** | TP / (TP + FP) | Of predicted positives, how many are correct? |
| **False Positive Rate (FPR)** | FP / (FP + TN) | Of actual negatives, how many did we incorrectly flag? |

### The Rejection Option

Sometimes the best decision is to **not decide**.

**Setup**:
- Cost of error: $\lambda_e$
- Cost of rejection: $\lambda_r$ (where $\lambda_r < \lambda_e$)

**Optimal policy**:
- Predict if confident: $\max_c P(Y=c|x) \geq 1 - \frac{\lambda_r}{\lambda_e}$
- Reject (abstain) if uncertain

**Use case**: Route uncertain cases to human experts.

---

## ROC Curves

The **Receiver Operating Characteristic** curve shows the trade-off between sensitivity and specificity across all classification thresholds.

### Construction

For each threshold τ:
1. Compute TPR (sensitivity) and FPR (1 - specificity)
2. Plot the point (FPR, TPR)

### Interpretation

- **Perfect classifier**: Goes through (0, 1) — 100% TPR, 0% FPR
- **Random classifier**: Diagonal line from (0, 0) to (1, 1)
- **Better models**: Curves closer to upper-left corner

### AUC (Area Under the ROC Curve)

A single number summarizing performance:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC > 0.7: Generally acceptable

**Interpretation**: The probability that a randomly chosen positive example ranks higher than a randomly chosen negative example.

### Equal Error Rate (EER)

The point where FPR = FNR. Lower is better.

---

## Precision-Recall Curves

### When to Use

ROC curves can be misleading with **class imbalance** (when one class is much more common).

**Example**: In fraud detection, 99.9% of transactions are legitimate. A model that predicts "not fraud" always achieves:
- 99.9% accuracy
- 100% specificity
- 0% recall (catches no fraud!)

### PR Curve Construction

For each threshold:
1. Compute Precision and Recall
2. Plot the point (Recall, Precision)

### Key Properties

- No dependence on TN (unlike ROC)
- Sensitive to class imbalance
- Baseline is the positive class proportion

### Summary Metrics

**Precision @ K**: Precision when retrieving top K results

**Average Precision (AP)**: Area under the (interpolated) PR curve

**mAP**: Mean AP across multiple queries/classes

**F-Score**: Harmonic mean of precision and recall
$$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 P + R}$$

- F₁: Equal weight to precision and recall
- F₂: Weights recall higher
- F₀.₅: Weights precision higher

**Why harmonic mean?** It penalizes if either P or R is very low.

---

## Regression Losses

### Common Loss Functions

| Loss | Formula | Properties |
|------|---------|------------|
| **MSE** | $\frac{1}{N}\sum(y - \hat{y})^2$ | Sensitive to outliers; corresponds to Gaussian likelihood |
| **MAE** | $\frac{1}{N}\sum\|y - \hat{y}\|$ | Robust to outliers; corresponds to Laplace likelihood |
| **Huber** | MSE for small errors, MAE for large | Best of both worlds |

### Quantile Loss

For predicting the q-th quantile:
$$L_q(y, \hat{y}) = \begin{cases} q \cdot (y - \hat{y}) & \text{if } y > \hat{y} \\ (1-q) \cdot (\hat{y} - y) & \text{if } y \leq \hat{y} \end{cases}$$

**Asymmetric penalty**: Different costs for over- vs under-prediction.

---

## Model Calibration

A model is **well-calibrated** if its predicted probabilities match actual frequencies.

**Example**: Among all predictions with confidence 80%, about 80% should be correct.

### Reliability Diagrams

- **x-axis**: Predicted probability (binned)
- **y-axis**: Actual proportion of positives in each bin
- **Perfect calibration**: Points fall on the diagonal

### Why Calibration Matters

- For decision making, we need accurate probabilities
- Many models (especially neural networks) are overconfident
- Calibration can be fixed post-hoc (Platt scaling, isotonic regression)

---

## Bayesian Model Selection

### The Bayesian Approach

Choose the model m that maximizes posterior probability:
$$p(m | D) \propto p(D | m) \cdot p(m)$$

**Marginal likelihood** (evidence):
$$p(D | m) = \int p(D | \theta, m) \cdot p(\theta | m) d\theta$$

This integral automatically penalizes complexity (Occam's Razor).

### Model Comparison Criteria

**AIC (Akaike Information Criterion)**
$$\text{AIC} = -2 \cdot \text{LL} + 2k$$

Where k = number of parameters.

**Intuition**: Approximates out-of-sample predictive performance.

**BIC (Bayesian Information Criterion)**
$$\text{BIC} = -2 \cdot \text{LL} + k \cdot \log N$$

Where N = number of observations.

**Intuition**: Approximates Bayesian model evidence; penalizes complexity more heavily than AIC.

### AIC vs BIC

| Criterion | Penalty | Selects | Best For |
|-----------|---------|---------|----------|
| AIC | 2k | Larger models | Prediction |
| BIC | k log N | Smaller models | Finding "true" model |

### MDL (Minimum Description Length)

Information-theoretic view:
$$\text{MDL} = L(\text{model}) + L(\text{data}|\text{model})$$

Choose the model that gives the shortest description of the data.

---

## Frequentist Decision Theory

### Risk of an Estimator

$$R(\theta, \hat{\theta}) = \mathbb{E}_{p(D|\theta)}[L(\theta, \hat{\theta}(D))]$$

The expected loss when the true parameter is θ.

### Types of Risk

**Bayes Risk**: Average over prior on θ
$$R_B = \int R(\theta, \hat{\theta}) p(\theta) d\theta$$

**Maximum Risk**: Worst-case over all θ
$$R_{max} = \max_\theta R(\theta, \hat{\theta})$$

### Empirical Risk Minimization

**Population Risk**: Expected loss on true distribution
$$R(f) = \mathbb{E}_{p^*(x,y)}[\ell(y, f(x))]$$

**Empirical Risk**: Average loss on training data
$$\hat{R}(f) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, f(x_i))$$

**The gap**: Estimation error = $\hat{R}(f) - R(f)$

### Structural Risk Minimization

Add complexity penalty to prevent overfitting:
$$\hat{f} = \arg\min_f \left[\hat{R}(f) + \lambda C(f)\right]$$

---

## Statistical Learning Theory

### PAC Learning

A concept is **Probably Approximately Correct (PAC) learnable** if:
- With high probability (1 - δ)
- We can find a hypothesis with low error (≤ ε)
- Using polynomial time and data

### VC Dimension

Measures the "capacity" or complexity of a hypothesis class.

**Definition**: Maximum number of points that can be **shattered** (perfectly classified for any labeling).

**Examples**:
- Linear classifiers in d dimensions: VC = d + 1
- A line in 2D: VC = 3

### Generalization Bounds

VC theory gives bounds like:
$$R(f) \leq \hat{R}(f) + O\left(\sqrt{\frac{\text{VC}}{N}}\right)$$

**Implication**: Lower VC dimension → better generalization.

---

## Hypothesis Testing

### The Setup

- **Null hypothesis** $H_0$: Default assumption (e.g., "no effect")
- **Alternative hypothesis** $H_1$: What we want to show

### Error Types

| | H₀ True | H₁ True |
|---|---|---|
| **Reject H₀** | Type I Error (α) | Correct |
| **Accept H₀** | Correct | Type II Error (β) |

- **Significance level** α: P(reject H₀ | H₀ true)
- **Power** 1 - β: P(reject H₀ | H₁ true)

### p-value

The probability, under the null hypothesis, of observing a test statistic at least as extreme as what was observed.

**Common misconception**: p-value is NOT P(H₀ is true | data)!

### Likelihood Ratio Test

Compare how well each hypothesis explains the data:
$$\Lambda = \frac{p(D | H_0)}{p(D | H_1)}$$

**Neyman-Pearson Lemma**: The likelihood ratio test is the most powerful test for a given significance level.

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **Decision Rule** | Map probabilities to actions |
| **Cost-Sensitive** | Different errors have different costs |
| **ROC Curve** | Trade-off between TPR and FPR |
| **PR Curve** | Better for imbalanced classes |
| **Calibration** | Predicted probabilities should match reality |
| **AIC/BIC** | Trade-off between fit and complexity |
| **VC Dimension** | Theoretical measure of model complexity |
| **Hypothesis Testing** | Formal framework for statistical evidence |

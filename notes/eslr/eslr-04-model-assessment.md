# Model Assessment and Selection

This chapter addresses one of the most important questions in machine learning: **How do we know if our model is any good?** Training error can be misleading, so we need principled ways to estimate how well our model will perform on new, unseen data.

## The Big Picture

When building models, we face a fundamental tension:
- **Simple models** may miss important patterns (underfitting)
- **Complex models** may memorize noise (overfitting)

The goal is to find the sweet spot — a model complex enough to capture real patterns but simple enough to generalize to new data.

---

## Understanding Generalization

### What We Really Care About

We don't just want a model that fits our training data well. We want a model that predicts well on **new data** it hasn't seen before. This is called **generalization**.

### Expected Test Error

The quantity we want to minimize:

$$\text{Err}_T = E[L(Y, \hat{f}(X)) | T]$$

Where:
- **T** = the training set used to build the model
- **L** = a loss function measuring prediction error
- The expectation is over new (X, Y) pairs

### Common Loss Functions

**For Regression:**
- **Squared Error**: $L(y, \hat{f}(x)) = (y - \hat{f}(x))^2$
  - Most common; penalizes large errors heavily
- **Absolute Error**: $L(y, \hat{f}(x)) = |y - \hat{f}(x)|$
  - More robust to outliers

**For Classification:**
- **0-1 Loss**: $L(y, \hat{G}(x)) = I(y \neq \hat{G}(x))$
  - Simply counts misclassifications
- **Deviance (Log-loss)**: $L(y, \hat{p}(x)) = -2\log\hat{p}(x)$
  - Penalizes confident wrong predictions heavily

---

## The Bias-Variance Tradeoff

This is perhaps the most important concept in all of machine learning!

### Decomposing Prediction Error

For squared error loss, we can decompose the expected prediction error at a point $x_0$:

$$\text{Err}(x_0) = E[(Y - \hat{f}(x_0))^2]$$

Assuming $Y = f(X) + \epsilon$ where $E[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2_\epsilon$:

$$\text{Err}(x_0) = \underbrace{\sigma^2_\epsilon}_{\text{Irreducible}} + \underbrace{[E[\hat{f}(x_0)] - f(x_0)]^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2]}_{\text{Variance}}$$

### What Each Term Means

**Irreducible Error (σ²ε)**
- The inherent randomness in Y that no model can predict
- Even with infinite data and a perfect model, you can't beat this
- Sets the floor for prediction error

**Bias²**
- How far off is our model *on average*?
- Measures systematic error — does the model consistently over or underpredict?
- Simple models tend to have high bias (they make strong assumptions that may be wrong)

**Variance**
- How much does our model *fluctuate* across different training sets?
- If you trained on different samples, how different would your predictions be?
- Complex models tend to have high variance (they're sensitive to specific training data)

### The Tradeoff Visualized

```
Error
  │    ╲                      ╱
  │     ╲    Total Error    ╱
  │      ╲                 ╱
  │       ╲_______________╱
  │        \             /
  │    Bias² ─────      
  │                ╲   ╱ Variance
  │                 ╲ ╱
  │                  ─────────
  └─────────────────────────────→ Model Complexity
       Simple            Complex
```

- **Simple models**: High bias, low variance (underfitting)
- **Complex models**: Low bias, high variance (overfitting)
- **Optimal**: Balance that minimizes total error

### Practical Implications

1. **U-shaped test error curve**: As complexity increases, test error first decreases (reducing bias) then increases (increasing variance)

2. **Training error is misleading**: It always decreases with complexity — it can't detect overfitting!

3. **More data helps variance**: With more training data, variance decreases (we get more reliable estimates)

4. **Signal-to-noise ratio matters**: When noise is high, simpler models often win

---

## Bias-Variance for Linear Models

For linear regression with p predictors:

**Variance** $\propto p$
- More parameters = more things to estimate = more variance

**Bias² = Model Bias² + Estimation Bias²**
- **Model Bias**: Difference between the true function and the best linear approximation
- **Estimation Bias**: Difference between what we estimate and the best linear approximation

For **OLS**: Estimation bias is zero (OLS gives unbiased estimates), but variance can be high.

For **Ridge Regression**: We *introduce* estimation bias deliberately to reduce variance. If the variance reduction outweighs the bias increase, we get lower overall error!

---

## Why Training Error is Optimistic

### The Fundamental Problem

Training error uses the same data to fit the model and evaluate it. The model is specifically tuned to do well on this data, so training error underestimates how the model will perform on new data.

### Quantifying the Optimism

Define:
- $\bar{\text{err}}$ = average training error
- $\text{Err}_{in}$ = expected error on training points (but with new Y values)

The **optimism** is:

$$\text{op} = \text{Err}_{in} - \bar{\text{err}} = \frac{2}{N}\sum_{i=1}^N \text{Cov}(y_i, \hat{y}_i)$$

**Interpretation**: Optimism measures how much each training label influences its own prediction. The more the model "memorizes" training labels, the more optimistic training error becomes.

### What Affects Optimism?

- **More parameters** → More optimism (more opportunity to fit training data specifically)
- **More training samples** → Less optimism per point (each point has less influence)

---

## Model Selection vs. Model Assessment

These are related but distinct tasks:

**Model Selection**
- Goal: Choose the best model from a set of candidates
- Question: Which model will generalize best?
- Uses validation data

**Model Assessment**
- Goal: Estimate how well the chosen model performs
- Question: What's the expected error on new data?
- Uses test data (must be kept separate from selection!)

### The Standard Split

| Set | Purpose | Typical Size |
|-----|---------|--------------|
| Training | Fit models | 50-60% |
| Validation | Select best model | 20-25% |
| Test | Estimate final performance | 20-25% |

**Critical rule**: Never use test data for model selection! This leads to optimistic error estimates.

---

## Analytical Estimates of Prediction Error

When data is scarce, we'd rather not set aside a validation set. These methods estimate test error analytically.

### Cp Statistic (Mallow's Cp)

$$C_p = \bar{\text{err}} + \frac{2p}{N}\hat{\sigma}^2_\epsilon$$

Where p = effective number of parameters.

**Interpretation**: Start with training error, add a penalty for complexity. The penalty estimates the optimism.

### AIC (Akaike Information Criterion)

$$\text{AIC} = -\frac{2}{N}\ell + \frac{2p}{N}$$

Where $\ell$ = log-likelihood of the model.

**How to use**: Fit multiple models, compute AIC for each, choose the one with lowest AIC.

**Properties**:
- Derived from information theory (minimizing KL divergence)
- Tends to select slightly complex models
- Works for any model with a likelihood (not just linear)

### BIC (Bayesian Information Criterion)

$$\text{BIC} = -\frac{2}{N}\ell + \frac{\log N}{N} \times p$$

**Difference from AIC**: BIC penalizes complexity more heavily (especially for large N).

**Properties**:
- Derived from Bayesian model comparison
- **Consistent**: Will select the true model as N → ∞ (if it's among the candidates)
- Tends to select simpler models than AIC

### AIC vs BIC: When to Use Which?

| Criterion | Penalty | Tends to Select | Best When |
|-----------|---------|-----------------|-----------|
| AIC | 2p/N | Larger models | Prediction is goal |
| BIC | (log N)p/N | Smaller models | Want to find "true" model |

### Effective Number of Parameters

For complex models, "number of parameters" isn't straightforward.

**General definition**: For any smoother $\hat{y} = Sy$, the effective degrees of freedom is:

$$\text{df} = \text{trace}(S)$$

**For OLS**: df = p (the number of predictors + intercept)

**For Ridge Regression**:

$$\text{df}(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$$

Where $d_j$ are the eigenvalues of $X^TX$. As λ increases, effective df decreases.

---

## VC Dimension

For non-linear models, counting parameters is inadequate. The **VC (Vapnik-Chervonenkis) dimension** provides a general measure of model complexity.

### The Concept of Shattering

A set of points is **shattered** by a class of functions if, for every possible labeling of the points, some function in the class can achieve that labeling perfectly.

**Example**: 3 points in 2D
- Linear classifiers can separate any labeling of 3 non-collinear points
- So linear classifiers in 2D can shatter 3 points
- But they cannot shatter 4 points (XOR configuration is impossible)

### VC Dimension Defined

The VC dimension of a class of functions is the **largest number of points** that can be shattered.

**Examples**:
- Linear classifiers in d dimensions: VC = d + 1
- Polynomials of degree k: VC = k + 1
- Neural networks: depends on architecture (roughly proportional to number of parameters)

### Structural Risk Minimization

The VC dimension gives bounds on generalization error:

$$\text{Test Error} \leq \text{Training Error} + \sqrt{\frac{h(\log(2N/h) + 1) - \log(\eta/4)}{N}}$$

Where h = VC dimension, N = sample size, η = confidence level.

**Implication**: Models with smaller VC dimension have tighter bounds — they're more likely to generalize well.

---

## Cross-Validation

When you can't afford separate validation data (or want to use all data efficiently), **cross-validation** provides a powerful alternative.

### K-Fold Cross-Validation

1. Randomly divide data into K equal parts (folds)
2. For k = 1 to K:
   - Train on all folds except k
   - Test on fold k
   - Record error
3. Average the K test errors

$$\text{CV}_K = \frac{1}{N}\sum_{i=1}^N L(y_i, \hat{f}^{-\kappa(i)}(x_i))$$

Where $\hat{f}^{-\kappa(i)}$ means the model trained without observation i's fold.

### Common Choices of K

**K = N (Leave-One-Out CV)**
- Train on N-1 points, test on 1 point, repeat N times
- Nearly unbiased (training set almost full size)
- High variance (test sets overlap heavily)
- Computationally expensive (unless there's a shortcut formula)

**K = 5 or 10 (The Sweet Spot)**
- Good balance of bias and variance
- Computationally reasonable
- Empirically shown to work well
- Most commonly recommended

**K = 2 (Repeated Random Splits)**
- High variance
- Not commonly used except with many repetitions

### Bias-Variance in Cross-Validation

| K | Training Size | Bias | Variance |
|---|---------------|------|----------|
| Small (e.g., 2) | ~N/2 | High (small training sets) | Lower |
| Large (e.g., N) | N-1 | Low (big training sets) | High (overlapping test sets) |
| 5-10 | ~80-90% of N | Moderate | Moderate |

---

## Bootstrap Methods

The bootstrap is a general technique for estimating uncertainty by resampling.

### The Bootstrap Idea

1. Draw B samples of size N **with replacement** from training data
2. Fit a model to each bootstrap sample
3. Use the variation across fits to estimate uncertainty

### Key Property

Each bootstrap sample contains about 63.2% unique observations:

$$P(\text{observation } i \text{ is in bootstrap sample}) = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - e^{-1} \approx 0.632$$

### Estimating Prediction Error

**Naive approach**: Average error on training points across bootstrap samples.
- **Problem**: Observations often appear in both training and test — leakage!

**Out-of-Bag (OOB) error**: For each observation, only use predictions from models where that observation wasn't in the training sample.

$$\hat{\text{Err}}^{\text{OOB}} = \frac{1}{N}\sum_{i=1}^N \frac{1}{|C^{-i}|}\sum_{b \in C^{-i}} L(y_i, \hat{f}^{*b}(x_i))$$

Where $C^{-i}$ = bootstrap samples not containing observation i.

**Properties of OOB error**:
- No leakage — honest estimate
- Similar to leave-one-out CV
- Free when using bagging/random forests

### The .632 Bootstrap Estimator

OOB error can be slightly pessimistic (training sets are only ~63% of full data). A correction:

$$\hat{\text{Err}}^{.632} = 0.368 \times \bar{\text{err}} + 0.632 \times \hat{\text{Err}}^{\text{OOB}}$$

This averages training error (too optimistic) with OOB error (slightly pessimistic).

---

## Summary: Choosing an Estimation Method

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Train/Val/Test Split** | Lots of data | Simple, unbiased | Wastes data |
| **AIC/BIC** | Comparing similar models, need speed | Fast, no retraining | Approximate, limited scope |
| **Cross-Validation** | Limited data, want reliability | Uses all data for training AND testing | Computationally expensive |
| **Bootstrap** | Need uncertainty estimates | Versatile, works for any statistic | Can be biased, requires care |

### Practical Recommendations

1. **If you have lots of data**: Use a held-out test set for final assessment
2. **For model selection with limited data**: Use 5-fold or 10-fold CV
3. **For very small samples**: Use leave-one-out CV or bootstrap
4. **Quick comparisons**: AIC/BIC are fast approximations
5. **Always**: Keep a final test set untouched until the very end!

### Common Pitfalls

- **Using test data for selection**: Invalidates your error estimate
- **Ignoring the bias-variance tradeoff**: Don't just minimize training error!
- **Forgetting about randomness**: Always think about how results would vary with different data

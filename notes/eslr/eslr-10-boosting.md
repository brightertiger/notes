# Boosting and Additive Trees

Boosting is one of the most powerful and widely-used machine learning techniques. The core idea is simple yet profound: combine many "weak" learners (models that are only slightly better than random guessing) into one strong learner.

## The Big Picture

Imagine you're trying to predict whether a customer will churn. A single simple rule ("customers with low usage churn") might be 55% accurate. Not impressive! But what if you combine 100 such simple rules, each focusing on different aspects? The combination can be remarkably accurate.

**Boosting** does exactly this — it sequentially builds simple models, each one focusing on the mistakes of the previous ones.

---

## AdaBoost: The Original Boosting Algorithm

AdaBoost (Adaptive Boosting) was one of the first successful boosting algorithms, and it remains one of the most elegant.

### The Setup

- **Goal**: Binary classification with $Y \in \{-1, +1\}$
- **Weak learners**: Simple classifiers $G_m(x)$ (e.g., decision stumps — trees with just one split)
- **Final classifier**: Weighted vote of weak learners

$$G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$$

### The Algorithm Step by Step

**1. Initialize**: Give all observations equal weight
$$w_i = \frac{1}{N} \quad \text{for } i = 1, 2, ..., N$$

**2. For m = 1 to M rounds:**

   a) **Fit** a weak classifier $G_m(x)$ using weights $w_i$
   
   b) **Compute weighted error**:
   $$\text{err}_m = \frac{\sum_{i=1}^N w_i \cdot I\{y_i \neq G_m(x_i)\}}{\sum_{i=1}^N w_i}$$
   
   c) **Compute classifier weight**:
   $$\alpha_m = \log\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)$$
   
   d) **Update observation weights**:
   $$w_i \leftarrow w_i \cdot \exp\left(\alpha_m \cdot I\{y_i \neq G_m(x_i)\}\right)$$
   
   e) **Normalize** weights so they sum to 1

**3. Output**: $G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$

### Understanding the Weight Updates

The key insight is in step 2d:
- **Correctly classified points**: Weight stays the same
- **Misclassified points**: Weight increases by factor $\exp(\alpha_m)$

Since $\alpha_m$ is larger when $\text{err}_m$ is smaller (i.e., when the classifier is better), good classifiers upweight their mistakes MORE. This forces the next classifier to focus on the hard cases!

### Understanding Classifier Weights

The weight $\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m}$ has nice properties:
- $\text{err}_m = 0.5$ (random guessing): $\alpha_m = 0$ (no contribution)
- $\text{err}_m < 0.5$ (better than random): $\alpha_m > 0$ (positive contribution)
- $\text{err}_m$ near 0 (very accurate): $\alpha_m$ is large (strong contribution)

### Why AdaBoost Works: Key Properties

1. **Adaptive**: Each round focuses on previously misclassified points
2. **Resistant to overfitting**: Empirically works well even with many rounds
3. **Theoretical guarantee**: Training error decreases exponentially with rounds:
   $$\text{Training error} \leq \prod_{m=1}^M \sqrt{4 \cdot \text{err}_m \cdot (1-\text{err}_m)}$$

---

## Forward Stagewise Additive Modeling

AdaBoost can be understood as a special case of a general framework called **forward stagewise additive modeling**.

### The General Framework

We want to build a model as a sum of basis functions:
$$f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m)$$

Where:
- $b(x; \gamma)$ is a basis function parameterized by $\gamma$
- $\beta_m$ is the coefficient for the m-th term

### The Stagewise Algorithm

Instead of optimizing all parameters at once (hard!), we build the model **greedily**:

For m = 1 to M:
1. Fix previous terms $f_{m-1}(x)$
2. Find new $\beta_m, \gamma_m$ to minimize:
   $$\sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \beta_m b(x_i; \gamma_m))$$
3. Update: $f_m(x) = f_{m-1}(x) + \beta_m b(x_i; \gamma_m)$

### L2 Loss: Fitting Residuals

For squared error loss $L(y, f) = (y - f)^2$:

$$\min_{\beta,\gamma} \sum_{i=1}^N (y_i - f_{m-1}(x_i) - \beta b(x_i; \gamma))^2$$

Let $r_{im} = y_i - f_{m-1}(x_i)$ be the **residual** from previous rounds. Then:

$$\min_{\beta,\gamma} \sum_{i=1}^N (r_{im} - \beta b(x_i; \gamma))^2$$

**Insight**: With squared error, each round fits the **residuals** from the previous round!

### Robust Loss Functions

Squared error is sensitive to outliers. Alternatives:

**Huber Loss**:
$$L(y, f) = \begin{cases} 
\frac{1}{2}(y-f)^2 & \text{if } |y-f| \leq \delta \\
\delta|y-f| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

- Quadratic for small errors (smooth optimization)
- Linear for large errors (robust to outliers)

### Exponential Loss: Back to AdaBoost

For exponential loss $L(y, f) = \exp(-y \cdot f)$:

$$\min_{\beta,G} \sum_{i=1}^N \exp(-y_i(f_{m-1}(x_i) + \beta G(x_i)))$$

Let $w_i^{(m)} = \exp(-y_i f_{m-1}(x_i))$. Then:

$$\min_{\beta,G} \sum_{i=1}^N w_i^{(m)} \exp(-y_i \beta G(x_i))$$

Solving this optimization gives exactly the AdaBoost update rules!

### Why Exponential Loss?

The population minimizer of exponential loss is:
$$f^*(x) = \frac{1}{2}\log\frac{P(Y=1|X=x)}{P(Y=-1|X=x)}$$

This is half the log-odds! So $\text{sign}(f(x))$ gives the Bayes optimal classifier.

---

## Gradient Boosting

Gradient boosting generalizes boosting to any differentiable loss function by viewing the problem as **gradient descent in function space**.

### The Key Insight

Think of the fitted values $\mathbf{f} = [f(x_1), f(x_2), ..., f(x_N)]$ as parameters we're optimizing.

To minimize $L(\mathbf{f}) = \sum_{i=1}^N L(y_i, f(x_i))$:
- Compute the gradient: $g_i = \frac{\partial L(y_i, f)}{\partial f}\bigg|_{f=f_{m-1}(x_i)}$
- Update: $\mathbf{f}_{new} = \mathbf{f}_{old} - \rho \cdot \mathbf{g}$

### The Problem

We can compute optimal $f$ values for training points, but we need to generalize to new points!

### The Solution: Fit a Model to the Gradient

Instead of using the gradient directly, we **fit a base learner to approximate the negative gradient**:

1. Compute negative gradient (pseudo-residuals):
   $$r_{im} = -\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\bigg|_{f=f_{m-1}}$$
2. Fit a base learner $h_m(x)$ to the pseudo-residuals
3. Find optimal step size: $\rho_m = \arg\min_\rho \sum_i L(y_i, f_{m-1}(x_i) + \rho h_m(x_i))$
4. Update: $f_m(x) = f_{m-1}(x) + \rho_m h_m(x)$

### Gradients for Common Loss Functions

| Loss | Formula | Negative Gradient (Pseudo-residual) |
|------|---------|-------------------------------------|
| **L2 (Squared)** | $(y-f)^2$ | $y_i - f(x_i)$ (ordinary residual) |
| **L1 (Absolute)** | $|y-f|$ | $\text{sign}(y_i - f(x_i))$ |
| **Deviance (Classification)** | $-y\log p - (1-y)\log(1-p)$ | $y_i - p(x_i)$ |
| **Huber** | Piecewise | Trimmed residual |

### Gradient Boosting with Trees

The most common base learner is a **regression tree**. This combination is called **Gradient Boosted Trees (GBT)** or **Gradient Boosted Decision Trees (GBDT)**.

**Algorithm**:
1. Initialize $f_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$
2. For m = 1 to M:
   - Compute pseudo-residuals: $r_{im} = -\frac{\partial L}{\partial f(x_i)}|_{f_{m-1}}$
   - Fit tree $h_m$ to pseudo-residuals
   - For each leaf j, compute optimal value: $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_j} L(y_i, f_{m-1}(x_i) + \gamma)$
   - Update: $f_m(x) = f_{m-1}(x) + \nu \sum_j \gamma_{jm} I(x \in R_{jm})$

The parameter $\nu$ is called the **learning rate** or **shrinkage parameter**.

---

## Modern Implementations

Gradient boosting has evolved into several highly-optimized implementations:

### XGBoost (Extreme Gradient Boosting)

- Uses second-order Taylor approximation for faster convergence
- Built-in regularization on tree complexity
- Handles missing values automatically
- Parallel tree construction
- Very popular in competitions!

### LightGBM

- **Gradient-based One-Side Sampling (GOSS)**: Focus on high-gradient (hard) examples
- **Exclusive Feature Bundling (EFB)**: Combine sparse features
- Leaf-wise tree growth (can be faster but risks overfitting)

### CatBoost

- Superior handling of categorical features (ordered boosting)
- Reduces target leakage
- Often requires less tuning

---

## Regularization and Tuning

Gradient boosting can overfit! Key regularization techniques:

### 1. Shrinkage (Learning Rate)

Scale each update by $\nu \in (0, 1)$:
$$f_m(x) = f_{m-1}(x) + \nu \cdot h_m(x)$$

**Smaller $\nu$**:
- Requires more trees (more iterations)
- But generalizes better!
- Typical values: 0.01 to 0.1

### 2. Subsampling

Train each tree on a random subset of the training data:
- Reduces variance
- Speeds up training
- Typical values: 50-80% of data

### 3. Early Stopping

Monitor performance on a validation set. Stop when validation error stops improving:
- Prevents overfitting
- Saves computation
- Most practical stopping criterion

### 4. Tree Constraints

Limit tree complexity:
- **Max depth**: Typically 3-8
- **Min samples per leaf**: Prevents tiny leaves
- **Max leaves**: Direct control on complexity

---

## AdaBoost vs. Gradient Boosting

| Aspect | AdaBoost | Gradient Boosting |
|--------|----------|-------------------|
| **Loss** | Exponential only | Any differentiable |
| **Weight update** | Reweight observations | Fit pseudo-residuals |
| **Robustness** | Sensitive to outliers | Can use robust losses |
| **Flexibility** | Classification focus | Regression and classification |
| **Historical** | First boosting algorithm | Modern standard |

---

## Summary

### Key Concepts

1. **Weak learners + boosting = strong learner**: The magic of combining simple models

2. **Sequential focusing**: Each round corrects previous mistakes

3. **Gradient descent in function space**: Gradient boosting's unifying principle

4. **Regularization is crucial**: Learning rate, subsampling, early stopping

5. **Trees are the workhorses**: Gradient boosted trees dominate tabular data

### When to Use Boosting

**Great for**:
- Tabular data
- When you need maximum accuracy
- When you can tune hyperparameters

**Not ideal for**:
- When interpretability is paramount
- Very small datasets
- When training time is extremely limited

### Practical Tips

1. Start with a small learning rate (0.01-0.1)
2. Use early stopping based on validation error
3. Tune max_depth (3-8) and n_estimators together
4. Consider subsampling for large datasets
5. Try XGBoost, LightGBM, or CatBoost — they're all excellent!

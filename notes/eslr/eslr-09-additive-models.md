# Additive Models, Trees, and Related Methods

This chapter introduces flexible methods that can capture non-linear relationships while remaining interpretable. We start with generalized additive models, then dive deep into decision trees — one of the most intuitive and widely-used machine learning methods.

## The Big Picture

Linear models are simple and interpretable but can miss important non-linear patterns. At the other extreme, highly flexible methods (like neural networks) can fit anything but are hard to interpret.

**Additive models and trees** offer a middle ground: they capture non-linear relationships while remaining interpretable.

---

## Generalized Additive Models (GAMs)

### The Limitation of Linear Models

In linear regression, we model:
$$E[Y|X] = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p$$

Each predictor has a simple linear effect. But what if the relationship is curved? What if income affects health differently for low vs. high earners?

### The GAM Solution

Replace linear terms with **flexible smooth functions**:

$$E[Y|X] = \alpha + f_1(X_1) + f_2(X_2) + ... + f_p(X_p)$$

Where each $f_j$ is learned from data — typically a smooth curve like a **spline**.

### More Generally: Link Functions

For non-normal responses (binary, count data), we use a link function:

$$g[E[Y|X]] = \alpha + f_1(X_1) + f_2(X_2) + ... + f_p(X_p)$$

Common link functions:
- **Identity** (linear regression): $g(\mu) = \mu$
- **Logit** (logistic regression): $g(\mu) = \log(\mu/(1-\mu))$
- **Log** (Poisson regression): $g(\mu) = \log(\mu)$

### Interpretability

**Key advantage**: Each $f_j$ shows exactly how predictor $X_j$ affects the response. You can plot $f_j(X_j)$ and see the relationship!

- Is it linear? Curved? Has a threshold?
- The shape is learned from data, not assumed

### Fitting GAMs

GAMs are fit by minimizing **Penalized Residual Sum of Squares (PRSS)**, which balances fitting the data with keeping functions smooth:

$$\text{PRSS} = \sum_{i=1}^N \left(y_i - \alpha - \sum_j f_j(x_{ij})\right)^2 + \sum_j \lambda_j \int f_j''(t)^2 dt$$

The penalty term punishes "wiggly" functions (large second derivatives).

---

## Decision Trees

Decision trees are perhaps the most intuitive machine learning method. They partition the feature space into regions and fit simple models (often just constants) in each region.

### The Core Idea

Think of playing "20 Questions" with your data:
- "Is income > $50K?" → split data into two groups
- "Is age > 40?" → further split
- Keep splitting until groups are "pure" (mostly one class/similar values)

The result is a tree structure that's easy to understand and explain.

### Regression Trees

For continuous outcomes, we:
1. **Partition** the feature space into rectangles $R_1, R_2, ..., R_M$
2. **Fit a constant** in each region: $c_m = \text{average of } y_i \text{ in } R_m$

The model is:
$$f(X) = \sum_{m=1}^M c_m \cdot I\{X \in R_m\}$$

Where $I\{\cdot\}$ is the indicator function (1 if true, 0 otherwise).

### How to Find the Best Splits

We use a **greedy algorithm** — at each node, find the split that most reduces error.

For a split on variable $X_j$ at value $s$:
- Left region: $R_1 = \{X | X_j \leq s\}$
- Right region: $R_2 = \{X | X_j > s\}$

Choose j and s to minimize:
$$\min_{j,s} \left[\min_{c_1}\sum_{X_i \in R_1}(y_i - c_1)^2 + \min_{c_2}\sum_{X_i \in R_2}(y_i - c_2)^2\right]$$

The inner minimizations are easy — just take averages in each region!

### Classification Trees

For categorical outcomes, each leaf predicts the **most common class** in that region:

$$\hat{G}_m = \text{majority class in } R_m$$

The proportion of class k in node m is:
$$\hat{p}_{mk} = \frac{1}{N_m}\sum_{i \in R_m} I\{y_i = k\}$$

### Splitting Criteria for Classification

We need to measure how "impure" a node is. Several options:

**Misclassification Error**: $1 - \max_k \hat{p}_{mk}$
- Simple but not differentiable — hard to optimize

**Gini Index**: $\sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk})$
- Measures probability of misclassifying a randomly chosen element
- Equals variance of Bernoulli distribution when K=2
- Most commonly used

**Cross-Entropy (Deviance)**: $-\sum_{k=1}^K \hat{p}_{mk}\log\hat{p}_{mk}$
- Information-theoretic measure of impurity
- Similar to Gini in practice

**Note**: Gini and entropy are more sensitive to node purity changes than misclassification error, making them better for tree growing.

### Handling Categorical Predictors

If variable X has L categories, there are $2^{L-1} - 1$ possible binary splits. This seems exponential, but for classification with 2 classes:

**Trick**: Order categories by proportion of class 1, then treat as ordered. This reduces to checking L-1 splits!

### Handling Missing Values

Two common approaches:

**1. Missing as a Category**: Create a new category for missing values.

**2. Surrogate Splits**: Find alternative splits that mimic the primary split. At prediction time, if the primary split variable is missing, use the surrogate.

The surrogate approach leverages correlations between predictors to minimize information loss.

---

## Pruning: Controlling Tree Size

### The Problem

Trees that grow too large:
- Overfit the training data
- Have high variance
- Are harder to interpret

### Option 1: Stop Early

Split only if improvement exceeds a threshold.

**Problem**: A bad split now might enable great splits later! This is short-sighted.

### Option 2: Grow and Prune (Better!)

1. **Grow** a large tree (until leaves have few observations)
2. **Prune** back to find the best subtree

### Cost-Complexity Pruning

Define a cost that balances fit and complexity:

$$C_\alpha(T) = \sum_{m=1}^{|T|} N_m Q_m(T) + \alpha|T|$$

Where:
- $|T|$ = number of terminal nodes (leaves)
- $Q_m$ = impurity measure for node m (e.g., RSS/N_m for regression)
- $\alpha$ = complexity penalty parameter

**Small α**: Prefer large trees (focus on fit)
**Large α**: Prefer small trees (focus on simplicity)

**Algorithm**:
1. For each α, find the subtree that minimizes $C_\alpha(T)$
2. Use cross-validation to select the best α

---

## Evaluating Classification Performance

### The Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Key Metrics

**Sensitivity (Recall, True Positive Rate)**:
$$\text{Sensitivity} = \frac{TP}{TP + FN}$$
"Of all actual positives, what fraction did we catch?"

**Specificity (True Negative Rate)**:
$$\text{Specificity} = \frac{TN}{TN + FP}$$
"Of all actual negatives, what fraction did we correctly identify?"

**Precision**:
$$\text{Precision} = \frac{TP}{TP + FP}$$
"Of all predicted positives, what fraction are correct?"

### The ROC Curve

The **Receiver Operating Characteristic (ROC)** curve shows the tradeoff between sensitivity and specificity as you vary the classification threshold.

- **X-axis**: 1 - Specificity (False Positive Rate)
- **Y-axis**: Sensitivity (True Positive Rate)

**AUC (Area Under the ROC Curve)**:
- AUC = 1: Perfect classifier
- AUC = 0.5: Random guessing
- AUC > 0.7: Generally acceptable

**Interpretation**: AUC is the probability that a randomly chosen positive example ranks higher than a randomly chosen negative example.

---

## MARS: Multivariate Adaptive Regression Splines

MARS extends tree ideas to regression with smoother, continuous functions.

### The Idea

Instead of piecewise constant regions, use **piecewise linear basis functions**:

$$(x - t)_+ = \max(0, x-t)$$
$$(t - x)_+ = \max(0, t-x)$$

These are "hockey stick" functions that are zero until a knot t, then linear.

### MARS Model

$$f(X) = \beta_0 + \sum_{m=1}^M \beta_m h_m(X)$$

Where each $h_m$ is a product of basis functions (allowing interactions).

### Connection to Trees

MARS is like a regression tree with smoother predictions at boundaries. Trees have sharp jumps; MARS transitions smoothly.

---

## PRIM: Patient Rule Induction Method

PRIM takes a different approach: find regions (boxes) with unusually high (or low) response values.

### The Algorithm

1. Start with a box containing all data
2. **Peeling**: Shrink the box by removing a thin slice from one face, choosing the slice that maximizes mean response
3. **Pasting**: Try expanding the box if it improves the mean
4. Repeat to find multiple boxes

### Use Case

Useful for finding "hot spots" — regions where the response is particularly high. Applications include fraud detection, medical diagnosis, quality control.

---

## Mixture of Experts

This is a probabilistic generalization of decision trees.

### The Idea

Instead of hard splits (left or right), use **soft probabilistic splits**:
- Each observation has some probability of going to each child node
- "Gating networks" at internal nodes determine these probabilities
- "Expert" models at leaves make predictions
- Final prediction is a weighted average

### Structure

**Gating Networks** (internal nodes):
- Soft decision functions
- Output probabilities for each branch

**Experts** (terminal nodes):
- Fit local models (often linear regression)
- Each expert specializes in a region

### Fitting

Use the **EM algorithm**:
- E-step: Compute responsibilities (how much each expert contributes to each observation)
- M-step: Update expert parameters and gating parameters

### Advantages

- Smoother predictions than hard trees
- Naturally provides uncertainty estimates
- Can capture complex interactions

---

## Summary: Choosing a Method

| Method | Best For | Interpretability | Flexibility |
|--------|----------|------------------|-------------|
| **GAM** | Understanding non-linear effects | High (plot each effect) | Medium |
| **Decision Tree** | Simple rules, categorical outcomes | Very High | Medium |
| **MARS** | Regression with interactions | Medium | Medium-High |
| **PRIM** | Finding high-response regions | High | Low |
| **Mixture of Experts** | Complex boundaries with uncertainty | Low | High |

### Key Takeaways

1. **Trees are intuitive**: Easy to explain and visualize
2. **Pruning is essential**: Unpruned trees overfit badly
3. **GAMs maintain interpretability**: Each predictor's effect is visible
4. **Splitting criteria matter**: Gini/entropy better than misclassification for tree growing
5. **These methods form building blocks**: Trees are the foundation for Random Forests and Boosting!

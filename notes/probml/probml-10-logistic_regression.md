# Logistic Regression

Logistic regression is the foundational discriminative model for classification. Despite its name, it's a classification algorithm (not regression) that directly models posterior class probabilities.

## The Big Picture

Unlike generative models (which model how data is generated per class), logistic regression directly models:
$$p(y | x, \theta)$$

This "discriminative" approach focuses on the decision boundary rather than the full data distribution.

---

## Binary Logistic Regression

### The Model

For binary classification with $y \in \{0, 1\}$:

$$p(y = 1 | x, \theta) = \sigma(w^T x + b)$$

Where σ is the **sigmoid function**:
$$\sigma(a) = \frac{1}{1 + e^{-a}}$$

The probability of class 0 is simply:
$$p(y = 0 | x, \theta) = 1 - \sigma(w^T x + b) = \sigma(-(w^T x + b))$$

### Alternative Notation

For $y \in \{-1, +1\}$:
$$p(y | x, \theta) = \sigma(y \cdot (w^T x + b))$$

This compact form handles both classes with one equation.

### The Decision Boundary

Predict $y = 1$ if $p(y = 1 | x) > 0.5$, which occurs when:
$$w^T x + b > 0$$

This is a **hyperplane** in feature space — logistic regression produces linear decision boundaries.

**Geometric interpretation**:
- **w** is the normal vector to the decision boundary
- **b** determines the offset from the origin
- Distance from boundary relates to confidence

---

## Maximum Likelihood Estimation

### The Likelihood

For dataset $\{(x_i, y_i)\}_{i=1}^N$:
$$L(\theta) = \prod_{i=1}^N p(y_i | x_i, \theta)$$

### Negative Log-Likelihood (Binary Cross-Entropy)

$$\text{NLL}(\theta) = -\sum_{i=1}^N [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]$$

Where $\hat{y}_i = \sigma(w^T x_i + b)$.

For $y \in \{-1, +1\}$ notation:
$$\text{NLL}(\theta) = \sum_{i=1}^N \log(1 + \exp(-y_i(w^T x_i + b)))$$

### Computing the Gradient

The gradient has a beautiful form:
$$\nabla_w \text{NLL} = \sum_{i=1}^N (\hat{y}_i - y_i) x_i = X^T(\hat{y} - y)$$

**Intuition**: The gradient is a weighted sum of input vectors, where the weights are the prediction errors.

### Optimization

**Good news**: The NLL is **convex** (Hessian is positive semi-definite).

**Methods**:
1. **Gradient Descent / SGD**: Simple, works for large datasets
2. **Newton's Method**: Faster convergence for smaller problems
3. **IRLS**: Iteratively Reweighted Least Squares — Newton's method specialized for logistic regression

---

## Regularization

### The Overfitting Problem

MLE can overfit, especially with:
- High-dimensional features
- Small datasets
- Linearly separable data (weights → ∞)

### L2 Regularization (Ridge)

Add Gaussian prior on weights:
$$p(w) = \mathcal{N}(0, \lambda^{-1} I)$$

**Regularized objective**:
$$L(w) = \text{NLL}(w) + \lambda \|w\|^2$$

**Effect**: Penalizes large weights, improves generalization.

### L1 Regularization (Lasso)

Use Laplace prior for sparse solutions:
$$L(w) = \text{NLL}(w) + \lambda \|w\|_1$$

**Effect**: Some weights become exactly zero — automatic feature selection.

### Practical Notes

- **Standardize features** before applying regularization (features should be on same scale)
- **Don't regularize the bias** term
- Choose λ via cross-validation

---

## Multinomial Logistic Regression

### Extending to Multiple Classes

For $y \in \{1, 2, ..., C\}$:

$$p(y = c | x, \theta) = \frac{\exp(w_c^T x + b_c)}{\sum_{j=1}^C \exp(w_j^T x + b_j)} = \text{softmax}(a)_c$$

Where $a_c = w_c^T x + b_c$ are the **logits**.

### Overparameterization

Note: We have C sets of weights, but only C-1 are needed (one class can be the reference).

For binary case with softmax:
$$p(y = 0 | x) = \frac{e^{a_0}}{e^{a_0} + e^{a_1}} = \sigma(a_0 - a_1)$$

This reduces to standard logistic regression with $w = w_0 - w_1$.

### Maximum Entropy Classifier

When features depend on both x and the class c:
$$p(y = c | x, w) \propto \exp(w^T \phi(x, c))$$

This is common in NLP where features might include "word X appears AND class is Y".

---

## Handling Special Cases

### Hierarchical Classification

When classes have taxonomy (e.g., animal → mammal → dog):

**Label smearing**: Propagate positive labels to parent categories.

**Approach**: Multi-label classification where an example can belong to multiple levels.

### Many Classes

**Hierarchical softmax**: Organize classes in a tree; predict by traversing tree.
- Reduces computation from O(C) to O(log C)
- Put frequent classes near root

### Class Imbalance

When some classes are much more common:

**Resampling strategies**:
$$p_c = \frac{N_c^q}{\sum_j N_j^q}$$

- q = 1: Instance-balanced (original distribution)
- q = 0: Class-balanced (equal weight per class)
- q = 0.5: Square-root sampling (compromise)

---

## Robust Logistic Regression

### Handling Outliers and Label Noise

Standard logistic regression is sensitive to mislabeled examples.

**Mixture model approach**:
$$p(y | x) = \pi \cdot \text{Ber}(0.5) + (1 - \pi) \cdot \text{Ber}(\sigma(w^T x + b))$$

Mix predictions with uniform noise — mislabeled points have less impact.

### Bi-tempered Logistic Loss

Two modifications for robustness:
1. **Tempered cross-entropy**: Handles outliers far from boundary
2. **Tempered softmax**: Handles noise near boundary

### Probit Regression

Replace sigmoid with Gaussian CDF (probit function):
$$p(y = 1 | x) = \Phi(w^T x + b)$$

Similar shape to logistic but different tails — can be more robust in some cases.

---

## Summary

| Aspect | Key Points |
|--------|------------|
| **Model** | $p(y=1\|x) = \sigma(w^Tx + b)$ |
| **Loss** | Binary cross-entropy (NLL) |
| **Optimization** | Convex — guaranteed global optimum |
| **Boundary** | Linear (hyperplane) |
| **Regularization** | L2 (shrink) or L1 (sparse) |
| **Multiclass** | Softmax over C classes |
| **Robustness** | Mixture models, tempered losses |

### When to Use Logistic Regression

**Good for**:
- Binary and multiclass classification
- When interpretability matters (coefficients are meaningful)
- As a baseline before trying complex models
- When computational resources are limited

**Limitations**:
- Linear decision boundaries
- May underfit complex data
- Sensitive to outliers (without modifications)

**Pro tip**: Start with logistic regression. If it works well, you may not need anything more complex!

# Decision Trees and Ensembles

Decision trees are intuitive models that partition the feature space into regions. While single trees are prone to overfitting, ensemble methods (Random Forests, Boosting) combine many trees for powerful, robust predictions.

## The Big Picture

**Trees**: Recursively partition space with simple rules.
- Highly interpretable
- But high variance (unstable)

**Ensembles**: Combine many trees.
- Random Forests: Reduce variance through averaging
- Boosting: Reduce bias through sequential correction

---

## Decision Tree Structure

### The Model

$$f(x) = \sum_{j=1}^{J} w_j \cdot I\{x \in R_j\}$$

Where:
- $R_j$ are disjoint regions (leaves)
- $w_j$ is the prediction for region j
- I{} is indicator function

### Building a Tree

At each node i:
1. Select a feature $d_i$
2. Select a threshold $t_i$
3. Split: left if $x_{d_i} \leq t_i$, right otherwise

**Result**: Axis-parallel partitions of the feature space.

### Leaf Predictions

**Regression**: Average of training labels in region
$$w_j = \frac{\sum_{n: x_n \in R_j} y_n}{\sum_{n: x_n \in R_j} 1}$$

**Classification**: Majority vote or probability distribution

---

## Finding Optimal Splits

### The Greedy Algorithm

Tree optimization is NP-hard. We use a greedy approach:

At each node, find the best split by minimizing:
$$L = \frac{|D_L|}{|D|} C_L + \frac{|D_R|}{|D|} C_R$$

Where C is the cost (impurity) and D is the data reaching that node.

### Splitting Criteria

**For Regression (MSE)**:
$$C = \frac{1}{|D|}\sum_{i \in D}(y_i - \bar{y})^2$$

**For Classification**:

*Gini Index*:
$$C = \sum_{c=1}^C \hat{p}_c(1 - \hat{p}_c)$$
Probability of misclassifying a randomly chosen element.

*Cross-Entropy*:
$$C = -\sum_{c=1}^C \hat{p}_c \log \hat{p}_c$$
Information-theoretic measure of impurity.

### Why Binary Splits?

- More splits = more data fragmentation
- Binary splits are sufficient (can always split further)
- Simpler to optimize

---

## Regularization (Preventing Overfitting)

### Option 1: Early Stopping

Stop growing when:
- Maximum depth reached
- Minimum samples per leaf
- Improvement below threshold

**Problem**: May stop too early (miss good splits downstream).

### Option 2: Grow and Prune

1. Grow full tree (until pure leaves or minimum samples)
2. Prune back using cost-complexity criterion:

$$C_\alpha(T) = \sum_{j=1}^{|T|} N_j \cdot C_j + \alpha |T|$$

Where $|T|$ is number of leaves and α is complexity penalty.

Use cross-validation to select optimal α.

### Handling Missing Features

**Categorical**: Treat "missing" as new category.

**Continuous**: Use surrogate splits — find alternative splits that best mimic the primary split.

---

## Pros and Cons of Trees

### Advantages

- **Interpretable**: Easy to visualize and explain
- **Minimal preprocessing**: Handles mixed types, no normalization needed
- **Fast**: Prediction is O(log nodes)
- **Robust to outliers**: Splits don't depend on magnitudes

### Disadvantages

- **High variance**: Small data changes → different tree
- **Axis-aligned only**: Can't capture diagonal boundaries efficiently
- **Prone to overfitting**: Without regularization

---

## Ensemble Learning

### The Core Idea

Combine multiple models to reduce errors.

**Regression**: Average predictions
$$\hat{y} = \frac{1}{M}\sum_{m=1}^M f_m(x)$$

**Classification**: Majority vote or average probabilities

### Why Ensembles Work

For M independent classifiers each with accuracy p > 0.5:
$$P(\text{majority correct}) = \sum_{k > M/2} \binom{M}{k} p^k (1-p)^{M-k}$$

As M → ∞, this probability → 1!

**Key requirement**: Classifiers must be diverse (uncorrelated errors).

### Stacking

Learn weights for combining models:
$$\hat{y} = \sum_m w_m f_m(x)$$

Train weights on held-out data to avoid overfitting.

---

## Bagging (Bootstrap Aggregating)

### Algorithm

1. Create B bootstrap samples (sample with replacement)
2. Train a tree on each bootstrap sample
3. Average predictions (regression) or vote (classification)

### Key Properties

- Each bootstrap sample contains ~63% of unique points:
  $$P(\text{included}) = 1 - (1 - 1/N)^N \approx 1 - 1/e \approx 0.632$$

- **OOB Error**: Evaluate each tree on its out-of-bag samples (free cross-validation!)

### Variance Reduction

$$\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Where ρ is correlation between trees.

- More trees (larger B) → second term vanishes
- Less correlation (smaller ρ) → first term shrinks

---

## Random Forests

### Beyond Bagging

Bagging helps, but trees from similar data are correlated.

**Random Forest innovation**: Add randomness to splits.

At each split:
1. Randomly select m features (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression)
2. Find best split among only those m features

### Why It Works

- Forces trees to use different features
- Reduces correlation between trees
- Combined with bagging → powerful ensemble

### Extra Trees

Even more randomness:
- Random feature subset (like RF)
- Random threshold selection (not optimized)
- Faster training, often similar performance

---

## Boosting

### The Key Idea

Sequentially fit weak learners, each focusing on previous mistakes.

$$F_m(x) = F_{m-1}(x) + \beta_m f_m(x)$$

**Boosting reduces bias** (unlike bagging which reduces variance).

### AdaBoost

For classification with exponential loss:
$$L(y, F) = \exp(-yF(x)), \quad y \in \{-1, +1\}$$

**Algorithm**:
1. Initialize equal weights on examples
2. For each round:
   - Train weak learner on weighted examples
   - Increase weights on misclassified examples
   - Compute learner weight based on accuracy

### Gradient Boosting

Generalize to any differentiable loss:

1. Initialize: $F_0 = \arg\min_\gamma \sum L(y_i, \gamma)$
2. For m = 1 to M:
   - Compute pseudo-residuals: $r_i = -\frac{\partial L(y_i, F)}{\partial F}|_{F_{m-1}}$
   - Fit weak learner to pseudo-residuals
   - Line search for step size
   - Update: $F_m = F_{m-1} + \eta \cdot f_m$

**For MSE loss**: Pseudo-residuals = actual residuals!

**Regularization via shrinkage**: Small learning rate η (0.01-0.1) + more trees.

### Stochastic Gradient Boosting

Subsample data at each round:
- Faster training
- Better generalization (regularization effect)

---

## XGBoost

### Innovations

1. **Regularized objective**:
$$L = \sum L(y_i, F(x_i)) + \gamma J + \frac{\lambda}{2}\sum w_j^2$$

2. **Second-order approximation** (use Hessian):
$$L \approx \sum [g_i F(x_i) + \frac{1}{2}h_i F(x_i)^2] + \text{regularization}$$

3. **Optimal leaf weights**:
$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

Where $G_j = \sum_{i \in j} g_i$ and $H_j = \sum_{i \in j} h_i$.

4. **Split gain**:
$$\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} - \gamma$$

γ acts as regularization — won't split unless gain exceeds γ.

---

## Feature Importance

### Mean Decrease in Impurity

Sum the impurity decrease at all splits using feature k:
$$R_k = \sum_{\text{nodes using } k} \Delta \text{impurity}$$

Average across all trees.

**Caveat**: Biased toward high-cardinality features.

### Permutation Importance

1. Compute baseline accuracy
2. For each feature k:
   - Permute (shuffle) feature k's values
   - Compute accuracy drop
3. Importance = accuracy drop

More reliable but slower.

### Partial Dependence Plots

Visualize effect of feature on prediction:
$$\bar{f}_k(x_k) = \frac{1}{N}\sum_{i=1}^N f(x_k, x_{i,-k})$$

Average over all other features.

---

## Summary

| Method | Reduces | Training | Key Hyperparameters |
|--------|---------|----------|---------------------|
| **Single Tree** | — | Fast | max_depth, min_samples |
| **Bagging** | Variance | Parallel | n_estimators |
| **Random Forest** | Variance | Parallel | n_estimators, max_features |
| **Boosting** | Bias | Sequential | n_estimators, learning_rate, max_depth |

### Practical Recommendations

1. **Start with Random Forest**: Works well with minimal tuning
2. **Try XGBoost/LightGBM**: Often best for tabular data
3. **Tune carefully**: Learning rate and n_estimators together
4. **Early stopping**: Monitor validation error
5. **Feature importance**: Helps interpretability

# Exemplar-Based Methods

Exemplar methods (also called instance-based or memory-based) keep training data around and use it directly for prediction. The classic example is K-Nearest Neighbors (KNN).

## The Big Picture

**Parametric models**: Learn parameters θ, discard training data at test time.
- Parameters: Fixed, doesn't grow with data

**Non-parametric models**: Keep training data, use it directly.
- Model complexity grows with data size
- Can adapt to arbitrary complexity

---

## Instance-Based Learning

### The Approach

1. Store training examples
2. At test time, find similar training examples
3. Predict based on their labels

**Key ingredient**: A good **distance/similarity measure**.

---

## K-Nearest Neighbors (KNN)

### Classification

Find K closest training points; vote on the label:
$$p(y = c | x, D) = \frac{1}{K} \sum_{i \in N_K(x)} I\{y_i = c\}$$

**Hyperparameter K**:
- K = 1: Highly flexible, noisy
- K = N: Predicts majority class always
- Typical: K = 5-10, or tune via cross-validation

### Regression

Average the labels of K nearest neighbors:
$$\hat{y} = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$

### Distance Metrics

**Euclidean distance**:
$$d(x, x') = \|x - x'\|_2 = \sqrt{\sum_j (x_j - x'_j)^2}$$

**Mahalanobis distance** (accounts for correlations):
$$d_M(x, x') = \sqrt{(x - x')^T M (x - x')}$$

Where M is a positive definite matrix (often $M = \Sigma^{-1}$).

**If M = I**: Reduces to Euclidean distance.

---

## The Curse of Dimensionality

### The Problem

In high dimensions, distances become meaningless:
- All points become approximately equidistant
- Local neighborhoods become empty
- Need exponentially more data to fill space

### Example

Consider the fraction of volume within 10% of the edges:
- 1D: 20%
- 10D: 89%
- 100D: 99.99999...%

Almost all points are near the boundary!

### Solutions

1. **Dimensionality reduction** (PCA, autoencoders)
2. **Feature selection**
3. **Metric learning** (learn a better distance)

---

## Computational Efficiency

### Naive Approach

For each query, compute distance to all N training points.
- Time: O(Nd) per query
- Infeasible for large datasets

### Approximate Nearest Neighbors

**KD-Trees**:
- Binary tree that partitions space
- O(log N) for low dimensions
- Degrades in high dimensions

**Locality-Sensitive Hashing (LSH)**:
- Hash similar items to same bucket
- Sublinear query time
- Approximate, not exact

---

## Open Set Recognition

**Standard classification**: All test classes seen during training.

**Open set**: New, unseen classes may appear at test time.

**KNN advantage**: Can handle novel classes naturally by looking at nearest neighbors.

**Applications**:
- Person re-identification
- Anomaly detection
- Few-shot learning

---

## Learning Distance Metrics

### Motivation

Euclidean distance may not reflect true similarity.

**Goal**: Learn a distance metric M that captures task-relevant similarity.

### Large Margin Nearest Neighbors (LMNN)

Learn M such that:
1. Points with same label are close
2. Points with different labels are far (by margin m)

**Constraint**: $M = W^T W$ ensures positive definiteness.

---

## Deep Metric Learning

### The Idea

Learn an embedding function $f(x; \theta)$ such that:
- Similar examples are close in embedding space
- Dissimilar examples are far

### Siamese Networks

Two copies of same network process two inputs.

**Contrastive Loss**:
$$L = I\{y_i = y_j\} \cdot d(x_i, x_j)^2 + I\{y_i \neq y_j\} \cdot [m - d(x_i, x_j)]_+^2$$

- Same class: Minimize distance
- Different class: Push apart (up to margin m)

### Triplet Loss

Use triplets: (anchor, positive, negative)

$$L = [m + d(a, p) - d(a, n)]_+$$

**Goal**: Anchor should be closer to positive than negative by margin m.

### Hard Negative Mining

Random negatives are too easy (already far from anchor).

**Solution**: Sample hard negatives that are close to anchor but from different class.

### Connection to Cosine Similarity

For normalized embeddings:
$$\|e_1 - e_2\|^2 = 2(1 - \cos\theta)$$

Euclidean distance and cosine similarity are equivalent for unit vectors.

---

## Kernel Density Estimation

### The Idea

Estimate the probability density by placing kernels at each data point:
$$\hat{p}(x) = \frac{1}{N} \sum_{n=1}^N K_h(x - x_n)$$

**Gaussian kernel**:
$$K_h(x) = \frac{1}{h\sqrt{2\pi}} \exp\left(-\frac{x^2}{2h^2}\right)$$

### Bandwidth h

Controls smoothness:
- Small h: Spiky, overfitting
- Large h: Over-smooth, underfitting

Choose via cross-validation.

### Connection to GMM

KDE is like a GMM where:
- Each point is its own cluster center
- All clusters have same (spherical) covariance
- No mixing proportions to learn

### KDE for Classification

Use Bayes rule with class-conditional densities:
$$p(y = c | x) \propto \pi_c \cdot \hat{p}(x | y = c)$$

---

## KDE vs KNN

**Connection**: Both use local neighborhoods.

- **KDE**: Fixed bandwidth, variable number of neighbors
- **KNN**: Variable bandwidth (grows until K neighbors), fixed number of neighbors

**Dual view**: KNN adapts to local density automatically.

---

## Summary

| Method | Key Idea | Pros | Cons |
|--------|----------|------|------|
| **KNN** | Vote of K nearest neighbors | Simple, no training | Slow at test time |
| **Metric Learning** | Learn task-specific distance | Better than Euclidean | Requires labeled pairs |
| **Deep Metric** | Embed + distance | Handles high dimensions | Needs lots of data |
| **KDE** | Kernels at each point | Density estimation | Curse of dimensionality |

### When to Use Exemplar Methods

**Good for**:
- Few training examples per class (few-shot learning)
- Classes change over time (no retraining needed)
- Interpretable predictions ("similar to example X")
- Baseline before trying complex methods

**Challenges**:
- High-dimensional data (curse of dimensionality)
- Large training sets (computational cost)
- Need good distance metric

# Random Forests

Random Forests are one of the most successful and widely-used machine learning algorithms. They combine the simplicity of decision trees with the power of ensemble methods to create a robust, accurate, and easy-to-use classifier.

## The Big Picture

Decision trees are intuitive and interpretable, but they have a major flaw: **high variance**. A small change in the training data can produce a completely different tree. Random Forests solve this by building many trees and averaging their predictions.

**Key insight**: Many imperfect models, when combined intelligently, can outperform a single "perfect" model.

---

## Why Averaging Helps

### The Bias-Variance View

Individual decision trees (especially deep ones) have:
- **Low bias**: They can fit complex patterns
- **High variance**: They're sensitive to the specific training data

Averaging reduces variance while maintaining low bias!

### Mathematical Intuition

Consider B random variables (predictions from B trees), each with:
- Individual variance: $\sigma^2$
- Pairwise correlation: $\rho$

The variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B X_b\right) = \rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2$$

**Two key insights**:

1. **More trees (larger B) always helps**: The second term shrinks toward 0
2. **Lower correlation ($\rho$) helps even more**: The first term shrinks

**This is why Random Forests can't overfit by adding more trees!** (Unlike boosting, which can overfit with too many rounds.)

---

## Bagging: The Foundation

**Bagging** (Bootstrap Aggregating) is the simpler ancestor of Random Forests.

### The Algorithm

1. **Bootstrap**: Draw B samples of size N with replacement from training data
2. **Train**: Fit a decision tree to each bootstrap sample
3. **Aggregate**: 
   - Regression: Average predictions
   - Classification: Majority vote

### Why Bootstrap?

Each bootstrap sample is slightly different from the original data:
- Contains ~63.2% of unique observations
- Some observations appear multiple times
- Others don't appear at all

This variation creates diversity among trees — different trees make different mistakes!

---

## Random Forests: Adding More Randomness

Random Forests add an additional source of randomness to further **decorrelate** the trees.

### The Key Innovation

At each split, instead of considering all features, consider only a **random subset** of m features.

**Why this helps**: 
- In bagging, if one feature is very strong, every tree uses it at the root → trees are correlated
- Random feature selection forces trees to use different features → less correlation

### Typical Values for m

| Task | Recommended m |
|------|---------------|
| Classification | $\sqrt{p}$ |
| Regression | $p/3$ |

Where p = total number of features.

### The Complete Algorithm

1. For b = 1 to B trees:
   - Draw a bootstrap sample of size N
   - Grow a tree:
     - At each node, randomly select m features
     - Find the best split among those m features
     - Split the node
     - Repeat until stopping criterion (min node size)
2. Output: Ensemble of B trees

For prediction:
- **Regression**: $\hat{f}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x)$
- **Classification**: $\hat{G}(x) = \text{majority vote of } \hat{G}_b(x)$

---

## Out-of-Bag (OOB) Error

One of the most elegant features of Random Forests: **free cross-validation**!

### The Idea

Each bootstrap sample leaves out ~36.8% of observations. For each observation i:
1. Find all trees where i was NOT in the training sample
2. Use only those trees to predict for i
3. This is an honest prediction — no leakage!

### OOB Error Estimate

$$\text{OOB Error} = \frac{1}{N}\sum_{i=1}^N L(y_i, \hat{y}_i^{\text{OOB}})$$

Where $\hat{y}_i^{\text{OOB}}$ is the prediction using only trees that didn't train on observation i.

### Properties

- **Essentially equivalent to leave-one-out cross-validation**
- **Computed for free** during training
- **No need for separate validation set**
- **Honest estimate** of generalization error

---

## Variable Importance

Random Forests provide built-in measures of which features matter most.

### Method 1: Mean Decrease in Impurity

For each feature j:
1. At each split on feature j, record the decrease in impurity (Gini, entropy, or MSE)
2. Sum across all splits and all trees
3. Normalize

**Pros**: Fast, computed during training
**Cons**: Biased toward high-cardinality categorical features

### Method 2: Permutation Importance

A more reliable approach:

1. Compute OOB accuracy for the original data
2. For each feature j:
   - Randomly shuffle (permute) feature j's values
   - Recompute OOB accuracy
   - Record the decrease in accuracy
3. Average across all trees

**Interpretation**: If shuffling feature j destroys accuracy, that feature was important!

**Pros**: Unbiased, captures complex dependencies
**Cons**: Computationally more expensive

### When Variables Are Correlated

Both importance measures can be misleading with correlated features:
- Importance may be split among correlated features
- A single feature from a correlated group may show low importance

---

## Proximity Measures

Random Forests can measure similarity between observations.

### Computing Proximities

For each pair of observations (i, k):
1. Count how often they end up in the same terminal node
2. Across all trees
3. Normalize by number of trees

This creates an N × N **proximity matrix**.

### Uses

- **Visualization**: Use multidimensional scaling (MDS) to plot proximities in 2D
- **Outlier detection**: Points with low average proximity to their class may be outliers
- **Missing value imputation**: Fill in missing values using weighted averages of similar observations
- **Clustering**: Use proximity as a similarity measure

---

## Advantages of Random Forests

### 1. Accuracy
- Often among the best performing methods "out of the box"
- Works well on many types of data without much tuning

### 2. Robustness
- Handles missing values gracefully
- Not sensitive to outliers (median of trees is robust)
- Works with both categorical and continuous features

### 3. Scalability
- Parallelizes naturally (trees are independent)
- Handles large datasets efficiently

### 4. Interpretability (relative to other ensembles)
- Variable importance provides insights
- Individual trees can be examined
- Proximities enable visualization

### 5. Built-in Validation
- OOB error provides honest generalization estimate
- No need for separate cross-validation

---

## Limitations

### 1. Less Interpretable Than Single Trees
- Can't see a single "path" to a prediction
- Hard to extract simple rules

### 2. Memory and Speed
- Storing many trees requires memory
- Prediction time scales with number of trees

### 3. Extrapolation
- Can't extrapolate beyond the range of training data
- Predictions for extreme values will be bounded by training range

### 4. Imbalanced Classes
- May be biased toward majority class
- May need class weights or sampling adjustments

---

## Hyperparameter Tuning

Random Forests have relatively few hyperparameters:

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|--------|
| **n_estimators** | Number of trees | 100-1000 | More is better (diminishing returns) |
| **max_features** | Features per split | sqrt(p), p/3 | Lower → more diversity, more variance |
| **max_depth** | Tree depth | None (grow full), or 10-30 | Deeper → lower bias, higher variance |
| **min_samples_leaf** | Min samples in leaf | 1-10 | Higher → smoother, simpler trees |

### Practical Tips

1. **Start with defaults**: They often work well!
2. **More trees rarely hurt**: Just costs compute time
3. **max_features is most important**: Try sqrt(p), log(p), and p/3
4. **Deeper trees are fine**: Unlike single trees, Random Forests resist overfitting

---

## Random Forests vs. Boosting

| Aspect | Random Forests | Gradient Boosting |
|--------|----------------|-------------------|
| **Training** | Parallel (independent trees) | Sequential (each tree corrects errors) |
| **Overfitting** | Very resistant | Can overfit with too many rounds |
| **Tuning** | Minimal tuning needed | Requires careful tuning |
| **Accuracy** | Very good | Often slightly better (with tuning) |
| **Speed** | Fast (parallelizable) | Slower (sequential) |
| **Interpretability** | Variable importance | Less interpretable |

### When to Use Which?

**Random Forests**: 
- Quick baseline
- Limited tuning time
- Parallel computing available
- Want OOB error estimate

**Gradient Boosting**:
- Maximum accuracy needed
- Time for hyperparameter tuning
- Tabular data competitions

---

## Summary

### Key Takeaways

1. **Averaging reduces variance**: The fundamental principle behind Random Forests

2. **Decorrelation is key**: Random feature selection makes trees diverse

3. **Can't overfit by adding trees**: Unlike many methods, more trees is (almost) always better

4. **OOB error is free cross-validation**: Built-in generalization estimate

5. **Variable importance provides insights**: Understand which features matter

6. **Minimal tuning required**: Works well out of the box

### The Random Forest Workflow

```
1. Train Random Forest with default settings
2. Check OOB error for baseline performance
3. Examine variable importance for insights
4. If needed, tune max_features and min_samples_leaf
5. For final model, use more trees (500-1000)
```

Random Forests remain one of the best "first try" algorithms for tabular data — accurate, robust, and easy to use!

# Decision Trees

Decision trees are among the most intuitive machine learning algorithms. They make predictions by asking a series of yes/no questions about the features, essentially building a flowchart for decision-making. This mirrors how humans often make decisions—through a sequence of simple questions.

## Decision Trees

**The Core Idea**: Recursively split the input/feature space using simple rules (called "stubs"). Each split divides the data into more homogeneous groups.

**Key Characteristics**:
- Splits are always parallel to the feature axes (like drawing vertical or horizontal lines)
- Each path from root to leaf represents a conjunction of conditions
- The tree structure naturally captures feature interactions

**Mathematical Representation**:
- Each leaf defines a region: $R_j = \{x : d_1 \leq t_1, d_2 \geq t_2, ...\}$
- Prediction for a point: $\hat{Y}_i = \sum_j w_j I\{x_i \in R_j\}$
- Leaf weights (for regression): $w_j = \frac{\sum_i y_i I\{x_i \in R_j\}}{\sum_i I\{x_i \in R_j\}}$ (just the average of y values in that leaf)

**Types of Decision Trees**:

*Binary Splits* (most common):
- **CART** (Classification and Regression Trees): The most widely used, always binary splits
- **C4.5**: Successor to ID3, handles continuous attributes, missing values

*Multi-way Splits*:
- **CHAID** (Chi-Square Automatic Interaction Detection): Uses statistical tests for splits
- **ID3**: Original algorithm, only handles categorical features

## Splitting

The key question: How do we decide which feature to split on and where?

**For Classification Trees—Measuring Impurity**:

We want each split to create more "pure" nodes (nodes dominated by one class).

**Gini Impurity**:
- $\text{Gini} = 1 - \sum_C p_i^2$
- Intuition: The probability of misclassifying a randomly chosen element if we labeled it randomly according to the class distribution in the node
- For a given class $i$: Probability of picking class $i$ and misclassifying it = $p_i \times (1 - p_i)$
- Sum across all classes: $\sum_C p_i(1 - p_i) = 1 - \sum_C p_i^2$
- Range: 0 (pure node, all one class) to $(K-1)/K$ for K classes (max 0.5 for binary)
- Example: If a node has 50% class A and 50% class B, Gini = $1 - (0.5^2 + 0.5^2) = 0.5$

**Entropy Criterion**:
- Based on information theory—measures uncertainty
- If event E is very likely ($P(E) \approx 1$): No surprise when it happens
- If event E is unlikely ($P(E) \approx 0$): Huge surprise when it happens
- Information content: $I(E) = \log(1/P(E)) = -\log(P(E))$
- Entropy = expected information content: $H(E) = -\sum P(E) \log P(E)$
- Range: 0 (pure) to $\log_2(K)$ for K classes (max 1 for binary)
- Maximum entropy when all outcomes equally likely

**For Regression Trees—Measuring Error**:
- **Sum of Squared Errors (SSE)**: $\sum_i (Y_i - \bar{Y})^2$
- This is just the variance times $N$ within the node
- We want splits that minimize the total SSE across child nodes

**Finding the Best Split**:

For each candidate split:
1. Calculate the weighted average reduction in impurity/error
2. Weights = number of observations flowing to each child node

**Example of Gini Reduction**:
- Starting Gini at root: $\text{Gini}_{\text{Root}}$ with $N_{\text{Root}}$ samples
- After split into Left and Right:
    - $\text{Gini}_{\text{New}} = \frac{N_{\text{Left}}}{N_{\text{Root}}} \times \text{Gini}_{\text{Left}} + \frac{N_{\text{Right}}}{N_{\text{Root}}} \times \text{Gini}_{\text{Right}}$
- Choose the split that minimizes $\text{Gini}_{\text{New}}$
- Note: $\text{Gini}_{\text{New}} \leq \text{Gini}_{\text{Root}}$ always (splits never increase impurity)

**The Algorithm**: Greedy search through all features and all possible split points to find the best split at each node.

## Bias-Variance Trade-off

Understanding this trade-off is essential for all of machine learning.

**Bias**:
- Measures how well the algorithm can model the true relationship
- High bias = making strong/restrictive assumptions
    - Example: Using a linear model for a parabolic relationship
- Low bias = fewer assumptions, more flexible

**Variance**:
- Measures how much the model changes across different training datasets
- High variance = model is very sensitive to the specific training data
- Low variance = model is stable across different samples

**Irreducible Error** (Bayes Error):
- The inherent noise in the data
- Cannot be reduced no matter how good the model

**The Trade-off**:
- $\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$
- Simple models: High bias, low variance (underfit)
- Complex models: Low bias, high variance (overfit)
- Goal: Find the sweet spot that minimizes total error

**Decision Trees and the Trade-off**:
- Deep trees have **low bias** (can fit complex patterns)
- But **high variance** (very sensitive to training data)
- This makes them prone to **overfitting**, especially with:
    - Noisy samples
    - Small data samples in deep nodes

**Tree Pruning** addresses overfitting by adding a complexity penalty:
- Objective: $\text{Tree Score} = \text{SSR} + \alpha T$
- Where $T$ = number of leaves, $\alpha$ = complexity parameter
- As the tree grows, SSR reduction must offset the complexity cost

## Nature of Decision Trees

**Strengths**:

*Non-linear Relationships*:
- Decision trees naturally model complex, non-linear decision boundaries
- Unlike splines, which add indicator variables but require continuous boundaries
- Trees can create completely discontinuous predictions

*No Feature Scaling Required*:
- Tree algorithms only care about the ordering of values, not their magnitude
- No need to normalize or standardize features

*Robustness to Outliers*:
- For input feature outliers: Splits simply ignore extreme values
- For output outliers (in regression): Some impact, but less than linear regression
- Note: High-leverage points in regression can have extreme influence; trees are more robust

**Weaknesses**:

*Extrapolation*:
- Trees cannot extrapolate beyond the range of training data
- For values outside training range, prediction = nearest leaf's value
- Linear models can extrapolate (for better or worse)

*Time Series*:
- Trees cannot capture linear trends or seasonality naturally
- Each leaf is a constant prediction—no notion of "trend"

## Bagging

**Bootstrap Aggregation** (Bagging) is a technique to reduce variance.

**The Bootstrap**:
- Given a dataset of size $N$
- Create a new dataset by sampling $N$ points *with replacement*
- Probability that point $i$ is never selected: $(1 - \frac{1}{N})^N \approx \frac{1}{e} \approx 0.37$
- So each bootstrap sample contains ~63% of unique original points

**Bagging for Trees**:
1. Create many bootstrap samples
2. Fit a tree to each
3. Average predictions (regression) or vote (classification)

**Why It Works**:
- Individual trees are unstable (high variance)
- Averaging independent predictions reduces variance
- If predictions were perfectly independent: Variance would decrease by factor of $n$

## Random Forest

Random Forest = Bagging + Random Feature Selection

**The Algorithm**:
1. Create bootstrap samples (the "random" part of the data)
2. At each split, consider only a random subset of features:
    - Classification: typically $\sqrt{p}$ features
    - Regression: typically $p/3$ features
3. Combine predictions: majority vote (classification) or average (regression)

**Why Random Feature Selection?**:
- Without it, all trees would be very similar (correlated)
- If one strong feature dominates, all trees split on it first
- Random selection decorrelates the trees

**Variance Reduction Math**:
- Let $\hat{y}_i$ be prediction from tree $i$, with variance $\sigma^2$
- Let $\rho$ be the correlation between trees
- Variance of average: $V\left(\frac{1}{n}\sum_i \hat{y}_i\right) = \rho\sigma^2 + \frac{1-\rho}{n}\sigma^2$
- As $n \to \infty$: Variance approaches $\rho\sigma^2$
- Lower correlation $\rho$ → lower variance → better!

**Bias vs Variance**:
- Bias remains the same as a single tree (no improvement)
- Variance decreases with more trees
- This is why random forests are so powerful: low bias AND low variance

**Out-of-Bag (OOB) Error**:
- ~37% of data not used in each tree (OOB samples)
- Use these to estimate test error—like free cross-validation!
- For each point, average predictions from trees that didn't train on it
- OOB error typically close to leave-one-out cross-validation error

**Proximity Matrix**:
- For OOB observations, count how often each pair lands in the same leaf
- Creates a similarity measure between observations
- Useful for clustering, visualization, missing value imputation

## ExtraTrees

**Extremely Randomized Trees**—taking randomization even further.

**Key Differences from Random Forest**:

| Aspect | Random Forest | ExtraTrees |
|--------|---------------|------------|
| Data sampling | Bootstrap (63%) | Entire dataset (100%) |
| Split thresholds | Optimized search | Randomly selected |
| Feature selection | Random subset | Random subset |

**Algorithm**:
1. Use the full training set (no bootstrapping)
2. At each node, for each candidate feature:
    - Select a random threshold uniformly between min and max
    - Evaluate the split
3. Choose the best feature-threshold combination

**Trade-off**:
- Even more randomness → even lower variance
- But slightly higher bias than Random Forest
- Much faster training (no optimization of thresholds)

## Variable Importance

Understanding which features matter is often as important as making predictions.

**Split-Based Importance** (built into tree algorithms):
- For each feature $j$, sum the Gini/entropy reduction across all splits using $j$
- Alternative: Count the number of times feature is used for splitting
- **Limitation**: Biased toward continuous features (more possible split points)
- **Limitation**: Biased toward high-cardinality categorical features

**Permutation-Based Importance** (model-agnostic):
1. Calculate baseline accuracy on OOB samples
2. For each feature $j$:
    - Randomly shuffle feature $j$'s values (breaks its relationship with target)
    - Calculate new accuracy
    - Importance = decrease in accuracy
3. Average across all trees

**Why Permutation Importance is Better**:
- Measures actual predictive value, not just usage
- Accounts for redundancy: If feature $j$ has a good surrogate, permuting $j$ won't hurt much
- Like setting the coefficient to 0 in regression

**Partial Dependence Plots (PDPs)**:
- Show the marginal effect of a feature on predictions
- Algorithm: For each value $x_s$ of feature $s$:
    - Set all observations to have $x_s$ for that feature
    - Average the predictions: $\hat{f}(x_s) = \frac{1}{N}\sum_i f(x_s, x_{i,-s})$
- **Assumption**: Features are not correlated (can be misleading otherwise)
- Can identify interactions using Friedman's H-statistic

**Other Importance Methods**:
- **SHAP (Shapley Values)**: Game-theoretic approach, model-agnostic, handles interactions
- **LIME**: Local interpretable model-agnostic explanations—explains individual predictions

## Handling Categorical Variables

**Binary Categorical**: Easy—just a yes/no split

**Multi-Category Variables—Options**:

*One-Hot Encoding*:
- Create a binary feature for each category
- Pro: Simple, works with any algorithm
- Con: Increases dimensionality; for trees, biases toward these features

*Label Encoding*:
- Assign ordinal numbers (1, 2, 3, ...)
- Pro: No dimensionality increase
- Con: Imposes an artificial ordering

*Native Handling* (in tree algorithms):
- Consider all possible subsets of categories for binary splits
- CART: Finds optimal binary grouping
- C4.5, CHAID: Can create multi-way splits (one branch per category)
- Pro: Optimal splits; Con: Exponential search space

## Tree Pruning

Pruning prevents overfitting by limiting tree complexity.

**Pre-Pruning** (Early Stopping):
- Stop growing before the tree is fully expanded
- Criteria:
    - Maximum depth
    - Minimum samples per leaf
    - Minimum impurity decrease
    - Maximum leaf nodes
- Pro: Fast, simple
- Con: Might stop too early (a bad split might enable good later splits)

**Post-Pruning** (Grow then Prune):
- Grow a full tree, then remove unhelpful branches
- Methods:
    - **Cost-Complexity Pruning** (CART): Minimize $\text{SSE} + \alpha \times (\text{number of leaves})$
    - **Reduced Error Pruning (REP)**: Remove nodes that don't improve validation error
    - **Pessimistic Error Pruning (PEP)**: Use statistical adjustments on training error
- Pro: Considers the full tree structure
- Con: More computationally expensive

**Selecting the Pruning Level**:
- Use cross-validation to find optimal $\alpha$ (complexity parameter)
- Plot training/validation error vs. $\alpha$ to visualize the trade-off
- The "right" amount of pruning balances underfitting and overfitting

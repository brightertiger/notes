# XGBoost

XGBoost (eXtreme Gradient Boosting) is arguably the most successful machine learning algorithm for structured/tabular data. It's a highly optimized implementation of gradient boosting that has won countless Kaggle competitions. What makes it special? Regularization to prevent overfitting, computational tricks for speed, and smart handling of missing values.

## Mathematical Details

**The Objective Function**:

XGBoost adds explicit regularization to the gradient boosting objective:

$$\text{Objective} = \sum_i L(y_i, \hat{y}_i) + \underbrace{\gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2}_{\text{Regularization}}$$

Where:
- $L(y_i, \hat{y}_i)$ = Loss function (e.g., MSE, log-loss)
- $T$ = Number of leaves in the tree
- $w_j$ = Output value (weight) of leaf $j$
- $\gamma$ = Penalty per leaf (controls tree complexity)
- $\lambda$ = L2 regularization on leaf weights

**Common Loss Functions**:
- **MSE** (Regression): $L = \frac{1}{2}(y_i - \hat{y}_i)^2$
- **Log-loss** (Classification): $L = -[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$

**Prediction Update**:
$$\hat{y}_i = \hat{y}_i^{(0)} + \sum_{j=1}^T w_j \cdot I(x_i \in R_j)$$

Where $\hat{y}_i^{(0)}$ is the prediction from previous rounds.

**Second-Order Taylor Approximation**:

The key insight: Approximate the loss with a second-order Taylor expansion for efficient optimization.

For adding a new tree with output value $O$:
$$L(y_i, \hat{y}_i^{(0)} + O) \approx L(y_i, \hat{y}_i^{(0)}) + g_i \cdot O + \frac{1}{2}H_i \cdot O^2$$

Where:
- $g_i = \frac{\partial L}{\partial \hat{y}_i}$ (gradient, first derivative)
- $H_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}$ (Hessian, second derivative)

**Why Second-Order?**:
- First-order (like regular gradient boosting): Only tells direction
- Second-order: Also tells curvature, enabling more accurate steps
- Newton's method vs gradient descent—converges faster!

**Simplified Objective**:

After substitution and dropping constants:
$$\text{Objective} = \sum_i g_i \cdot O + \gamma T + \frac{1}{2}\left(\sum_i H_i + \lambda\right) O^2$$

**Optimal Leaf Value**:

Taking derivative with respect to $O$ and setting to zero:
$$O^* = -\frac{\sum g_i}{\sum H_i + \lambda}$$

**For Different Loss Functions**:

| Loss | $g_i$ | $H_i$ |
|------|-------|-------|
| MSE | $\hat{y}_i - y_i$ | 1 |
| Log-loss | $\hat{y}_i - y_i$ | $\hat{y}_i(1-\hat{y}_i)$ |

**The Role of $\lambda$**:
- Appears in denominator: $O^* = -\frac{\sum g_i}{\sum H_i + \lambda}$
- High $\lambda$ → pushes leaf values toward 0
- Prevents any single leaf from having extreme predictions
- Acts like L2 regularization in ridge regression

## Regression

**Similarity Score** (for evaluating potential splits):
$$\text{Similarity} = \frac{G^2}{H + \lambda} = \frac{(\sum g_i)^2}{\sum H_i + \lambda}$$

For MSE loss:
$$\text{Similarity} = \frac{(\sum r_i)^2}{N + \lambda}$$

Where $r_i = y_i - \hat{y}_i$ is the residual and $N$ is the number of samples.

**Intuition**:
- Numerator: Total residual squared (how much signal?)
- Denominator: Count + regularization (how much noise?)
- Higher similarity = more "pure" node (good for prediction)

**Effect of $\lambda$**:
- Large $\lambda$ → smaller similarity scores
- Reduces sensitivity to individual observations
- More pruning, simpler trees

**Gain from a Split**:
$$\text{Gain} = \text{Similarity}_{\text{left}} + \text{Similarity}_{\text{right}} - \text{Similarity}_{\text{parent}}$$

**Split Criterion**:
- Only split if $\text{Gain} > \gamma$
- $\gamma$ controls minimum improvement required
- Even with $\gamma = 0$, pruning still happens (due to regularization)!

**Pruning Mechanisms**:
- Maximum depth
- Minimum cover (sum of Hessians, i.e., $N$ for regression)
- Trees are grown fully, then pruned backward
- Key difference from pre-pruning: A "bad" split might enable good subsequent splits

**Final Predictions**:
- Leaf output: $\frac{\sum r_i}{N + \lambda}$
- Ensemble: Initial Prediction + $\eta \times \text{(Tree 1 output)} + \eta \times \text{(Tree 2 output)} + ...$
- Initial prediction = mean of target values
- $\eta$ = learning rate (typically 0.01-0.3)

## Classification

For classification, XGBoost works with log-odds (logits) and uses log-loss.

**Similarity Score**:
$$\text{Similarity} = \frac{(\sum r_i)^2}{\sum p_i(1-p_i) + \lambda}$$

Where:
- $r_i = y_i - p_i$ (residual: actual minus predicted probability)
- $p_i$ = current probability estimate
- Denominator: $\sum p_i(1-p_i)$ comes from the Hessian of log-loss

**Gain Calculation**: Same as regression.

**Cover/Minimum Weight**:
- For regression: Just $N$ (number of samples)
- For classification: $\sum p_i(1-p_i)$
- This is the sum of Hessians—measures "effective sample size"
- Points with $p \approx 0.5$ contribute most (most uncertain)

**Leaf Output**:
$$O = \frac{\sum r_i}{\sum p_i(1-p_i) + \lambda}$$

**Ensemble Prediction**:
1. Initial prediction = $\log\left(\frac{\bar{y}}{1-\bar{y}}\right)$ (log-odds of class proportion)
2. Add tree contributions with learning rate
3. Final output is log-odds
4. Convert to probability: $p = \frac{1}{1 + e^{-\text{log-odds}}}$

## Optimizations

XGBoost's speed comes from several clever tricks:

**Approximate Greedy Algorithm**:
- Exact algorithm: Try every possible split point (slow for large data)
- Approximate: Bucket continuous features into quantiles
- Only consider bucket boundaries as split candidates
- Much faster with minimal accuracy loss

**Quantile Sketch Algorithm**:
- Need to find quantiles in a distributed setting
- XGBoost uses weighted quantiles (weighted by Hessian/cover)
- Points with higher uncertainty (higher $H_i$) get more weight
- More granular splits where they matter most

**Sparsity-Aware Split Finding**:
- Real data often has missing values
- XGBoost learns optimal direction for missing values:
    1. Compute split gain sending missing values left
    2. Compute split gain sending missing values right
    3. Choose direction with higher gain
- This "default direction" is learned during training
- Also works for zero values in sparse data

**Cache-Aware Access**:
- Gradients and Hessians stored in cache for fast access
- Block structure for efficient memory access
- Out-of-core computation for data that doesn't fit in memory

## Comparisons

**XGBoost**:
- Stochastic gradient boosting (row/column subsampling)
- No native handling of categorical variables (need encoding)
- Depth-wise tree growth (all nodes at same depth split together)
- Level-by-level: Explores all possibilities at each depth

**LightGBM**:
- **GOSS** (Gradient-based One-Side Sampling): Oversample high-gradient points
- Native encoding for categorical variables
- **EFB** (Exclusive Feature Bundling): Combines mutually exclusive features
- Histogram-based splitting (faster)
- **Leaf-wise growth**: Splits the leaf with highest gain
    - Faster convergence but can overfit more easily

**CatBoost**:
- **MVS** (Minimum Variance Sampling): More statistically sound sampling
- Superior categorical encoding (ordered target encoding to prevent leakage)
- **Symmetric trees**: All nodes at same depth use the same split
    - Faster inference, natural regularization
- Handles missing values and categorical features natively

## XGBoost vs. Traditional Gradient Boosting

**System Optimizations**:
- **Parallelization**: Tree construction parallelized (not tree-to-tree, but within trees)
- **Cache-aware access**: Block structure for efficient memory usage
- **Out-of-core computation**: Can handle datasets larger than memory

**Algorithmic Enhancements**:
- **Regularization**: Built-in L1 and L2 on weights
- **Missing values**: Learned default directions
- **Newton boosting**: Second-order optimization (faster convergence)
- **Weighted quantile sketch**: Approximate split finding

**Result**: XGBoost is often 10-100x faster than sklearn's GradientBoostingClassifier while achieving similar or better accuracy.

## Handling Missing Values

**The Problem**: Most ML algorithms require complete data, forcing imputation.

**XGBoost's Approach**:
- Treat missing values as a special category
- During training: Learn whether missing → left or missing → right
- The "default direction" is chosen to maximize gain
- No preprocessing needed!

**Why This Works Better**:
- Imputation (mean, median) assumes missing is similar to observed
- XGBoost learns what missing values *actually* mean
- Different features can have different optimal directions

**Comparison to Traditional Approaches**:
| Method | Approach | Limitation |
|--------|----------|------------|
| Mean imputation | Replace with mean | Reduces variance |
| Indicator variable | Add "is_missing" feature | Doubles features |
| XGBoost native | Learn optimal direction | None—learned from data |

## Hyperparameter Tuning

**Key Hyperparameters**:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_estimators` | Number of trees | 100-10000 |
| `learning_rate` ($\eta$) | Step size shrinkage | 0.01-0.3 |
| `max_depth` | Maximum tree depth | 3-10 |
| `min_child_weight` | Minimum sum of Hessians in leaf | 1-10 |
| `gamma` ($\gamma$) | Minimum gain for split | 0-5 |
| `subsample` | Fraction of rows per tree | 0.5-1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.5-1.0 |
| `lambda` ($\lambda$) | L2 regularization | 0-10 |
| `alpha` ($\alpha$) | L1 regularization | 0-10 |

**Tuning Strategy**:

1. **Fix learning rate low** (e.g., 0.1), tune other params
2. **Control complexity**: `max_depth`, `min_child_weight`, `gamma`
3. **Add randomness**: `subsample`, `colsample_bytree`
4. **Add regularization**: `lambda`, `alpha`
5. **Lower learning rate** and increase `n_estimators`

**Common Approaches**:
- **Grid Search**: Exhaustive but expensive
- **Random Search**: Often just as good, much faster
- **Bayesian Optimization**: Intelligent exploration of parameter space
- **Early Stopping**: Use validation set, stop when performance plateaus

**Rule of Thumb**:
- Lower `learning_rate` + more `n_estimators` = better but slower
- Start with defaults, then tune `max_depth` and `learning_rate`
- Use cross-validation to avoid overfitting to validation set

# Decision Trees

## Decision Trees

-   Recursively split the input / feature space using stubs i.e. decision rules
    -   Splits are parallel to the axis
-   Mathematical Representation
    -   $R_j = \{ x : d_1 <= t_1, d_2 >= t_2 ... \}$\
    -   $\hat Y_i = \sum_j w_j I\{x_i \in R_j\}$
    -   $w_j = \frac{\sum_i y_i I \{x_i \in R_j\}}{\sum_i I \{x_i \in R_j\}}$
-   Types of Decision Trees
    -   Binary Splits
        -   Classification and Regression Trees (CART)
        -   C4.5
    -   Multiple Splits:
        -   CHAID (Chi-Square Automatic Interaction Detection)
        -   ID3

## Splitting

-   Split Criteria for Classification Trees
    -   The nodes are split to decrease impurity in classification
    -   Gini Criterion
        -   $1 - \sum_C p_{i}^2$
        -   Probability that observation belongs to class i: $p_i$
        -   Misclassification:
        -   For a given class (say i):
            -   $p_i \times p_{k \ne i} = p_i \times (1 - p_i)$
        -   Across all classes:
        -   $\sum_C p_i \times (1 - p_i)$
        -   $\sum_C p_i - \sum_C p_{i}^2$
        -   $1 - \sum_C p_{i}^2$
        -   Ranges from 0 (pure node) to 0.5 (binary), or more generally (K-1)/K for K classes
    -   Entropy Criterion
        -   Measure of uncertainty of a random variable
        -   Given an event E
            -   p(E) = 1 $\implies$ No Surprise
            -   p(E) = 0 $\implies$ Huge Surprise
            -   Informaion Content: $I(E) = \log(1 / p(E))$
        -   Entropy is the expectation of this information content
            -   $H(E) = - \sum p(E) \log(p(E))$
            -   Maximum when all outcomes have same probability of occurrence
        -   Ranges from 0 (pure node) to log₂(K) for K classes (max 1 for binary)
-   Split Criteria for Regression Trees
    -   Sum-Squared Error
    -   $\sum_i (Y_i - \bar Y)^2$
-   Finding the Split
    -   For any candidate value:
        -   Calculate the weighted average reduction in impurity / error
        -   Weights being the number of observations flowing in the child nodes
    -   Starting Gini
        -   $\text{Gini}_{\text{Root}}$
        -   $N_{\text{Root}}$
    -   After Split
        -   Child Nodes
            -   $\text{Gini}_{\text{Left}}, N_{\text{Left}}$
            -   $\text{Gini}_{\text{Right}}, N_{\text{Right}}$
        -   Updated Gini
            -   $\frac{N_{\text{Left}}}{N_{\text{Root}}} \times \text{Gini}_{\text{Left}} + \frac{N_{\text{Right}}}{N_{\text{Root}}} \times \text{Gini}_{\text{Right}}$
    -   Find the split, the results in minimum updated Gini
    -   Updated Gini \<= Starting Gini
    -   Greedy algorithms to find the best splits

## Bias-Variance Trade-off
-   Bias
    -   Measures ability of an ML algorithm to model true relationship between features and target
    -   Simplifying assumptions made by the model to learn the relationship
        - Example: Linear vs Parabolic relationship
    -   Low Bias: Less restrictive assumptions
    -   High Bias: More restrictive assumptions
- Variance
    - The difference in model performance across different datasets drawn from the same distribution
    -   Low Variance: Small changes to model performance with changes in datasets
    -   High Variance: Large changes to model performance with changes in datasets
-   Irreducible Error
    -   Bayes error
    -   Cannot be reduced irrespective of the model form
-   Best model minimizes: $\text{MSE} = \text{bias}^2 + \text{variance}$
-   Decision trees have low bias and high variance
-   Decision trees are prone to overfitting
    -   Noisy Samples
    -   Small data samples in nodes down the tree
    -   Tree Pruning solves for overfitting
        -   Adding a cost term to objective which captures tree complexity
        -   $\text{Tree Score} = SSR + \alpha T$
        -   As the tree grows in size, the reduction in SSR has to more than offset the complexity cost

## Nature of Decision Trees

-   Decision Trees can model non-linear relationships (complex decision boundaries)
-   Spline regressions cannot achieve the same results
    -   Spline adds indicator variables to capture interactions and create kinks
    -   But the decision boundary has to be continuous
    -   The same restriction doesn't apply to decision trees
-   Decision Trees don't require feature scaling
-   Decision Trees are less sensitive to outliers
    -   Outliers are of various kinds:
        -   Outliers: Points with extreme values
            -   Input Features
                -   Doesn't impact Decision Trees
                -   Split finding will ignore the extreme values
            -   Output / Target
        -   Influential / High-Leverage Points: Undue influence on model
-   Decision Trees cannot extrapolate well to ranges outside the training data
-   Decision trees cannot capture linear time series based trends / seasonality

## Bagging

-   Bootstrap Agrregation
-   Sampling with repetition
    -   Given Dataset of Size N
    -   Draw N samples with replacement
    -   Probability that a point (say i) never gets selected
        -   $(1 - \frac{1}{N})^N \approx \frac{1}{e}$
    -   Probability that a point (say i) gets selected atleast once
        -   $1 - \frac{1}{e} \approx 63\%$

## Random Forest

-   Use bootstrap aggregation (bagging) to create multiple datasets
    -   "Random" subspace of dataset
-   Use subset of variables for split at each node
    -   sqrt for classification
    -   m//3 for regression
-   Comparison to single decision tree
    -   Bias remains the same
    -   Variance decreases
    -   Randomness in data and splits reduces the correlation in prediction across trees
    -   Let $\hat y_i$ be the prediction from ith tree in the forest
    -   Let $\sigma^2$ be the variance of $\hat y_i$
    -   Let $\rho$ be the correlation between two trees in the forest
    -   $V(\sum_i \hat y_i) = \sum V(\hat y_i) + 2 \sum\sum COV(\hat y_i, \hat y_j)$
    -   $V(\sum_i \hat y_i) = n \sigma^2 + n(n-1) \rho \sigma^2$
    -   $V( \frac{1}{n} \sum_i \hat y_i) = \rho \sigma^2 + \frac{1-\rho}{n} \sigma^2$
    -   Variance goes down as more trees are added, but bias stays put
-   Output Combination
    -   Majority Voting for Classification
    -   Averaging for Regression
-   Out-of-bag (OOB) Error
    -   Use the non-selected rows in bagging to estimate model performance
    -   Comparable to cross-validation results
-   Proximity Matrix
    -   Use OOB observations
    -   Count the number of times each pair goes to the same terminal node
    -   Identifies observations that are close/similar to each other

## ExtraTrees

-   Extremely Randomized Trees
-   Key differences from Random Forest:
    -   Bagging: No (uses entire training set for each tree, unlike RF which uses bootstrap samples)
    -   Split thresholds: Randomly selected instead of searching for optimal splits
-   Multiple trees are built using:
    -   Random variable subset for splitting (same as Random Forest)
    -   Random threshold selection for each variable (different from Random Forest)
-   More randomness leads to reduced variance but slightly higher bias compared to Random Forest

## Variable Importance

-   Split-based importance
    -   If variable j is used for split
        -   Calculate the improvement in Gini at the split
    -   Sum this improvement across all trees and splits wherever jth variable is used
    -   Alternate is to calculate the number of times variable is used for splitting
    -   Biased in favour of continuous variables which can be split multiple times
-   Permutation-based importance / Boruta
    -   Use OOB samples to calculate variable importance
    -   Take bth tree:
        -   Pass the OOB samples and calculate accuracy
        -   Permute jth variable and calculate the decrease in accuracy
    -   Average this decrease in accuracy across all trees to calculate variable importance for j
    -   Effect is similar to setting the coefficient to 0 in regression
    -   Takes into account if good surrogates are present in the dataset
-   Partial Dependence Plots
    -   Marginal effect of of a feature on target
    -   Understand the relationship between feature and target
    -   Assumes features are not correlated
    -   $\hat f(x_s) =\frac{1}{C} \sum f(x_s,x_i)$
    -   Average predictions over all other variables
    -   Can be used to identify important interactions
        -   Friedman's H Statistic
        -   If features don't interact Joint PDP can be decomposed into marginals
-   Shapley Values
    -   Model agnostic feature importance
-   LIME 

## Handling Categorical Variables

-   Binary categorical variables are easily incorporated into decision trees
-   For multi-category variables:
    -   One-hot encoding (creates a binary feature for each category)
    -   Label encoding (assigns an ordinal value to each category)
-   Trees can directly handle categorical variables by considering all possible subsets for splitting
    -   CART typically uses binary splits (creates a binary question from categorical features)
    -   C4.5 and CHAID can create multi-way splits

## Tree Pruning

-   Decision trees are prone to overfitting
    -   Noisy Samples
    -   Small data samples in nodes down the tree
    -   Tree Pruning solves for overfitting
        -   Adding a cost term to objective which captures tree complexity
        -   $\text{Tree Score} = SSR + \alpha T$
        -   As the tree grows in size, the reduction in SSR has to more than offset the complexity cost
-   Pre-pruning vs. Post-pruning:
    -   Pre-pruning: Stops tree growth early using criteria like:
        -   Minimum samples per leaf
        -   Maximum depth
        -   Minimum impurity decrease
    -   Post-pruning: Grows a full tree and then removes branches that don't improve generalization
        -   Cost-complexity pruning (used in CART)
        -   Reduced Error Pruning (REP)
        -   Pessimistic Error Pruning (PEP)
-   Cross-validation can be used to determine optimal pruning level 
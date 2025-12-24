# Additive Models

## Generalized Additive Models

-   Linear models fail to capture non-linear trends
-   Additive models an alternative
    -   $g[\mu(X)] = \alpha + f(X_1) + f(X_2)....$
    -   $f(x)$ are non-parametric smoothing functions (say cubic splines)
    -   $\mu(x)$ is the conditional mean
    -   $g(x)$ is the link functions
        -   identity, logit, log-linear etc.\
-   Estimation using Penalized Sum Squares (PRSS)
-   The coefficients of the regression are replaced with a flexible function (say spline)
    -   Allows for modeling non-linear relationships

## Tree-based Methods

-   Partition the feature space into rectangles and fit a simple model in each partition
-   Regression Setting
-   $f(X) = \sum c_i I\{X \in R_i\}$
-   $c_m = ave(y_i | X_i \in R_m)$
-   Greedy Algorithms to find best splits
    -   $R_1 = \{X | X_j \le s\}; \; R_2 = \{X | X_j > s\}$\
    -   $\min_{j,s} \min \sum (y_i - c_i)^2 I\{X \in R_i\}$
-   Tree size is a hyperparameter
-   Pruning
    -   Option-1
        -   Split only if delta is greater than some threshold
        -   Short Sighted, the node may lead to a better split down the line\
    -   Option 2
        -   Grow the tree to full size (say depth 5)
        -   $N_m$ \# of observations in m'th node
        -   $C_m = \sum y_i / N_m$
        -   $Q_m = {1 \over N_m }\sum (y_i - C_m)^2$
        -   Cost-Complexity Pruning
        -   $C = \sum_T N_m Q_m(T) + \alpha |T|$
        -   $\alpha$ governs the trade-off, large value leads to smaller trees
-   Classification Setting
-   $p_{mk} = {1 \over N_m}\sum_{R_m} I\{y_i = k\}$
-   Splitting Criteria
    -   Miss-classification Error: $1 - \hat p_{mk}$
    -   Gini Index: $\sum_K p_{mk}(1 - \hat p_{mk})$
        -   Probability of misclassification
        -   Variance of Binomial Distribution
    -   Cross-Entropy: $- \sum_K p_{mk} \log (p_{mk})$
    -   Gini Index and Cross Entropy more sensitive to node probabilities
-   Splitting categorical variable
    -   N levels, $2^{N-1} - 1$ possible paritions
    -   Order the categories by proportion
    -   Treat the variable as continuous
-   Missing Values
    -   Create a new level within the original corresponding to missing observations
    -   Create a surrogate variable for missing values
        -   Split by non-missing values
        -   Leverage the correlation between predictors and surrogates to minimize loss of information
-   Evaluation
    -   $L_{xy} =$ Loss for predicting class x object as k
    -   $L_{00}, L_{11} = 0$
    -   $L_{10} =$ False Negative
    -   $L_{01} =$ False Positive
    -   Sensitivity:
        -   Prediciting disease as disease (Recall)
        -   TP / TP + FN
        -   $L_{11} / (L_{11} + L_{10})$
    -   Specificity:
        -   Predicting non-disease as non-disease
            -   TN / TN + FP
            -   $L_{00} / (L_{00} + L_{01})$\
    -   AUC-ROC
        -   How Sensitivity (y) and Specificity (x) vary with thresholds
        -   Area under ROC Curve is the C-statistic
        -   Equivalent to Mann-Whitney U Test, Wilcoxin rank-sum test
        -   Median Difference in prediction scores for two groups
-   MARS
    -   High dimension regression
    -   Piece-wise Linear basis Functions
    -   Analogous to decision tree splits
    -   Can handle interactions

## PRIM

-   Patient Rule Induction Method
-   Boxes with high response rates
-   Non-tree partitioning structure
-   Start with a large box and
    -   Peeling: compress the side that gives the largest mean
    -   Pasting: expand the bix dimensions that gives the largest mean

## Mixture of Experts

-   Tree splits are not hard decisions but soft probabilities
-   Terminal nodes are called experts
    -   A linear model is fit in each terminal node
-   Non-terminal nodes are called gating networks
-   The decision of experts is combined by gating networks
-   Estimation via EM Algorithm 
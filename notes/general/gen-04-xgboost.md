# XGBoost

-   Extreme Gradient Boosting
-   Introduces regularization to reduce overfitting

## Mathematical Details

-   Loss Function
    -   $L(y_i, p_i)$
    -   MSE
        -   ${1 \over 2}\sum{(y_i - p_i)^2}$
    -   NLL Loss
        -   $- \sum {y_i \log p_i + (1 - y_i) \log (1 -p_i)}$
-   In XGBoost, the objective has regularization terms
    -   $\sum_i L(y_i, p_i) + \gamma T + {1 \over 2} \lambda \sum_{j=1}^T w_j^2$
    -   $\gamma$ is the complexity cost per leaf
    -   $\lambda$ is the L2 regularization term on leaf weights
    -   $T$ is the number of leaves in the tree
    -   $p_i = p_i^0 + \sum_{j=1}^T w_j I(x_i \in R_j)$
    -   $p_i^0$ is the initial prediction / prediction from previous round
-   High values of $\lambda$ will push the optimal output values close to 0
-   Second-order Taylor approximation to simplify the objective
    -   $L(y_i, p_i^0 + O_{value})$
    -   $L(y_i, p_i^0) + \frac{dL}{dO_{value}} O_{value} + {1 \over 2} \frac{d^2L}{dO_{value}^2} O_{value}^2$
    -   $L(y_i, p_i^0) + g O_{value} + {1 \over 2} H O_{value}^2$
    -   $L(y_i, p_i^0)$ is constant
    -   $\sum_i L(y_i, p_i) = \sum_i g_i O_{value} + {1 \over 2} \sum H_i O_{value}^2$
-   Objective Function
    -   $\sum_i L(y_i, p_i) + \gamma T + {1 \over 2} \lambda O_{value}^2$
    -   $\sum_i g_i O_{value} + \gamma T + {1 \over 2} (\sum H_i + \lambda) O_{value}^2$
-   Optimal output value
    -   Differentiate objective function wrt $O_{value}$
    -   $O_{value}^* = - \frac{\sum g_i}{\sum H_i + \lambda}$
    -   For MSE:
        -   $g_i = - (y_i - p_i^0)$
        -   $H_i = 1$
    -   For NLL
        -   Output value is log(odds)
        -   $g_i = - (y_i - p_i)$
        -   $H_i = p_i (1 - p_i)$
-   Splitting Criteria
    -   Objective value at optimal output
    -   $\sum_i g_i O_{value} + \gamma T + {1 \over 2} (\sum H_i + \lambda) O_{value}^2$
    -   ${1 \over 2}{\sum_i g_i^2 \over \sum H_i + \lambda} + \gamma T$

## Regression

-   Calculate similarity score
    -   $G^2 / (H + \lambda)$
    -   $\lambda$ is the regularization parameter
    -   Reduces sensitivity to a particular observation
    -   Large values will result in more pruning (shrinks similarity scores)
    -   In case of MSE loss function
        -   $\sum_i r_i^2 / (N + \lambda)$
        -   $r$ is the residual
        -   $N$ is the number of observations in the node
-   Calculate Gain for a split
    -   $\mathrm{Gain} = \mathrm{Similarity_{left}} + \mathrm{Similarity_{right}} - \mathrm{Similarity_{root}}$\
-   Split criterion
    -   $\mathrm{Gain} - \gamma > 0$
    -   $\gamma$ controls tree complexity
    -   Helps prevent over fitting
    -   Setting $\gamma = 0$ doesn't turn-off pruning
-   Pruning
    -   Max-depth
    -   Cover / Minimum weight of leaf node
        -   N for regression
    -   Trees are grown fully before pruning
        -   If a child node satisfies minimum Gain but root doesn't, the child will still exist
-   Output Value of Tree
    -   $\sum_i r_i / (N + \lambda)$\
-   Output Value of Ensemble
    -   Initial Prediction + $\eta$ Output Value of 1st Tree ....
    -   Initial prediction is the simple average of target
    -   $\eta$ is the learning rate

## Classification

-   Calculate similarity score
    -   $G^2 / (H + \lambda)$
    -   In case of Log loss function
        -   $\sum r_i^2 / (\sum{p_i (1-p_i)} + \lambda)$
        -   $r$ is the residual
        -   $p$ is the previous probability estimate
-   Calculate Gain for a split
    -   $\mathrm{Gain} = \mathrm{Similarity_{left}} + \mathrm{Similarity_{right}} - \mathrm{Similarity_{root}}$\
-   Split criterion
    -   $\mathrm{Gain} - \gamma > 0$
-   Pruning
    -   Max Depth
    -   Cover / Minimum weight of leaf node
        -   $\sum{p_i (1-p_i)}$
-   Output Value of Tree
    -   $\sum r_i / (\sum{p_i (1-p_i)} + \lambda)$
-   Output Value of Ensemble
    -   Initial prediction
        -   Simple average of target\
        -   Convert the value to log(odds)
    -   Initial Prediction + $\eta$ Output Value of 1st Tree ....
    -   Output is log(odds)
    -   Transform the value to probability

## Optimizations

-   Approximate Greedy Algorithm
    -   Finding splits faster
    -   Histogram based splits by bucketing the variables
-   Quantile Sketch Algorithm
    -   Approximately calculate the quantiles in parallel
    -   Quantiles are weighted by cover / hessian\
-   Sparsity Aware Split Finding
    -   Calculate the split based on known data values of the variable
    -   For missing data:
        -   Send the observations to left node and calculate the Gain
        -   Send the observations to right node and calculate the Gain
    -   Evaluate which path gives maximum Gain
-   Cache Aware Access
    -   Stores gradients and hessians in Cache
    -   Compress the data and store on hard-drive for faster access

## Comparisons

-   XGBoost
    -   Stochastic Gradient Boosting
    -   No Treatment for categorical variables
    -   Depth-wise tree growth
-   LightGBM
    -   Gradient One-Side Sampling (GOSS)
        -   Maximum Gradient Observation are oversampled
    -   Encoding for categorical variables
    -   Exclusive Feature Bundling to reduce number of features
    -   Histrogram based splitting
    -   Leaf-wise tree growth
-   CatBoost
    -   Minimum Variance Sampling
    -   Superior encoding techniques for categorical variables
        -   Target encoding
    -   Symmetric tree growth 

## XGBoost vs. Traditional Gradient Boosting

-   Key Improvements in XGBoost:
    -   System Optimizations:
        -   Parallelized tree construction
        -   Cache-aware access patterns
        -   Out-of-core computation for large datasets
    -   Algorithmic Enhancements:
        -   Regularization to prevent overfitting
        -   Built-in handling of missing values
        -   Newton boosting (using second-order derivatives)
        -   Weighted quantile sketch for approximate split finding
    -   These improvements make XGBoost significantly faster and more memory-efficient than traditional gradient boosting implementations

## Handling Missing Values

-   XGBoost has a built-in method for handling missing values
-   For each node in a tree:
    -   It learns whether missing values should go to the left or right branch
    -   Direction is determined by which path optimizes the objective function
    -   This approach allows XGBoost to handle missing values without preprocessing
-   Contrast with traditional approaches:
    -   Imputation (mean, median, mode replacement)
    -   Creating indicator variables
    -   XGBoost's approach often performs better as it learns the optimal direction during training

## Hyperparameter Tuning

-   Key hyperparameters to tune:
    -   `n_estimators`: Number of boosting rounds
    -   `learning_rate`: Step size shrinkage to prevent overfitting
    -   `max_depth`: Maximum depth of trees
    -   `min_child_weight`: Minimum sum of instance weight needed in a child
    -   `gamma`: Minimum loss reduction required for a split
    -   `subsample`: Fraction of samples used for fitting trees
    -   `colsample_bytree`: Fraction of features used for fitting trees
    -   `lambda`: L2 regularization term on weights
    -   `alpha`: L1 regularization term on weights
-   Common tuning approaches:
    -   Grid search with cross-validation
    -   Random search
    -   Bayesian optimization 
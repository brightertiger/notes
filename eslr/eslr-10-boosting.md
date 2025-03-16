# Boosting

## AdaBoost

-   Sequentially apply weak classifiers on modified versions of the data
-   Data modifications involve re-weighting the observations
    -   Errors from previous round are given more weight
    -   Focus on hard-to-classify examples
-   $Y \in \{-1,+1\}$
-   $G(x_i) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x_i)\right)$
    -   $G_m(.)$ is weak classifier with an accuracy slightly better than random
    -   $\alpha_m$ is calculated by the boosting algorithm
    -   Final classifier is weighted vote of weak classifiers
-   Algorithm
    -   Initial weights $w_i = 1/N$
    -   For m = 1 to M rounds:
        -   Fit classifier $G_m(x)$ using weights $w_i$
        -   Compute Error $err_m = \frac{\sum_{i=1}^N w_i I\{y_i \neq G_m(x_i)\}}{\sum_{i=1}^N w_i}$
        -   Compute $\alpha_m = \log((1 - err_m) / err_m)$
        -   Update weights $w_i \leftarrow w_i \exp(\alpha_m I\{y_i \neq G_m(x_i)\})$
-   Properties
    -   Adaptive: focus shifts to harder examples
    -   Resistant to overfitting (in practice)
    -   Theoretical error bound: $\prod_m \sqrt{4 \cdot err_m \cdot (1-err_m)}$
    -   Works well with decision trees as base learners

## Additive Models

-   Boosting fits a forward stagewise additive model
-   $f(x) = \sum_{m=1}^M \beta_m b(x, \gamma_m)$
-   $\min_{\beta_m,\gamma_m} \sum_{i=1}^N L(y_i, \beta_m b(x_i, \gamma_m))$
    -   Optimal values of $\beta$ and $\gamma$ can be found iteratively
    -   $\min_{\beta,\gamma} \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \beta b(x_i, \gamma))$
    -   Stagewise: optimize $\beta, \gamma$ given fixed $f_{m-1}$
-   L2 Loss Function
    -   $\min_{\beta,\gamma} \sum_{i=1}^N (y_i - f_{m-1}(x_i) - \beta b(x_i, \gamma))^2$
    -   $\min_{\beta,\gamma} \sum_{i=1}^N (r_{im} - \beta b(x_i, \gamma))^2$
    -   Fit on residuals from previous rounds
    -   Equivalent to gradient descent in function space with squared error
    -   Robust Loss Functions for regression
        -   Huber Loss
        -   L2 penalty for large errors
        -   L1 penalty for small errors
        -   Less sensitive to outliers
-   Exponential Loss
    -   $L(y, f(x)) = \exp(-y f(x))$
    -   Equivalent to using deviance
    -   The optimal $f$ that minimizes this loss is $\frac{1}{2}\log\frac{P(Y=1|X=x)}{P(Y=-1|X=x)}$
    -   Hence justified to use the sign of $f(x)$ for prediction
    -   $\min_{\beta,G} \sum_{i=1}^N \exp(-y_i(f_{m-1}(x_i) + \beta G(x_i)))$
    -   $\min_{\beta,G} \sum_{i=1}^N w_i^{(m)} \exp(-y_i \beta G(x_i))$
    -   $\min_{\beta,G} \exp(-\beta) \sum_{y_i=G(x_i)} w_i^{(m)} + \exp(\beta) \sum_{y_i\neq G(x_i)} w_i^{(m)}$
    -   Optimal $\beta = \frac{1}{2}\log\frac{1-err_m}{err_m}$
    -   AdaBoost is equivalent to forward stagewise additive modeling with exponential loss

## Gradient Boosting

-   Gradient Descent in function space
-   Minimize $L(f) = \sum L(y_i, f(x_i))$
-   $\arg \min L(\mathbf f); \; \mathbf f = \{f(x_1), f(x_2)....\}$
-   Additive Models $ \mathbf f = \sum_m h_m$
-   Steepest Descent $h_m = \rho_m g_m$
    -   $g_{im} = -\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}|_{f=f_{m-1}}$
    -   Negative gradient of loss function
-   Algorithm:
    1. Initialize $f_0(x) = \arg\min_{\gamma}\sum_{i=1}^N L(y_i, \gamma)$
    2. For m = 1 to M:
       - Compute negative gradient: $r_{im} = -\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}|_{f=f_{m-1}}$
       - Fit a base learner (e.g., regression tree) to $r_{im}$
       - Line search for optimal step size: $\rho_m = \arg\min_{\rho}\sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \rho h_m(x_i))$
       - Update: $f_m(x) = f_{m-1}(x) + \rho_m h_m(x)$
-   Line Search for optimal step size
    -   $\rho_m = \arg \min L(f_{m-1} - \rho g_m)$
    -   Can be solved analytically for some loss functions
    -   Numerical optimization for others
-   Gradients for common loss functions
    -   L2 Loss: Residual $y_i - f(x_i)$
    -   L1 Loss: Sign of Residual $\text{sign}(y_i - f(x_i))$
    -   Classification / Deviance: $y_i - p(x_i)$
    -   Huber: Combination of L1 and L2 depending on residual magnitude
-   Popular implementations
    -   XGBoost: Optimized implementation with additional regularization
    -   LightGBM: Gradient-based One-Side Sampling (GOSS) for efficiency
    -   CatBoost: Better handling of categorical variables
-   Regularization techniques
    -   Shrinkage: Multiply each update by learning rate η (0 < η < 1)
    -   Subsampling: Use random subset of training data for each iteration
    -   Early stopping: Stop when validation performance degrades
    -   Tree constraints: Limit depth, minimum samples per leaf, etc. 
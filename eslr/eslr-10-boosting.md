# Boosting

## AdaBoost

-   Sequentially apply the weak classifers on modified versions of the data
-   Data modifications involve re-weighting the observations
    -   Errors from previous round are given more weight
-   $Y \in \{-1,+1\}$
-   $G(x_i) = sign(\sum \alpha_m G_m(x_i))$
    -   G(.) is weak classifier with an accuracy slightly better than random
    -   $\alpha$ is calcualted by the boosting algorithm
-   Algorithm
    -   Initial weights $w_i = 1/N$
    -   For m rounds
        -   Fit classifier $G_i(x)$ using $w_i$
        -   Compute Error $\bar{err} = {1 \over \sum w_i}\sum I\{y_i \ne G(x_i)\}$
        -   Compute $alpha_m = \log((1 - err_m) / err_m)$
        -   Compute new weights $w_i = w_i \exp \alpha_m I\{y_i \ne G(x_i)\}$

## Additive Models

-   Boosting fits a forward stagewise additive model
-   $f(x) = \sum \beta_m b(x, \gamma_m)$
-   $\min L(y_i, \beta_m b(x, \gamma_m))$
    -   Optimal values of beta and gama can be found iteratively
    -   $\min L(y_i, f_{m-1}(x) + \beta b(x, \gamma))$
-   L2 Loss Funciton
    -   $\min L(y_i, f_{m-1}(x) + \beta b(x, \gamma))$
    -   $\sum (y_i - f_{m-1}(x) - \beta b(x, \gamma))^2$
    -   $\sum (r_{im} - \beta b(x, \gamma))^2$
    -   Fit on residuals from previous rounds
    -   Robust Loss Functions for regression
        -   Huber Loss
        -   L2 penalty for large errors
        -   L1 penalty for small errors
-   Exponential Loss
    -   $L(y_i, f(x)) = \exp (-y f(x))$
    -   Equivalent to using deviance
    -   The optimal f that minimizes this loss is 1/2 log-odds
    -   Hence justified to use the sign of f(x) for prediction
    -   $\min \sum \exp (-y f_{m-1}(x) + -y \beta G(x))$
    -   $\min \sum w_i^m \exp (-y \beta G(x))$
    -   $\min \exp -\beta \sum_{\text{correct}} w_i^m + \exp \beta \sum_{\text{incorrect}} w_i^m$
    -   $\beta = {1 \over 2}\log({1- err \over err})$

## Gradient Boosting

-   Gradient Descent in function space
-   Minimize $L(f) = \sum L(y_i, f(x_i))$
-   $\arg \min L(\mathbf f); \; \mathbf f = \{f(x_1), f(x_2)....\}$
-   Additive Models $ \mathbf f = \sum_m h_m$
-   Steepest Descent $h_m = \rho_m g_m$
    -   $g_{im} = \delta L(y_i, f(x_i)) / \delta f(x_i)$
-   Line Search for optimal step size
    -   $\rho_m = \arg \min L(f_{m-1} - \rho g_m)$
-   Gradients
    -   L2 Loss: Residual
    -   L1 Loss: Sign of Residual
    -   Classification / Deviance: Error
    -   Huber 
# Classification

## Decision Boundary

-   Classificaiton approach is to learn a discriminant function $\delta_k(x)$ for each class
    -   Classify x to the class with largest discriminant value
-   The decision boundary is linear if
    -   $\delta_k(x)$ is linear
    -   Posterior prbability is linear $P(G=k|X=x)$
    -   Their monotonic transformation is linear
-   Linear Decision Boundary: $f_k(x) = \beta_{0k} + \beta_k x$
-   Decision Boundary between two classes (k, l) is the set of points where $f_k(x) = f_l(x)$
    -   $\{x : (\beta_{0k} - \beta_{0l}) + (\beta_k - \beta_l) x = 0\}$
    -   Affine set or a hyperplane
-   Example: Binary Logistic Regression
    -   $P({G=1 \over X=x}) = \frac{\exp(\beta x)}{1 + \exp(\beta x)}$
    -   $P({G=0\over X=x}) = \frac{1}{1 + \exp(\beta x)}$
    -   $\log({P(G=1 | X=x) \over p(G=0 | X=x)}) = x \beta$
    -   Log-odds transformation gives linear decision boundary
    -   Decsion boundary is the set of points $\{x| \beta x = 0\}$

## Linear Probability Model

-   Encode each of the k classes with an indicator function $Y_{N \times K}$
-   Fit a regression model to each of the classes simulatneously
    -   $\hat \beta = (X'X)^{-1}(X'Y)$
    -   $\hat Y = X \hat \beta$
-   Drawbacks
    -   Predictions can be outside range (0,1)
    -   Classes can be masked by others
        -   Large number of classes with small number of features
        -   Possible that one of the classes (say 2) gets dominated thoughout by the other classes (1,3)
        -   The model will never predict for class 2

## Linear and Quadratic Discriminant Analysis

-   Bayes theorem
    -   posterior $\propto$ prior x likelihood
    -   $P(G = k | X= x) = \frac{f_k(x) \times \pi_k}{\sum_k f_k(x) \times \pi_k}$
    -   $f_k(x)$ is the discriminant function
    -   $\pi_k$ is the prior estimate
-   Naive Bayes assumes each of the class densities are product of marginal densities.
    -   Inputs are conditionally independent of each class
-   LDA (and QDA) assumes the discriminant function to have MVN probability density function
-   LDA makes the assumption that the covariance martix (for MVN) is common for all the classes
-   Discrimination Function
    -   $f_k(x) = \frac{1}{(2\pi)^{p/2} \Sigma^{1/2}} \exp\{(X - \mu)^T \Sigma^{-1} (X - \mu)\}$
-   Decision Boundary
    -   $\log(\frac{P(G=k | X=x)}{P(G=l | X=x)}) = C + X^T \Sigma^{-1}(\mu_k - \mu_l)$
    -   Linear in X
    -   The constant terms can be grouped together because of common covariance matrix
-   Estimation
    -   $\pi_k = N_k / N$
    -   $\mu_k = \sum_{i \in K} x_i / N_k$
    -   $\Sigma = \sum_k \sum_{i \in K} (x_i - \mu_k)^T(x_i - \mu_k) / N_k$
-   QDA relaxes the assumtion of contant covariance matrix
    -   It assumes a class specific covariance matrix
    -   Discrimination function becomes quadratic in x
    -   The number of parameters to be estimated grows considerably
-   Regularization
    -   Compromise between LDA and QDA
    -   Shrink the individual covariances of QDA towards LDA
    -   $\alpha \Sigma_k + (1 - \alpha) \Sigma$
-   Computation
    -   Simplify the calculation by using eigen decomposition of the covariance matrix $\Sigma$

## Logistic Regression

-   Model posterior probabilities via separate functions while ensuring the output remains in the range \[0,1\]
    -   $P({G=1 \over X=x}) = \frac{\exp(\beta x)}{1 + \exp(\beta x)}$
    -   $P({G=0\over X=x}) = \frac{1}{1 + \exp(\beta x)}$
-   Estimation is done by maximizing conditional log-likelihood
    -   $LL(\beta) = \sum y_i(\log(p(x_i, \beta)) + (1 - y_i) (1 - \log(p(x_i, \beta))$
    -   $LL(\beta) = \sum y (x \beta) + \log(1 + \exp x \beta)$
    -   Normal Equation
        -   $\frac{\delta LL}{\delta \beta} = \sum x_i (y_i - p(x_i, \beta)) = 0$
-   Optimization
    -   Non-linear function of parameters
    -   Use Newton-Raphson method
    -   Seond Order Derivative or Hessian
    -   
        -   $\frac{\delta^2 LL}{\delta \beta^2} = \sum x_i x_i^T p(x_i, \beta) (1 - p(x_i, \beta))$
    -   The second order derivative is positive, hence it's a convex optimization problem
    -   IRLS (Iteratively Weighted Least Squares) algorithm
-   Goodness of Fit
    -   Deviance = $-2 (\log L_M - \log L_S)$
    -   L(M): LL of Current Model
    -   L(S) LL of Saturated Model
        -   Model that perfectly fits the data, Constant for a given dataset
    -   Compare two different models by looking at change in deivance
-   Regularization
    -   Lasso penalties can be added to the objective function
    -   Intercept term isn't penalized

## Comparison between LDA and Logistic Regression

-   Both Logistic and LDA return linear decision boundaries
-   Difference lies in the way coefficients are estimated
-   Logistic Regression makes less stringent assumptions
    -   LR maximizes conditional log-likelihood
    -   LDA maximizes full log-likelihood (i.e. joint desnity)
-   LDA makes more restrictive assumptions about the distributions
    -   Efficiency is estimation
    -   Less robust to outliers

## Percepton Learning Algorithm

-   Minimize the distance of missclassified points to the separating hyperplane
-   $D(\beta) = - \sum y (x^T \beta)$
-   Use SGD to estimate the parameters
-   When the data is separable, there are many solutions that exist. The final convergence depends on the initialization.
-   When data isn't separable, there is no convergence.

## Maximum Margin Classifiers

-   Maximize the distance of of points from either class to the hyperplane.
-   $L = \max_{\beta, ||\beta|| = 1} M \, \, \text{subject to} \, y_i \times x_i \beta >= M \, \forall \, i \in N$
-   The final parameters can be arbitrarily scaled.
-   $L = \max {1 \over 2}||\beta||^2 \, \, \text{subject to} \, y_i \times x_i \beta >= 1 \, \forall \, i \in N$
-   Lagrangian Multiplier
-   $L = \max {1 \over 2}||\beta||^2 - \sum \alpha_i (y_i \times x_i \beta) - 1)$
-   Taking derivative wrt to $\beta$
    -   $\beta = \sum \alpha_i y_i x_i$
    -   Parameter is a linear combination of points where the constraints are active $\alpha_i > 0$ 
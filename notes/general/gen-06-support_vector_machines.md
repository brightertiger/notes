# Support Vector Machines

## Linear SVM

-   Classification setting
-   Find the maximum-margin hyperplane that can separate the data
-   Best hyperplane is the one that maximizes the margin
    -   Margin the distance of the hyperplane to closest data points from both classes
    -   Hyperplane: $H : wx +b = 0$
-   Distance of a point (x) to a hyperplane (h):
    -   $d = \frac{|Wx + b|}{||W||}$
-   Margin is defined by the point closest to the hyperplane
    -   $\gamma(W,b) = \min_{x \in D} \frac{|Wx + b|}{||W||^2}$
    -   Margin is scale invariant
-   SVM wants to maximize this margin
    -   For margin to be maximized, hyperplane must lie right in the middle of the two classes
    -   Otherwise it can be moved towards data points of the class that is further away and be further increased
-   Mathematics
    -   Binary Classification
        -   $y_i \in \{+1,-1\}$
    -   Need to find a separating hyperplane such that
        -   $(Wx_i + b) > 0 \; \forall \; y_i = +1$
        -   $(Wx_i + b) < 0 \; \forall \; y_i = -1$
        -   $y_i(Wx_i + b) > 0$
    -   SVM posits that the best hyperplane is the one that maximizes the margin
        -   Margin acts as buffer which can lead to better generalization
    -   Objective
        -   $\max_{W,b} \gamma(W,b) \; \text{subject to} \; y_i(Wx_i + b) > 0$
        -   $\max_{W,b} \min_{x \in D} \frac{|Wx + b|}{||W||^2} \; \text{subject to} \; y_i(Wx_i + b) > 0$
        -   A max-min optimization problem
    -   Simplification
        -   The best possible hyperplace is scale invariant
        -   Add a constraint such that $|Wx +b| = 1$
    -   Updated objective
        -   $\max \frac{1}{||W||^2} \; \text{subject to} \; y_i(Wx_i + b) \ge 0 \; ; |Wx +b| = 1$
        -   $\min ||W||^2 \; \text{subject to} \; y_i(Wx_i + b) \ge 0 \; ; |Wx +b| = 1$
    -   Combining the contraints
        -   $y_i(Wx_i + b) \ge 0\; ; |Wx +b| = 1 \implies y_i(Wx_i + b) \ge 1$
        -   Holds true because the objective is trying to minimize W
    -   Final objective
        -   $\min ||W||^2 \; \text{subject to} \; y_i(Wx_i + b) \ge 1$\
    -   Quadratic optimization problem
        -   Can be solved quickly unlike regression which involves inverting a large matrix
        -   Gives a unique solution unlike perceptron
    -   At the optimal solution, some training points will lie of the margin
        -   $y_i(Wx_i + b) = 1$
        -   These points are called support vectors
-   Soft Constraints
    -   What if the optimization problem is infeasible?
        -   No solution exists
    -   Add relaxations i.e. allow for some misclassification
        -   Original: $y_i(Wx_i + b) \ge 1$
        -   Relaxed: $y_i(Wx_i + b) \ge 1 - \xi_i \; ; \xi_i > 0$
        -   $\xi_i = \begin{cases} 1 - y_i(Wx_i + b), & \text{if } y_i(Wx_i + b) < 1\\0, & \text{otherwise} \end{cases}$
        -   Hinge Loss $\xi_i = \max (1 - y_i(Wx_i + b), 0)$
    -   Objective: $\min ||W||^2 + C \sum_i \max (1 - y_i(Wx_i + b), 0)$
        -   C is the regularization parameter that calculates trade-off
        -   High value of C allows for less torelance on errors
-   Duality
    -   Primal problem is hard to solve
    -   Convert the problem to a Dual, which is easier to solve and also provides near-optimal solution to primal
    -   The gap is the optimality that arises in this process is the duality gap
    -   Lagrangian multipliers determine if strong suality exists
    -   Convert the above soft-margin SVM to dual via Lagrangian multipliers
    -   $\sum \alpha_i + \sum\sum \alpha_i \alpha_j y_i y_j x_i^T x_j$
    -   $\alpha$ is the Lagrangian multiplier
-   Kernelization
    -   Say the points are not separable in lower dimension
        -   Transform them via kernels to project them to a higher dimension
        -   The points may be separable the higher dimension
        -   Non-linear feature transformation
        -   Solve non-linear problems via Linear SVM
    -   Polynomial Kernel
        -   $K(x_i, x_j) = (x_i^T x_j + c)^d$
        -   The d regers to the degree of the polynomial
        -   Example: 2 points in 1-D (a and b) transformerd via second order polynomial kernel
            -   $K(a,b) = (ab + 1)^2 = 2ab+ a^2b^2 + 1 = (\sqrt{2a}, a, 1)(\sqrt{2b}, b, 1)$
        -   Calculates similarity between points in higher dimension
    -   RBF Kernel
        -   $K(x_i, x_j) = \exp \{\gamma |x_i - x_j|^2\}$
        -   The larger the distance between two observations, the less is the similarity
        -   Radial Kernel determines how much influence each observation has on classifying new data points\
        -   Transforms points to an infinite dimension space
            -   Tayloy Expansion of exponential term shows how RBF is a polynomial function with inifnite dimensions
        -   2 points in 1-D (a and b) transformerd via RBF
            -   $K(a,b) = (1, \sqrt{\frac{1}{1!}}a, \sqrt{\frac{1}{2!}}a^2...)(1, \sqrt{\frac{1}{1!}}b, \sqrt{\frac{1}{2!}}b^2...)$
    -   Kernel Trick
        -   Transforming the original dataset via Kernels and training SVM is expensive
        -   Convert Dot-products of support vectors to dot-products of mapping functions
        -   $x_i^T x_j \implies \phi(x_i)^T \phi(x_j)$
        -   Kernels are chosen in a way that this is feasible
-   SVM For Regression
    -   Margins should cover all data points (Hard) or most data points (Soft)
    -   The boundary now lies in the middle of the margins
        -   The regression model to estimate the target values
    -   The objective is to minimize the the distance of the points to the boundary
    -   Hard SVM is sensitive to outliers 

## Kernel Selection

-   Choosing the right kernel:
    -   Linear kernel: $K(x_i, x_j) = x_i^T x_j$
        -   Efficient for high-dimensional data
        -   Works well when number of features exceeds number of samples
        -   Simplest kernel with fewest hyperparameters
    -   Polynomial kernel: $K(x_i, x_j) = (x_i^T x_j + c)^d$
        -   Good for normalized training data
        -   Degree d controls flexibility (higher d = more complex decision boundary)
        -   Can capture feature interactions
    -   RBF/Gaussian kernel: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
        -   Most commonly used non-linear kernel
        -   Works well for most datasets
        -   Gamma parameter controls influence radius (higher gamma = more complex boundary)
    -   Sigmoid kernel: $K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$
        -   Similar to neural networks (hyperbolic tangent activation)
        -   Less commonly used in practice

-   Cross-validation should be used to select the optimal kernel and hyperparameters

## SVM Hyperparameter Tuning

-   C parameter (regularization strength):
    -   Controls trade-off between maximizing margin and minimizing training error
    -   Smaller C: Wider margin, more regularization, potential underfitting
    -   Larger C: Narrower margin, less regularization, potential overfitting
-   Gamma parameter (for RBF kernel):
    -   Controls influence radius of support vectors
    -   Smaller gamma: Larger radius, smoother decision boundary
    -   Larger gamma: Smaller radius, more complex decision boundary
-   Practical suggestions:
    -   Start with RBF kernel, grid search over C and gamma
    -   Try logarithmic scale for both C and gamma (e.g., 0.001, 0.01, 0.1, 1, 10, 100)
    -   Use cross-validation to evaluate performance 
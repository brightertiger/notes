# Kernel Methods

## Kernel Density Estimation

-   A random sample $[x_0, x_1, ... x_n]$ is drawn from probability distribution $f_X(x)$
-   Parzen Estimate of $\hat f_X(x)$
    -   $\hat f_X(x_0) = {1 \over N \lambda} \,\, \# x_i \in N(x_0)$
    -   $\lambda$ is width of the neighbourhood
    -   The esitmate is bumpy
-   Gaussian Kernel of width $\lambda$ can be a choice of Kernel
    -   $\hat f_X(x_0) = {1 \over N \lambda} K_{\lambda}(x_i, x_0)$
    -   $K_{\lambda}(x_i, x_0) = \phi(\|x_i - x_0\|) / \lambda$
    -   Weight of point $x_i$ descreases as distance from $x_0$ increases
-   New estimate for density is
    -   $\hat f_X(x_0) = {1 \over N } \phi_{\lambda}(x_ - x_0)$
    -   Convolution of Sample empirical distribution with Gaussian Kernel

## Kernel Desnity Classification

-   Bayes' Theorem
-   $P(G=j | X=x_0) \propto \hat \pi_j \hat f_j(x_0)$
-   $\hat \pi_j$ is the sample proportion of the class j
-   $\hat f_j(x_0)$ is the Kernel density estimate for class j
-   Learning separate class densities may be misleading
    -   Dense vs Non-dense regions in feature space
    -   Density estimates are critical only near the decision boundary

## Naive Bayes Classifier

-   Applicable when dimension of feature space is high
-   Assumption: For a given class, the features are independent
    -   $f_j(x) = \prod_p f_{jp}(x_p)$
    -   Rarely holds true in real world dataset
-   $\log \frac{P(G=i|X=X)}{P(G=j|X=X)} = \log \frac{\pi_i \prod f_{ip}(x_p)}{\pi_j \prod f_{jp}(x_p)}$
    -   $\log \frac{\pi_i}{\pi_j} + \sum \log \frac{f_{ip}(x_p)}{f_{jp}(x_p)}$

## Radial Basis Functions

-   Basis Functions $f(x) = \sum \beta h(x)$
-   Transform lower dimension features to high dimensions
-   Data which is not linearly separable in lower dimension may become linearly separable in higher dimensions
-   RBF treats gaussian kernel functions as basis functions
    -   Taylor series expansion of $\exp(x)$
    -   Polynomial basis function with infinite dimensions
-   $f(x) = \sum_j K_{\lambda_j}(\xi_j, x) \beta_j$
-   $f(x) = \sum_j D({\|x_i - \xi_j\| \over \lambda _j}) \beta_j$
    -   D is the standard normal gaussian density function
-   For least square regression, SSE can be optimized wrt to $\beta, \xi, \lambda$
    -   Non-Linear Optimization
    -   Use greedy approaches like SGD
-   Simplify the calculations by assuming $\xi, \lambda$ to be hyperparameters
    -   Use unsupervised learning to estimate them
    -   Assuming constant variance simplifies calculations
        -   It can create "holes" where none of the kernels have high density estimate

## Mixture Models

-   Extension of RBF
-   $f(x) = \sum_j \alpha_j \phi(x, \mu_j, \Sigma_j)$
-   $\sum \alpha_j = 1$, are the mixing proporitons
-   Gaussian mixture models use Gaussian kernel in place of $\phi$
-   Parameters are fit using Maximum Likelihood
-   If the covariance martix is restricted to a diagonal matrix $\Sigma = \sigma^2 I$, then it reduces to radial basis expansion
-   Classification can be done via Bayes Theorem
    -   Separate density estimation for each class
    -   Probability is $\propto \hat \pi_i f_j(x)$ 
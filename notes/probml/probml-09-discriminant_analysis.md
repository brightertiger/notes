# Discriminant Analysis

-   Generative Models specify a way to generate features for each class
    -   $p(y = c |x, \theta) \propto p(x | y = c, \theta) \times p(y = c)$
    -   $p(x | y, \theta)$ is the class conditional density
    -   $p(y)$ is the prior over class labels
-   Discriminative Models estimate the posterior class probability
    -   $p(y | x, \theta)$

-   Gaussian Discriminant Analysis
    -   Class Conditional Densities are multivariate Gaussians
    -   $p(x | y=c, \theta) = N(\mu_c, \Sigma_c)$
    -   $\log p(y = c | x, \theta )$ will be quadratic in $\mu_c$, $\Sigma_c$ (QDA)
    -   If the covariance matrices are shared across class labels, the decision boundary will become linear in $\mu_c$ (LDA)
    -   LDA can be refactored to be similar to logistic regression
    -   Models are fitted via MLE.
        -   $\Sigma_c$ estimates often lead to overfitting
        -   Tied covariances i.e. LDA solve this problem
        -   MAP estimation can introduce some regularization
    -   Class assignment is based on nearest centroid based on the estimates of $\mu_c$

-   Naive Bayes Classifiers
    -   Work on Naive Bayes Assumption
    -   Features are conditionally independent given the class label
        -   $p(\mathbf x | y = c) = \prod p(x_d | y = c)$
    -   The assumption is naive since it will rarely hold true
    -   $p(y = c | x, \theta) \propto \pi(y = c) \prod p(x_d | y = c, \theta_{dc})$
    -   Model has very few parameters and is easy to estimate
    -   The distribution of $p(x_d | y = c)$ is
        -   Bernoulli for binary
        -   Categorical for categorical
        -   Gaussian for continuous

-   Generative Classifiers are better at handing missing data or unlabeled data
-   Discriminative Models give more robust estimates for posterior probabilities 
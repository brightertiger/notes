# Probability (Advanced Topics)

- Covariance measures the degree of linear association
    - $\text{COV}[X,Y] = E[(X - E[X])(Y - E[Y])]$
- Covariance is unscaled measure. Correlation scales covariance between -1, 1.
    - $\rho = \frac{\text{COV}[X,Y]}{\sqrt{V(X)} \sqrt{V(Y)}}$
- Independent variables are uncorrelated. But, vice-versa is not true.
- Correlation doesn't imply causation. Can be spurious.

- Simpson's Paradox
    - Statistical Trend that appears in groups of data can disappear or reverse when the groups are combined.

- Mixture Models
    - Convex combination of simple distributions
    - $p(y|\theta) = \sum \pi_k p_k(y)$
    - First sample a component and then sample points from the component
    - GMM: $p(y) = \sum_K \pi_k \mathcal N(y | \mu_k, \sigma_k)$
    - GMMs can be used for unsupervised soft clustering.
    - K Means clustering is a special case of GMMs
        - Uniform priors over components
        - Spherical Gaussians with identity matrix variance
        
- Markov Chains
    - Chain Rule of probability
    - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x1, x2)$
    - First-order Markov Chain: Future only depends on the current state.
    - y(t+1:T) is independent of y(1:t)
    - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x2)$
    - The p(y | y-1) function gives the state transition matrix
    - Relaxing these conditions gives bigram and trigram models. 
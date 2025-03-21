# Information Theory

- Entropy is a measure of uncertainty or lack of predictability associated with a distribution
- If entropy is high, it's difficult to predict the value of observation
- $H(X) = - \sum p(x) \log p(x) = - E[\log p(X)]$
- Uniform distribution has maximum entropy
- Dirac Delta distribution has minimum entropy

- Cross Entropy
    - $H(p,q) = -\sum p(x) \log q(x)$

- Joint Entropy
    - $H(X,Y) = -\sum p(x,y) \log p(x,y)$

- Conditional Entropy
    - $H(Y | X) = \sum_x p(x) H(Y | X=x)$
    - $H(Y|X) = H(X,Y) - H(X)$
    - Reduction in joint uncertainty (X,Y) given we observed X

- Perplexity
    - Exponentiated cross-entropy
    - Geometric mean of inverse probabilities
    - $\text{perplexity}(p) = 2^{H(p)} = \sqrt[N]{\prod \frac{1}{p(x_i)}}$
    - Used to evaluate the quality of generative language models
    - Weighted average of branching factor
        - Number of possible words that can follow a given word
        - Given vocab size is K
        - If some words are more frequent, perplexity is lower than K

- KL Divergence
    - Relative Entropy
    - Distance between two distribution
    - $KL(p||q) = H(p,q) - H(p)$
    - Extra bits needed when compressing data generated from p using q
    - Suppose objective is to minimize KL divergence
    - Empirical distribution puts probability mass on training data and zero mass every where else
        - $p = {1 \over N} \sum {\delta (x - x_n)}$
    - This reduces KL divergence to cross entropy or negative log likelihood.
        - $KL(p||q) = {1 \over N} \sum \log q(x_n)$
    - Data augmentation perturbs data samples to reflect natural variations. This spreads the probability mass over larger space. Prevents overfitting.

- Mutual Information
    - $I(X,Y) = KL(p(x,y) || p(x)p(y))$
    - KL Divergence between joint and factored marginal distribution
    - $I(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$
    - Reduction in uncertainty about X after observing Y
    - Generalized correlation coefficient that can capture non-linear trends.
    - Can be normalized to reduce scale effect
    - Data Processing Inequality: Transformation cannot increase the amount of information

- Fano's Inequality
    - Feature selection via high mutual information
    - Bounds probability of misclassification in terms of mutual information between features 
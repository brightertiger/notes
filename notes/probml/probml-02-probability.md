# Probability

- Frequentist View: Probability as the long-run relative frequency of an event in repeated experiments.
- Bayesian View: Probability as a quantification of subjective uncertainty or degree of belief.
    - Model Uncertainty: Epistemic uncertainty arising from incomplete knowledge of the underlying process
    - Data Uncertainty: Aleatoric uncertainty arising from inherent randomness in the system
    - Data uncertainty is irreducible and persists even with more data
- Event: Some state of the world (A) that either holds or doesn't hold.
    - $0 \le P(A) \le 1$ (probability is non-negative and bounded)
    - $P(A) + P(\bar A) = 1$ (law of total probability)
- Joint Probability: Probability that two events occur simultaneously
    - $P(A,B)$ is the probability that both A and B occur
    - If A and B are independent: $P(A,B) = P(A)P(B)$
    - $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion principle)
- Conditional Probability: Probability of event B occurring given that event A has already occurred
    - $P(B | A) = \frac{P(A \cap B)}{P(A)}$ where $P(A) > 0$
    - Allows updating beliefs based on new evidence
- Random Variables: Functions that map outcomes from a sample space to real numbers, allowing mathematical manipulation
    - Discrete random variables: Take values from a countable set (e.g., number of customers)
    - Continuous random variables: Take values from an uncountable set (e.g., precise weight)
- Probability Mass Function (PMF): Gives probability for each possible value of a discrete random variable
    - $0 \le p(x) \le 1$ for all x
    - $\sum_x p(x) = 1$ (probabilities sum to 1)
- Cumulative Distribution Function (CDF): Gives probability that a random variable is less than or equal to a value
    - $F_X(x) = P(X \le x)$
    - $P(a \le X \le b) = F_X(b) - F_X(a)$
    - Monotonically non-decreasing: $F_X(a) \le F_X(b)$ whenever $a \le b$
    - $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$
- Probability Density Function is the derivative of CDF
- Inverse CDF or Quantile Function
    - $P^{-1}(0.5)$ is the median
    - $P^{-1}(0.25); P^{-1}(0.75)$ are lower and upper quartiles
- Marginal Distribution of an random variable
    - $p(X=x) = \sum_y p(X=x, Y=y)$
- Conditional Distribution of a Random Variable
    - $p(Y=y | X=x) = {p(Y=y, X=x) \over p(X=x)}$
- Product Rule
    - $p(x,y) = p(y|x)p(x) = p(x|y) p(y)$
- Chain Rule
    - $p(x1,x2,x3) = p(x1) p(x2 | x1) p(x3 | x1, x2)$
- X and Y are independent
    - $X \perp Y \Rightarrow p(X,Y) = p(X) p(Y)$
- X and Y are conditionally independent of Z
    - $X \perp Y | Z \Rightarrow p(X,Y | Z) = p(X|Z) p(Y | Z)$

- Mean or Expected Value
    - First moment around origin
    - $\mathbf E(X) = \sum xp(x) \; \text{OR} \; \int_x xp(x) dx$
    - Linearity of Expectation: $\mathbf E(aX + b) = a \mathbf E(X) + b$

- Variance of a distribution
    - Second moment around mean
    - $\mathbf E(X-\mu)^2 = \sigma^2$ 
    - $\text{Var}(aX + b) = a^2 Var(X)$

- Mode of a distribution
    - Value with highest probability mass or probability density

- Law of Total / Iterated Expectation
    - $E(X) = E(E(X|Y))$

- Law of Total Variance
    - $V(X) = E(V(X | Y)) + V(E(X | Y))$

- Bayes' Rule
  - Compute probability distribution over some unknown quantity H given observed data Y
  - $P(H | Y) = {P(Y |H) P(H) \over P(Y)}$
  - Follows from product rule
  - p(H) is the prior distribution
  - p(Y | H) is the observation distribution
  - p(Y=y | H=h) is the likelihood
  - Bayesian Inference: $\text{posterior} \propto \text{prior} \times \text{likelihood}$
    
- Bernoulli and Binomial Distribution
    - Describes a binary outcome
    - $Y \sim Ber(\theta)$
    - $p(Y=y) = \theta^y (1 - \theta)^{1-y}$ for $y \in \{0, 1\}$
    - Binomial distribution is N repetitions of Bernoulli trials
    - $Bin(k | N,\theta) = {N \choose k} \theta^k (1 - \theta)^{N-k}$ where k is the number of successes

- Logistic Distribution
    - If we model a binary outcome using ML model, the range of f(X) is [0,1]
    - To avoid this constraint, use logistic function: $\sigma(a) = {1 \over 1 + e^{-a}}$
    - The quantity a is log-odds: $\log(p / (1-p))$
    - Logistic function maps log-odds to probability
    - $p(y=1|x, \theta) = \sigma(f(x, \theta))$
    - $p(y=0|x, \theta) = \sigma( - f(x, \theta))$
    - Binary Logistic Regression: $p(y|x, \theta) = \sigma(wX +b)$
    - Decision boundary: $p(y|x, \theta) = 0.5$
    - As we move away from decision boundary, model becomes more confident about the label

- Categorical Distribution
    - Generalizes Bernoulli to more than two classes
    - $\text{Cat}(y | \theta) = \prod \theta_c ^ {I(y=C)} \Rightarrow p(y = c | \theta) = \theta_c$
    - Categorical distribution is a special case of multinomial distribution. It drops the multinomial coefficient. 
    - The categorical distribution needs to satisfy
        - $0 \le f(X, \theta) \le 1$
        - $\sum f(X, \theta) = 1$
    - To avoid these constraints, its common to pass the raw logit values to a softmax function
        - ${e^x_1 \over \sum e^x_i} , {e^x_2 \over \sum e^x_i}....$
    - Softmax function is "soft-argmax"
        - Divide the raw logits by a constant T (temperature)
        - If T → 0 all the mass is concentrated at the most probable state, winner takes all
    - If we use categorical distribution for binary case, the model is over-parameterized.
        - $p(y = 0 | x) = {e^{a_0} \over e^{a_0} + e^{a_1}} = \sigma(a_0 - a_1)$

- Log-Sum-Exp Trick
    - If the raw logit values grow large, the denominator of softmax can enounter numerical overflow.
    - To avoid this:
        - $\log \sum \exp(a_c) = m + \log \sum \exp(a_c - m)$
        - if m is arg max over a, then we wont encounter overflow.
    - LSE trick is used in stable cross-entropy calculation by transforming the sigmoid function to LSE(0,-a).

- Gaussian Distribution
    - CDF of Gaussian is defined as
        - $\Phi(y; \mu, \sigma^2) = {1 \over 2} [ 1 + \text{erf}({z \over \sqrt(2)})]$
        - erf is the error function
    - The inverse of the CDF is called the probit function.
    - The derivative of the CDF gives the pdf of normal distribution
    - Mean, Median and Mode of gaussian is $\mu$
    - Variance of Gaussian is $\sigma^2$
    - Linear Regression uses conditional gaussian distribution
        - $p(y | x, \theta) = \mathcal N(y | f_\mu(x, \theta); f_\sigma(x, \theta))$
        - if variance does not depend on x, the model is homoscedastic. 
    - Gaussian Distribution is widely used because:
        - parameters are easy to interpret
        - makes least number of assumption, has maximum entropy
        - central limit theorem: sum of independent random variables are approximately gaussian
    - Dirac Delta function puts all the mass at the mean. As variance approaches 0, gaussian turns into dirac delta.
    - Gaussian distribution is sensitive to outliers. A robust alternative is t-distribution.
        - PDF decays as polynomial function of distance from mean.
        - It has heavy tails i.e. more mass
        - Mean and mode is same as gaussian.
        - Variance is $\nu \sigma^2 \over \nu -2$
        - As degrees of freedom increase, the distribution approaches gaussian.

- Exponential distribution describes times between events in Poisson process.
- Chi-Squared Distribution is sum-squares of Gaussian Random Variables.

- Transformations
  - Assume we have a deterministic mapping y = f(x)
  - In discrete case, we can derive the PMF of y by summing over all x
  - In continuous case:
      - $P_y(y) = P(Y \le y) = P(f(X) \le y) = P(X \le f^{-1}(y)) = P_x(f^{-1}(y))$
      - Taking derivatives of the equation above gives the result.
      - $p_y(y) = p_x(f^{-1}(y)) \cdot |{d f^{-1}(y) \over dy}|$ (change of variables formula)
      - In multivariate case, the derivative is replaced by absolute value of Jacobian determinant.

- Convolution Theorem
    - y = x1 + x2
    - $P(y \le y^*) = \int_{-\infty}^{\infty}p_{x_1}(x_1) dx_1 \int_{-\infty}^{y^* - x1}p_{x_2}(x_2)dx_2$
    - Differentiating under integral sign gives the convolution operator
    - $p(y) = \int p_1(x_1) p_2(y - x_1) dx_1$
    - In case x1 and x2 are gaussian, the resulting pdf from convolution operator is also gaussian. → sum of gaussians results in gaussian (reproducibility) 

- Central Limit Theorem
    - Suppose there are N random variables that are independently identically distributed with mean μ and variance σ².
    - As N increases, the distribution of the sample mean $\bar{X} = \frac{1}{N}\sum X_i$ approaches Gaussian:
        - $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/N)$
    - Equivalently, the standardized sum $\frac{\bar{X} - \mu}{\sigma/\sqrt{N}} \xrightarrow{d} \mathcal{N}(0, 1)$

- Monte-Carlo Approximation
    - It's often difficult ti compute the pdf of transformation y = f(x).
    - Alternative:
        - Draw a large number of samples from x
        - Use the samples to approximate y 
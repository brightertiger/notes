# Classification

Classification is one of the most common machine learning tasks. Unlike regression (where we predict a continuous number), in classification we predict which *category* or *class* an observation belongs to. Examples include: spam vs. not spam, disease vs. healthy, or recognizing handwritten digits (0-9).

## The Big Picture

In classification, we want to:
1. **Learn decision boundaries** that separate different classes
2. **Estimate probabilities** of class membership
3. **Make predictions** for new observations

The methods in this chapter produce *linear* decision boundaries — the simplest and most interpretable kind.

---

## Decision Boundaries

### What is a Decision Boundary?

A decision boundary is the dividing line (or surface, in higher dimensions) between regions assigned to different classes. When a new observation falls on one side, we predict one class; on the other side, we predict the other class.

### Linear Decision Boundaries

The boundary is linear if it's described by a linear equation in the features:

$$\{x : w_0 + w_1 x_1 + w_2 x_2 + ... + w_p x_p = 0\}$$

In 2D, this is a straight line. In 3D, it's a plane. In higher dimensions, it's a **hyperplane**.

### How Do We Get Linear Boundaries?

The decision boundary is linear if any of these conditions hold:
- The discriminant function $\delta_k(x)$ is linear in x
- The posterior probability $P(G=k|X=x)$ is linear in x
- Some monotonic transformation of the above is linear

### Discriminant Functions

We learn a **discriminant function** $\delta_k(x)$ for each class k. To classify a new point x:
- Compute $\delta_k(x)$ for all classes
- Assign x to the class with the highest discriminant value

For linear discriminant functions:

$$\delta_k(x) = \beta_{0k} + \beta_k^T x$$

The decision boundary between classes k and l is where $\delta_k(x) = \delta_l(x)$:

$$\{x : (\beta_{0k} - \beta_{0l}) + (\beta_k - \beta_l)^T x = 0\}$$

This is clearly linear in x — an affine set (hyperplane not necessarily through the origin).

---

## Example: Logistic Regression Boundary

Binary logistic regression models probabilities as:

$$P(G=1|X=x) = \frac{\exp(\beta^T x)}{1 + \exp(\beta^T x)} = \frac{1}{1 + \exp(-\beta^T x)}$$

$$P(G=0|X=x) = \frac{1}{1 + \exp(\beta^T x)}$$

The **log-odds** (also called logit) is:

$$\log\left(\frac{P(G=1|x)}{P(G=0|x)}\right) = \beta^T x$$

This is linear in x! So the decision boundary — where both classes are equally likely — is:

$$\{x : \beta^T x = 0\}$$

---

## Linear Probability Model (and Why It Fails)

### The Approach

One simple idea: encode classes as numbers (0/1 for binary, or indicator matrix Y for multiclass) and just run linear regression!

$$\hat{\beta} = (X^TX)^{-1}X^TY$$
$$\hat{Y} = X\hat{\beta}$$

### Problems with This Approach

1. **Predictions outside [0,1]**: Linear regression can predict negative probabilities or probabilities greater than 1 — nonsense for probabilities!

2. **Class masking**: With multiple classes and few features, one class can be "dominated" everywhere. Imagine three classes where class 2 always has lower predicted values than classes 1 or 3 — the model will never predict class 2!

**Takeaway**: Use methods designed for classification, not regression hacks.

---

## Linear Discriminant Analysis (LDA)

LDA is one of the oldest and most elegant classification methods. It takes a generative approach: model how the data is *generated* for each class, then use Bayes' theorem to classify.

### The Generative Model

Assume that within each class k, the data follows a multivariate normal distribution:

$$X | G=k \sim N(\mu_k, \Sigma)$$

**Key assumption for LDA**: All classes share the same covariance matrix Σ (but have different means μ_k).

### Using Bayes' Theorem

Once we model how data is generated, we can compute:

$$P(G=k | X=x) = \frac{f_k(x) \times \pi_k}{\sum_l f_l(x) \times \pi_l}$$

Where:
- $\pi_k$ = prior probability of class k (how common is this class?)
- $f_k(x)$ = class-conditional density (how likely is x given class k?)

### The Discriminant Function

For a multivariate normal:

$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\left\{-\frac{1}{2}(x - \mu_k)^T\Sigma^{-1}(x - \mu_k)\right\}$$

Taking the log and simplifying (using the common Σ assumption), we get the **linear discriminant function**:

$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k$$

This is linear in x, hence "Linear" Discriminant Analysis.

### Decision Boundary

The log-odds between classes k and l is:

$$\log\frac{P(G=k|x)}{P(G=l|x)} = C + x^T\Sigma^{-1}(\mu_k - \mu_l)$$

Linear in x! The constant terms combine nicely because Σ is shared across classes.

### Estimation in Practice

We estimate the parameters from training data:
- **Prior**: $\hat{\pi}_k = N_k / N$ (proportion of each class)
- **Mean**: $\hat{\mu}_k = \frac{1}{N_k}\sum_{i \in \text{class } k} x_i$ (class centroid)
- **Covariance**: $\hat{\Sigma} = \frac{1}{N}\sum_k\sum_{i \in \text{class } k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$ (pooled covariance)

---

## Quadratic Discriminant Analysis (QDA)

### Relaxing the Equal Covariance Assumption

What if each class has its own covariance structure? QDA allows:

$$X | G=k \sim N(\mu_k, \Sigma_k)$$

### Quadratic Discriminant Function

Now the discriminant becomes:

$$\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) + \log\pi_k$$

This is **quadratic** in x — the decision boundaries are curved (ellipses, parabolas, hyperbolas).

### Trade-offs: LDA vs QDA

| Aspect | LDA | QDA |
|--------|-----|-----|
| Boundaries | Linear | Quadratic (curved) |
| Parameters | ~Kp means + p(p+1)/2 covariance | ~Kp means + Kp(p+1)/2 covariances |
| Flexibility | Less flexible | More flexible |
| Variance | Lower (fewer parameters) | Higher (more parameters) |
| Best when | Classes have similar spread | Classes have different shapes/orientations |

### Regularized Discriminant Analysis

A compromise between LDA and QDA:

$$\hat{\Sigma}_k(\alpha) = \alpha\Sigma_k + (1-\alpha)\Sigma$$

- α = 0: LDA (shared covariance)
- α = 1: QDA (class-specific covariance)
- 0 < α < 1: Smooth interpolation

Choose α via cross-validation.

---

## Naive Bayes Classifier

### The Independence Assumption

Naive Bayes makes a strong (and usually wrong!) assumption: given the class, all features are **conditionally independent**:

$$f_k(x) = \prod_{j=1}^p f_{kj}(x_j)$$

### Why "Naive"?

This assumption rarely holds in practice. If predicting whether an email is spam, the presence of "free" and "money" are probably correlated. But Naive Bayes ignores this.

### Why Does It Still Work?

Despite the wrong assumption, Naive Bayes often works well because:
1. We only need to get the *ranking* of classes right, not exact probabilities
2. The errors from the independence assumption may cancel out
3. It's extremely fast and scales to high dimensions

### The Log-Odds

$$\log\frac{P(G=k|x)}{P(G=l|x)} = \log\frac{\pi_k}{\pi_l} + \sum_{j=1}^p\log\frac{f_{kj}(x_j)}{f_{lj}(x_j)}$$

Each feature contributes independently to the log-odds — simple and interpretable!

---

## Logistic Regression

Logistic regression is the workhorse of classification. Unlike LDA (which models P(X|G)), logistic regression directly models P(G|X).

### The Model

For binary classification:

$$P(G=1|X=x) = \frac{\exp(\beta^T x)}{1 + \exp(\beta^T x)} = \sigma(\beta^T x)$$

$$P(G=0|X=x) = \frac{1}{1 + \exp(\beta^T x)} = 1 - \sigma(\beta^T x)$$

Where σ(·) is the **sigmoid function** — it squashes any real number into (0,1).

### Why the Sigmoid?

The sigmoid ensures probabilities are always between 0 and 1, and the log-odds is linear:

$$\log\frac{P(G=1|x)}{P(G=0|x)} = \beta^T x$$

### Maximum Likelihood Estimation

We find β by maximizing the probability of the observed labels. The log-likelihood is:

$$\ell(\beta) = \sum_{i=1}^N \left[y_i \log p(x_i, \beta) + (1-y_i)\log(1-p(x_i, \beta))\right]$$

Or equivalently:

$$\ell(\beta) = \sum_{i=1}^N \left[y_i(\beta^T x_i) - \log(1 + \exp(\beta^T x_i))\right]$$

### Finding the Maximum

Taking the derivative (the **score function**):

$$\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^N x_i(y_i - p(x_i, \beta)) = X^T(y - p)$$

Setting this to zero: the predicted probabilities should "balance" with the actual labels.

### Optimization: Newton-Raphson / IRLS

The log-likelihood is non-linear in β — no closed-form solution. We use iterative methods.

The **Hessian** (second derivative) is:

$$\frac{\partial^2 \ell}{\partial \beta \partial \beta^T} = -\sum_{i=1}^N x_i x_i^T p(x_i, \beta)(1-p(x_i, \beta)) = -X^T W X$$

Where W is diagonal with $W_{ii} = p_i(1-p_i)$.

**Good news**: The Hessian is negative definite, so the log-likelihood is **concave** — there's a unique global maximum!

The algorithm is called **Iteratively Reweighted Least Squares (IRLS)** because each iteration looks like a weighted least squares problem.

### Measuring Goodness of Fit: Deviance

**Deviance** measures how far our model is from a "perfect" model:

$$\text{Deviance} = -2(\log L_M - \log L_S)$$

Where:
- $L_M$ = likelihood of our model
- $L_S$ = likelihood of the saturated model (perfect fit)

To compare models, look at the change in deviance (larger drops = better improvement).

### Regularization

Just like in regression, we can add penalties:
- **L2 penalty** (Ridge): Shrinks coefficients, helps with correlated predictors
- **L1 penalty** (Lasso): Shrinks some coefficients to exactly zero

$$\text{Maximize: } \ell(\beta) - \lambda\sum_{j=1}^p |\beta_j|$$

Note: Don't penalize the intercept!

---

## LDA vs. Logistic Regression

Both produce linear decision boundaries, but they differ in important ways:

| Aspect | LDA | Logistic Regression |
|--------|-----|---------------------|
| Approach | Generative (models P(X\|G)) | Discriminative (models P(G\|X)) |
| Likelihood | Full joint likelihood | Conditional likelihood |
| Assumptions | Normal class distributions, equal covariances | Just that log-odds is linear |
| Efficiency | More efficient if assumptions hold | More robust to violations |
| Outliers | Sensitive (Gaussian assumption) | More robust |
| When to use | Small samples, well-behaved data | Large samples, suspect non-normality |

**Rule of thumb**: If you have small samples and trust the Gaussian assumption, LDA can be more efficient. Otherwise, logistic regression is safer.

---

## Perceptron Learning Algorithm

The perceptron is a historically important algorithm (precursor to neural networks) that finds a separating hyperplane.

### The Objective

Minimize the total distance of misclassified points to the decision boundary:

$$D(\beta) = -\sum_{i \in \text{misclassified}} y_i(x_i^T\beta)$$

### Algorithm

Use stochastic gradient descent:
1. Initialize β
2. For each misclassified point i:
   - Update: $\beta \leftarrow \beta + \eta \cdot y_i \cdot x_i$
3. Repeat until convergence

### Limitations

- **Multiple solutions**: If data is separable, many hyperplanes work. The final answer depends on initialization and order of updates.
- **No convergence guarantee**: If data is NOT separable, the algorithm never converges — it just bounces around forever.

This motivates the **Support Vector Machine**, which finds the unique "best" separating hyperplane.

---

## Maximum Margin Classifiers

### The Margin Idea

If data is linearly separable, many hyperplanes separate the classes. Which is "best"?

**Intuition**: Choose the hyperplane with the largest **margin** — the distance from the hyperplane to the nearest points of either class. This gives the most "room for error" on new data.

### Mathematical Formulation

We want to maximize the margin M:

$$\max_{\beta, \|\beta\|=1} M \quad \text{subject to} \quad y_i(x_i^T\beta) \geq M \quad \forall i$$

**Reformulation** (dropping the norm constraint):

$$\min_\beta \frac{1}{2}\|\beta\|^2 \quad \text{subject to} \quad y_i(x_i^T\beta) \geq 1 \quad \forall i$$

This is a **quadratic program** — convex optimization with a unique solution.

### Lagrangian Formulation

$$L = \frac{1}{2}\|\beta\|^2 - \sum_{i=1}^N \alpha_i\left[y_i(x_i^T\beta) - 1\right]$$

Taking the derivative with respect to β:

$$\beta = \sum_{i=1}^N \alpha_i y_i x_i$$

**Key insight**: The solution is a linear combination of the training points! But only points where the constraint is active (on the margin boundary) contribute — these are called **support vectors**.

For most points, $\alpha_i = 0$. Only a few points determine the decision boundary. This makes SVMs efficient and robust.

---

## Summary: Choosing a Classification Method

| Method | Best For | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **LDA** | Small samples, Gaussian data | Efficient, stable | Assumes normality |
| **QDA** | Different class shapes | Flexible boundaries | More parameters |
| **Naive Bayes** | High dimensions, text data | Fast, scales well | Wrong independence assumption |
| **Logistic Regression** | General purpose | Robust, probabilistic | Requires tuning |
| **Perceptron** | Historical interest | Simple | Unstable, no probabilities |
| **SVM (Max Margin)** | Separable data, small samples | Unique solution, sparse | Hard to interpret |

For most practical applications, **logistic regression** is a great starting point. For high-dimensional text classification, **Naive Bayes** is surprisingly effective. For small samples with well-behaved data, **LDA** is worth trying.

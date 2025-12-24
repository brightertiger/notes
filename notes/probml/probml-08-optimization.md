# Optimization

Optimization is at the heart of machine learning — finding the parameters that minimize loss functions. This chapter covers the algorithms and techniques used to train models effectively.

## The Big Picture

**The optimization problem**:
$$\theta^* = \arg\min_\theta L(\theta)$$

We want to find parameters θ that minimize some loss function L.

**Challenges**:
- Non-convex landscapes (many local minima)
- High-dimensional parameter spaces (millions of parameters)
- Noisy gradients (from mini-batch sampling)
- Computational constraints

---

## Basic Concepts

### Optima

- **Global optimum**: Best solution in the entire parameter space
- **Local optimum**: Best solution in a neighborhood (not necessarily globally best)

### Optimality Conditions

For smooth functions, at a minimum:
1. **First-order condition**: Gradient is zero
   $$\nabla L(\theta^*) = 0$$

2. **Second-order condition**: Hessian is positive semi-definite
   $$\nabla^2 L(\theta^*) \succeq 0$$

### Convexity

A function is **convex** if any local minimum is also a global minimum.

For convex functions, optimization is "easy" — gradient descent will find the global optimum.

**Unfortunately**: Most deep learning losses are non-convex!

### Lipschitz Continuity

A function is L-Lipschitz if:
$$|f(x_1) - f(x_2)| \leq L |x_1 - x_2|$$

**Interpretation**: The function can't change too rapidly. This property is useful for proving convergence.

---

## First-Order Methods

Use only gradient information (first derivatives).

### Gradient Descent

The simplest optimization algorithm:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where η is the **learning rate** or step size.

**Intuition**: Move in the direction of steepest descent.

### Step Size Selection

Choosing η is crucial:
- **Too small**: Slow convergence
- **Too large**: Oscillations, divergence

**Options**:

1. **Constant step size**: Simple but suboptimal

2. **Line search**: Find optimal η at each step
   $$\eta_t = \arg\min_\eta L(\theta_t - \eta \nabla L(\theta_t))$$

3. **Learning rate schedule**: Decrease η over time
   - Must satisfy Robbins-Monro conditions for convergence

### Momentum

Gradient descent is slow in flat regions and oscillates in narrow valleys.

**Heavy ball momentum**:
$$m_t = \beta m_{t-1} + \nabla L(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} - \eta m_t$$

**Intuition**: Accumulate velocity like a ball rolling downhill. EWMA of gradients smooths out oscillations and accelerates in consistent directions.

**Typical value**: β = 0.9

### Nesterov Momentum

Momentum can overshoot. Nesterov adds "lookahead":

$$m_{t+1} = \beta m_t - \eta \nabla L(\theta_t + \beta m_t)$$

**Idea**: Compute gradient at the anticipated next position, not current position.

---

## Second-Order Methods

Use curvature information (Hessian).

### Newton's Method

Use quadratic approximation:
$$L(\theta) \approx L(\theta_t) + \nabla L^T(\theta - \theta_t) + \frac{1}{2}(\theta - \theta_t)^T H (\theta - \theta_t)$$

Optimal step:
$$\theta_{t+1} = \theta_t - H^{-1} \nabla L$$

**Advantages**: Faster convergence (quadratic vs. linear)

**Disadvantages**:
- Hessian H is expensive to compute (O(d²) storage, O(d³) inversion)
- Not scalable to deep learning

### Quasi-Newton Methods (BFGS)

Approximate the Hessian using gradient information:
- Build up Hessian approximation over iterations
- **L-BFGS**: Limited memory version; uses only recent gradients

Useful for smaller models where full-batch gradients are available.

---

## Stochastic Gradient Descent (SGD)

### The Key Insight

For finite-sum problems:
$$L(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, f(x_i; \theta))$$

Computing the full gradient requires summing over all N examples — expensive!

**SGD approximation**: Use a random mini-batch B ⊂ {1, ..., N}:
$$\nabla L(\theta) \approx \frac{1}{|B|}\sum_{i \in B} \nabla \ell(y_i, f(x_i; \theta))$$

### Properties

- **Unbiased**: Expected gradient equals true gradient
- **High variance**: Individual mini-batch gradients are noisy
- **Much faster**: Each step is O(|B|) instead of O(N)

### Mini-batch Size Trade-offs

| Batch Size | Gradient Quality | Computation | Generalization |
|------------|------------------|-------------|----------------|
| Small | Noisy | Fast per step | Often better (regularization effect) |
| Large | Accurate | Slow per step, but parallelizable | May overfit |

---

## Variance Reduction

Reduce noise in SGD gradient estimates.

### SVRG (Stochastic Variance Reduced Gradient)

Periodically compute full gradient; use it to correct mini-batch estimates:
$$g_t = \nabla \ell_i(\theta_t) - \nabla \ell_i(\tilde{\theta}) + \nabla L(\tilde{\theta})$$

### SAGA

Maintain running estimates of gradients for each example; update incrementally.

**Trade-off**: Extra memory for reduced variance.

---

## Adaptive Learning Rates

Different parameters may need different learning rates.

### AdaGrad

Adapt learning rate based on historical gradient magnitudes:

$$s_t = s_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$

**Effect**: Parameters with large gradients get smaller learning rates.

**Problem**: Learning rate decreases monotonically and may become too small.

### RMSProp

Use exponential moving average instead of sum:

$$s_t = \beta s_{t-1} + (1 - \beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t$$

**Prevents** learning rate from vanishing.

### AdaDelta

Like RMSProp, but also scales by historical update magnitudes:

$$\delta_t = \beta \delta_{t-1} + (1 - \beta) (\Delta\theta)^2$$
$$\theta_{t+1} = \theta_t - \frac{\sqrt{\delta_t + \epsilon}}{\sqrt{s_t + \epsilon}} g_t$$

### Adam (Adaptive Moment Estimation)

The most popular optimizer. Combines momentum with adaptive learning rates:

**First moment** (mean of gradients):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment** (mean of squared gradients):
$$s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction** (important for early iterations):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

**Update**:
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

**Default values**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## Constrained Optimization

### Lagrange Multipliers

Convert constrained to unconstrained optimization.

For equality constraint $h(\theta) = 0$:
$$\mathcal{L}(\theta, \lambda) = L(\theta) - \lambda h(\theta)$$

At optimum:
$$\nabla L = \lambda \nabla h$$

**Geometric interpretation**: Gradient of objective is parallel to gradient of constraint.

### KKT Conditions

For inequality constraints $g(\theta) \leq 0$:
$$\mathcal{L}(\theta, \mu) = L(\theta) + \mu g(\theta)$$

**Complementary slackness**:
- If constraint is active ($g(\theta) = 0$): $\mu > 0$
- If constraint is inactive ($g(\theta) < 0$): $\mu = 0$
- Always: $\mu \cdot g(\theta) = 0$

### Proximal Gradient Descent

For composite objectives with non-smooth terms (e.g., L1 regularization):
$$L(\theta) = f(\theta) + g(\theta)$$

Where f is smooth and g is non-smooth.

1. Gradient step on smooth part
2. **Proximal operator** to handle non-smooth part:
   $$\text{prox}_g(x) = \arg\min_z \left[g(z) + \frac{1}{2}\|z - x\|^2\right]$$

**Example**: For L1 penalty, proximal operator is soft-thresholding.

---

## EM Algorithm

For models with latent variables, direct MLE is difficult.

### The Problem

$$\log p(Y | \theta) = \log \sum_z p(Y, z | \theta)$$

The sum inside the log is intractable.

### The Solution

Iterate between:

**E-step**: Compute posterior over latent variables given current parameters
$$q(z) = p(z | Y, \theta^{old})$$

**M-step**: Maximize expected complete-data log-likelihood
$$\theta^{new} = \arg\max_\theta \mathbb{E}_{q(z)}[\log p(Y, z | \theta)]$$

### Properties

- **Monotonic**: Likelihood never decreases
- **Converges**: To a local maximum
- **May get stuck**: Multiple restarts recommended

### Example: GMM

- **E-step**: Compute responsibilities (soft cluster assignments)
- **M-step**: Update means, covariances, and mixing proportions

---

## Simulated Annealing

For non-differentiable or discrete optimization:

1. Start with high "temperature" T
2. Propose random moves
3. Accept improvements always; accept worse moves with probability $\exp(-\Delta L / T)$
4. Gradually decrease T

**Idea**: High T allows escaping local minima; low T focuses on refinement.

---

## Practical Tips

### Learning Rate

- Start with a reasonable default (e.g., 0.001 for Adam)
- Use learning rate warmup for large models
- Decay learning rate during training

### Initialization

- Poor initialization can prevent learning
- Xavier/Glorot: Scale by fan-in/fan-out
- He: For ReLU networks

### Gradient Clipping

Prevent exploding gradients by clipping:
$$g \leftarrow \min\left(1, \frac{\tau}{\|g\|}\right) g$$

### Early Stopping

Monitor validation loss; stop when it starts increasing.

---

## Summary

| Method | Key Idea | When to Use |
|--------|----------|-------------|
| **SGD** | Mini-batch gradients | Large datasets |
| **Momentum** | Accumulate velocity | Faster than vanilla SGD |
| **Adam** | Adaptive + momentum | Default for deep learning |
| **L-BFGS** | Quasi-Newton | Small-medium models, full batch |
| **Proximal** | Handle non-smooth terms | L1 regularization |
| **EM** | Latent variables | Mixture models |

**General advice**:
1. Start with Adam
2. Try SGD + momentum if Adam overfits
3. Use learning rate schedules
4. Watch for vanishing/exploding gradients

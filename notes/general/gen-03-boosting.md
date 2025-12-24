# Boosting

Boosting is one of the most powerful ideas in machine learning: take many "weak" models that are only slightly better than random guessing, and combine them into a "strong" model that achieves high accuracy. It's like combining many rough rules of thumb into a sophisticated decision-making system.

## Overview

**The Key Insight**: Ensemble methods combine multiple models for better predictions.

**Two Main Ensemble Paradigms**:
- **Bagging**: Build models in parallel on different data subsets, then average
    - Reduces variance (covered in Decision Trees notes)
- **Boosting**: Build models sequentially, each one focusing on previous mistakes
    - Reduces bias

**Boosting Formulation**:
- $F(x_i) = \sum_m \alpha_m f_m(x_i)$
- $f_m$ = the $m$-th weak learner (typically a small tree)
- $\alpha_m$ = weight given to that learner
- Each $f_m$ and $\alpha_m$ are fit jointly, considering what came before

**PAC Learning Framework**:
- PAC = Probably Approximately Correct
- Question: Is a problem "learnable"?
- A model is PAC-learnable if we can achieve error < $\epsilon$ with probability > $(1-\delta)$
- **Strong learner**: Achieves low error with high probability (but complex, needs lots of data)
- **Weak learner**: Only slightly better than random guessing

**Schapire's Breakthrough** (1990): "The Strength of Weak Learnability"
- Key theorem: If a problem can be solved by a strong learner, it can be solved by combining weak learners
- Original mechanism (three hypotheses):
    - H1: Train on complete data
    - H2: Train on a balanced sample of H1's correct and incorrect predictions
    - H3: Train on examples where H1 and H2 disagree
    - Final: Majority vote of H1, H2, H3
- Improved performance but couldn't scale easily → led to AdaBoost

**AdaBoost** (Adaptive Boosting):
- Construct many hypotheses (not just three)
- Key innovation: Sample weights "adapt" based on performance
- Correctly classified: weight decreases (pay less attention)
- Incorrectly classified: weight increases (focus on mistakes)

**Weight Update Mechanism**:
- Learner weight: $\alpha_m = \frac{1}{2}\log\left[\frac{1-\epsilon_m}{\epsilon_m}\right]$
- Where $\epsilon_m$ = weighted classification error
- Sample weights: 
    - Correctly classified: $w_i \leftarrow w_i \times e^{-\alpha}$
    - Incorrectly classified: $w_i \leftarrow w_i \times e^{\alpha}$

**Common Pitfalls**:
- **Underfitting**: Not enough weak learners in the ensemble
- **Overfitting**: Using learners that are too complex (not "weak" enough)

## Gradient Boosting

Gradient Boosting generalizes boosting to any differentiable loss function.

**The Key Idea**: Instead of reweighting samples, fit each new learner to the *negative gradient* of the loss function. The gradient tells us how to "fix" the current predictions.

**Why Gradients?**:
- Gradients point in the direction of steepest increase in loss
- Negative gradient = direction to decrease loss most rapidly
- The gradient for each point is a proxy for "how poorly is this point being predicted?"

**Connection to Gradient Descent**:
- Regular gradient descent: Update *parameters* in the negative gradient direction
- Gradient boosting: Add a *new function* that approximates the negative gradient
- Think of it as gradient descent in "function space"

**Mathematical Derivation**:

We want to minimize loss by adding a new function $f_m$:
- Current: $F(x_i) = \sum_{k=1}^{m-1} \alpha_k f_k(x_i)$
- Goal: Find $f_m$ to minimize $L(F(x_i) + \alpha f_m(x_i))$

**Taylor Approximation** (first-order):
- $L(F + \alpha f_m) \approx L(F) + \alpha f_m \cdot \frac{\partial L}{\partial F}$
- The first term is constant; we minimize the second term
- We want: $\min \sum_i \frac{\partial L}{\partial F(x_i)} \times \alpha f(x_i)$

**Pseudo-Residuals**:
- Define: $r_i = -\frac{\partial L}{\partial F(x_i)}$ (the negative gradient)
- Goal becomes: $\min -\sum_i r_i \times \alpha f(x_i)$
- The ensemble improves as long as $\sum_i r_i f(x_i) > 0$

**Why "Pseudo-Residuals"?**:
- For squared loss: $L = \frac{1}{2}(y - F)^2$
- Gradient: $\frac{\partial L}{\partial F} = -(y - F)$
- So $r_i = y_i - F(x_i)$ = actual residual!
- For other losses, $r_i$ is *like* a residual (hence "pseudo")

**Adapting for CART**:
- Decision trees minimize squared error naturally
- Transform the objective:
    - $\min \sum r_i^2 - 2\sum_i r_i \times \alpha f(x_i) + \sum (\alpha f(x_i))^2$
    - $\min \sum (r_i - \alpha f(x_i))^2$
- Now the tree can simply minimize squared error between predictions and pseudo-residuals!

**Optimal Step Size** (Line Search):
- $\alpha^* = \frac{\sum r_i f(x_i)}{\sum f(x_i)^2} \approx 1$

**The Gradient Boosting Algorithm**:

1. **Initialize** with a constant value: $F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$
    - For squared loss: just the mean of $y$

2. **For** $m = 1$ to $M$:
    a. Compute pseudo-residuals: $r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$
    b. Fit a tree to $(x_i, r_{im})$
    c. For each leaf $j$, compute optimal output: $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_j} L(y_i, F_m(x_i) + \gamma)$
    d. Update: $F_{m+1}(x) = F_m(x) + \nu \sum_j \gamma_{jm} I(x \in R_j)$

**The Shrinkage Parameter** ($\nu$):
- Also called learning rate
- Prevents overfitting by taking small steps
- Required because Taylor approximation only works for small changes
- Typical values: 0.01 to 0.3

## Extension to Classification

For classification, we predict log-odds and minimize negative log-likelihood.

**Log-Odds to Probability**:
- $p = \frac{e^{\log(\text{odds})}}{1 + e^{\log(\text{odds})}} = \frac{1}{1 + e^{-\log(\text{odds})}}$

**The Loss Function** (Negative Log-Likelihood):
- $\text{NLL} = -\sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$
- Rewriting in terms of log-odds:
- $\text{NLL} = -\sum [y_i \cdot \text{log-odds} - \log(1 + e^{\text{log-odds}})]$

**Computing Pseudo-Residuals**:
- $\frac{\partial \text{NLL}}{\partial \log(\text{odds})} = p_i - y_i$
- So $r_{im} = y_i - p_i$ (intuitive: how far off is our probability estimate?)

**Finding Optimal Leaf Values**:
- For log-loss, minimizing within each leaf isn't straightforward
- Use second-order Taylor approximation:
- $\gamma^* = \frac{\sum(y_i - p_i)}{\sum p_i(1-p_i)}$
- Numerator: sum of residuals (first derivative)
- Denominator: sum of $p(1-p)$ (second derivative, the Hessian)

**The Classification Algorithm**:

1. **Initialize**: Log-odds that minimizes NLL (e.g., $\log(\frac{\bar{y}}{1-\bar{y}})$)

2. **For** $m = 1$ to $M$:
    a. Compute residuals: $r_{im} = y_i - p_i$
    b. Fit a tree to $(x_i, r_{im})$
    c. For each leaf $j$: $\gamma_{jm} = \frac{\sum_{i \in R_j}(y_i - p_i)}{\sum_{i \in R_j} p_i(1-p_i)}$
    d. Update: $F_{m+1}(x) = F_m(x) + \nu \sum_j \gamma_{jm} I(x \in R_j)$

3. **Predict**: Convert final log-odds to probability

## Gradient Boosting vs AdaBoost

| Aspect | AdaBoost | Gradient Boosting |
|--------|----------|-------------------|
| Focus | Reweighting misclassified samples | Fitting negative gradient of loss |
| Loss function | Exponential loss | Any differentiable loss |
| Sample weights | Explicitly updated | Implicitly through gradients |
| Optimization | Coordinate descent | Gradient descent in function space |

**Key insight**: AdaBoost is a special case of Gradient Boosting with exponential loss!

## Common Loss Functions

**For Regression**:
- **L2 (Squared Error)**: $L(y, F) = \frac{1}{2}(y - F)^2$
    - Gradient = $y - F$ (the actual residual)
    - Sensitive to outliers
- **L1 (Absolute Error)**: $L(y, F) = |y - F|$
    - Gradient = $\text{sign}(y - F)$
    - More robust to outliers
- **Huber Loss**: Combines L1 and L2
    - Behaves like L2 for small errors, L1 for large errors
    - Best of both worlds

**For Classification**:
- **Log Loss**: $L(y, F) = -y\log(p) - (1-y)\log(1-p)$
    - Standard for probability estimation
- **Exponential Loss**: $L(y, F) = e^{-yF}$
    - What AdaBoost minimizes
    - Very sensitive to outliers

## Regularization in Gradient Boosting

**Learning Rate/Shrinkage** ($\nu$):
- Scales contribution of each tree
- Smaller $\nu$ → need more trees, but better generalization
- Trade-off: Training time vs. accuracy

**Subsampling** (Stochastic Gradient Boosting):
- Use only a fraction of data for each tree (e.g., 50-80%)
- Similar to mini-batch gradient descent
- Reduces overfitting and training time

**Early Stopping**:
- Monitor validation performance
- Stop adding trees when validation error stops improving
- Prevents overfitting without explicit regularization

**Tree Constraints**:
- Maximum depth (typically 3-8 for boosting)
- Minimum samples per leaf
- Maximum leaf nodes
- Shallow trees = weak learners = less overfitting

## AdaBoost for Classification

Let's derive AdaBoost from scratch to understand its mechanics.

**Setup**:
- Binary classification: $y \in \{-1, +1\}$
- Exponential loss: $L(y_i, f(x_i)) = e^{-y_i f(x_i)}$
- This is an upper bound on 0-1 loss

**Why Exponential Loss?**:
- If correct prediction: $y_i f(x_i) > 0$ → small loss
- If wrong prediction: $y_i f(x_i) < 0$ → exponentially large loss
- Forces the algorithm to focus hard on mistakes

**Objective Function**:
- Ensemble: $F(x) = \sum_m \alpha_m f_m(x)$
- Loss: $L = \sum_i e^{-y_i F(x_i)}$

**At Round $m$**:
- $L = \sum_i e^{-y_i \sum_{k=1}^m \alpha_k f_k(x)}$
- $L = \sum_i e^{-y_i \sum_{k=1}^{m-1} \alpha_k f_k(x)} \cdot e^{-y_i \alpha_m f_m(x)}$
- Let $w_i^m = e^{-y_i F_{m-1}(x_i)}$ (weights from previous rounds)
- $L = \sum_i w_i^m \cdot e^{-y_i \alpha_m f_m(x_i)}$

**Finding Optimal $\alpha_m$**:
- Split into correct and incorrect predictions:
- $L = \sum_{\text{correct}} w_i e^{-\alpha_m} + \sum_{\text{incorrect}} w_i e^{\alpha_m}$
- Let $\epsilon_m$ = weighted misclassification rate
- $L = (1-\epsilon_m)e^{-\alpha_m} + \epsilon_m e^{\alpha_m}$
- Taking derivative and setting to zero:
- $\alpha_m^* = \frac{1}{2}\log\left[\frac{1-\epsilon_m}{\epsilon_m}\right]$

**Interpreting $\alpha_m$**:
- If $\epsilon_m = 0$ (perfect): $\alpha_m \to \infty$ (trust this learner completely)
- If $\epsilon_m = 0.5$ (random): $\alpha_m = 0$ (ignore this learner)
- If $\epsilon_m > 0.5$ (worse than random): $\alpha_m < 0$ (flip predictions)

**The AdaBoost Algorithm**:

1. **Initialize**: $w_i = 1/N$ for all samples

2. **For** $m = 1$ to $M$:
    a. Fit weak learner $f_m$ minimizing weighted error
    b. Compute weighted error: $\epsilon_m = \frac{\sum_i w_i I(y_i \neq f_m(x_i))}{\sum_i w_i}$
    c. Compute learner weight: $\alpha_m = \frac{1}{2}\log\left[\frac{1-\epsilon_m}{\epsilon_m}\right]$
    d. Update sample weights: $w_i \leftarrow w_i \cdot e^{\alpha_m I(y_i \neq f_m(x_i))}$
    e. Normalize weights to sum to 1

3. **Final prediction**: $\text{sign}\left(\sum_m \alpha_m f_m(x)\right)$

**LogitBoost**: Similar to AdaBoost but minimizes logistic loss
- $\log(1 + e^{-y_i f(x_i)})$
- More robust to noise and outliers than exponential loss

## Notes

**Why Boosting Works**:
- Weak learners have high bias, low variance
- Boosting gradually reduces bias by focusing on mistakes
- Each iteration adds a small amount of complexity

**Historical Development**:
1. AdaBoost (1995) - Freund & Schapire
2. AdaBoost interpreted as gradient descent (1999) - Friedman et al.
3. Generalized to any gradient descent (Gradient Boosting)

**Gradient Descent vs Gradient Boosting**:

| Gradient Descent | Gradient Boosting |
|------------------|-------------------|
| Updates model *parameters* | Updates model *predictions* |
| Parameters: $\theta \leftarrow \theta - \eta \nabla L$ | Predictions: $F \leftarrow F + \nu f_m$ |
| Fixed model structure | Adds new functions |

**Gradient Boosting is a Meta-Model**:
- It's not a single model but a *framework* for combining weak learners
- The weak learner can be any model (trees are most common)
- The loss function can be customized for different tasks

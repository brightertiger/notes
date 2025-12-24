# Support Vector Machines

Support Vector Machines (SVMs) are powerful classifiers based on a beautifully geometric idea: find the hyperplane that separates classes with the **maximum margin**. This margin acts as a "safety buffer" that often leads to excellent generalization.

## Linear SVM

**The Core Question**: Given labeled data, what's the "best" hyperplane to separate the classes?

**The SVM Answer**: The best hyperplane is the one that maximizes the distance to the nearest points from each class. These nearest points are called **support vectors** because they "support" (define) the margin.

**Why Maximize the Margin?**:
- Think of the margin as a "no man's land" between classes
- Larger margin → more room for error on new data
- Intuitively, a decision boundary right at the edge of your training data is fragile
- A boundary with wide margins should generalize better

**Hyperplane Basics**:
- A hyperplane in $d$ dimensions: $H: \mathbf{w} \cdot \mathbf{x} + b = 0$
- $\mathbf{w}$ = normal vector (perpendicular to the hyperplane)
- $b$ = offset (distance from origin)
- Points with $\mathbf{w} \cdot \mathbf{x} + b > 0$ are on one side; $< 0$ on the other

**Distance from Point to Hyperplane**:
$$d = \frac{|\mathbf{w} \cdot \mathbf{x} + b|}{||\mathbf{w}||}$$

This is just the projection of the point onto the normal vector, normalized by the length of $\mathbf{w}$.

**The Margin**:
$$\gamma(\mathbf{w}, b) = \min_{x \in D} \frac{|\mathbf{w} \cdot \mathbf{x} + b|}{||\mathbf{w}||}$$

The margin is the distance to the *closest* point. It's scale-invariant: multiplying $\mathbf{w}$ and $b$ by any constant doesn't change the hyperplane or its margin.

**The Optimization Problem**:

For binary classification with $y_i \in \{+1, -1\}$:
- We need: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) > 0$ for all points (correct classification)
- We want: Maximize the margin

**Original Formulation**:
$$\max_{\mathbf{w}, b} \min_{x \in D} \frac{|\mathbf{w} \cdot \mathbf{x} + b|}{||\mathbf{w}||} \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) > 0$$

This is a max-min problem—tricky to solve directly.

**Clever Simplification**:
Since the hyperplane is scale-invariant, we can choose any scale. Let's fix:
$$|\mathbf{w} \cdot \mathbf{x} + b| = 1 \quad \text{for the points closest to the hyperplane}$$

Now the margin becomes $\frac{1}{||\mathbf{w}||}$, and maximizing margin = minimizing $||\mathbf{w}||$.

**Final SVM Objective** (Hard Margin):
$$\min \frac{1}{2}||\mathbf{w}||^2 \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i$$

**Why $\frac{1}{2}||\mathbf{w}||^2$?**:
- Equivalent to minimizing $||\mathbf{w}||$ (squared is easier mathematically)
- The $\frac{1}{2}$ makes derivatives cleaner

**This is a Quadratic Programming (QP) problem**:
- Quadratic objective, linear constraints
- Efficient solvers exist
- Unique global solution (unlike perceptron, which finds any separating hyperplane)

**Support Vectors**: At the optimal solution, some points satisfy $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$ exactly. These are the support vectors—they lie on the margin boundary and fully determine the solution.

## Soft Margin SVM

**The Problem**: What if data isn't perfectly separable?

**Hard margin SVM fails if**:
- Classes overlap
- There are outliers
- Data is noisy

**The Solution**: Allow some misclassifications, but penalize them.

**Slack Variables** ($\xi_i$):
- Original constraint: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$
- Relaxed constraint: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$ where $\xi_i \geq 0$

**What $\xi_i$ Represents**:
- $\xi_i = 0$: Point correctly classified and outside margin
- $0 < \xi_i < 1$: Point correctly classified but inside margin
- $\xi_i = 1$: Point exactly on the decision boundary
- $\xi_i > 1$: Point misclassified

**Soft Margin Objective**:
$$\min \frac{1}{2}||\mathbf{w}||^2 + C \sum_i \xi_i \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**The Hinge Loss**:
$$\xi_i = \max(1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b), 0)$$

This is the famous **hinge loss**—zero for points outside the margin, linear penalty for points inside or misclassified.

**The C Parameter**:
- High $C$: Little tolerance for errors → narrow margin, risk of overfitting
- Low $C$: More tolerance for errors → wide margin, risk of underfitting
- $C = \infty$: Hard margin SVM

## Duality

**The Primal Problem** (what we wrote above) can be hard to solve directly. Converting to the **dual problem** has advantages:
- Often easier to solve
- Reveals the kernel trick

**Lagrangian Formulation**:
Introduce Lagrange multipliers $\alpha_i$ for each constraint:
$$L = \frac{1}{2}||\mathbf{w}||^2 - \sum_i \alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1]$$

**The Dual Problem** (after taking derivatives):
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T \mathbf{x}_j)$$
$$\text{subject to} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

**Key Insight**: The data only appears as dot products $\mathbf{x}_i^T \mathbf{x}_j$!

**Support Vectors and $\alpha_i$**:
- If $\alpha_i = 0$: Point is not a support vector (doesn't affect solution)
- If $\alpha_i > 0$: Point is a support vector

Most $\alpha_i$ are zero → the solution depends only on support vectors.

## Kernelization

**The Problem**: What if data isn't linearly separable in the original space?

**The Solution**: Map data to a higher-dimensional space where it might be linearly separable.

**Feature Mapping**:
- Original data: $\mathbf{x}$
- Mapped data: $\phi(\mathbf{x})$ in higher (possibly infinite) dimension
- Find a hyperplane in this new space

**The Kernel Trick**:
Remember, the dual problem only uses dot products $\mathbf{x}_i^T \mathbf{x}_j$.

If we work in the mapped space, we need $\phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$.

**Key insight**: We can define a **kernel function** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$ that computes this dot product *without ever explicitly computing $\phi$*.

**Common Kernels**:

**Polynomial Kernel**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$$

Example: 1D points $a$ and $b$, degree 2:
- $K(a, b) = (ab + 1)^2 = a^2b^2 + 2ab + 1$
- This equals $\phi(a)^T \phi(b)$ where $\phi(x) = (x^2, \sqrt{2}x, 1)$

The kernel implicitly maps to a 3D space!

**RBF (Gaussian) Kernel**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2\right)$$

Properties:
- Similarity decreases exponentially with distance
- Maps to **infinite-dimensional** space (via Taylor expansion of exponential)
- Very flexible—can fit complex boundaries
- $\gamma$ controls the "reach" of each training point

**Why Kernels are Powerful**:
1. Compute high-dimensional dot products efficiently
2. Don't need to explicitly represent the high-dimensional vectors
3. Can work in infinite-dimensional spaces (RBF)
4. Turn linear methods into non-linear methods

## SVM for Regression (SVR)

**The Twist**: Instead of maximizing margin between classes, we define a "tube" around the regression line and minimize points outside it.

**$\epsilon$-Insensitive Loss**:
- No penalty if prediction is within $\epsilon$ of true value
- Linear penalty for points outside the tube

**Formulation**:
$$\min \frac{1}{2}||\mathbf{w}||^2 + C\sum_i (\xi_i + \xi_i^*)$$

Subject to:
- $y_i - (\mathbf{w} \cdot \mathbf{x}_i + b) \leq \epsilon + \xi_i$
- $(\mathbf{w} \cdot \mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^*$
- $\xi_i, \xi_i^* \geq 0$

**Intuition**:
- The regression line lies in the middle of the tube
- Points inside the tube (within $\epsilon$) don't contribute to the solution
- Only points outside the tube become support vectors

## Kernel Selection

**Linear Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
- Fastest, most interpretable
- Best when: Many features relative to samples (text classification)
- No hyperparameters beyond $C$

**Polynomial Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$
- Captures feature interactions
- Higher $d$ = more complex boundaries
- Parameters: degree $d$, coefficient $c$

**RBF Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$
- Most versatile, usually a good default
- Higher $\gamma$ = tighter fit around training points
- Parameters: $\gamma$ (often $\gamma = 1/(2\sigma^2)$)

**Sigmoid Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i^T \mathbf{x}_j + c)$
- Similar to neural network activation
- Less commonly used; can have convergence issues

**Rule of Thumb**:
1. Start with RBF (most flexible)
2. If that works well, try linear (faster, more interpretable)
3. Use cross-validation to select kernel and hyperparameters

## SVM Hyperparameter Tuning

**The C Parameter**:
- Trade-off: margin width vs. training accuracy
- Low $C$: Wide margin, more misclassifications allowed, simpler model
- High $C$: Narrow margin, few misclassifications, complex model (may overfit)

**The $\gamma$ Parameter** (RBF kernel):
- Controls influence radius of each support vector
- Low $\gamma$: Large radius, smoother boundary, points influence far away
- High $\gamma$: Small radius, complex boundary, points only influence locally

**The Trade-off**:

| | Low $C$ | High $C$ |
|---|---------|----------|
| **Low $\gamma$** | Very smooth (underfit) | Smooth but fits training well |
| **High $\gamma$** | Wiggly but regularized | Very complex (overfit) |

**Tuning Strategy**:
1. Grid search over $C$ and $\gamma$ on logarithmic scale
2. Example: $C \in \{0.001, 0.01, 0.1, 1, 10, 100, 1000\}$
3. Example: $\gamma \in \{0.001, 0.01, 0.1, 1, 10\}$
4. Use cross-validation to evaluate each combination
5. Select combination with best validation performance

**Practical Tips**:
- Scale your features! SVMs are sensitive to feature scales
- RBF kernel with well-tuned $C$ and $\gamma$ is often competitive with more complex methods
- For large datasets, consider linear SVM (much faster) or approximations

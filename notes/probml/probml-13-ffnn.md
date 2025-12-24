# Feed-Forward Neural Networks

Neural networks are powerful function approximators that learn hierarchical representations. This chapter covers the fundamentals of deep learning.

## The Big Picture

**Key insight**: Compose simple functions to create complex ones.
$$f(x) = f_L(f_{L-1}(...f_2(f_1(x))...))$$

Each layer transforms its input, extracting progressively more abstract features.

---

## From Linear Models to Neural Networks

### Limitations of Linear Models

Linear models: $f(x) = Wx + b$

**Problem**: Can only represent linear decision boundaries.

### Feature Engineering

One solution: Transform features first:
$$f(x) = W\phi(x) + b$$

Where $\phi(x)$ are hand-crafted features (polynomials, interactions, etc.).

**Problem**: Requires domain expertise; doesn't scale.

### The Neural Network Solution

**Learn the features automatically!**
$$f(x) = W_L \cdot \sigma(W_{L-1} \cdot \sigma(...\sigma(W_1 x + b_1)...) + b_{L-1}) + b_L$$

Each layer learns a useful transformation.

---

## Activation Functions

Non-linear functions applied after each layer. Without them, the network would collapse to a single linear transformation.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Output: (0, 1)
- **Problem**: Vanishing gradients (saturates for large |x|)
- **Problem**: Not zero-centered

### Tanh

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

- Output: (-1, 1)
- Zero-centered (better than sigmoid)
- Still has vanishing gradient problem

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

- Output: [0, ∞)
- **Pros**: Non-saturating, computationally efficient, sparse activations
- **Cons**: "Dead ReLU" — neurons can get stuck at 0

### Leaky ReLU

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

- Small slope α (e.g., 0.01) for negative inputs
- Prevents dead neurons
- **Parametric ReLU (PReLU)**: Learn α

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

Where Φ is the Gaussian CDF.
- Smooth approximation of ReLU
- Used in transformers (BERT, GPT)

### Swish

$$\text{Swish}(x) = x \cdot \sigma(x)$$

- Self-gated
- Works well in deep networks

---

## The XOR Problem

A classic example showing why we need hidden layers.

**XOR truth table**:
| x₁ | x₂ | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

**No single line can separate the classes!**

With one hidden layer (2 neurons), a neural network can solve XOR by:
1. First layer creates two linear separators
2. Second layer combines them

---

## Universal Approximation Theorem

**Statement**: A neural network with a single hidden layer of sufficient width can approximate any continuous function on a compact domain to arbitrary precision.

**Implication**: Neural networks are extremely powerful function approximators.

**In practice**: Deep (many layers) is often better than wide (many neurons per layer):
- More parameter efficient
- Learns hierarchical representations
- Better generalization

---

## Backpropagation

The algorithm that makes training deep networks possible.

### The Chain Rule

For composed functions $f = f_1 \circ f_2 \circ ... \circ f_L$:
$$\frac{\partial L}{\partial \theta_l} = \frac{\partial L}{\partial z_L} \cdot \frac{\partial z_L}{\partial z_{L-1}} \cdot ... \cdot \frac{\partial z_{l+1}}{\partial z_l} \cdot \frac{\partial z_l}{\partial \theta_l}$$

### Forward Pass

Compute activations layer by layer, storing intermediate values.

### Backward Pass

Compute gradients layer by layer, from output to input:
$$\frac{\partial L}{\partial z_l} = \frac{\partial L}{\partial z_{l+1}} \cdot \frac{\partial z_{l+1}}{\partial z_l}$$

### Automatic Differentiation

Modern frameworks (PyTorch, TensorFlow) build a computational graph and compute gradients automatically.

**Forward mode**: Efficient when few inputs, many outputs
**Reverse mode (backprop)**: Efficient when many inputs, few outputs (typical in ML)

### Example: Cross-Entropy Gradient

For softmax + cross-entropy:
$$\frac{\partial L}{\partial a_c} = p_c - y_c$$

Beautifully simple: just the prediction error!

### Example: ReLU Gradient

$$\frac{\partial}{\partial x}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

Gradient flows unchanged through positive regions, is blocked through negative.

---

## Training Challenges

### Vanishing Gradients

**Problem**: Gradients shrink exponentially in deep networks.

**Cause**: Chained derivatives of saturating activations (sigmoid, tanh).

**Solutions**:
- Use non-saturating activations (ReLU and variants)
- Residual connections
- Careful initialization
- Batch/layer normalization

### Exploding Gradients

**Problem**: Gradients grow exponentially.

**Solutions**:
- Gradient clipping: $g \leftarrow \min(1, \frac{\tau}{\|g\|}) g$
- Careful initialization

### Mathematical Perspective

Gradient through L layers:
$$\frac{\partial L}{\partial z_1} = \prod_{l=1}^{L-1} J_l \cdot g_L$$

If eigenvalues of Jacobians are:
- < 1: Gradients vanish
- > 1: Gradients explode

---

## Residual Connections

**Key innovation** (ResNet): Add skip connections.

$$z_{l+1} = z_l + f_l(z_l)$$

**Benefits**:
- Gradients flow directly through skip connection
- Learn small perturbations instead of full transformations
- Enables training very deep networks (100+ layers)

**Gradient flow**:
$$\frac{\partial L}{\partial z_l} = \frac{\partial L}{\partial z_L}\left(1 + \frac{\partial}{\partial z_l}\sum_{i=l}^{L-1} f_i(z_i)\right)$$

The "1" term ensures gradients always flow, even if the other term vanishes.

---

## Initialization

Poor initialization can prevent learning entirely.

### The Problem

If weights are too large or too small:
- Activations explode or vanish
- Gradients explode or vanish

### Xavier/Glorot Initialization

For linear activations:
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**Maintains variance** of activations and gradients across layers.

### He Initialization

For ReLU activations:
$$w \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

Accounts for ReLU killing half the activations.

---

## Regularization

### Early Stopping

Stop training when validation error starts increasing.
- Implicit regularization
- Prevents overfitting

### Weight Decay (L2)

Add penalty on weight magnitudes:
$$L = L_{data} + \lambda \sum_l \|W_l\|^2$$

Equivalent to Gaussian prior on weights (MAP estimation).

### Dropout

Randomly "drop" neurons during training with probability p.

$$h_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{otherwise} \end{cases}$$

**Interpretation**:
- Prevents co-adaptation of neurons
- Approximate ensemble of subnetworks
- At test time: use full network (or Monte Carlo dropout for uncertainty)

### Data Augmentation

Create modified versions of training data:
- Images: rotations, flips, crops, color jitter
- Text: synonym replacement, back-translation

---

## Layer Normalization

Normalize activations to stabilize training:

$$\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\tilde{z} = \gamma \hat{z} + \beta$$

Where γ and β are learnable parameters.

**Batch Norm**: Normalize across batch dimension
**Layer Norm**: Normalize across feature dimension (better for RNNs, Transformers)

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Layers** | Transform representations |
| **Activations** | Add non-linearity |
| **Backprop** | Compute gradients efficiently |
| **Residual connections** | Enable deep networks |
| **Normalization** | Stabilize training |
| **Dropout** | Prevent overfitting |
| **Initialization** | Start training successfully |

### Practical Recipe

1. Start with standard architecture (ResNet, etc.)
2. Use ReLU or GELU activations
3. Xavier/He initialization
4. Adam optimizer
5. Batch/Layer normalization
6. Dropout if overfitting
7. Data augmentation for images
8. Early stopping based on validation loss

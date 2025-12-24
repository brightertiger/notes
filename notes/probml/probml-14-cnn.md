# Convolutional Neural Networks

CNNs are specialized neural networks designed for processing grid-structured data, especially images. They're the foundation of modern computer vision.

## The Big Picture

**Problem with MLPs for images**:
- Different image sizes → different input dimensions
- Translation invariance is hard to learn
- Too many parameters (e.g., 1000×1000 image = 3 million inputs!)

**CNN solution**:
- Local connectivity (each neuron sees small region)
- Weight sharing (same filter applied everywhere)
- Translation equivariance built in

---

## The Convolution Operation

### 1D Convolution

$$[w \star x]_i = \sum_{u=0}^{L-1} w_u \cdot x_{i+u}$$

Slide a **filter** (kernel) across the input and compute dot products.

### 2D Convolution

$$[W \star X]_{i,j} = \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} w_{u,v} \cdot x_{i+u, j+v}$$

**Interpretation**: Template matching. High response where input matches the filter pattern.

### Key Insight: Weight Sharing

Same filter weights used at every location → huge parameter reduction!

**Example**: 3×3 filter has 9 parameters, regardless of image size.

---

## Convolution as Matrix Multiplication

Convolution can be expressed as multiplication by a **Toeplitz matrix**:
$$y = Cx$$

Where C has a special sparse structure with repeated weights.

This equivalence is useful for:
- Understanding computational cost
- Implementing on hardware

---

## Convolution Variants

### Valid Convolution

No padding; output shrinks:
- Input: $(H, W)$
- Filter: $(f_H, f_W)$
- Output: $(H - f_H + 1, W - f_W + 1)$

### Same (Zero) Padding

Pad input with zeros to maintain size:
- Padding: $p = (f - 1) / 2$
- Output same size as input

### Strided Convolution

Skip positions to downsample:
- Stride $s$: move filter by s pixels
- Output size: $\lfloor(H + 2p - f)/s + 1\rfloor$

---

## Multi-Channel Convolutions

### Input with Multiple Channels

For RGB images (3 channels), the filter is 3D:
$$z_{i,j} = \sum_c \sum_u \sum_v x_{i+u, j+v, c} \cdot w_{u,v,c}$$

Each filter produces one output channel.

### Multiple Filters

To detect multiple features, use multiple filters:
- Weight tensor: $(f_H, f_W, C_{in}, C_{out})$
- Each filter produces one channel of output

**Output**: Stack of feature maps (one per filter).

### 1×1 Convolution

Special case: filter size = 1×1
- Acts only across channels, not spatial
- Like a per-pixel fully-connected layer
- Used to change number of channels cheaply

---

## Pooling Layers

### Purpose

- Reduce spatial dimensions
- Achieve translation **invariance** (small shifts don't matter)
- Reduce parameters and computation

### Max Pooling

Take maximum value in each window:
$$y_{i,j} = \max_{(u,v) \in \text{window}} x_{i+u, j+v}$$

Most common: 2×2 window with stride 2 (halves dimensions).

### Average Pooling

Take mean instead of max.

### Global Average Pooling

Average over entire spatial dimensions:
- Input: $(H, W, C)$ → Output: $(1, 1, C)$
- Often used before final classifier

---

## Dilated (Atrous) Convolution

Insert "holes" in the filter:
- Dilation rate r: sample every r-th pixel
- Increases **receptive field** without increasing parameters
- Useful for dense prediction (segmentation)

---

## Transposed Convolution

"Upsampling" convolution for:
- Autoencoders
- Generative models
- Semantic segmentation

Increases spatial dimensions (opposite of regular conv).

---

## Normalization

### Batch Normalization

Normalize across the batch dimension:
$$\hat{z}_n = \frac{z_n - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$\tilde{z}_n = \gamma \hat{z}_n + \beta$$

**Per channel**: Compute μ, σ over (N, H, W) for each channel.

**Benefits**:
- Stabilizes training
- Allows higher learning rates
- Some regularization effect

**Issues**:
- Depends on batch statistics → problems with small batches
- Different behavior at train vs. test time

### Layer Normalization

Normalize across channels (and spatial dims):
- Independent of batch size
- Better for RNNs and Transformers

### Instance Normalization

Normalize per sample, per channel:
- Used in style transfer

---

## Common Architectures

### ResNet (Residual Networks)

**Key innovation**: Skip connections
$$y = F(x) + x$$

**Residual block**:
```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
```

Enables training 100+ layer networks.

### DenseNet

**Key idea**: Connect each layer to all subsequent layers
$$x_l = [x_0, f_1(x_0), f_2(x_0, x_1), ...]$$

**Benefits**:
- Feature reuse
- Strong gradient flow

**Drawback**: Memory intensive

### EfficientNet

**Key insight**: Scale depth, width, and resolution together
- Neural Architecture Search (NAS) to find optimal scaling

---

## Adversarial Examples

### White-Box Attacks

Attacker has full access to model.

**FGSM** (Fast Gradient Sign Method):
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$$

Add small perturbation in gradient direction.

**PGD** (Projected Gradient Descent):
Iterative version of FGSM; stronger attack.

### Black-Box Attacks

No access to model internals:
- Query-based attacks
- Transfer attacks (adversarial examples transfer across models)

### Defenses

- Adversarial training
- Input preprocessing
- Certified defenses (provable robustness)

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Convolution** | Local feature detection with weight sharing |
| **Pooling** | Downsample, add invariance |
| **Stride** | Alternative to pooling for downsampling |
| **Padding** | Control output size |
| **1×1 Conv** | Channel mixing |
| **Skip connections** | Enable deep networks |
| **Normalization** | Stabilize training |

### Why CNNs Work for Images

1. **Local structure**: Nearby pixels are related
2. **Translation equivariance**: Features can appear anywhere
3. **Hierarchical composition**: Simple features → complex objects
4. **Parameter efficiency**: Weight sharing dramatically reduces parameters

### Practical Tips

1. Use pre-trained models when possible (transfer learning)
2. Start with proven architectures (ResNet, EfficientNet)
3. Data augmentation is crucial
4. Batch normalization helps training
5. Global average pooling instead of flattening before classifier

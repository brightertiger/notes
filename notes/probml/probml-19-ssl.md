# Self-Supervised and Semi-Supervised Learning

These techniques leverage unlabeled data to improve learning. In a world where labels are expensive but data is abundant, these methods are increasingly important.

## The Big Picture

**The label bottleneck**: Labeling data is expensive and time-consuming.

**The opportunity**: Vast amounts of unlabeled data are available.

**The goal**: Learn useful representations from unlabeled data that transfer to downstream tasks.

---

## Data Augmentation

### The Idea

Create modified versions of training examples that preserve the label.

**Effect**: Expands training set, improves robustness.

### Common Augmentations

**Images**:
- Rotation, flipping, cropping
- Color jitter, brightness changes
- Random erasing, cutout

**Text**:
- Synonym replacement
- Back-translation
- Random insertion/deletion

**Audio**:
- Pitch shifting
- Time stretching
- Adding noise

### Theoretical View: Vicinal Risk Minimization

Instead of minimizing risk at exact training points, minimize in a **vicinity** around them:

$$R = \int L(y, f(x')) p(x' | x) dx'$$

Where p(x'|x) is the augmentation distribution.

---

## Transfer Learning

### The Problem

Task A has lots of data; Task B has little data.

### The Solution

1. **Pretrain** on large source dataset (Task A)
2. **Fine-tune** on small target dataset (Task B)

### Fine-tuning Strategies

**Feature extraction**: Freeze pretrained layers, train only new head.
- Best when: Very small target data, similar domains

**Full fine-tuning**: Update all parameters.
- Best when: More target data, different domains

**Gradual unfreezing**: Start from top layers, progressively unfreeze.
- Often best practice

### Parameter-Efficient Fine-tuning

For large models, updating all parameters is expensive.

**Adapters**: Small bottleneck layers inserted between frozen transformer blocks.

**LoRA**: Low-rank updates to weight matrices.

**Prompt tuning**: Learn soft prompts while keeping model frozen.

---

## Self-Supervised Learning

### The Core Idea

Create supervisory signals from the data itself — no human labels needed.

### Pretext Tasks

**Reconstruction**: Predict masked or corrupted parts
- Autoencoders
- Masked language modeling (BERT)
- Masked image modeling (MAE)

**Contrastive**: Learn to distinguish similar from dissimilar
- Positive pairs: Same image, different augmentations
- Negative pairs: Different images

**Predictive**: Predict properties of the data
- Rotation prediction (images)
- Next word prediction (language)

---

## Contrastive Learning

### The Framework

1. Create two views of same example (via augmentation)
2. Push their representations together
3. Push representations of different examples apart

### SimCLR (Simple Contrastive Learning)

**Pretraining**:
1. Take image x
2. Apply two augmentations: $x_1, x_2$
3. Encode both: $z_1 = g(f(x_1)), z_2 = g(f(x_2))$
4. Contrastive loss (NT-Xent):
   $$L = -\log \frac{\exp(\text{sim}(z_1, z_2)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_1, z_k)/\tau)}$$

**Fine-tuning**: Discard projection head g, fine-tune encoder f.

### Key Insights

- **Projection head** is crucial during pretraining but discarded after
- **Large batch sizes** help (more negatives)
- **Strong augmentation** is important

### Challenges

- Need many negative examples
- Batch size dependence
- Risk of feature collapse

---

## Non-Contrastive Methods

### BYOL (Bootstrap Your Own Latent)

No negative examples needed!

**Architecture**:
- **Online network** (student): Updated by gradient descent
- **Target network** (teacher): Exponential moving average of online

**Loss**: Predict target representation from online prediction.

**Why no collapse?** Asymmetric architecture + momentum updates prevent trivial solutions.

### Masked Autoencoders (MAE)

Inspired by BERT's success:
1. Mask large portion of image (75%!)
2. Encode visible patches
3. Decode to reconstruct masked patches
4. For downstream: Use only encoder

**Why mask so much?** Forces learning high-level features, not just copying.

---

## Semi-Supervised Learning

Use both labeled and unlabeled data together.

### Self-Training

1. Train on labeled data
2. Predict on unlabeled data (pseudo-labels)
3. Add confident predictions to training set
4. Repeat

**Connection to EM**: Pseudo-labels are the E-step!

### Noise Student Training

Self-training + noise:
1. Train teacher on labeled data
2. Generate pseudo-labels for unlabeled data
3. Train student on all data with noise (dropout, augmentation)
4. Student becomes new teacher; repeat

**Key insight**: Noise makes student more robust than teacher.

### Consistency Regularization

Model predictions should be consistent under small input changes:
$$L = L_{supervised} + \lambda \cdot d(f(x), f(\text{aug}(x)))$$

**FixMatch**: Combine pseudo-labeling with consistency.

---

## Label Propagation

### Graph-Based Approach

1. Build graph: Nodes = data points, edges = similarity
2. Propagate labels from labeled to unlabeled nodes
3. Use resulting labels for training

### Algorithm

- T: Transition matrix (normalized edge weights)
- Y: Label matrix (N × C)
- Iterate: $Y = TY$ until convergence
- Use propagated labels for supervised learning

### Assumptions

- Similar points should have similar labels
- Cluster structure in data reflects label structure

---

## Generative Self-Supervised Learning

### Variational Autoencoders (VAE)

**Generative model**:
1. Sample latent: $z \sim p(z)$
2. Generate: $x \sim p(x|z)$

**Training**: Maximize ELBO (Evidence Lower Bound)
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

**Use for SSL**: Learn representations z for downstream tasks.

### GANs

**Generator**: Maps noise to data.
**Discriminator**: Distinguishes real from fake.

**Semi-supervised extension**: Discriminator predicts K classes + "fake".

---

## Active Learning

### The Idea

If we must label data, label the most informative examples.

### Strategies

**Uncertainty sampling**: Label examples where model is least confident.
$$x^* = \arg\max_x H[p(y|x)]$$

**BALD** (Bayesian Active Learning by Disagreement): Label where model's predictions are most diverse (across ensemble/dropout samples).

**Query by committee**: Multiple models vote; label where they disagree.

---

## Few-Shot Learning

### The Challenge

Learn to classify new classes from very few examples (1-5 per class).

### Meta-Learning Approach

Train model to learn quickly:
- **Training**: Many "episodes" with different class subsets
- **Testing**: New classes, few examples each

### Metric Learning Approach

Learn embedding space where similarity = class membership.
- Prototypical networks: Classify by nearest class prototype
- Matching networks: Weighted nearest neighbor

---

## Weak Supervision

### When Labels Are Imperfect

- Noisy labels (some wrong)
- Soft labels (probability distributions)
- Aggregate from multiple labelers

### Label Smoothing

Instead of hard labels [0, 1, 0]:
$$y_{smooth} = (1 - \epsilon) \cdot y + \epsilon / K$$

Prevents overconfidence, improves calibration.

---

## Summary

| Method | Uses Unlabeled Data | Key Idea |
|--------|---------------------|----------|
| **Data Augmentation** | No (extends labeled) | Transform while preserving label |
| **Transfer Learning** | Pre-training stage | Leverage large datasets |
| **Contrastive Learning** | Yes | Pull similar, push dissimilar |
| **Non-Contrastive** | Yes | Predict across views (no negatives) |
| **Semi-Supervised** | Yes (with some labels) | Pseudo-labels + consistency |
| **Active Learning** | Selects what to label | Query most informative |

### Practical Recommendations

1. **Always use data augmentation** — almost free improvement
2. **Start with pretrained models** — rarely worth training from scratch
3. **Try self-training** for semi-supervised (simple and effective)
4. **Large-scale pretraining** for best representations (if compute allows)
5. **Match pretraining and downstream domains** for best transfer

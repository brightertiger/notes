# Dimensionality Reduction

High-dimensional data is everywhere: images (thousands of pixels), text (thousands of words), genomics (thousands of genes). Dimensionality reduction compresses this data into a smaller number of meaningful features while preserving important structure.

## Background

**The Curse of Dimensionality**:

As dimensions increase, data becomes increasingly sparse. Consider:
- A 1×1 square patch covers 1% of a 10×10 square
- A 1×1×1 cubic patch covers 0.1% of a 10×10×10 cube
- In general: Volume grows exponentially with dimension

**Why This Matters**:
- Machine learning needs enough data to "fill" the space
- Data requirements grow exponentially with dimensions
- Distances become less meaningful (everything is "far" from everything)
- Many algorithms break down

**Two Approaches to Reduce Dimensions**:

1. **Feature Selection**: Keep a subset of original features
   - Pros: Interpretable, simple
   - Cons: May lose important combinations

2. **Feature Extraction/Latent Features**: Create new features from combinations of originals
   - Linear methods: PCA, LDA
   - Non-linear methods: t-SNE, UMAP, autoencoders
   - Pros: Can capture more complex structure
   - Cons: New features may be hard to interpret

## Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in the data and projects onto them. It's the most widely used dimensionality reduction technique.

**The Idea**:
- Find a linear projection from high dimension ($D$) to low dimension ($L$)
- Preserve as much variance as possible
- The directions of maximum variance are called **principal components**

**Mathematical Setup**:
- Original data: $\mathbf{x} \in \mathbb{R}^D$
- Projection matrix: $\mathbf{W} \in \mathbb{R}^{D \times L}$ (columns are orthonormal)
- **Encode**: $\mathbf{z} = \mathbf{W}^T \mathbf{x} \in \mathbb{R}^L$
- **Decode**: $\hat{\mathbf{x}} = \mathbf{W}\mathbf{z}$

**Objective**: Minimize reconstruction error
$$L(\mathbf{W}) = \frac{1}{N}\sum_i ||\mathbf{x}_i - \hat{\mathbf{x}}_i||^2$$

**Derivation** (projecting to 1D):

We want to find $\mathbf{w}_1$ that minimizes reconstruction error:
$$L = \frac{1}{N}\sum_i ||\mathbf{x}_i - z_{i1}\mathbf{w}_1||^2$$

Expanding and using $\mathbf{w}_1^T\mathbf{w}_1 = 1$ (orthonormality):
$$L = \frac{1}{N}\sum_i [\mathbf{x}_i^T\mathbf{x}_i - 2z_{i1}\mathbf{w}_1^T\mathbf{x}_i + z_{i1}^2]$$

Taking derivative w.r.t. $z_{i1}$:
$$\frac{\partial L}{\partial z_{i1}} = -2\mathbf{w}_1^T\mathbf{x}_i + 2z_{i1} = 0$$

**Optimal encoding**: $z_{i1} = \mathbf{w}_1^T\mathbf{x}_i$ (project onto $\mathbf{w}_1$)

Substituting back:
$$L = C - \frac{1}{N}\sum_i z_{i1}^2 = C - \frac{1}{N}\mathbf{w}_1^T\boldsymbol{\Sigma}\mathbf{w}_1$$

Where $\boldsymbol{\Sigma}$ is the covariance matrix of $\mathbf{X}$.

**Key Insight**: Minimizing reconstruction error = Maximizing variance of projections!

Using Lagrange multipliers with constraint $\mathbf{w}_1^T\mathbf{w}_1 = 1$:
$$\frac{\partial}{\partial \mathbf{w}_1}\left[\mathbf{w}_1^T\boldsymbol{\Sigma}\mathbf{w}_1 + \lambda(1 - \mathbf{w}_1^T\mathbf{w}_1)\right] = 0$$
$$\boldsymbol{\Sigma}\mathbf{w}_1 = \lambda\mathbf{w}_1$$

**The optimal $\mathbf{w}_1$ is an eigenvector of the covariance matrix!**

To maximize variance, choose the eigenvector with the **largest eigenvalue**.

**Geometric Interpretation**:
- Imagine the data as a cloud of points
- The first principal component is the direction of maximum spread
- The second PC is perpendicular and captures the next most variance
- And so on...

Think of it as finding the best "viewing angle" for your data.

**Eigenvalues = Variance Explained**:
- Each eigenvalue equals the variance along that principal component
- Sum of all eigenvalues = total variance
- Fraction explained by first $k$ components: $\frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^D \lambda_j}$

**Scree Plot**: Graph eigenvalues (or % variance) vs. component number
- Look for an "elbow" where variance drops off
- Components before the elbow are usually important

**Factor Loadings**:
- Each PC is a linear combination of original features
- Loadings = weights in this combination
- High loading = feature contributes strongly to that PC

**PCA + Regression**: Still interpretable!
- Run regression on principal components
- Use loadings to translate back to original features

**Computing PCA via SVD**:
- Singular Value Decomposition: $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^T$
- $\mathbf{V}$ contains the principal components (eigenvectors)
- $\mathbf{S}$ contains singular values (square roots of eigenvalues)
- More numerically stable than eigendecomposition

**Limitations of PCA**:
- Only captures **linear** relationships
- Sensitive to **outliers** (they inflate variance)
- Can't handle **missing data** (need imputation first)
- **Unsupervised**: Doesn't use label information

**Alternatives**:
- **Kernel PCA**: Non-linear version using the kernel trick
- **Factor Analysis**: Assumes latent factors + noise
- **LDA**: Supervised, maximizes between-class variance

## Stochastic Neighbor Embedding (SNE)

**The Idea**: Preserve local neighborhood structure rather than global distances. Points that are neighbors in high-D should be neighbors in low-D.

**Manifold Hypothesis**:
- High-dimensional data often lies on a lower-dimensional "manifold"
- Think: Earth's surface is a 2D manifold in 3D space
- We want to "unfold" this manifold

**Algorithm**:

1. **Convert distances to probabilities** (high-D):
$$p_{j|i} \propto \exp\left(-\frac{||\mathbf{x}_i - \mathbf{x}_j||^2}{2\sigma_i^2}\right)$$

This is the probability that point $i$ would pick point $j$ as its neighbor.

**Adaptive variance** ($\sigma_i$):
- Dense regions get smaller $\sigma_i$ (be more selective)
- Sparse regions get larger $\sigma_i$ (include more distant neighbors)
- Controlled by **perplexity** parameter (roughly, the number of effective neighbors)

2. **Initialize low-D coordinates** $\mathbf{z}_i$ randomly

3. **Compute low-D probabilities**:
$$q_{j|i} \propto \exp\left(-||\mathbf{z}_i - \mathbf{z}_j||^2\right)$$

4. **Minimize KL divergence** between $p$ and $q$:
$$L = \sum_i \sum_j p_{j|i} \log\frac{p_{j|i}}{q_{j|i}}$$

**Why KL Divergence?**:
- If $p$ is high but $q$ is low: **Large penalty** (neighbors in high-D are far in low-D — bad!)
- If $p$ is low but $q$ is high: **Small penalty** (distant points end up close — not as bad)
- Prioritizes preserving local structure

**Symmetric SNE**: Make distances symmetric: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2}$

## t-SNE

**The Problem with SNE**: Crowding. Gaussian probability decays quickly, pushing moderately distant points too close together in low-D.

**The Solution**: Use the **t-distribution** (fat tails) for low-D probabilities:
$$q_{ij} \propto (1 + ||\mathbf{z}_i - \mathbf{z}_j||^2)^{-1}$$

**Why t-Distribution?**:
- Heavier tails than Gaussian
- Points that are moderately far in high-D can be *very* far in low-D
- Creates well-separated clusters with tight internal structure

**t-SNE Properties**:
- Excellent for visualization (2D or 3D)
- Creates visually appealing, well-separated clusters
- Preserves local structure well

**Limitations**:
- **Computationally expensive**: $O(N^2)$ for pairwise distances
- **Random initialization**: Results vary between runs
- **Not invertible**: Can't go from low-D back to high-D
- **Coordinates are meaningless**: Only relative positions matter
- **Global structure distorted**: Distances between clusters are not meaningful

**Hyperparameters**:
- **Perplexity**: Balance between local and global (typical: 5-50)
- **Learning rate**: Step size for gradient descent
- **Iterations**: Usually 1000+

## UMAP

**Uniform Manifold Approximation and Projection**: Like t-SNE but faster and (arguably) better at preserving global structure.

**Key Differences from t-SNE**:

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Pairwise distances | All pairs | Only neighbors |
| Initialization | Random | Spectral embedding |
| Updates | All points every iteration | Stochastic (subsets) |
| Speed | Slow ($O(N^2)$) | Much faster |
| Global structure | Often distorted | Better preserved |

**UMAP Algorithm**:

1. **Build neighborhood graph** in high-D
   - Compute distance to $k$ nearest neighbors
   - Convert to similarity scores (exponential decay from nearest neighbor)

2. **Make similarities symmetric**: $s_{ij} = s_{i|j} + s_{j|i} - s_{i|j} \cdot s_{j|i}$

3. **Initialize low-D with spectral embedding** (decomposition of graph Laplacian)

4. **Compute low-D similarities** using t-distribution variant:
$$q_{ij} \propto (1 + \alpha \cdot d^{2\beta})^{-1}$$

5. **Minimize cross-entropy** between high-D and low-D graphs

**UMAP Advantages**:
- **Much faster** than t-SNE (especially for large datasets)
- **Better global structure**: Preserves relative cluster distances better
- **`transform` method**: Can embed new points without recomputing everything
- **Flexible**: Can use any distance metric

**When to Use Which**:
- **Small data, visualization only**: Either works, t-SNE often prettier
- **Large data**: UMAP (t-SNE too slow)
- **Need to embed new points**: UMAP
- **Need reproducibility**: UMAP (more stable with same parameters)

## Applications of Dimensionality Reduction

**Data Visualization**:
- Reduce to 2D or 3D for plotting
- Discover clusters, outliers, patterns visually
- t-SNE and UMAP are standard tools

**Noise Reduction**:
- Signal variance > noise variance
- First few PCs capture signal, later PCs capture noise
- Reconstructing from top PCs filters out noise

**Preprocessing for ML**:
- Reduces curse of dimensionality
- Speeds up training
- Can improve performance (by removing noise)
- Essential for algorithms sensitive to dimensionality (e.g., k-NN)

**Feature Extraction**:
- Create more informative features
- Example: Face recognition—first few PCs are "eigenfaces"

**Multicollinearity**:
- Highly correlated predictors cause problems in regression
- PCA creates uncorrelated components
- Principal Component Regression (PCR) solves this

## Comparing Techniques

| Method | Linear? | Preserves | Best For | Interpretable? |
|--------|---------|-----------|----------|----------------|
| PCA | Yes | Global variance | Preprocessing, compression | Yes (loadings) |
| t-SNE | No | Local neighborhoods | 2D/3D visualization | No |
| UMAP | No | Local + some global | Large-scale visualization | No |
| LDA | Yes | Class separation | Supervised reduction | Yes |

**Selection Guide**:
1. **Know you need visualization?** → t-SNE or UMAP
2. **Need to preprocess for ML?** → PCA
3. **Have class labels?** → Consider LDA
4. **Need interpretability?** → PCA
5. **Very large data?** → UMAP or randomized PCA
6. **Complex non-linear structure?** → UMAP, t-SNE, or kernel PCA

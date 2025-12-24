# Clustering

Clustering is the task of grouping similar objects together without being told what the groups should be. It's **unsupervised learning**—we have no labels, only data. The algorithm discovers structure on its own.

**Why Clustering Matters**:
- Customer segmentation: Group customers by behavior
- Image segmentation: Group pixels by color/texture
- Document organization: Group articles by topic
- Anomaly detection: Normal patterns vs. outliers
- Data compression: Represent data by cluster centers

## Hierarchical Agglomerative Clustering

**The Idea**: Start with each point as its own cluster, then repeatedly merge the two most similar clusters until only one remains.

**Algorithm**:
1. Start: Each data point is its own cluster
2. Compute all pairwise distances between clusters
3. Merge the two closest clusters
4. Repeat steps 2-3 until one cluster remains

**This produces a dendrogram**—a tree showing the merge history. Cut the tree at any height to get that many clusters.

**Defining "Similarity" Between Clusters**:

| Linkage Method | Definition | Properties |
|----------------|------------|------------|
| **Single Link** | Distance between two *closest* members | Tends to create long, chain-like clusters |
| **Complete Link** | Distance between two *farthest* members | Creates compact, spherical clusters |
| **Average Link** | Average distance between all pairs | Balance between single and complete |
| **Ward's Method** | Increase in total within-cluster variance | Minimizes variance, popular choice |

**When to Use**:
- When you don't know the number of clusters ahead of time
- When you want to see hierarchical structure
- For small to medium datasets (doesn't scale well)

**Limitation**: Very slow! $O(n^3)$ time complexity makes it impractical for large datasets.

## K-Means Clustering

**The Idea**: Partition data into exactly $K$ clusters, each represented by its centroid (center of mass).

**Algorithm**:
1. Choose $K$ initial cluster centers (randomly or using K-Means++)
2. **Assign** each point to nearest center: $z_n^* = \arg\min_k ||x_n - \mu_k||^2$
3. **Update** centers as mean of assigned points: $\mu_k = \frac{1}{N_k}\sum_{n: z_n=k} x_n$
4. Repeat steps 2-3 until assignments don't change

**Objective Function** (Distortion/Inertia):
$$L = \sum_n ||x_n - \mu_{z_n}||^2$$

This is the total squared distance from each point to its assigned center.

**Key Properties**:
- Each iteration decreases (or maintains) the objective
- Guaranteed to converge, but not necessarily to global optimum
- Converges quickly in practice

**The Initialization Problem**:
- K-Means is sensitive to initial centers
- Bad initialization → stuck in poor local minimum
- Solution: **Multiple restarts**—run many times, keep best result

**K-Means++ Initialization** (the smart way):
1. Choose first center randomly from data points
2. For each subsequent center:
   - Compute distance $D(x)$ from each point to nearest existing center
   - Choose new center with probability proportional to $D(x)^2$
3. This spreads out initial centers—points far from existing centers more likely to be chosen

**Result**: Much better starting point, often finds better solutions.

**K-Medoids Algorithm** (PAM - Partitioning Around Medoids):
- Centers must be actual data points (medoids), not means
- More robust to outliers
- Assignment: $z_n^* = \arg\min_k d(x_n, \mu_k)$ (any distance metric)
- Update: Find point with smallest total distance to all other cluster members
- **Swap step**: Try swapping current medoid with non-medoid, keep if cost decreases

**Choosing the Number of Clusters ($K$)**:

This is one of the hardest problems in clustering!

**Elbow Method**:
1. Plot distortion vs. $K$
2. Look for "elbow" where distortion stops decreasing rapidly
3. Often subjective—the elbow isn't always clear

**Silhouette Score**:
For each point $i$:
- $a_i$ = average distance to other points in same cluster (cohesion)
- $b_i$ = average distance to points in nearest other cluster (separation)
- $S_i = \frac{b_i - a_i}{\max(a_i, b_i)}$

Interpretation:
- $S_i \approx 1$: Point is well-clustered
- $S_i \approx 0$: Point is on boundary between clusters
- $S_i \approx -1$: Point may be in wrong cluster

Average silhouette score across all points measures overall clustering quality.

**K-Means is EM with Hard Assignments**:
- Expectation-Maximization (EM) uses "soft" assignments (probabilities)
- K-Means uses "hard" assignments (0 or 1)
- Both assume spherical clusters with equal variance
- K-Means is a special case of Gaussian Mixture Models

**Limitations of K-Means**:
- **Assumes spherical clusters**: Can't find elongated or irregular shapes
- **Assumes equal-sized clusters**: Tends to split large clusters
- **Sensitive to outliers**: Means are pulled by extreme values
- **Requires specifying $K$**: Must know number of clusters beforehand
- **Non-convex clusters**: Fails on clusters with complex shapes
- **Euclidean distance**: May not be appropriate for all data types

**Interpreting Results**:
- **Cluster centers**: "Prototypical" members, useful for understanding what each cluster represents
- **Cluster sizes**: Imbalanced sizes might indicate poor clustering or natural structure
- **Within-cluster variance**: Measures homogeneity
- **Between-cluster variance**: Measures separation

## Spectral Clustering

**The Idea**: Use the eigenvalues (spectrum) of a similarity graph to find clusters. Works when K-Means fails on non-convex shapes.

**Graph Perspective**:
- Data points are nodes
- Edges connect similar points (weighted by similarity)
- Goal: Find a partition that minimizes edges cut between groups

**Algorithm**:
1. Build similarity graph from data (e.g., k-nearest neighbors or Gaussian similarity)
2. Compute the Graph Laplacian: $L = D - A$
   - $D$ = degree matrix (diagonal, $D_{ii}$ = sum of edges from node $i$)
   - $A$ = adjacency matrix (edge weights)
3. Find eigenvectors of $L$ corresponding to smallest eigenvalues
4. Use these eigenvectors as new features, run K-Means

**Why the Laplacian?**:
- Smallest eigenvalue is always 0 (corresponding to all-ones vector)
- Second smallest eigenvalue (Fiedler value) indicates best cut
- For $K$ clusters, use the $K$ smallest eigenvectors

**When to Use**:
- Non-convex cluster shapes (crescents, rings)
- When you have a natural similarity/distance matrix
- Graph-structured data

## DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

**The Idea**: Clusters are dense regions separated by sparse regions. Points in sparse regions are "noise."

**Key Concepts**:
- **Core Point**: Has at least `minPts` points within radius $\epsilon$
- **Border Point**: Within $\epsilon$ of a core point, but not itself core
- **Noise Point**: Neither core nor border

**Reachability**:
- **Direct Density Reachable**: Point $q$ is within $\epsilon$ of core point $p$
- **Density Reachable**: There's a chain of core points connecting $p$ to $q$
- **Density Connected**: Both $p$ and $q$ are density reachable from some core point

**Algorithm**:
1. For each point, determine if it's a core point
2. Create clusters by connecting core points that are within $\epsilon$ of each other
3. Assign border points to nearby clusters
4. Mark remaining points as noise

**Parameters**:
- $\epsilon$ (epsilon): Neighborhood radius
- `minPts`: Minimum points to form dense region

**Choosing Parameters**:
- `minPts`: Often set to $\text{dimensionality} + 1$ or higher
- $\epsilon$: Plot k-distance graph (distance to k-th nearest neighbor), look for elbow

**Advantages**:
- No need to specify number of clusters
- Finds arbitrarily shaped clusters
- Robust to outliers (identifies them as noise)
- Only two intuitive parameters

**Disadvantages**:
- Struggles with clusters of varying density
- Parameter selection can be tricky
- Doesn't work well in high dimensions (curse of dimensionality)
- Can't handle clusters that are close together

**Extensions**:
- **OPTICS** (Ordering Points To Identify Clustering Structure):
    - Creates an ordering of points based on density
    - Can extract clusters at different density levels
    - More robust to parameter choices

- **HDBSCAN** (Hierarchical DBSCAN):
    - Automatically finds clusters of varying densities
    - Combines benefits of DBSCAN and hierarchical clustering
    - More robust, fewer parameters to tune
    - Often the best choice in practice

## Choosing a Clustering Algorithm

**Decision Guide**:

| Situation | Recommended Algorithm |
|-----------|----------------------|
| Know number of clusters, spherical shapes | K-Means |
| Don't know number, want hierarchy | Hierarchical Agglomerative |
| Arbitrary shapes, want to identify noise | DBSCAN or HDBSCAN |
| Non-convex shapes, know number | Spectral Clustering |
| Large dataset | K-Means or Mini-batch K-Means |
| Want to visualize cluster relationships | Hierarchical with dendrogram |

**Validation Approaches**:
- **Internal metrics** (no ground truth): Silhouette score, Davies-Bouldin index, Calinski-Harabasz index
- **External metrics** (with ground truth): Adjusted Rand Index, Normalized Mutual Information
- **Stability**: Do clusters persist with data perturbations?

**Remember**: Clustering is exploratory—there's often no "right" answer. Different algorithms reveal different structure. The best choice depends on your data and goals.

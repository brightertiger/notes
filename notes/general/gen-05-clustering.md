# Clustering

-   Unsupervised learning technique
-   Assign similar data points to a single group / cluster

## Hierarchical Agglomerative Clustering

-   At each step, merge the two most similar groups
-   Keep going until there is a single group left
-   Similarity between groups
    -   Single Link: Distance between the two closest members of each group
    -   Complete Link: Distance between the two farthest members of each group
    -   Average Link: Average Distance between all pairs

## K-Means Clustering

-   Hierarchical Clustering is very slow
-   Algorithm
    -   Assume there are K (hyperparameter) clusters
    -   Assign each data point to its nearest cluster center
        -   $z_n^* = \arg \min_k ||x_n - \mu_k ||^2$
    -   Update the cluster centers at the end of assignments
        -   $\mu_k = {1 \over N_k}\sum_{n: z_n=k} x_n$
-   Objective
    -   Minimize distortion
    -   $L = \sum_{n} ||x_n - \mu_{z_n}||^2$
-   Non-Convex objective, sensitive to intialization
-   Multiple Restarts to control randomness
-   K-Means++ Algorithm
    -   Pick centers sequentially to cover the data
    -   Pick initial points randomly
    -   For subsequent rounds, initialize with points picked with probability proportional to the distance from it's cluster center
    -   Points far away from the cluster center are more likely to be picked in subsequent iterations
-   K-Medoids Algorithm
    -   More robust to outliers
    -   Dont update the cluster center with mean
    -   Use average dissimilarity to all other points in the cluster (i.e. medoid)
    -   $z_n^* = \arg \min d(x_n,\mu_k)$
    -   $\mu_k^* = \arg \min_n \sum_{n'} d(x_n,x_n')$
    -   Point has smallest sum of distances to all other points
    -   Partitioning around medoid
        -   Swap the current medoid center with a non-medoid to see if the cost decreases
-   Selecting the number of Clusters
    -   Minimize Distortion
        -   Use a validation dataset
        -   Select the parameter that minimizes distortion on validation
        -   But usually it decreases monotonically with number of clusters
    -   Elbow method
        -   Rate at which distortion goes down with number of clusters
    -   Silhouette Coefficient
        -   How similar object is to it's own cluster compared to other clusters
        -   Measures Compactness
        -   Given data point i
            -   $a_i$ = (Mean distance to observations in own cluster)
            -   $b_i$ = (Mean Distance ot observations in the next closest cluster)
        -   $S_i = (b_i - a_i) / \max(a_i, b_i)$
        -   Average the score for all the K clusters
        -   Ideal value is 1, worst value is -1
-   K-Means is a variant of EM
    -   K-Means assumes that clusters are spherical
    -   Hard assignment in K-Means vs Soft Assignment in EM

-   Limitations of K-Means:
    -   Assumes clusters are spherical and equally sized
    -   Sensitive to outliers (means are affected by extreme values)
    -   Requires number of clusters (K) to be specified
    -   May converge to local optima
    -   Cannot handle non-convex clusters
    -   Uses Euclidean distance, which may not be appropriate for all data types

-   Interpreting Clustering Results:
    -   Cluster centers: Represent "prototypical" members of each cluster
    -   Cluster sizes: Distribution of data across clusters
    -   Within-cluster variation: Measure of cluster homogeneity
    -   Between-cluster variation: Measure of cluster separation
    -   Silhouette score: Combines cohesion and separation metrics
    -   Visualization: Use dimensionality reduction (PCA, t-SNE) to visualize clusters

## Spectral Clustering

-   Clusters in a graph
-   Find a subgraph
    -   Maxmimum number of within cluster connections
    -   Minimum number of between cluster connections
-   Calculate degree and adjacency matrix
-   Calculate the graph Laplacian $L=D-A$
    -   0 if the nodes are not connected
    -   -1 if the nodes are connected
-   Second smallest eigenvalue and eigenvector of L gives the best cut for graph partition
    -   The smallest value will be zero
    -   Group the nodes using second smallest eigenvector
-   Applicable to pairwise similarity matrix
    -   Graph Representation
        -   Node is the object
        -   Distance denotes the edge

## DBSCAN

-   Density based spatial clustering
-   Clusters are dense regions in space separated by regions with low density
-   Recusrvely expand the cluster based on dense connectivity
-   Can find clusters of arbitrary shape
-   Two parameters:
    -   $\epsilon$ radius
    -   mininum \# points to be contained in the neighbourhood
-   Core Point
    -   Point that has mininum \# points in $\epsilon$ radius
-   Direct Density Reachble
    -   Points in $\epsilon$ radius of core point
-   Density Reachable
    -   Chain connects the two points
    -   Chain is formed by considering many different core points
-   Border Point
    -   Point is DDR but not core
-   Expand the clusters recursively by collapsing DR and DDR points

-   Advantages of DBSCAN:
    -   Does not require specifying number of clusters
    -   Can find arbitrarily shaped clusters
    -   Robust to outliers (identifies them as noise)
    -   Only needs two parameters: epsilon and minimum points
-   Disadvantages of DBSCAN:
    -   Struggles with varying density clusters
    -   Selection of parameters can be challenging
    -   Not efficient for high-dimensional data due to curse of dimensionality
    -   May have difficulty with datasets where clusters are close to each other

-   Extensions to DBSCAN:
    -   OPTICS: Ordering points to identify clustering structure
    -   HDBSCAN: Hierarchical DBSCAN that extracts clusters from varying densities 
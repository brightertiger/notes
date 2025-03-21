# Exemplar Methods

- Non-parametric Models
    - Keep the training data around
    - Effective number of model parameters grow with |D|

- Instance-based Learning
    - Models keep training examples around test time
    - Define similarity between training points and test input
    - Assign the label based on the similarity

- KNN
    - Classify the new input based on K closest examples in the training set
    - $p(y = c | x, D) = \frac{1}{K} \sum_{i \in N_K(x)} I\{y_i=c\}$
    - The closest point can be computed using Mahalanobis Distance
    - $d_M(x,\mu) = \sqrt{(x-\mu)^TM(x-\mu)}$
    - M is positive definite matrix
    - If M = I, then distance reduces to Euclidean distance

- Curse of Dimensionality
    - Space volume grows exponentially with increase in dimension
        - Suppose inputs are uniformly distributed
        - As we move from square to cube, 10% edge covers less region

- Speed and Memory Requirements
    - Finding K nearest neighbors slow
    - KD Tree / LSH to speed up approximate neighbor calculation
        - KD Tree:
            - K dimensional binary search tree
        - LSH: Similar objects go to same hash bucket
            - Shingling, Minhash, LSH

- Open Set recognition
    - New classes appear at test time
        - Person Re-identification
        - Novelty Detection

- Learning Distance Matrix
    - Treat M is the distance matrix as a parameter
    - Large Margin Nearest Neighbors
    - Find M such that
        - $M = W^T W$ (Positive Definite)
        - Similar points have minimum distance
        - Dissimilar points are at least m units away (margin) 

- Deep Metric Learning
    - Reduce the curse of dimensionality
    - Project he input from high dimension space to lower dimension via embedding
    - Normalize the embedding
    - Compute the distance
        - Euclidean or Cosine, both are related
        - $|e_1 - e_2|^2 = |e1|^2 + |e_2|^2 - 2e_1 e_2$
        - Euclidean = 2 ( 1 - Cosine)
        - $\cos \theta = {a \dot b \over ||a|| ||b||}$
        - Derivation via trigonometry 
            - $\ cos \theta = a^2 + b ^ 2 - c^2 / 2 a b$
    - Learn an embedding function such that similar examples are close and dissimar examples are far
    - Loss functions:
        - Classification Losses
            - Only learn to push examples on correct side of the decision boundary
        - Pairwise Loss
            - Simaese Neural Network
            - Common Backbone to embed the inputs
            - $L(\theta, x_i, x_j) =  I \{y_i =y_j\} d(x_i, x_j) +  I \{y_i \ne y_j\} [m - d(x_i, x_j)]_+$
            - If same class, minimize the distance
            - If different class, maximize the distance with m margin (Hinge Loss)
        - Triplet Loss
            - In Pairwise Loss: positive and negative examples siloed
            - $L(\theta, x_i, x^+, x^-) = [m + d(x_i, x_+) - d(x_i, x_-)]_+$
            - Minimize the distance between anchor and positive
            - Maximize the distance between anchor and negative
            - m is the safety margin
            - Need to mine hard negative examples that are close to the positive pairs
            - Computationally slow
            - Use proxies to represent each class and speed up the training

- Kernel Density Estimation
    - Density Kernel
        - Domain: R
        - Range: R+
        - $\int K(x)dx = 1$
        - $K(-x) = K(x)$
        - $\int x K(x-x_n) dx = x_n$
    - Gaussian Kernel
        - $K(x) = {1 \over \sqrt{2\pi}} \exp(-{1\over2}x^2)$
        - RBF: Generalization to vector valued inputs
    - Bandwitdth
        - Parameter to control the width of the kernel
    - Density Estimation
        - Extend the concept of Gaussian Mixture models to the extreme
        - Each point acts as an individual cluster
            - Mean $x_n$ 
            - Constant variance
            - No covariance
            - Var-Cov matrix is $\sigma^2 I$
            - $p(x|D) = {1 \over N}\sum K_h(x - x_n)$
            - No model fitting is required
    - KDE vs KNN
        - KDE and KNN are closely related
        - Essentially, in KNN we grow the volume around a point till we encounter K neighbors. 
# Recommendation Systems

-   Feedback Types
    -   Explicit
        -   Ratings, likes, or dislikes provided directly by users
        -   Sparse, values often missing (not missing at random)
        -   Higher quality but harder to collect
    -   Implicit
        -   Inferred from user behavior - clicks, watches, purchases, etc.
        -   Sparse, positive-only feedback (absence doesn't mean dislike)
        -   Easier to collect but noisier
        
-   Collaborative Filtering
    -   Users collaborate indirectly to help recommend items
    -   User-based: Recommend items that similar users liked
        -   $\hat{y}_{ui} = \frac{\sum_{u'} \text{sim}(u, u') y_{u', i}}{\sum_{u'} \text{sim}(u, u')}$
    -   Item-based: Recommend items that are similar to what the user already likes
    -   Similarity calculations typically use common ratings between users/items
    -   Data sparsity makes similarity calculation challenging
    
    -   Matrix Factorization
        -   View the problem as matrix completion
        -   Predict all missing entries in the ratings matrix
        -   Optimization: $L = ||Z - Y||^2$ (for observed entries only)
        -   Break up Z into low-rank matrices: $Z = U^TV$
        -   U represents user embeddings, V represents item embeddings
        -   Can't use SVD directly due to missing values
        -   Use ALS (Alternating Least Squares): Estimate U given V, then V given U
        -   Add biases to account for user/item rating tendencies:
            -   $\hat{y}_{ui} = \mu + b_u + c_i + u_u^T v_i$
                -   $\mu$ is global average rating
                -   $b_u$ is user bias (rating tendency)
                -   $c_i$ is item bias (intrinsic quality)
                -   $u_u^T v_i$ is user-item interaction
            -   $L = \sum_{(u,i) \in \text{observed}} (y_{ui} - \hat{y}_{ui})^2 + \lambda(\sum ||u_u||^2 + \sum ||v_i||^2)$
            
    -   Probabilistic Matrix Factorization
        -   Bayesian approach to matrix factorization
        -   $p(Y=y_{ui}) = \mathcal{N}(\mu + b_u + c_i + u_u^T v_i, \sigma^2)$
        -   Can add priors on user/item vectors

-   Bayesian Personalized Ranking (BPR)
    -   For implicit feedback data
    -   Ranking approach: Model ranks items user interacted with (positive set) ahead of others (negative set)
    -   $p(y = (u,i,j) | \theta) = \sigma(f(u,i;\theta) - f(u,j;\theta))$
        -   $f(u,i;\theta)$ is predicted score for user u and item i
    -   Use hinge or logistic loss to estimate parameters
    -   Samples triplets (user, positive item, negative item) for training

-   Factorization Machines (FM)
    -   Generalization of matrix factorization that can incorporate side features
    -   Represent user-item pairs as feature vectors:
        -   $x = \text{concat}[\text{user\_features}, \text{item\_features}]$
    -   Model:
        -   $f(x) = \mu + \sum w_i x_i + \sum_{i<j} (v_i \cdot v_j) x_i x_j$
    -   The dot product term captures pairwise interactions between features
    -   Uses low-rank approximation to reduce parameter count
    -   Can easily incorporate context (time, location, etc.) and user/item metadata
    -   Loss functions:
        -   Explicit feedback: MSE or MAE
        -   Implicit feedback: Ranking loss (BPR)

-   Cold-Start Problem
    -   Difficult to generate predictions for new users or items with no history
    -   Solutions:
        -   Content-based approaches using item/user metadata
        -   Hybrid models combining collaborative and content-based methods
        -   Active learning to efficiently collect initial preferences
        -   Transfer learning from related domains

-   Exploration-Exploitation Tradeoff
    -   Counterfactual challenge: Users might like items they never see
    -   Recommendation systems must balance:
        -   Exploitation: Recommend items likely to be relevant based on history
        -   Exploration: Recommend items with uncertain but potentially high value
    -   Approaches:
        -   Multi-armed bandits (Thompson Sampling, UCB)
        -   Contextual bandits for personalized exploration
        -   Diversity promotion in recommendations 
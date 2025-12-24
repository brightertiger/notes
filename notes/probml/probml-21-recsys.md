# Recommendation Systems

Recommendation systems predict user preferences for items (movies, products, songs, etc.). They power personalization across the internet — from Netflix to Amazon to Spotify.

## The Big Picture

**The problem**: Users interact with a tiny fraction of available items. Can we predict what they'd like?

**Key challenge**: The user-item matrix is extremely sparse (>99% missing).

**Goal**: Fill in the missing entries (predict ratings) or rank items for each user.

---

## Types of Feedback

### Explicit Feedback

Users directly express preferences:
- Star ratings (1-5)
- Thumbs up/down
- Reviews

**Pros**: Clear signal about preferences.
**Cons**: Sparse (users rarely rate), not missing at random (users rate things they care about).

### Implicit Feedback

Inferred from user behavior:
- Clicks, views, purchases
- Time spent
- Add to cart

**Pros**: Abundant, always available.
**Cons**: Noisy, positive-only (absence doesn't mean dislike).

---

## Collaborative Filtering

### The Core Idea

Users "collaborate" to help recommend items:
- Users with similar preferences in the past will have similar preferences in the future
- Items liked by similar users are likely to be liked by the target user

### User-Based CF

For target user u and item i:
$$\hat{y}_{ui} = \frac{\sum_{u'} \text{sim}(u, u') \cdot y_{u'i}}{\sum_{u'} \text{sim}(u, u')}$$

**Steps**:
1. Find users similar to u (based on rating patterns)
2. Aggregate their ratings for item i

### Item-Based CF

For target user u and item i:
$$\hat{y}_{ui} = \frac{\sum_{i'} \text{sim}(i, i') \cdot y_{ui'}}{\sum_{i'} \text{sim}(i, i')}$$

**Steps**:
1. Find items similar to i (based on who rated them)
2. Aggregate user u's ratings for those items

**In practice**: Item-based often preferred (item similarities more stable than user similarities).

### Challenges

- **Sparsity**: Few common ratings for similarity calculation
- **Scalability**: Computing all pairwise similarities is expensive
- **Cold start**: No history for new users/items

---

## Matrix Factorization

### The Idea

The rating matrix R can be approximated as a product of low-rank matrices:
$$R \approx U \cdot V^T$$

Where:
- U is user matrix (N × K): Each row is a user's "embedding"
- V is item matrix (M × K): Each row is an item's "embedding"
- K << N, M (typically K = 10-200)

### Interpretation

Each dimension captures a latent "factor":
- Movie factors might capture: Action content, Romance level, Production year...
- User factors capture: Preference for action, romance, etc.

**Prediction**: $\hat{y}_{ui} = u_u^T v_i$ (dot product of embeddings)

### Training

Minimize squared error on observed ratings:
$$L = \sum_{(u,i) \in \text{observed}} (y_{ui} - u_u^T v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$$

**Note**: Can't use SVD directly (missing values). Use:
- **Alternating Least Squares (ALS)**: Fix U, solve for V; fix V, solve for U
- **SGD**: Stochastic gradient descent on observed entries

### Adding Biases

Users and items have inherent tendencies:
$$\hat{y}_{ui} = \mu + b_u + c_i + u_u^T v_i$$

Where:
- μ: Global average rating
- $b_u$: User bias (some users rate higher on average)
- $c_i$: Item bias (some items are generally liked more)

---

## Probabilistic Matrix Factorization

### Bayesian Approach

Model ratings as:
$$p(y_{ui} | u_u, v_i) = \mathcal{N}(\mu + b_u + c_i + u_u^T v_i, \sigma^2)$$

Add priors on embeddings:
$$u_u \sim \mathcal{N}(0, \sigma_u^2 I)$$
$$v_i \sim \mathcal{N}(0, \sigma_v^2 I)$$

**Benefits**:
- Principled handling of uncertainty
- Regularization from priors
- Can incorporate side information

---

## Bayesian Personalized Ranking (BPR)

### For Implicit Feedback

**Problem**: With implicit data, we only have positive examples.

**Approach**: Learn to rank positives above negatives.

**Assumption**: User prefers items they interacted with over items they didn't.

### The Loss

For triplet (user, positive item, negative item):
$$L = -\log \sigma(f(u, i^+) - f(u, i^-))$$

Where f is the prediction score (e.g., $u_u^T v_i$).

**Training**: Sample triplets and optimize.

---

## Factorization Machines

### Beyond Matrix Factorization

Matrix factorization only captures user-item interactions.

**Factorization Machines** generalize to any features:
$$f(x) = \mu + \sum_{j=1}^d w_j x_j + \sum_{j < k} (v_j \cdot v_k) x_j x_k$$

Where:
- x: Feature vector (one-hot user, one-hot item, plus any other features)
- v_j: Embedding for feature j

### Advantages

- Can incorporate **side information**: User demographics, item attributes, context (time, location)
- Handles **cold start** better
- Same framework for different feature types

### Connection to MF

When features are just user and item IDs:
$$f = \mu + b_u + c_i + u_u^T v_i$$

Exactly matrix factorization with biases!

---

## The Cold Start Problem

### The Challenge

New users or items have no interaction history.

### Solutions

**Content-based**: Use features instead of collaborative signal
- New user: Ask preferences, use demographics
- New item: Use item attributes, description

**Hybrid methods**: Combine collaborative and content-based

**Active learning**: Ask strategic questions to new users

**Transfer learning**: Leverage data from related domains

---

## Exploration-Exploitation Trade-off

### The Problem

If we only recommend what users already like, they never discover new interests.

**Counterfactual**: Users might love items they never see!

### Approaches

**Multi-Armed Bandits**:
- **Thompson Sampling**: Sample from posterior, act greedily
- **UCB**: Optimism in the face of uncertainty

**Contextual Bandits**: Personalized exploration based on user features

**Diversity**: Ensure recommendations aren't all the same type

---

## Deep Learning for RecSys

### Neural Collaborative Filtering

Replace dot product with neural network:
$$\hat{y}_{ui} = f_{neural}([u_u; v_i])$$

**Benefit**: Captures non-linear interactions.

### Sequential Recommendations

Model user's sequence of interactions with RNN/Transformer:
$$h_t = f(x_t, h_{t-1})$$

**Benefit**: Captures temporal dynamics.

### Two-Tower Models

Separate encoders for users and items:
- Fast serving (pre-compute item embeddings)
- Easy to add features

---

## Evaluation Metrics

### For Rating Prediction

- **RMSE**: $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$
- **MAE**: $\frac{1}{N}\sum |y - \hat{y}|$

### For Ranking

- **Precision@K**: Fraction of top-K recommendations that are relevant
- **Recall@K**: Fraction of relevant items in top-K
- **NDCG**: Accounts for position (higher rank = more important)
- **MAP**: Mean average precision

### Offline vs. Online

**Offline**: Historical data, fast to compute but may not reflect real preferences.

**Online (A/B testing)**: Real users, gold standard but expensive and slow.

---

## Summary

| Method | Key Idea | Best For |
|--------|----------|----------|
| **User-Based CF** | Similar users → similar items | Small, stable user base |
| **Item-Based CF** | Similar items | Most practical CF |
| **Matrix Factorization** | Low-rank approximation | General purpose |
| **BPR** | Learn to rank | Implicit feedback |
| **Factorization Machines** | Feature interactions | Side information |
| **Neural Methods** | Non-linear patterns | Large data, complex patterns |

### Practical Recommendations

1. **Start simple**: Matrix factorization with biases often hard to beat
2. **Add biases**: Critical for good performance
3. **Handle implicit correctly**: Use ranking losses, not rating losses
4. **Address cold start**: Hybrid methods or feature-based fallback
5. **Evaluate carefully**: Ranking metrics often more meaningful than RMSE
6. **Consider fairness**: Avoid filter bubbles, ensure diverse recommendations

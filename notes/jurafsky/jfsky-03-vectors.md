# Vector Semantics and Word Embeddings

How do we represent word meaning computationally? This chapter covers the evolution from sparse count-based vectors to dense neural embeddings — one of the most important advances in NLP.

## The Big Picture

**The Problem**: Computers need numerical representations of words.

**Key Insight** (Distributional Hypothesis):
> "You shall know a word by the company it keeps" — J.R. Firth

Words that appear in similar contexts have similar meanings.

**The Evolution**:
```
One-hot vectors → Count-based vectors → Neural embeddings
(sparse, no similarity)   (sparse, some similarity)   (dense, learned similarity)
```

---

## Challenges of Lexical Semantics

Why is meaning hard?

| Challenge | Example |
|-----------|---------|
| **Word forms** | sing, sang, sung (same lemma "sing") |
| **Polysemy** | "bank" = river bank or financial bank |
| **Synonymy** | couch ≈ sofa (same meaning) |
| **Relatedness** | coffee ~ cup (not synonyms, but related) |
| **Semantic frames** | "A bought from B" ≈ "B sold to A" |
| **Connotation** | "slender" vs. "skinny" (same denotation, different feeling) |

---

## Vector Space Models

### The Core Idea

Represent words as vectors in a high-dimensional space where:
- **Similar words** are **close together**
- **Dissimilar words** are **far apart**

### Document Vectors (Term-Document Matrix)

| | Doc1 | Doc2 | Doc3 |
|---|---|---|---|
| cat | 3 | 0 | 1 |
| dog | 2 | 4 | 0 |
| pet | 1 | 2 | 1 |

- **Rows**: Words (vocabulary of size V)
- **Columns**: Documents (D documents)
- **Cell**: Count of word in document

**Use case**: Information retrieval (find similar documents).

### Word Vectors (Term-Term Matrix)

| | cat | dog | pet | food |
|---|---|---|---|---|
| cat | - | 15 | 20 | 8 |
| dog | 15 | - | 25 | 12 |
| pet | 20 | 25 | - | 10 |

- **Rows and Columns**: Words
- **Cell**: Co-occurrence count (how often words appear together)

**Result**: Each word is a V-dimensional vector.

---

## Measuring Similarity

### Cosine Similarity

Normalized dot product — measures angle between vectors:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}$$

**Interpretation**:
- cos = 1: Identical direction (most similar)
- cos = 0: Perpendicular (unrelated)
- cos = -1: Opposite direction (antonyms, in some cases)

**Why cosine over Euclidean?**
- Handles different vector magnitudes
- A long document and short document can still be similar

### For Unit Vectors

When vectors are normalized (length 1):
$$||\vec{a} - \vec{b}||^2 = 2(1 - \cos\theta)$$

Euclidean distance and cosine become equivalent!

---

## TF-IDF Weighting

Raw counts have problems:
- Common words ("the", "is") dominate
- Rare but meaningful words get drowned out

### Term Frequency (TF)

How often does word appear in document?

**Raw TF**: $\text{tf}_{t,d} = \text{count}(t, d)$

**Log TF** (dampens large counts):
$$\text{tf}_{t,d} = \log(1 + \text{count}(t, d))$$

### Inverse Document Frequency (IDF)

How rare is the word across documents?

$$\text{idf}_t = \log\left(\frac{N}{\text{df}_t}\right)$$

Where:
- N = total number of documents
- df_t = number of documents containing term t

**Effect**: Common words (low IDF) get downweighted.

### TF-IDF

Combine both:
$$w_{t,d} = \text{tf}_{t,d} \times \text{idf}_t$$

**High TF-IDF**: Word appears often in this document but rarely overall → distinctive!

---

## Pointwise Mutual Information (PMI)

### The Intuition

Are two words appearing together more than we'd expect by chance?

$$\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) \cdot P(y)}$$

**Interpretation**:
- PMI > 0: Words co-occur more than expected (associated)
- PMI = 0: Words co-occur as expected (independent)
- PMI < 0: Words co-occur less than expected (avoid each other)

### From Counts

$$\text{PMI}(x, y) = \log_2 \frac{\text{count}(x, y) \cdot N}{\text{count}(x) \cdot \text{count}(y)}$$

### Positive PMI (PPMI)

Negative PMI values are unreliable (rare events).

$$\text{PPMI}(x, y) = \max(0, \text{PMI}(x, y))$$

---

## From Sparse to Dense: Word2Vec

### The Problem with Count Vectors

- **Very high dimensional** (vocabulary size)
- **Very sparse** (mostly zeros)
- **No generalization** between similar words

### The Neural Solution

Learn **dense, low-dimensional** vectors (typically 100-300 dimensions).

**Key properties**:
- Similar words have similar vectors
- Relationships are captured geometrically

### Static vs. Contextual Embeddings

| Type | Same word = same vector? | Examples |
|------|--------------------------|----------|
| **Static** | Yes | Word2Vec, GloVe, FastText |
| **Contextual** | No (depends on context) | ELMo, BERT, GPT |

---

## Skip-Gram with Negative Sampling (SGNS)

The most popular Word2Vec algorithm.

### The Task

Given a target word, predict surrounding context words.

**Example**: "The quick **brown** fox jumps"
- Target: "brown"
- Context (window=2): "The", "quick", "fox", "jumps"

### Training Setup

1. **Positive examples**: (target, context) pairs from real text
2. **Negative examples**: (target, random_word) pairs — fake associations

### The Objective

Maximize probability of real pairs, minimize probability of fake pairs:

$$L = \log \sigma(v_w \cdot v_c) + \sum_{i=1}^{k} \mathbb{E}_{c_i \sim P_n}[\log \sigma(-v_w \cdot v_{c_i})]$$

Where:
- $\sigma$ is sigmoid function
- $v_w$ is target word vector
- $v_c$ is context word vector
- k is number of negative samples (typically 5-20)

### Negative Sampling Distribution

Don't sample uniformly — would get too many rare words.

$$P(w) \propto \text{freq}(w)^{0.75}$$

The 0.75 power smooths the distribution (gives rare words a better chance than pure frequency).

### Two Embeddings Per Word

Each word has:
- **Target embedding**: When it's the center word
- **Context embedding**: When it appears in context

Final embedding is often their sum or average.

---

## Enhancements and Variations

### FastText (Subword Embeddings)

**Problem**: What about unknown words like "ungooglable"?

**Solution**: Represent words as bag of character n-grams.

"where" → {<wh, whe, her, ere, re>}

Word vector = sum of n-gram vectors.

**Benefit**: Can handle any word, even unseen ones!

### GloVe (Global Vectors)

Combines advantages of count-based and neural methods.

Uses global co-occurrence statistics + optimization:
$$J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Often comparable to Word2Vec in practice.

---

## Word Analogies

Famous Word2Vec property:

**"king" - "man" + "woman" ≈ "queen"**

Find word that completes analogy a:b :: a':?

$$b' = \arg\min_{x} \text{distance}(x, b - a + a')$$

**Works for**:
- Gender: king:queen :: man:woman
- Capitals: Paris:France :: Tokyo:Japan
- Tense: walking:walked :: swimming:swam

---

## Bias in Word Embeddings

### The Problem

Word embeddings learn biases present in training data.

**Examples**:
- "doctor" closer to "man" than "woman"
- "homemaker" closer to "woman" than "man"
- Names associated with certain ethnic groups linked to negative words

### Types of Harm

**Allocation harm**: System makes unfair decisions
- Resume screening favoring male-associated names

**Representation harm**: Reinforces stereotypes
- Search results, autocomplete suggestions

### Mitigation Strategies

- Debias during training or post-hoc
- Careful data curation
- Evaluation for fairness

---

## Summary

| Representation | Pros | Cons |
|----------------|------|------|
| **Count-based (TF-IDF)** | Interpretable, simple | Sparse, high-dimensional |
| **PMI** | Captures associations | Sparse, noisy for rare words |
| **Word2Vec** | Dense, captures analogy | Static, no context |
| **FastText** | Handles OOV words | Still static |
| **Contextual** | Word sense disambiguation | Computationally expensive |

### Key Takeaways

1. **Words can be represented as vectors** in semantic space
2. **Distributional similarity** = semantic similarity
3. **Dense embeddings** outperform sparse for most tasks
4. **Context matters** — motivates contextual embeddings (BERT, etc.)
5. **Beware of biases** inherited from training data

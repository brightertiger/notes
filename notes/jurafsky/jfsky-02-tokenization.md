# N-Grams and Language Models

Language models assign probabilities to sequences of words. They answer: "How likely is this sentence?" This fundamental capability underlies spell checking, machine translation, speech recognition, and text generation.

## The Big Picture

**Key Question**: What's the probability of a word sequence?

$$P(\text{"the cat sat on the mat"})$$

**Why This Matters**:
- Spell checking: P("the cat") > P("the kat")
- Machine translation: Choose more fluent translation
- Speech recognition: Distinguish "recognize speech" from "wreck a nice beach"
- Text generation: Sample likely continuations

---

## Language Model Fundamentals

### Joint Probability of Words

We want to compute:
$$P(w_1, w_2, ..., w_n)$$

Using the **chain rule** of probability:
$$P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_1,w_2) \times ... \times P(w_n|w_1,...,w_{n-1})$$

More compactly:
$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_{1:i-1})$$

**Problem**: As sequences get longer, we need infinitely many parameters!

### The Markov Assumption

**Key insight**: Approximate by using only recent history.

For a **bigram model** (2-gram):
$$P(w_n | w_1, w_2, ..., w_{n-1}) \approx P(w_n | w_{n-1})$$

**Assumption**: The next word depends only on the previous word.

### N-Gram Models

| Model | Conditioning Context | Example |
|-------|---------------------|---------|
| Unigram | None | P(the) |
| Bigram | Previous 1 word | P(cat \| the) |
| Trigram | Previous 2 words | P(sat \| the cat) |
| 4-gram | Previous 3 words | P(on \| the cat sat) |

**Trade-off**: Larger n = better context but more parameters and sparser data.

---

## Estimating N-Gram Probabilities

### Maximum Likelihood Estimation

Use **relative frequency** (counting):

**Bigram probability**:
$$P(w_n | w_{n-1}) = \frac{\text{count}(w_{n-1}, w_n)}{\text{count}(w_{n-1})}$$

**Example**: Computing P(sat | the) from a corpus:
- Count "the sat" occurrences: 100
- Count "the ___" occurrences: 10,000
- P(sat | the) = 100/10,000 = 0.01

### Handling Sentence Boundaries

Add special tokens:
- **\<s\>**: Beginning of sentence (BOS)
- **\</s\>**: End of sentence (EOS)

**Example**: "\<s\> the cat sat \</s\>"

This allows us to model:
- P(the | \<s\>) — how likely is "the" to start a sentence?
- P(\</s\> | sat) — how likely is "sat" to end a sentence?

### Practical Note: Log Probabilities

Multiplying many small probabilities → numerical underflow!

**Solution**: Work in log space:
$$\log(p_1 \times p_2) = \log(p_1) + \log(p_2)$$

Add log probabilities instead of multiplying probabilities.

---

## Perplexity

### What Is Perplexity?

The standard metric for evaluating language models.

**Definition**:
$$\text{PP}(W) = P(w_1, w_2, ..., w_n)^{-1/n}$$

Equivalently:
$$\text{PP}(W) = \sqrt[n]{\prod_{i=1}^{n} \frac{1}{P(w_i | w_{1:i-1})}}$$

### Intuition: Weighted Branching Factor

Perplexity ≈ average number of equally likely choices at each step.

**Example**:
- Perplexity of 100 ≈ model is choosing between 100 equally likely words
- Perplexity of 10 ≈ model is choosing between 10 equally likely words

**Lower perplexity = better model** (more confident predictions).

### Connection to Information Theory

Perplexity relates to **entropy**:
$$\text{PP}(W) = 2^{H(W)}$$

Where entropy H measures uncertainty:
$$H(W) = -\frac{1}{n} \log_2 P(w_1, ..., w_n)$$

### Important Caveats

**Perplexities are only comparable when**:
- Using the same vocabulary
- Using the same test set

Adding rare words to vocabulary increases perplexity (more choices).

---

## The Unknown Word Problem

### The Problem

What if we see a word we've never seen before?
$$P(\text{unigoogleable} | w_{n-1}) = 0$$

Zero probability breaks everything:
- Product becomes zero
- Perplexity becomes infinite

### Solution: \<UNK\> Token

1. Define a vocabulary (e.g., most frequent 50,000 words)
2. Replace all other words with \<UNK\>
3. Treat \<UNK\> as just another word

**Training**: "I saw a brachiosaurus" → "I saw a \<UNK\>"
**Testing**: Same replacement

**Caveat**: Smaller vocabulary = lower perplexity (fewer choices). Not always fair to compare!

---

## Smoothing

### The Zero Probability Problem

Even with \<UNK\>, we'll see **new n-grams**:
$$P(\text{cat} | \text{the green}) = 0 \text{ (if never seen together)}$$

**Smoothing** redistributes probability mass to unseen events.

### Laplace (Add-One) Smoothing

**Idea**: Pretend we saw everything at least once.

**Unigram**:
$$P(w_i) = \frac{\text{count}(w_i) + 1}{N + V}$$

**Bigram**:
$$P(w_i | w_j) = \frac{\text{count}(w_j, w_i) + 1}{\text{count}(w_j) + V}$$

Where V is vocabulary size.

**Problem**: Add-1 is too aggressive — steals too much from seen events.

**Solution**: Add-k smoothing (k < 1, e.g., k = 0.01).

### Backoff

**Idea**: If no evidence for trigram, use bigram. If no bigram, use unigram.

$$P_{BO}(w_i|w_{i-2},w_{i-1}) = \begin{cases} 
P(w_i|w_{i-2},w_{i-1}) & \text{if count > 0} \\
\alpha \cdot P_{BO}(w_i|w_{i-1}) & \text{otherwise}
\end{cases}$$

Where α is a normalization factor.

### Interpolation

**Idea**: Always mix all n-gram levels.

$$P(w_i | w_{i-2}, w_{i-1}) = \lambda_1 P(w_i) + \lambda_2 P(w_i | w_{i-1}) + \lambda_3 P(w_i | w_{i-2}, w_{i-1})$$

Where $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

Learn λ values from held-out data.

### Kneser-Ney Smoothing

**State-of-the-art for n-gram models**.

**Key insight**: Use *continuation probability* — how likely is a word to appear in new contexts?

$$P_{KN}(w_i | w_{i-1}) = \frac{\max(\text{count}(w_{i-1}, w_i) - d, 0)}{\sum_v \text{count}(w_{i-1}, v)} + \lambda(w_{i-1}) P_{continuation}(w_i)$$

**Intuition**: 
- Words like "Francisco" have high count but appear after "San" — low continuation probability
- Words like "the" appear in many contexts — high continuation probability

---

## Efficiency Considerations

N-gram models can be huge! Practical tricks:

| Technique | Purpose |
|-----------|---------|
| **Quantization** | Store probabilities with fewer bits |
| **Tries** | Efficient storage and lookup |
| **String hashing** | Reduce memory for n-grams |
| **Bloom filters** | Fast membership testing |
| **Stupid Backoff** | Simple, fast approximation |

---

## Summary

| Concept | Key Idea |
|---------|----------|
| **Language Model** | Assign probabilities to word sequences |
| **N-gram** | Approximate using last n-1 words |
| **MLE** | Estimate from counts (relative frequency) |
| **Perplexity** | Model evaluation metric (lower = better) |
| **Smoothing** | Handle unseen n-grams |
| **Kneser-Ney** | Best n-gram smoothing technique |

### The Bigger Picture

N-gram models are:
- **Simple and interpretable**
- **Capture local syntax** (word order)
- **Fast** to train and use

But they have **limitations**:
- **Fixed context** (can't capture long-range dependencies)
- **Sparse data** (even trigrams need lots of data)
- **No semantic understanding** (just counting patterns)

This motivates **neural language models** (covered in later chapters).

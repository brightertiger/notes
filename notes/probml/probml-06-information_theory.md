# Information Theory

Information theory provides mathematical tools for quantifying information, uncertainty, and the relationships between random variables. Originally developed for communication systems, it's now fundamental to machine learning.

## The Big Picture

Information theory answers questions like:
- How much uncertainty is in a distribution?
- How different are two distributions?
- How much does knowing X tell us about Y?

These concepts are central to understanding loss functions, model evaluation, and feature selection.

---

## Entropy

### Definition

**Entropy** measures the uncertainty or "unpredictability" of a random variable:

$$H(X) = -\sum_x p(x) \log p(x) = -\mathbb{E}[\log p(X)]$$

**Intuition**: How many bits do we need, on average, to encode samples from this distribution?

### Key Properties

- **Non-negative**: $H(X) \geq 0$
- **Maximum for uniform distribution**: If all outcomes equally likely, uncertainty is maximized
- **Minimum for deterministic**: $H(X) = 0$ when outcome is certain (Dirac delta)

### Examples

**Fair coin**: $H = -\frac{1}{2}\log\frac{1}{2} - \frac{1}{2}\log\frac{1}{2} = 1$ bit

**Biased coin (p=0.9)**: $H = -0.9\log 0.9 - 0.1\log 0.1 \approx 0.47$ bits

**More predictable → lower entropy!**

---

## Cross-Entropy

### Definition

Cross-entropy measures the average number of bits needed to encode data from distribution p using a code optimized for distribution q:

$$H(p, q) = -\sum_x p(x) \log q(x)$$

**Intuition**: How well does q approximate p?

### Key Properties

- $H(p, q) \geq H(p)$ (equality when p = q)
- Cross-entropy is what we minimize in classification!

### Connection to Machine Learning

When training a classifier:
- **p** = true distribution (one-hot labels)
- **q** = predicted distribution (softmax outputs)

**Cross-entropy loss**:
$$\mathcal{L} = -\sum_c y_c \log \hat{y}_c$$

For one-hot labels, this simplifies to: $-\log \hat{y}_{true}$

---

## Joint and Conditional Entropy

### Joint Entropy

Uncertainty about both X and Y together:
$$H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$$

### Conditional Entropy

Remaining uncertainty about Y after observing X:
$$H(Y | X) = \sum_x p(x) H(Y | X = x) = -\sum_{x,y} p(x,y) \log p(y|x)$$

### Chain Rule

$$H(X, Y) = H(X) + H(Y | X)$$

**Intuition**: Total uncertainty = uncertainty in X + remaining uncertainty in Y given X.

---

## Perplexity

### Definition

Perplexity is the exponentiated cross-entropy:
$$\text{Perplexity}(p, q) = 2^{H(p, q)}$$

Or for a sequence of N tokens:
$$\text{Perplexity} = \sqrt[N]{\prod_{i=1}^N \frac{1}{p(x_i)}}$$

**Interpretation**: The weighted average number of choices (branching factor) the model is uncertain between.

### Use in Language Models

- **Lower perplexity** = better model
- Perplexity of 1 = perfect prediction
- Perplexity of V (vocabulary size) = random guessing

**Example**: Perplexity of 50 means the model is, on average, choosing between 50 equally likely next words.

---

## KL Divergence

### Definition

**Kullback-Leibler divergence** measures how different distribution q is from distribution p:

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

**Intuition**: Extra bits needed when using code for q to encode data from p.

### Key Properties

- **Non-negative**: $D_{KL}(p \| q) \geq 0$ (Gibbs' inequality)
- **Zero iff p = q**: Perfect match
- **Asymmetric**: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general
- **Not a true distance** (asymmetric, no triangle inequality)

### Connection to MLE

When we minimize NLL, we're minimizing:
$$\text{NLL} = -\frac{1}{N}\sum_i \log q(x_i)$$

If samples come from empirical distribution $\hat{p}$:
$$\text{NLL} = H(\hat{p}, q) = H(\hat{p}) + D_{KL}(\hat{p} \| q)$$

Since $H(\hat{p})$ is constant, **minimizing NLL = minimizing KL divergence**!

### Forward vs. Reverse KL

**Forward KL** $D_{KL}(p \| q)$: p is the reference
- Mode-covering: q tries to cover all of p's mass
- Penalizes q for missing modes of p

**Reverse KL** $D_{KL}(q \| p)$: q is the reference
- Mode-seeking: q concentrates on modes of p
- Okay to miss some modes, but penalizes placing mass where p has none

---

## Mutual Information

### Definition

How much does knowing X tell us about Y?

$$I(X; Y) = D_{KL}(p(x, y) \| p(x)p(y))$$

Or equivalently:
$$I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)$$

**Intuition**: Reduction in uncertainty about X from knowing Y.

### Key Properties

- **Symmetric**: $I(X; Y) = I(Y; X)$
- **Non-negative**: $I(X; Y) \geq 0$
- **Zero iff independent**: $I(X; Y) = 0 \Leftrightarrow X \perp Y$

### As Generalized Correlation

Mutual information captures **any** dependence (not just linear), making it more general than Pearson correlation.

### Data Processing Inequality

If X → Y → Z forms a Markov chain:
$$I(X; Z) \leq I(X; Y)$$

**Implication**: Processing cannot increase information. You can only lose information through transformations!

---

## Fano's Inequality

### Statement

For any estimator $\hat{X}$ of X based on Y:
$$H(X | Y) \leq H(P_e) + P_e \log(|X| - 1)$$

Where $P_e = P(\hat{X} \neq X)$ is the error probability.

### Implications

- **Lower bound on error**: If $H(X|Y)$ is high, error must be high
- **Feature selection**: Features with high mutual information with the target reduce classification error

---

## Applications in ML

### Loss Functions

**Cross-entropy loss** minimizes KL divergence between true and predicted distributions.

### Variational Inference

Approximate intractable posterior by minimizing KL divergence.

### Information Bottleneck

Find representations that maximally compress input while retaining relevant information about the output.

### Data Augmentation

Spreads probability mass over larger input space, reducing overfitting.

---

## Summary

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Entropy** | $H(X) = -\mathbb{E}[\log p(X)]$ | Uncertainty in X |
| **Cross-Entropy** | $H(p, q) = -\mathbb{E}_p[\log q]$ | Bits to encode p using q |
| **KL Divergence** | $D_{KL}(p\|q) = H(p,q) - H(p)$ | Extra bits; difference between distributions |
| **Mutual Information** | $I(X;Y) = H(X) - H(X\|Y)$ | Information shared between X and Y |
| **Perplexity** | $2^{H(p,q)}$ | Effective vocabulary size |

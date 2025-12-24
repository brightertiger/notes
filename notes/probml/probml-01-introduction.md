# Introduction to Machine Learning

Machine learning is the science of getting computers to learn from data without being explicitly programmed. This chapter introduces the fundamental concepts, problem types, and challenges that define the field.

## What is Machine Learning?

**Tom Mitchell's Definition**:
> A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

**In plain English**: A machine learning system gets better at a task as it sees more data.

**Example**: A spam filter
- **Task (T)**: Classify emails as spam or not spam
- **Experience (E)**: A dataset of labeled emails
- **Performance (P)**: Accuracy on new emails

---

## Supervised Learning

In supervised learning, we have input-output pairs and want to learn the mapping between them.

### The Setup

- **Inputs** (X): Also called features, covariates, or predictors
  - Example: Pixel values of an image, words in an email
- **Outputs** (Y): Also called labels, targets, or responses
  - Example: "cat" or "dog", spam or not spam

**Goal**: Learn a function $f: X \rightarrow Y$ that generalizes to new, unseen examples.

### Classification

When the output is a **discrete category**:

$$L(\theta) = \frac{1}{N}\sum_{i=1}^N I\{y_i \neq f(x_i, \theta)\}$$

This is the **misclassification rate** — the fraction of examples we get wrong.

**Key concepts**:
- **Empirical Risk**: Average loss on training data
- **Empirical Risk Minimization (ERM)**: Find parameters that minimize training loss
- **Generalization**: The real goal is to perform well on *new* data, not just training data

### Dealing with Uncertainty

Models can't predict with 100% certainty. There are two types of uncertainty:

**Model Uncertainty (Epistemic)**
- Arises from lack of knowledge about the true mapping
- Can be reduced with more data
- Example: We don't know if a blurry image is a cat or dog

**Data Uncertainty (Aleatoric)**
- Arises from inherent randomness in the data
- Cannot be reduced even with infinite data
- Example: A coin flip is inherently random

### Probabilistic Predictions

Instead of just predicting a class, predict a probability distribution over classes:

$$p(y | x, \theta)$$

**Why probabilities?**
- Quantify confidence
- Enable better decision making
- Allow principled handling of uncertainty

**Negative Log-Likelihood (NLL)**:

$$\text{NLL}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log p(y_i | f(x_i, \theta))$$

Minimizing NLL is equivalent to **Maximum Likelihood Estimation (MLE)** — finding parameters that make the observed data most probable.

### Regression

When the output is a **continuous value**:

$$L(\theta) = \frac{1}{N}\sum_{i=1}^N (y_i - f(x_i, \theta))^2$$

This is **Mean Squared Error (MSE)** — the average squared difference between predictions and true values.

**Connection to Probability**: If we assume Gaussian noise:

$$p(y | x, \theta) = \mathcal{N}(y | f(x, \theta), \sigma^2)$$

Then minimizing NLL is equivalent to minimizing MSE!

### Types of Regression Models

| Model | Description | Flexibility |
|-------|-------------|-------------|
| **Linear** | $f(x) = w^Tx + b$ | Low |
| **Polynomial** | Includes $x^2, x^3$, etc. | Medium |
| **Neural Networks** | Nested nonlinear functions | High |

---

## Overfitting and Generalization

### The Overfitting Problem

A model that perfectly fits training data but fails on new data is **overfitting**. It has memorized the training set rather than learning the underlying pattern.

**Signs of overfitting**:
- Training error much lower than test error
- Model is very complex relative to data size
- Model captures noise as if it were signal

### Understanding the Errors

**Population Risk**: Theoretical expected loss on the true data generating process
$$R(\theta) = \mathbb{E}_{(x,y) \sim p^*}[L(y, f(x, \theta))]$$

**Empirical Risk**: Average loss on training data
$$\hat{R}(\theta) = \frac{1}{N}\sum_{i=1}^N L(y_i, f(x_i, \theta))$$

**Generalization Gap**: Difference between population and empirical risk
$$\text{Gap} = R(\theta) - \hat{R}(\theta)$$

A large gap indicates overfitting.

### The U-Shaped Test Error Curve

```
Error
  │    
  │  ╲                   ╱ Training Error
  │   ╲      ___________╱  (keeps decreasing)
  │    ╲____╱
  │     
  │         ╱─────────╲
  │        ╱    Test   ╲
  │───────╱    Error    ╲───── (U-shaped)
  │       
  └─────────────────────────→ Model Complexity
     Simple          Complex
```

- **Underfitting** (left): Model too simple, high bias
- **Sweet spot** (middle): Good balance
- **Overfitting** (right): Model too complex, high variance

---

## No Free Lunch Theorem

**There is no single best model that works for all problems.**

Every model makes assumptions about the data. When those assumptions match reality, the model works well. When they don't, it fails.

**Implication**: Understanding your problem domain is crucial for choosing the right model.

---

## Unsupervised Learning

In unsupervised learning, we only have inputs X — no labels.

**Goal**: Discover hidden structure in data.

### Common Tasks

**Clustering**: Group similar data points together
- Example: Customer segmentation, document grouping

**Dimensionality Reduction**: Find lower-dimensional representations
- Example: Compress images while preserving important information

**Density Estimation**: Model the probability distribution $p(x)$
- Example: Anomaly detection (low probability = anomalous)

**Self-Supervised Learning**: Create proxy tasks from unlabeled data
- Example: Predict missing words in text (BERT), predict next frame in video

### Evaluation Challenge

Without labels, how do we evaluate? Common approaches:
- Likelihood of held-out data
- Performance on downstream tasks
- Human evaluation of quality

---

## Reinforcement Learning

An **agent** learns to interact with an **environment** to maximize cumulative **reward**.

**Key differences from supervised learning**:
- No explicit labels — only reward signals
- Rewards are often delayed (sparse feedback)
- Agent's actions affect future states (sequential decision making)

**Analogy**: 
- Supervised learning = learning with a teacher who gives correct answers
- Reinforcement learning = learning with a critic who only says "good" or "bad"

---

## Data Preprocessing

### Text Data

Raw text needs transformation before ML models can process it.

**Bag of Words (BoW)**
- Represent document as vector of word counts
- Loses word order but captures content

**Problem**: Common words ("the", "a") dominate counts.

**TF-IDF** (Term Frequency - Inverse Document Frequency):
$$\text{TF-IDF} = \log(1 + \text{TF}) \times \text{IDF}$$

Where:
- TF = term frequency (how often word appears in document)
- IDF = $\log\frac{N}{1 + \text{DF}}$ (inverse of how many documents contain the word)

**Effect**: Downweight common words, upweight distinctive words.

**Word Embeddings**: Map words to dense vectors that capture semantic meaning
- Similar words have similar vectors
- "king" - "man" + "woman" ≈ "queen"

**Handling Unknown Words**:
- UNK token: Replace rare/unseen words with a special token
- Subword units (BPE): Break words into common pieces

### Missing Data

How data is missing matters!

| Type | Description | Example | Handling |
|------|-------------|---------|----------|
| **MCAR** | Missing Completely At Random | Random sensor failures | Easier to handle |
| **MAR** | Missing At Random (depends on observed data) | Older people less likely to report income | Model the missingness |
| **NMAR** | Not Missing At Random | Sick people skip health surveys | Most challenging |

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **ML Definition** | Learning improves with experience |
| **Supervised Learning** | Learn input-output mapping from labeled data |
| **Classification** | Predict discrete categories |
| **Regression** | Predict continuous values |
| **Probabilistic View** | Quantify uncertainty with probability distributions |
| **Overfitting** | Memorizing training data instead of learning patterns |
| **Generalization** | The ultimate goal: perform well on new data |
| **Unsupervised Learning** | Find structure without labels |
| **RL** | Learn from reward signals through interaction |

The probabilistic perspective unifies these concepts — learning is inference under uncertainty.

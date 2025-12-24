# Sequence Architectures: RNNs, LSTMs, and Attention

Language is inherently sequential. This chapter covers neural architectures designed to process sequences: from basic RNNs to LSTMs to the attention mechanism that revolutionized NLP.

## The Big Picture

**The Problem**: Language has dependencies across arbitrary distances.
- "The **cat** that I saw yesterday **was** cute" (subject-verb agreement)
- Standard feedforward networks have fixed input size

**The Solution**: Architectures with **memory** that process sequences step by step.

**The Evolution**:
```
FFNNs → RNNs → LSTMs/GRUs → Attention → Transformers
(fixed context)  (memory)  (better memory)  (direct connections)
```

---

## Why Not Feedforward Networks?

**Limitation**: Fixed context window.

A bigram FFNN can only see the previous word. But language has long-range dependencies:
- "The students who did well on the exam **were** happy"
- Verb agrees with "students", not "exam"

**We need**: Variable-length context.

---

## Recurrent Neural Networks (RNNs)

### The Core Idea

Add a **recurrent connection** — the hidden state from the previous step feeds into the current step.

$$h_t = g(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**The hidden state $h_t$ acts as memory!**

### Intuition

Think of reading a sentence word by word:
- You update your understanding as each word arrives
- Your current understanding depends on what you've read so far
- That's exactly what $h_t$ does

### Training: Backpropagation Through Time (BPTT)

1. **Unroll** the network across time steps
2. **Forward pass**: Compute all hidden states and outputs
3. **Backward pass**: Compute gradients through the unrolled graph
4. **Update**: Sum gradients for shared weights

**For long sequences**: Use truncated BPTT (limit how far back gradients flow).

### RNN Language Model

$$e_t = E x_t \quad \text{(word embedding)}$$
$$h_t = g(W_{hh} h_{t-1} + W_{he} e_t)$$
$$P(w_{t+1}) = \text{softmax}(W_{hy} h_t)$$

**Training**: Teacher forcing — use true previous word, not predicted word.

**Weight tying**: Share parameters between input embedding E and output layer.

---

## RNN Task Variants

### Sequence Labeling (Many-to-Many, aligned)

**Task**: Label each token (NER, POS tagging).

```
Input:  John  loves  New   York
Output: B-PER O      B-LOC I-LOC
```

Predict at each step based on hidden state.

### Sequence Classification (Many-to-One)

**Task**: Classify entire sequence (sentiment analysis).

```
Input:  This movie was great!
Output: POSITIVE
```

Use final hidden state (or pooled states) for classification.

### Sequence Generation (One-to-Many or Many-to-Many)

**Task**: Generate text.

```
Input:  <BOS>
Output: The cat sat on the mat <EOS>
```

Autoregressive: Each output becomes next input.

---

## RNN Architectures

### Stacked RNNs

Multiple RNN layers:
```
Layer 3: h3_t = f(h3_{t-1}, h2_t)
Layer 2: h2_t = f(h2_{t-1}, h1_t)  
Layer 1: h1_t = f(h1_{t-1}, x_t)
```

**Benefit**: Different abstraction levels at each layer.

### Bidirectional RNNs

Process sequence both ways:
- Forward: $\vec{h}_t$ (left to right)
- Backward: $\overleftarrow{h}_t$ (right to left)
- Combined: $h_t = [\vec{h}_t; \overleftarrow{h}_t]$

**Benefit**: Each position has access to full context.

**Limitation**: Can't use for autoregressive generation (need future that doesn't exist yet).

---

## The Vanishing Gradient Problem

### The Problem

Gradients shrink exponentially as they flow backward through time:

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

If $\frac{\partial h_t}{\partial h_{t-1}} < 1$ consistently → **vanishing gradients**.

**Consequence**: RNNs struggle to learn long-range dependencies.

### Why It Happens

- Sigmoid derivative: max 0.25
- Tanh derivative: max 1.0
- Repeated multiplication through many steps → exponential decay

### Solutions

1. **Gradient clipping** (for exploding gradients)
2. **Better architectures**: LSTM, GRU
3. **Skip connections**: Allow gradients to flow directly

---

## LSTM (Long Short-Term Memory)

### The Innovation

Add **explicit memory management** through **gates**.

Two types of state:
- **Cell state $c_t$**: Long-term memory (conveyor belt)
- **Hidden state $h_t$**: Working memory / output

### The Three Gates

| Gate | Purpose | Controls |
|------|---------|----------|
| **Forget** | What to erase from memory | $f_t$ |
| **Input** | What new info to add | $i_t$ |
| **Output** | What to expose as output | $o_t$ |

### LSTM Equations

**Forget gate** (what to keep from old memory):
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

**Input gate** (how much of new info to add):
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

**Candidate values** (new info to potentially add):
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

**Cell state update** (the key equation!):
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate** (what to output):
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

**Hidden state**:
$$h_t = o_t \odot \tanh(c_t)$$

### Why LSTM Works

The cell state update is **additive**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Gradients can flow through unchanged (when forget gate ≈ 1).

---

## GRU (Gated Recurrent Unit)

Simplified LSTM with fewer parameters:
- Combines forget and input gates into **update gate**
- No separate cell state

Often performs comparably to LSTM with less computation.

---

## Attention Mechanism

### The Bottleneck Problem

In encoder-decoder models, all information must pass through a fixed-size vector.

**Problem**: Information gets lost for long sequences.

### The Attention Solution

Let the decoder **look at all encoder states** when making each prediction.

### How Attention Works

For each decoder step:

1. **Score**: Compute similarity between decoder state and each encoder state
   $$e_{ij} = \text{score}(s_{i-1}, h_j)$$

2. **Normalize**: Convert scores to weights (softmax)
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

3. **Combine**: Weighted sum of encoder states
   $$c_i = \sum_j \alpha_{ij} h_j$$

4. **Use**: Context vector informs prediction
   $$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

### Scoring Functions

| Type | Formula |
|------|---------|
| Dot product | $s^T h$ |
| Scaled dot product | $\frac{s^T h}{\sqrt{d}}$ |
| MLP | $v^T \tanh(W_1 s + W_2 h)$ |

### Benefits of Attention

1. **Long-range dependencies**: Direct connections regardless of distance
2. **Interpretability**: Attention weights show what the model focuses on
3. **Alignment**: Helpful for translation (which source words map to which target words)

---

## Self-Attention and Transformers

### From Attention to Self-Attention

**Regular attention**: Query from decoder, keys/values from encoder.

**Self-attention**: Query, keys, values all from same sequence.

$$\text{output}_i = \text{Attention}(x_i, (x_1, x_1), (x_2, x_2), ..., (x_n, x_n))$$

Each position attends to all positions in the same sequence.

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- Q = queries (what I'm looking for)
- K = keys (what I offer for matching)
- V = values (what I actually provide)
- $\sqrt{d_k}$ scaling prevents softmax saturation

### Multi-Head Attention

Run multiple attention operations in parallel:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

**Benefit**: Each head can capture different types of relationships.

### Positional Encoding

Attention is **permutation invariant** — doesn't know word order!

**Solution**: Add position information to embeddings.

**Sinusoidal encoding**:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

### BERT Architecture

**Base model**:
- 12 attention heads
- 12 layers
- 768 hidden size (12 × 64)
- 110M parameters

---

## Summary

| Architecture | Memory | Long-range | Parallelizable |
|--------------|--------|------------|----------------|
| **RNN** | Hidden state | Limited | No |
| **LSTM** | Cell + hidden | Better | No |
| **Attention RNN** | + context | Good | Partially |
| **Transformer** | Attention only | Excellent | Yes |

### Key Takeaways

1. **RNNs** process sequences with memory but struggle with long dependencies
2. **LSTMs** use gates to control information flow, solving vanishing gradients
3. **Attention** provides direct connections between any positions
4. **Transformers** replace recurrence with pure attention, enabling parallelism
5. Modern NLP is dominated by **Transformer-based models** (BERT, GPT, etc.)

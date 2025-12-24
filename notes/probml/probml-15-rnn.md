# Recurrent Neural Networks and Transformers

RNNs process sequential data by maintaining a hidden state that carries information across time steps. Transformers, which use attention mechanisms, have largely replaced RNNs for many tasks.

## The Big Picture

**Sequential data** is everywhere: text, speech, time series, video.

**The challenge**: Variable-length inputs with temporal dependencies.

**RNN solution**: Maintain a memory (hidden state) that updates as new inputs arrive.

**Transformer solution**: Use attention to relate all positions directly.

---

## Core RNN Architecture

### The Basic Update

$$h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

**Components**:
- $x_t$: Input at time t
- $h_{t-1}$: Previous hidden state (the "memory")
- $W_{xh}$: Input-to-hidden weights
- $W_{hh}$: Hidden-to-hidden weights (recurrent connection)
- $\phi$: Non-linearity (usually tanh)

**Key insight**: Same weights used at every time step (weight sharing through time).

---

## Types of Sequence Tasks

### Seq2Vec (Many-to-One)

Variable-length input → Fixed output

**Examples**: Sentiment analysis, document classification

**Approach**: Use final hidden state (or aggregate all states) as representation.

### Vec2Seq (One-to-Many)

Fixed input → Variable-length output

**Examples**: Image captioning, music generation

**Approach**: Condition on input vector, generate sequence autoregressively.

### Seq2Seq (Many-to-Many)

Variable input → Variable output

**Examples**: Machine translation, summarization

**Approach**: Encoder-decoder architecture.

---

## Bidirectional RNNs

Process sequence in both directions:

**Forward**: $\vec{h}_t = f(x_t, \vec{h}_{t-1})$
**Backward**: $\overleftarrow{h}_t = f(x_t, \overleftarrow{h}_{t+1})$

**Final state**: Concatenate both: $h_t = [\vec{h}_t; \overleftarrow{h}_t]$

**Benefit**: Each position has access to both past and future context.

**Limitation**: Can't be used for autoregressive generation (need future that doesn't exist yet).

---

## The Vanishing/Exploding Gradient Problem

### The Problem

Gradient through L time steps:
$$\frac{\partial L}{\partial h_0} = \prod_{t=1}^{L} \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial L}{\partial h_L}$$

If $\|W_{hh}\| < 1$: Gradients vanish exponentially
If $\|W_{hh}\| > 1$: Gradients explode exponentially

### Exploding Gradient Solution

**Gradient clipping**:
$$g \leftarrow \min\left(1, \frac{\tau}{\|g\|}\right) g$$

### Vanishing Gradient Solutions

Use architectures with **additive updates** instead of multiplicative:
- LSTM
- GRU
- Skip connections

---

## LSTM (Long Short-Term Memory)

### The Key Innovation

Separate **cell state** $C_t$ that flows through time with minimal transformation.

### Gates

Three gates control information flow:

**Forget Gate**: What to discard from cell state
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input Gate**: What new information to add
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**Output Gate**: What to output from cell state
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

### Update Equations

**Candidate cell state**:
$$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Cell state update** (additive!):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Hidden state output**:
$$h_t = o_t \odot \tanh(C_t)$$

### Why LSTM Works

The cell state acts like a "conveyor belt" — gradients can flow through unchanged if the forget gate is open.

---

## GRU (Gated Recurrent Unit)

Simplified version of LSTM with fewer parameters.

**Two gates**:
- **Update gate** $z_t$: How much to update hidden state
- **Reset gate** $r_t$: How much past state to forget

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Trade-off**: Fewer parameters, competitive performance.

---

## Backpropagation Through Time (BPTT)

### The Algorithm

1. Unroll network through time
2. Forward pass: Compute all hidden states
3. Backward pass: Compute gradients through the unrolled graph
4. Sum gradients for shared weights across time

### Truncated BPTT

For long sequences:
- Don't backprop through entire sequence
- Truncate to manageable length (e.g., 100 steps)
- Trade-off: Can't learn very long-range dependencies

---

## Decoding Strategies

### Greedy Decoding

At each step, pick the most likely token:
$$y_t = \arg\max_y P(y | y_{<t}, x)$$

**Problem**: Locally optimal choices may not be globally optimal.

### Beam Search

Keep top-K candidates at each step:
1. Expand each candidate in all possible ways
2. Keep the K highest-probability sequences
3. Continue until all sequences end

**Benefit**: Better than greedy; balances quality and computation.

### Sampling

For generation diversity:

**Top-K sampling**: Sample from top K tokens only

**Top-P (nucleus) sampling**: Sample from smallest set with cumulative probability > p

**Temperature**: Scale logits before softmax
- Low T: More deterministic
- High T: More random

---

## Attention Mechanism

### The Problem with Basic RNNs

All information must flow through the bottleneck of the hidden state.

For long sequences, early information gets "washed out."

### The Attention Solution

Allow the decoder to "look at" all encoder states:

$$\text{Attention}(q, (k_1,v_1), ..., (k_m,v_m)) = \sum_{i=1}^m \alpha_i \cdot v_i$$

Where attention weights $\alpha_i$ depend on similarity between query $q$ and keys $k_i$.

### Scaled Dot-Product Attention

$$\alpha_i = \text{softmax}\left(\frac{q^T k_i}{\sqrt{d}}\right)$$

**Scaling by $\sqrt{d}$**: Prevents softmax saturation when dimensions are large.

### Seq2Seq with Attention

Instead of using only the final encoder state:
- Query: Current decoder hidden state
- Keys & Values: All encoder hidden states

Context at each decoding step:
$$c_t = \sum_i \alpha_i(h_t^{dec}, h_i^{enc}) \cdot h_i^{enc}$$

---

## Transformers

### The Revolution

**Key insight**: Attention is all you need — no recurrence required!

**Benefits**:
- Parallelizable (no sequential dependency)
- Direct connections between all positions
- Scales to very long sequences

### Self-Attention

Each position attends to all positions (including itself):
$$y_i = \text{Attention}(x_i, (x_1,x_1), (x_2,x_2), ..., (x_n,x_n))$$

**Query, Key, Value**: All derived from same input via learned projections.

### Multi-Head Attention

Run multiple attention operations in parallel:
$$h_i = \text{Attention}(W_i^Q x, W_i^K x, W_i^V x)$$
$$\text{output} = \text{Concat}(h_1, ..., h_H) W^O$$

**Benefit**: Capture different types of relationships.

### Positional Encoding

Attention is permutation-invariant — it doesn't know position!

**Solution**: Add position information to inputs:
$$x_{pos} = x + \text{PE}(pos)$$

**Sinusoidal encoding** (original Transformer):
- Different frequencies for different dimensions
- Can generalize to unseen lengths

**Learned embeddings** (common in practice).

### Transformer Architecture

**Encoder block**:
1. Multi-head self-attention + residual + LayerNorm
2. Feed-forward network + residual + LayerNorm

**Decoder block**:
1. Masked self-attention (can't see future)
2. Cross-attention to encoder
3. Feed-forward network

---

## Pre-trained Language Models

### ELMo

Concatenate forward and backward LSTM representations.

### BERT (Bidirectional Encoder)

**Pre-training tasks**:
- Masked Language Modeling (MLM): Predict masked tokens
- Next Sentence Prediction

**Fine-tuning**: Add task-specific head.

### GPT (Generative Pre-Training)

**Architecture**: Decoder-only transformer (causal masking)

**Pre-training**: Autoregressive language modeling

**Key insight**: Scale up model and data → emergent capabilities.

### T5 (Text-to-Text Transfer Transformer)

**Unifying framework**: Every task as text-to-text
- Classification: "classify: text → label"
- Translation: "translate: source → target"

---

## Summary

| Architecture | Key Feature | Best For |
|--------------|-------------|----------|
| **Basic RNN** | Recurrent hidden state | Short sequences |
| **LSTM/GRU** | Gates + additive updates | Medium sequences |
| **Bidirectional** | Both directions | When future is available |
| **Attention** | Direct access to all positions | Long-range dependencies |
| **Transformer** | Self-attention + parallelism | Everything (modern default) |

### When to Use What

- **RNN/LSTM**: Small data, limited compute, streaming data
- **Transformer**: Large data, sufficient compute, best performance
- **Pre-trained models**: Almost always start here and fine-tune!

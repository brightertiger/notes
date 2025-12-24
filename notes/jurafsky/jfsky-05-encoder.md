# Encoder-Decoder Models

Encoder-decoder (seq2seq) models transform one sequence into another. They power machine translation, summarization, question answering, and many other NLP tasks.

## The Big Picture

**The Problem**: Input and output sequences have different lengths and structures.

**Example (Translation)**:
- English: "The cat sat on the mat" (6 words)
- German: "Die Katze saß auf der Matte" (6 words, different order)
- Japanese: 猫がマットの上に座った (different structure entirely)

**The Solution**: 
1. **Encoder**: Compress input into a representation
2. **Decoder**: Generate output from that representation

---

## Encoder-Decoder Architecture

### The Two Components

```
Input Sequence → [ENCODER] → Context Vector → [DECODER] → Output Sequence
```

**Encoder**:
- Processes input sequence
- Produces contextualized hidden states
- Creates a "summary" of the input

**Decoder**:
- Uses encoder output as initial context
- Generates output tokens one at a time
- Autoregressive: each output depends on previous outputs

---

## Sequence-to-Sequence with RNNs

### Encoder

Process input token by token:
$$h_t^{enc} = f(h_{t-1}^{enc}, x_t)$$

The final hidden state $h_T^{enc}$ summarizes the entire input.

### Context Vector

Simple approach: Use final encoder hidden state.
$$c = h_T^{enc}$$

**Problem**: All information must squeeze through this bottleneck!

### Decoder

Initialize with context, generate autoregressively:
$$h_t^{dec} = f(h_{t-1}^{dec}, y_{t-1}, c)$$
$$P(y_t) = \text{softmax}(W h_t^{dec})$$

### Training: Teacher Forcing

During training, use **ground truth** previous tokens, not predicted ones.

**Without teacher forcing**: Errors compound (predicted mistake → more mistakes).
**With teacher forcing**: More stable training, faster convergence.

**The drawback**: Exposure bias — model never sees its own mistakes during training.

---

## The Attention Solution

### The Bottleneck Problem

As sequences get longer, the fixed-size context vector struggles to capture everything.

### Dynamic Context

Instead of one context vector, compute a **different context for each decoder step**:

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j^{enc}$$

Where $\alpha_{ij}$ are attention weights — how much should decoder step i focus on encoder position j?

### Computing Attention Weights

1. **Score** each encoder state against decoder state:
   $$e_{ij} = \text{score}(s_{i-1}^{dec}, h_j^{enc})$$

2. **Normalize** to get weights:
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

3. **Combine** to get context:
   $$c_i = \sum_j \alpha_{ij} h_j^{enc}$$

4. **Decode** using context:
   $$s_i^{dec} = f(s_{i-1}^{dec}, y_{i-1}, c_i)$$

### Benefits of Attention

| Benefit | Explanation |
|---------|-------------|
| **Long sequences** | No information bottleneck |
| **Alignment** | Learns which source words map to which target words |
| **Interpretability** | Can visualize what the model focuses on |
| **Gradient flow** | Direct paths for gradients |

---

## Transformer Encoder-Decoder

### Key Difference: Cross-Attention

The decoder has three types of attention:
1. **Self-attention** on encoder (bidirectional)
2. **Masked self-attention** on decoder (causal — can't see future)
3. **Cross-attention**: Queries from decoder, Keys/Values from encoder

### Cross-Attention Mechanism

$$\text{CrossAttn}(Q^{dec}, K^{enc}, V^{enc}) = \text{softmax}\left(\frac{Q^{dec} (K^{enc})^T}{\sqrt{d}}\right) V^{enc}$$

Decoder queries look up relevant information from encoder.

---

## Tokenization for Seq2Seq

### The Challenge

Different languages have different:
- Writing systems
- Word boundaries
- Vocabulary sizes

### Subword Tokenization

Use **BPE** or **WordPiece** for both languages:
- Handles rare words gracefully
- Shares subwords across similar languages
- Reduces vocabulary size

---

## Evaluation Metrics

### Human Evaluation (Gold Standard)

**Adequacy**: Is the meaning preserved?
- 1 = None, 5 = All meaning captured

**Fluency**: Is it grammatically correct and natural?
- 1 = Incomprehensible, 5 = Native quality

**Problem**: Expensive, slow, not reproducible.

### Automatic Metrics

#### BLEU (Bilingual Evaluation Understudy)

The classic MT metric.

**Core idea**: Count n-gram matches between output and reference.

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ = precision for n-grams (typically n=1,2,3,4)
- $w_n$ = weights (usually uniform: 0.25 each)
- BP = brevity penalty (penalizes short translations)

**Brevity Penalty**:
$$BP = \min\left(1, \exp\left(1 - \frac{r}{c}\right)\right)$$

Where r = reference length, c = candidate length.

**BLEU ranges**: 0 to 100 (100 = perfect match).

**Limitations**:
- Doesn't capture meaning (just surface n-grams)
- One reference may miss valid alternatives
- Insensitive to word order issues

#### chrF (Character F-Score)

Uses character n-grams instead of word n-grams.

**Benefits**:
- Works for languages without clear word boundaries
- More robust to morphological variation
- Often correlates better with human judgment

#### BERTScore

Uses neural embeddings for semantic matching.

1. Embed each token in reference and hypothesis using BERT
2. Greedily match tokens by cosine similarity
3. Compute precision, recall, F1 from matches

**Benefits**:
- Captures semantic similarity, not just surface form
- "automobile" matches "car" even though different words

---

## Decoding Strategies

### The Challenge

At each step, we have a probability distribution over the entire vocabulary.

**Goal**: Find the most likely complete sequence.

**Problem**: Exhaustive search is intractable (V^T possibilities).

### Greedy Decoding

Pick highest probability token at each step:
$$y_t = \arg\max_y P(y | y_{<t}, x)$$

**Pros**: Fast, simple.
**Cons**: Locally optimal ≠ globally optimal. High-probability first word might lead to low-probability continuation.

### Beam Search

Keep top-K hypotheses at each step:

1. Start with K copies of \<BOS\>
2. Expand each hypothesis with all possible next tokens
3. Keep top K by total probability
4. Repeat until all hypotheses end with \<EOS\>
5. Return highest-scoring complete hypothesis

**Beam width K**:
- K=1 is greedy decoding
- K=5-10 typical for translation
- Larger K = better but slower

**Length normalization** (prevent bias toward short sequences):
$$\text{score} = \frac{\log P(Y|X)}{|Y|^\alpha}$$

Where α ≈ 0.6-0.7.

### Sampling Strategies (for Generation)

For creative text generation, we want **diversity**, not just the most likely output.

**Top-K Sampling**:
1. Keep only top K most probable tokens
2. Redistribute probability among them
3. Sample from this truncated distribution

**Top-P (Nucleus) Sampling**:
1. Sort tokens by probability
2. Keep smallest set with cumulative probability > p
3. Sample from this set

**Temperature** (before softmax):
$$P(y) = \frac{\exp(z_y / T)}{\sum_j \exp(z_j / T)}$$

- T < 1: More peaked (confident, less diverse)
- T = 1: Original distribution
- T > 1: Flatter (more random, more diverse)

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Encoder** | Compress input to representation |
| **Decoder** | Generate output autoregressively |
| **Attention** | Dynamic context at each step |
| **Cross-attention** | Transformer way of connecting encoder to decoder |
| **Beam search** | Better than greedy, tractable search |
| **BLEU/BERTScore** | Automatic evaluation |

### Key Takeaways

1. **Encoder-decoder** is the standard architecture for sequence transduction
2. **Attention** solves the bottleneck problem and provides interpretability
3. **Evaluation is hard** — automatic metrics are imperfect proxies for quality
4. **Decoding strategy matters** — beam search for accuracy, sampling for diversity

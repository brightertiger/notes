# Transfer Learning and Pre-trained Models

Transfer learning has revolutionized NLP. Pre-train a large model on massive text data, then fine-tune for specific tasks. This chapter covers BERT and the pre-train/fine-tune paradigm.

## The Big Picture

**The Old Way**: Train a model from scratch for each task.
- Requires lots of labeled data
- Each task starts from zero

**The New Way**: Pre-train → Fine-tune.
- Pre-train once on huge unlabeled corpus
- Fine-tune with small labeled dataset per task
- Transfer linguistic knowledge across tasks

---

## Key Concepts

### Contextual Embeddings

**Static embeddings** (Word2Vec): Same vector for "bank" regardless of context.

**Contextual embeddings** (BERT): Different vector for "river bank" vs. "bank account".

The same word gets different representations based on surrounding words.

### The Pre-train → Fine-tune Paradigm

```
[MASSIVE UNLABELED TEXT] → Pre-training → [GENERAL LANGUAGE MODEL]
                                                    ↓
[SMALL LABELED DATA] → Fine-tuning → [TASK-SPECIFIC MODEL]
```

**Pre-training**: Self-supervised learning on vast text (books, Wikipedia, web).

**Fine-tuning**: Supervised learning on task-specific data.

### Language Model Types

| Type | Direction | Example | Best For |
|------|-----------|---------|----------|
| **Causal** | Left-to-right | GPT | Generation |
| **Bidirectional** | Both directions | BERT | Understanding |
| **Encoder-Decoder** | Both + generation | T5 | Both |

---

## Bidirectional Transformers (BERT)

### Why Bidirectional?

For many tasks, we can see the entire input at once.

**Causal models** (GPT): Can only see left context.
```
"The cat sat on the [???]" - what comes next?
```

**Bidirectional models** (BERT): See full context.
```
"The cat sat on the [MASK]" - what's missing?
```

### BERT Architecture

**Self-attention** across entire sequence:
$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**BERT Base**:
- Vocabulary: 30K subwords (WordPiece)
- Hidden size: 768 (12 heads × 64 dims)
- Layers: 12
- Parameters: 110M
- Max sequence: 512 tokens

**Compute note**: Attention is O(n²) in sequence length — limits max length.

---

## Pre-training Objectives

### Masked Language Modeling (MLM)

**The task**: Predict randomly masked tokens.

```
Input:  The cat [MASK] on the mat
Target: sat
```

**Masking strategy** (for 15% of tokens):
- 80%: Replace with [MASK]
- 10%: Replace with random word
- 10%: Keep original

**Why this mix?**
- [MASK] never appears at fine-tuning → train on real words too
- Random replacement adds noise, prevents overfitting
- Keeping some originals helps learn bidirectional context

### Span Masking (SpanBERT)

Mask contiguous spans instead of random tokens.

```
Input:  The [MASK] [MASK] [MASK] the mat
Target: cat sat on
```

**Benefits**:
- Better for tasks requiring span understanding (QA, NER)
- Span Boundary Objective: Predict span from boundary tokens

### Next Sentence Prediction (NSP)

**The task**: Are these sentences adjacent in the original text?

```
[CLS] The cat sat [SEP] It was happy [SEP] → IsNext
[CLS] The cat sat [SEP] I like pizza [SEP] → NotNext
```

**Training data**: 50% real pairs, 50% random pairs.

**Note**: Later models (RoBERTa, ALBERT) found NSP less helpful than expected.

---

## Pre-training Data

BERT was trained on:
- **BooksCorpus**: 800M words
- **English Wikipedia**: 2.5B words

Later models use more:
- CommonCrawl (filtered web text)
- News articles
- Code repositories

**Compute requirement**: Days to weeks on TPU/GPU clusters.

---

## Fine-tuning

### The Basic Recipe

1. Load pre-trained model
2. Add task-specific **classification head** (usually 1-2 layers)
3. Train on labeled data with small learning rate

### Fine-tuning Strategies

| Strategy | What's Updated | Best For |
|----------|---------------|----------|
| **Full fine-tuning** | All parameters | Lots of data, maximum performance |
| **Feature extraction** | Only head | Very small data |
| **Adapter tuning** | Small inserted modules | Efficient multi-task |
| **Prompt tuning** | Soft prompts only | Very large models |

### Hyperparameters

**Learning rate**: 2e-5 to 5e-5 (much smaller than training from scratch!)

**Epochs**: 2-4 (often sufficient)

**Batch size**: 16-32

---

## Task-Specific Architectures

### Sequence Classification

**Task**: Classify entire input (sentiment, topic).

```
Input:  [CLS] I loved this movie [SEP]
Output: Use [CLS] representation → linear → softmax → class
```

### Sentence Pair Classification

**Task**: Classify relationship between two texts (NLI, similarity).

```
Input:  [CLS] Premise text [SEP] Hypothesis text [SEP]
Output: [CLS] representation → linear → entailment/contradiction/neutral
```

### Token Classification (NER, POS)

**Task**: Label each token.

```
Input:  [CLS] John lives in New York [SEP]
Labels:       B-PER O     O  B-LOC I-LOC
```

Each token gets its own classification head output.

**WordPiece handling**:
- Training: Expand labels to all subword tokens
- Evaluation: Use label of first subword

### Span Prediction (QA)

**Task**: Find answer span in context.

```
Input:  [CLS] Where is Paris? [SEP] Paris is in France [SEP]
Output: Predict start position (index 5: "Paris")
        Predict end position (index 8: "France")
```

Two classifiers: one for start, one for end position.

---

## Modern Variants

### RoBERTa (Robustly Optimized BERT)

Key changes:
- Remove NSP objective
- Larger batches, more data
- Dynamic masking (different masks each epoch)

### ALBERT (A Lite BERT)

Parameter reduction:
- Factorized embedding parameters
- Cross-layer parameter sharing
- Sentence order prediction instead of NSP

### DistilBERT

Knowledge distillation:
- 40% smaller, 60% faster
- 97% of BERT performance

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Contextual embeddings** | Same word → different vectors in different contexts |
| **Pre-training** | Self-supervised on massive text |
| **Fine-tuning** | Task-specific with small labeled data |
| **MLM** | Predict masked tokens (BERT's main objective) |
| **[CLS] token** | Aggregate representation for classification |
| **[SEP] token** | Separate segments in input |

### The Revolution

Transfer learning changed NLP:

| Before | After |
|--------|-------|
| Train from scratch per task | Pre-train once, fine-tune many times |
| Need lots of labeled data | Works with small labeled data |
| Shallow features | Deep contextual understanding |
| Task-specific architectures | One architecture, many tasks |

### Practical Tips

1. **Start with pre-trained models** — rarely worth training from scratch
2. **Try multiple learning rates** — this is the most important hyperparameter
3. **Don't over-fine-tune** — 2-4 epochs is often enough
4. **Consider model size** — DistilBERT for production, BERT-large for best accuracy
5. **Domain matters** — SciBERT for science, BioBERT for biomedical, etc.

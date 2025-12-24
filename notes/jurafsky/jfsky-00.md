# Speech and Language Processing Notes

These notes are based on "Speech and Language Processing" by Dan Jurafsky and James H. Martin — the definitive textbook for Natural Language Processing. This guide makes NLP concepts accessible to undergraduates while building the foundation for advanced study.

## What is Natural Language Processing?

NLP is the field of computer science focused on enabling computers to understand, interpret, and generate human language. It bridges linguistics, computer science, and machine learning.

**Why is language hard for computers?**
- Language is **ambiguous** (bank of a river vs. bank for money)
- Language is **context-dependent** (meaning changes with context)
- Language is **creative** (infinite sentences from finite vocabulary)
- Language has **implicit knowledge** (common sense, world knowledge)

## Topics Covered

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| 2 | **Regular Expressions & Text Processing** | Pattern matching, tokenization, normalization |
| 3 | **N-Grams & Language Models** | Probabilistic models of word sequences |
| 6 | **Vector Semantics** | Word embeddings, similarity, Word2Vec |
| 9 | **Sequence Models** | RNNs, LSTMs, and attention mechanisms |
| 10 | **Encoder-Decoder Models** | Seq2Seq, machine translation |
| 11 | **Transfer Learning** | BERT, pre-training, fine-tuning |

## The Evolution of NLP

```
Rule-based    →    Statistical    →    Neural    →    Pre-trained
(1950s-1980s)      (1990s-2000s)     (2013-2018)    (2018-present)

Hand-crafted       Probabilistic      Deep learning   Transformers
grammars          models (HMMs,       (RNNs, LSTMs)   (BERT, GPT)
                  n-grams)
```

## Core NLP Tasks

**Understanding Language**:
- Text classification (spam detection, sentiment)
- Named entity recognition (finding names, places, dates)
- Parsing (sentence structure)
- Semantic analysis (meaning extraction)

**Generating Language**:
- Machine translation
- Text summarization
- Question answering
- Dialogue systems

## Prerequisites

- **Programming**: Python, basic data structures
- **Math**: Probability basics, linear algebra fundamentals
- **ML Basics**: Helpful but not strictly required

## How to Use These Notes

1. **Start with fundamentals**: Regex and n-grams build the foundation
2. **Understand the progression**: From sparse to dense representations, from RNNs to Transformers
3. **Connect to practice**: Try implementing concepts in Python
4. **See the big picture**: Modern NLP combines all these ideas

Let's dive into the fascinating world of language and computation!

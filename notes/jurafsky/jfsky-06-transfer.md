# Transfer Learning

-   Contextual Embeddings: Representation of words in context. Same word can have different embeddings based on the context in which it appears.
-   Pretraining: Learning contextual embeddings from vast amounts of text data.
-   Fine-tuning: Taking generic contextual representations and tweaking them to a specific downstream task by using a NN classifier head.
-   Transfer Learning: Pretrain-Finetune paradigm is called transfer learning.
-   Language Models:
    -   Causal: Left-to-Right transformers
    -   Bidirectional: Model can see both left and right context

# Bidirectional Transformer Models

-   Causal transformers are well suited for autoregressive problems like text generation
-   Sequence classification and labeling problems can relax this restriction
-   Bidirectional encoders allow self attention mechanism to cover the entire input sequence
-   Map the input sequence embeddings to output embeddings of the same length with expanded context
-   Self Attention
    -   $$\text{SA} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
-   BERT Base
    -   Subword vocabulary of 30K using WordPiece
    -   Hidden layer size of 768 (12 \* 64)
    -   12 Layers, 12 attention heads
    -   110M+ parameters
    -   Max Sequence length is 512 tokens
-   Size of input layer dictates the complexity of the model
-   Computational time and memory grow quadratically with input sequence length

## Pre-Training

-   Fill-in-the-blank or Cloze task approach
-   Predict the "masked" words (Masked Language Modeling - MLM)
-   Use cross-entropy loss over the vocabulary to drive training
-   Self-supervised Learning (creates supervision from unlabeled data)
-   Masked Language Modeling (BERT approach)
    -   Requires large unannotated text corpus
    -   Random sample (15%) of tokens is chosen to be masked for learning
    -   For these 15% tokens:
        -   80% replaced with [MASK] token
        -   10% replaced with random word (adds noise, prevents overfitting)
        -   10% left unchanged (helps model learn bidirectional context)
    -   Predict original token for each of the masked positions
    -   Model must use bidirectional context to make accurate predictions
-   Span-based Approaches (e.g., SpanBERT)
    -   Mask contiguous sequences of tokens rather than individual tokens
    -   Span length selected from geometric distribution
    -   Starting location is selected from uniform distribution
    -   Once the span is selected, all words within it are masked
    -   Learning objectives:
        -   Masked Language Modeling (predict masked tokens)
        -   Span Boundary Objective (SBO): Predict internal span words using only boundary tokens
    -   Better for tasks requiring span representations (QA, coreference)
-   Next Sentence Prediction (NSP)
    -   Additional pre-training task in original BERT
    -   Helps with discourse understanding tasks
    -   Training data consists of:
        -   50% actual pairs of adjacent sentences
        -   50% random sentence pairs (negative examples)
    -   Model must distinguish true pairs from random pairs
    -   Later models (RoBERTa, ALBERT) found this less helpful than expected
-   Training Data
    -   Large diverse text corpora:
        -   Books corpus (800M words)
        -   English Wikipedia (2.5B words)
        -   Additional data in later models (CommonCrawl, etc.)
    -   Computationally expensive (days/weeks on TPU/GPU clusters)

## Fine-Tuning

-   Creation of task-specific models on top of pretrained models leveraging generalizations from self-supervised learning
-   Advantages:
    -   Requires limited amount of labeled data
    -   Much faster than training from scratch
    -   Often better performance than task-specific architectures
-   Methods:
    -   Full fine-tuning: Update all parameters of pretrained model
    -   Adapter tuning: Keep most parameters frozen, add small trainable modules
    -   Prompt tuning: Frame downstream task as a language modeling problem
-   Task-specific modifications:
    -   Add classification head (typically 1-2 layers)
    -   Adjust learning rate (typically 2e-5 to 5e-5)
    -   Train for fewer epochs (2-4 typically sufficient)
-   Sequence Classification
    -   Add special [CLS] token at beginning of input
    -   Use [CLS] token's final-layer representation
    -   Add classification head: linear layer + softmax
    -   Fine-tune with labeled examples
-   Natural Language Inference (NLI)
    -   Recognize contradiction, entailment, or neutral relationship between text pairs
    -   Input format: [CLS] premise [SEP] hypothesis [SEP]
    -   Use [CLS] representation for classification
-   Sequence Labeling (NER, POS tagging)
    -   Prediction for each token (token-level classification)
    -   Add softmax layer over label classes for each token
    -   Use BIO tagging scheme (Beginning, Inside, Outside)
    -   WordPiece tokenization creates challenges:
        -   Training: Expand the labels to all subword tokens
        -   Evaluation: Use tag assigned to the first subword token
-   Span-based Tasks (QA, extraction)
    -   For tasks requiring identifying spans in text
    -   Generate span representations:
        -   Concatenate [start, end, pooled-span] embeddings
        -   Or use span boundary representations
    -   Use regression or classification to predict start and end positions
    -   SQuAD format: [CLS] question [SEP] context [SEP] 
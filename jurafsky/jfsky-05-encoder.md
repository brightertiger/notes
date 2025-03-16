# Encoder-Decoder Models

## Overview

-   Encoder-Decoder or Seq2Seq architecture
-   Can be implemented using Transformers or RNNs
-   Output sequence is a complex transformation of the entire input sequence
-   In Machine Translation, the sequence order may not always agree
    -   Word order topology changes from language to language (subject-verb-object)
    -   Same vocabulary may not exist. Words map to phrases.
-   Encoder block takes an input sequence and creates a contextualized vector representation
-   Decoder block uses this representation to generate the output sequence
-   Architecture
    -   Encoder: Input is sequence and output is contextualized hidden states
    -   Context Vector: A transformation of the contextualized hidden states
    -   Decoder: Uses context vector to generate arbitrary length sequences

## Sequence Models

-   Models are autoregressive by nature
-   Add a BOS token \<s\> for conditional generation
-   Keep sampling till EOS token \</s\>
-   RNN / LSTM
    -   Encoder
        -   Process the input sequence token by token
    -   Context vector
        -   Use the final hidden state of LSTM as the context vector
    -   Decoder
        -   Use the context vector for initialization
        -   Use BOS token for generation
    -   Drawback: Influence of context vector wanes as longer sequences are generated
        -   Solution is to make context vector available for each timestep of the decoder
    -   Training happens via teacher forcing
-   Transformers
    -   Uses Cross-Attention for decoding
    -   Keys and values come from encoder but query comes from decoder
    -   Allows decoder to attend to each token of the input sequence
-   Tokenization
    -   BPE / Wordpiece tokenizer

## Evaluation

-   Human Evaluation
    -   Adequacy: How accurately the meaning is preserved
    -   Fluency: Grammatical correctness and naturalness
    -   Time-consuming and expensive but still gold standard
-   Automatic Evaluation
    -   chrF Score: Character F-Score
        -   Compares character n-grams between reference and hypothesis
        -   Works well for morphologically rich languages
        -   Less affected by exact word match requirements
    -   BLEU: Bilingual Evaluation Understudy
        -   Modified n-gram precision: Compares n-grams in output with reference
        -   Clips each n-gram count to maximum count in any reference
        -   Combines precision for different n-gram sizes (usually 1-4)
        -   Adds brevity penalty (BP) for short translations: $BP = \min(1, e^{1-r/c})$
        -   $BLEU = BP \cdot \exp(\sum_{n=1}^{N} w_n \log p_n)$
        -   Where r is reference length, c is candidate length
    -   BERTScore
        -   Uses contextual embeddings from BERT
        -   Compute embeddings for each token in reference and hypothesis
        -   Compute cosine similarity between each pair of tokens
        -   Match tokens greedily based on similarity
        -   Compute precision, recall and F1 from these matches
        -   Better semantic matching than n-gram based metrics

## Attention

-   Final hidden state of encoder acts as a bottleneck in basic seq2seq
-   Attention mechanism helps the decoder access all intermediate encoder states
-   Generate dynamic context vector for each decoder step
-   Process:
    1. Compute alignment scores between decoder state and all encoder states
       - $e_{ij} = f(s_{i-1}, h_j)$ where s is decoder state, h is encoder state
       - f can be dot product, MLP, or other similarity function
    2. Normalize scores using softmax to get attention weights
       - $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$
    3. Compute context vector as weighted sum of encoder states
       - $c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$
    4. Use context vector along with previous decoder state to predict next word
       - $s_i = f(s_{i-1}, y_{i-1}, c_i)$
-   Benefits: 
    -   Models long-range dependencies effectively
    -   Provides interpretability through attention weights
    -   Helps with word alignment in MT

## Decoding

-   Plain vanilla greedy decoding selects the highest probability token at each step
    -   Simple but often suboptimal for overall sequence probability
-   The overall results may be suboptimal because a high-probability token now may lead to low-probability continuations
-   Search trees represent all possible output sequences
    -   The most probable sequence may not be composed of argmax tokens at each step
    -   Exhaustive search is intractable (vocab_size^sequence_length possibilities)
-   Beam Search
    -   Select top-k possible tokens at each time step (beam width)
    -   Each of the "k" hypotheses is extended with each possible next token
    -   Keep only the k most probable extended sequences
    -   Continue until all beams produce EOS token or max length is reached
    -   Length penalty to avoid bias toward shorter sequences:
        -   $score(Y) = \frac{\log P(Y|X)}{length(Y)^\alpha}$ where α is typically around 0.6-0.7
    -   Usually k is between 5 and 10 (larger for harder problems)
-   Sampling Techniques (for text generation)
    -   Top-K Sampling
        -   Top-K tokens are selected and the probability mass is redistributed among them
        -   Reduces chance of selecting low-probability (nonsensical) tokens
    -   Top-P (Nucleus) Sampling
        -   Instead of selecting a fixed number of tokens, select the smallest set of tokens whose cumulative probability exceeds threshold p
        -   Adapts to the confidence of the model's predictions
        -   Typically p = 0.9 or 0.95 
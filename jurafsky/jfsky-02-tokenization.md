# Tokenization

## N-Grams

-   Language Models assign probabilities to sequence of words
    -   $P(w_1, w_2, ..., w_n)$
-   Simplify the calculation using chain rule
    -   $P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2 | w_1)..... \times P(w_n | w_1 w_2 .. w_{n-1})$
    -   $P(w_1, w_2, ..., w_n) = \prod P(w_i | w_{1:i-1})$
-   Joint probability can be expressed as a product of conditional probabilities
    -   Probability of a word given historical context
    -   $P(w_n | h)$
-   N-gram refers to a sequence of n words
-   N-gram model makes the Markov assumption
    -   $P(w | h)$ can be approximated using just the last n-1 words
-   For example in case of a bigram model
    -   $P(w_n | h) \approx P(w_n | w_{n-1})$
-   Estimate the probabilities using Maximum Likelihood
    -   Relative frequency
    -   $P(w_n | w_{n-1}) = {P(w_{n-1}, w_n) \over \sum_k P(w_{n-1}, w_k)}$
    -   $P(w_n | w_{n-1}) = P(w_{n-1}, w_n) / P(w_{n-1})$
-   BOS and EOS tokens to handle the edge cases
-   N-gram models apt at capturing syntactic features (noun-verb-adj etc)
-   To avoid numerical underflow, overflow problems use log probabilities
    -   $p_1 \times p_2 = \exp(\log p_1 + \log p_2)$

## Perplexity

-   Inverse probability normalized by the length of sequence
-   $PP(W) = P(w_1 w_2 ... w_n)^{ - {1 \over n}}$
-   $PP(W) = \sqrt[n]{\prod 1 / P(w_i | w_{1:i-1})}$
-   Higher the conditional probability, lower is the perplexity
-   Weighted average branching factor
    -   Branching factor refers to the number of possible words that can follow a particluar word
-   Perplxity of LMs comparable only if they use same vocabulary

## Perplexity and Entropy

-   Entropy is a measure of information
    -   Number of bits it takes to encode information (log base 2)
    -   $H(x) = - \sum p \log (p)$
-   Entropy Rate: Entropy // Seq Length
-   LMs can potentially consider infinite sequence length
    -   $H(W) = - \lim_{n \to \infty} {1 \over n} \sum p(w_{1:n}) \log(p_{1:n})$
    -   $H(W) \approx - {1 \over n} \log p(w_{1:n})$
-   $P(W) = 2^{H(W)}$

## Unknown Words

-   If probability of a word is zero, the perplexity is not defined.
-   Unknown words or OOV words (out of vocab)
-   Handle via pre-processing <UNK> token
    -   Replace rare words with this token in training corpus
-   LMs can achieve lower perplexity by selecting smaller vocab size

## Smoothing

-   Avoid assigning zero probabilities to unseen sequences
-   Laplace Smoothing
    -   Add smoothing constants while calculating relative frequencies
    -   1 to numerator
    -   V to denominator, V is the vocab size to ensure that probabilities sum up to 1
    -   $P(w_i) = (w_i + 1) / (N + V)$
    -   $P(w_i | w_n) = \text{count}(w_i, w_n) + 1 / \text{count}(w_n) + V$
    -   Discount some probability mass from seen phrases and save it for unseen phrases
    -   Generalization to "Add - k" smoothing
-   Back-off
    -   Use shorter sequences in case not enough support for full context
    -   Trigram if evidence is sufficent, otherwise use bigram
-   Interpolation
    -   Mix the probability estimates from all n-grams
    -   $P(w_1 | w_2 w_3) = \lambda_1 P(w_1) + \lambda_2 P(w_1 | w_2) + + \lambda_3 P(w_1 | w_2 w_3)$
-   Kneser-Ney Smoothing
    -   Absolute discounting
    -   $P(w_1 | w_2) = C(w_1 w_2) - d / \sum c(w_2) + \lambda P(w_1)$

## Efficiency

-   Reduce memory footprint
-   Quantization for probabilities
-   Reverse Tries for N-grams
-   String Hashing
-   Bloom filters
-   Stupid Backoff 
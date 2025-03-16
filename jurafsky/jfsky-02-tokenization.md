# Tokenization

## N-Grams

-   Language Models assign probabilities to sequence of words
    -   $P(w_1, w_2, ..., w_n)$
-   Simplify the calculation using chain rule
    -   $P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2 | w_1) \times ... \times P(w_n | w_1 w_2 ... w_{n-1})$
    -   $P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_{1:i-1})$
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
    -   $P(w_n | w_{n-1}) = \frac{P(w_{n-1}, w_n)}{\sum_k P(w_{n-1}, w_k)}$
    -   $P(w_n | w_{n-1}) = \frac{count(w_{n-1}, w_n)}{count(w_{n-1})}$
-   BOS and EOS tokens to handle the edge cases
-   N-gram models apt at capturing syntactic features (noun-verb-adj etc)
-   To avoid numerical underflow, overflow problems use log probabilities
    -   $p_1 \times p_2 = \exp(\log p_1 + \log p_2)$

## Perplexity

-   Inverse probability normalized by the length of sequence
-   $PP(W) = P(w_1 w_2 ... w_n)^{-\frac{1}{n}}$
-   $PP(W) = \sqrt[n]{\prod_{i=1}^{n} \frac{1}{P(w_i | w_{1:i-1})}}$
-   Higher the conditional probability, lower is the perplexity
-   Weighted average branching factor
    -   Branching factor refers to the number of possible words that can follow a particular word
    -   Lower perplexity means model is more confident in its predictions
-   Perplexity of LMs comparable only if they use same vocabulary
    -   Adding rare words increases perplexity

## Perplexity and Entropy

-   Entropy is a measure of information
    -   Number of bits it takes to encode information (log base 2)
    -   $H(X) = -\sum_{x} p(x) \log_2(p(x))$
-   Entropy Rate: Entropy per symbol in a sequence
    -   $H(W) = \lim_{n \to \infty} \frac{1}{n}H(w_1, w_2, ..., w_n)$
-   LMs can potentially consider infinite sequence length
    -   $H(W) = -\lim_{n \to \infty} \frac{1}{n} \sum_{w_{1:n}} p(w_{1:n}) \log_2(p(w_{1:n}))$
    -   $H(W) \approx -\frac{1}{n} \log_2 p(w_{1:n})$
-   Perplexity relates to entropy: $PP(W) = 2^{H(W)}$
    -   Perplexity can be interpreted as the average number of choices the model has when predicting the next word

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
    -   Add 1 to numerator (count)
    -   Add V to denominator (V is the vocab size) to ensure that probabilities sum up to 1
    -   $P(w_i) = \frac{\text{count}(w_i) + 1}{N + V}$
    -   $P(w_i | w_{j}) = \frac{\text{count}(w_i, w_{j}) + 1}{\text{count}(w_{j}) + V}$
    -   Discount some probability mass from seen phrases and save it for unseen phrases
    -   Generalization to "Add-k" smoothing (k can be less than 1)
-   Back-off
    -   Use shorter sequences if not enough support for full context
    -   Use trigram if evidence is sufficient, otherwise use bigram
    -   $P_{BO}(w_i|w_{i-2}w_{i-1}) = \begin{cases} 
      P(w_i|w_{i-2}w_{i-1}) & \text{if count}(w_{i-2}w_{i-1}w_i) > 0 \\
      \alpha(w_{i-2}w_{i-1}) \cdot P_{BO}(w_i|w_{i-1}) & \text{otherwise}
    \end{cases}$
    -   Where α is a normalization factor
-   Interpolation
    -   Mix the probability estimates from all n-grams
    -   $P(w_i | w_{i-2}w_{i-1}) = \lambda_1 P(w_i) + \lambda_2 P(w_i | w_{i-1}) + \lambda_3 P(w_i | w_{i-2}w_{i-1})$
    -   λ values sum to 1, typically learned from held-out data
-   Kneser-Ney Smoothing
    -   State-of-the-art n-gram smoothing technique
    -   Uses absolute discounting with a sophisticated back-off distribution
    -   $P_{KN}(w_i | w_{i-1}) = \frac{\max(c(w_{i-1}w_i) - d, 0)}{\sum_v c(w_{i-1}v)} + \lambda(w_{i-1})P_{continuation}(w_i)$
    -   Where d is discount (typically 0.75) and P_continuation captures how likely word appears in new contexts

## Efficiency

-   Reduce memory footprint
-   Quantization for probabilities
-   Reverse Tries for N-grams
-   String Hashing
-   Bloom filters
-   Stupid Backoff 
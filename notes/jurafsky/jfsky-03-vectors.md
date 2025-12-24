# Vectors

## Lexical Semantics

-   Issues that make it harder for syntactic models to scale well
-   Lemmas and word forms (sing vs sang vs sung are forms of the same lemma "sing")
-   Word Sense Disambiguation (mouse animal vs mouse hardware)
-   Synonyms with same propositional meaning (couch vs sofa)
-   Word Relatedness (coffee vs cup)
-   Semantic Frames (A buy from B vs B sell to A)
-   Connotation (affective meaning)

## Vector Semantics

-   Represent words using vectors called "embeddings"
-   Derived from co-occurrence matrix
-   Document Vectors
    -   Term-Document Matrix
    -   |V| × |D| Dimension
    -   Count of times a word shows up in a document
    -   Vector of the document in |V| dimension space
    -   Used for information retrieval
    -   Vector Space Model
-   Word Vectors
    -   Term-Term Matrix
    -   |V| × |V| dimension
    -   Number of times a word and context word show up in the same document
    -   Word-Word co-occurrence matrix
    -   Sparsity is a challenge
-   Cosine Similarity
    -   Normalized Dot Product
    -   Normalized by the L2-norm, to control for vector size
    -   $\cos \theta = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}$
    -   1 if vectors are in the same direction
    -   -1 if vectors are in opposite direction
    -   0 if vectors are perpendicular
    -   For normalized vectors, it's directly related to euclidean distance
    -   $|\vec{a} - \vec{b}|^2 = |\vec{a}|^2 + |\vec{b}|^2 - 2|\vec{a}||\vec{b}|\cos\theta = 2(1 - \cos \theta)$

## TF-IDF

-   Term Frequency
    -   Frequency of word t in document d
    -   $tf_{t,d} = \text{count}(t,d)$
    -   Smooth TF
    -   $tf_{t,d} = \log(1 + \text{count}(t,d))$
-   Document Frequency
    -   Number of documents in which term t appears
    -   $df_t$
-   Inverse Document Frequency
    -   $idf_t = \log(N / df_t)$
-   TF-IDF
    -   $w_{t,d} = tf_{t,d} \times idf_t$

## PMI

-   Point-wise Mutual Information measures the association between words
-   Ratio of:
    -   How often do x and y actually co-occur? (observed joint probability)
    -   How often would x and y co-occur if they were independent? (expected joint probability)
-   $PMI(x,y) = \log_2 \left(\frac{P(x,y)}{P(x)P(y)}\right)$
-   Ranges from negative infinity to positive infinity
    -   Positive: Words co-occur more than expected by chance
    -   Zero: Words co-occur exactly as expected by chance
    -   Negative: Words co-occur less than expected by chance
-   Positive PMI (PPMI): max(0, PMI) - often used to avoid negative values
-   In practice, we estimate probabilities from corpus counts:
    -   $PMI(x,y) = \log_2 \left(\frac{count(x,y) \cdot N}{count(x) \cdot count(y)}\right)$
    -   Where N is the total number of word pairs

## Vector Representation

-   For a given word T
    -   Term-Document Matrix
    -   Each word vector has |D| dimensions
    -   Each cell is weighted using TF-IDF logic
-   Document Vector
    -   Average of all word vectors appearing in the document
    -   Similarity is calculated by cosine distance

## Word2Vec

-   TF-IDF and PMI generate sparse vectors (mostly zeros)
-   Need for dense and more efficient representation of words
-   Static Embeddings
    -   Fixed vector for each word regardless of context
    -   Skipgram with Negative Sampling (SGNS)
    -   Continuous Bag of Words (CBOW) - predicts target word from context
-   Contextual Embeddings
    -   Dynamic embedding for each word
    -   Changes with context (word sense disambiguation)
    -   Examples: ELMo, BERT, GPT (covered in transfer learning)
-   Self-Supervised Learning
    -   No need for human-labeled data
    -   Creates supervised task from unlabeled text

## Skipgram

-   Algorithm
    -   For each word position t in text:
        -   Use current word w_t as target 
        -   Words within window of ±k as context words
    -   Treat target word and neighboring context word pairs as positive samples
    -   Randomly sample other words from vocab as negative samples
    -   Train neural network to distinguish positive from negative pairs
    -   Use the learned weights as embeddings
-   Positive Examples
    -   Context Window of Size 2
    -   All words ±2 positions from the given word
-   Negative Examples
    -   Sampled according to adjusted unigram frequency
    -   Downweighted to avoid sampling stop words too frequently
    -   $P(w_j) \propto f(w_j)^{0.75}$ (raising to 0.75 power reduces frequency skew)
-   Objective Function
    -   Maximize the similarity of positive pairs
    -   Minimize the similarity of negative pairs
    -   $L_{w,c} = \log \sigma(v_w \cdot v_c) + \sum_{i=1}^{k} \mathbb{E}_{c_i \sim P_n(w)}[\log \sigma(-v_w \cdot v_{c_i})]$
    -   Where σ is the sigmoid function
    -   Use SGD to update word vectors
-   Each word has two separate embeddings
    -   Target vectors (when word appears as w)
    -   Context vectors (when word appears as c)
    -   Final embedding is often the sum or average of the two

## Enhancements

-   Unknown / OOV words
    -   Use subwords models like FastText
    -   n-grams on characters
-   GloVe
    -   Global vectors
    -   Ratios of probabilities from word-word co-occurrence matrix
-   Similarity
    -   $a:b :: a':b'$
    -   $b' = \arg \min \text{distance}(x, b - a + a')$
-   Bias
    -   Allocation Harm
        -   Unfair to different groups
        -   father-doctor, mother - housewife
    -   Representational Harm
        -   Wrong association for marginal groups
        -   African-american names to negative sentiment words 
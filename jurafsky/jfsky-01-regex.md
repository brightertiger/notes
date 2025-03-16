# Regex

## Regex

-   Language for specifying text search strings
-   Algebraic notation for characterizing a set of strings
-   Basic regular expression
    -   Match the "word" /word/
    -   Match the "word" or "Word" /[wW]ord/
    -   Match single digit /[0-9]/
-   Ranges
    -   Capital Letters /[A-Z]/
    -   Lower Case Letters /[a-z]/
    -   Single Digit /[0-9]/
-   Caret
    -   Exclusions
    -   Not an upper case letter /[^A-Z]/
    -   Not a period /[^.]/
    -   If caret is not the first character, it's treated as any other character
-   Question Mark
    -   Preceding character or nothing
    -   "word" or "words" /words?/
    -   "colour" or "color" /colou?r/
-   Kleene \*
    -   Zero or more occurrences
    -   Zero or more "a" /a\*/
    -   Zero or more "a"s or "b"s /[ab]\*/
-   Kleene +
    -   One or more occurrences
    -   One or more digits /[0-9]+/
-   Wildcard
    -   Match any single expression
    -   Any character between "beg" and "n" /beg.n/
-   Anchors
    -   Start of the line ^
    -   Lines starting with "the" /^The/
    -   End of the line $
    -   Lines ending with period /\.$/
    -   Word boundary \b /\bthe\b/
-   Grouping
    -   Disjunction "|"
        -   Match either cat or dog /cat|dog/
    -   Parentheses ()
        -   Match "guppy" or "guppies" /gupp(y|ies)/
-   Example
    -   /(^|[^a-zA-Z])[tT]he([^a-zA-Z]|$)/
        -   At the start or a non-alphabetic character
        -   At the end or non-alphabetic character
        -   Look for "the" or "The"
-   Operators
    -   Any digit \d
    -   Any non-digit \D
    -   Whitespace \s
    -   Non-whitespace \S
    -   Any alphanumeric \w
    -   Non Alpha-numeric \W
-   Range
    -   Zero or more \*
    -   One or more +
    -   Exactly zero or one ?
    -   N Occurrences {n}
    -   N-to-M Occurrences {n,m}
    -   At least N Occurrences {n,}
    -   Up to M Occurrences {,m}

## Words

-   Utterance is the spoken correlate of a sentence
-   Disfluency
    -   Fragments: broken off words
    -   Fillers or Filled Pauses "um"
-   Lemma: Lexical form of the same word (cats vs cat)
-   Types (V): Number of distinct words
-   Tokens (N): Number of running words
-   Heap's Law: $V = K N^\beta$

## Text Normalization

-   Involves three steps
    -   Tokenizing Words
    -   Normalizing word formats
    -   Segmenting sentences
-   Tokenization
    -   Breaking up an utterance into tokens
    -   Penn Treebank Tokenization (standard in many NLP applications)
    -   NLTK Regex Tokenization (flexible rule-based approach)
    -   Byte Pair Encoding
        -   Empirically determine the tokens using data
        -   Useful in dealing with unseen words (OOV problem)
        -   Use subword tokens which are arbitrary substrings
        -   Token Learner: Creates vocabulary out of corpus
        -   Token Segmentor: Applies token learner on raw test data
        -   BPE Token Learner
            -   Starts with individual characters as vocab
            -   Merges the most frequently occurring pairs and adds them to vocab
            -   Repeats the count and merge process to create longer substrings
            -   Continues until target vocab size is reached
            -   No merging across word boundaries
        -   BPE Token Parser
            -   Run the token learner on test data
            -   Apply merges in the same order in which tokens were created
            -   First split into individual characters
            -   Merge the characters based on BPE vocab

## Word Normalization

-   Putting words and tokens in a standard format
-   Case Folding: Convert everything to lowercase
    -   Simple but loses information (US vs. us, Apple vs. apple)
-   Lemmatization: Reduce words to roots
    -   Maps variants to the same base form (am, are, is → be)
    -   Stemming (removes suffixes like -ing, -ed, etc.)
    -   Porter Stemming (algorithm for English stemming)
    -   Typically requires POS information for accuracy

## Edit Distance

-   Similarity between two strings
-   Minimum number of editing operations needed to transform one string into another
    -   Insertion, Deletion and Substitution
-   Levenshtein Distance: All three operations have the same cost (typically 1)
-   Dynamic Programming solution:
    -   Create a matrix D[i,j] representing distance between first i chars of string1 and first j chars of string2
    -   Initialize: D[i,0] = i, D[0,j] = j
    -   Fill matrix: D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+cost)
      - where cost = 0 if string1[i-1] = string2[j-1], otherwise 1
    -   Final distance is in D[m,n]
-   Viterbi Algorithm is a related DP algorithm used for finding most likely sequence of hidden states 
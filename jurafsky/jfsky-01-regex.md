# Regex

## Regex

-   Language for specifying text search strings
-   Algebraic notation for characterizing a set of strings
-   Basic regular expression
    -   Match the "word" /word/
    -   Match the "word" or "Word" /\[wW\]ord/
    -   Match single digit /\[1234567890\]/
-   Ranges
    -   Capital Letters /\[A-Z\]/
    -   Lower Case Letters /\[a-z\]/
    -   Single Digit /\[0-9\]/
-   Caret
    -   Exclusions
    -   Not an upper case letter /\[\^A-Z\]/
    -   Not a period /\[\^.\]/
    -   If caret is not the first character, it's treated as any other character
-   Question Mark
    -   Preceding character or nothing
    -   "word" or "words" /words?/
    -   "colour" or "color" /colou?r/
-   Kleene \*
    -   Zero or more occurances
    -   Zero or more "a" /a\*/
    -   Zero or more "a"s or "b"s /\[ab\]\*/
-   Kleene +
    -   One or more occurances
    -   One or more digits /\[0-9\]+/
-   Wildcard
    -   Match any single expression
    -   Any character between "beg" and "n" /beg.n/
-   Anchors
    -   Start of the line \^
    -   Lines starting with "the" /\^The/
    -   End of the line \$
    -   Lines ending with period /\\.\$/
    -   Word boundary \b /\bthe\b/
-   Grouping
    -   Disjunction "\|"
        -   Match either cat or dog /cat\|dog/
    -   Paranthesis ()
        -   Match "guppy" or "guppies" /gupp(y\|ies)/
-   Example
    -   /(ˆ\|\[ˆa-zA-Z\])\[tT\]he(\[ˆa-zA-Z\]\|\$)/
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
    -   N Occurances {n}
    -   N-to-M Occurances {n,m}
    -   Atleast N Occurances {n,}
    -   Upto M Occurances {,m}

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
    -   Tokenzing Words
    -   Normalizing word formats
    -   Segmenting sentences
-   Tokenization
    -   Breaing up a an utterance into tokens
    -   Penn Treebank Tokenization
    -   NLTK Regex Tokenization
    -   Byte Pair Encoding
        -   Emperically determine the tokens using data
        -   Useful in dealing with unseen words
        -   Use subwords tokens which are arbitrary substrings
        -   Token Learner: Creates vocabulary out of corpus
        -   Token Segementor: Applies token learner on raw test data
        -   BPE Token Learner
            -   Starts with individual characters as vocab
            -   Merges the most frequently occuring pairs and adds them to back vocab
            -   Repeats the count and merge process to create longer substrings
            -   Continues until vocab size is reached
            -   No merging across word boundries
        -   BPE Token Parser
            -   Run the token learner on test data
            -   Same order in which tokens were created
            -   First split into individual characters
            -   Merge the characters based on BPE vocab

## Word Normalization

-   Putting words and tokens in a standard format
-   Case Folding: Convert everything to lowercase
-   Lemmatization: Reduce words to roots
    -   Stemming (ing, ed etc.)
    -   Porter Stemming

## Edit Distance

-   Similarity between two strings
-   Minimum number of editing operations needed to transform one string into another
    -   Insertion, Deletion and Substitution
-   Levenstien Distance: All three operations ahve the same cost
-   Dynamic Programming
    -   Viterbi Algorithm 
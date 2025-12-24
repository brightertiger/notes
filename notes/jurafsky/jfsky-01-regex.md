# Regular Expressions and Text Processing

Text processing is the foundation of NLP. Before we can analyze language, we need to handle raw text: finding patterns, breaking text into words, and normalizing variations.

## The Big Picture

**The Problem**: Raw text is messy. We need systematic ways to:
- Find patterns in text
- Break text into meaningful units (tokens)
- Handle variations (color vs. colour, ran vs. running)

**The Tools**:
- **Regular expressions**: Powerful pattern matching
- **Tokenization**: Splitting text into words/subwords
- **Normalization**: Standardizing text formats

---

## Regular Expressions

### What Are Regular Expressions?

A **regex** is a language for specifying text patterns. Think of it as "find" on steroids.

**Example**: Find all email addresses in a document.
- Without regex: Write complex code with many if statements
- With regex: `/[\w.]+@[\w.]+\.\w+/`

### Basic Patterns

| Pattern | Meaning | Example Match |
|---------|---------|---------------|
| `/word/` | Exact match | "word" |
| `/[wW]ord/` | Either character | "word" or "Word" |
| `/[0-9]/` | Any digit | "0", "5", "9" |
| `/[a-z]/` | Any lowercase | "a", "m", "z" |
| `/[A-Z]/` | Any uppercase | "A", "M", "Z" |

### Negation with Caret

`[^...]` means "NOT these characters":

| Pattern | Meaning |
|---------|---------|
| `/[^A-Z]/` | Not an uppercase letter |
| `/[^0-9]/` | Not a digit |
| `/[^.]/` | Not a period |

**Note**: Caret only means negation when it's the FIRST character inside brackets!

### Quantifiers (How Many?)

| Symbol | Meaning | Example |
|--------|---------|---------|
| `?` | Zero or one | `/colou?r/` matches "color" and "colour" |
| `*` | Zero or more | `/a*/` matches "", "a", "aaa" |
| `+` | One or more | `/[0-9]+/` matches "1", "42", "12345" |
| `{n}` | Exactly n | `/a{3}/` matches "aaa" |
| `{n,m}` | Between n and m | `/a{2,4}/` matches "aa", "aaa", "aaaa" |
| `{n,}` | At least n | `/a{2,}/` matches "aa", "aaa", ... |

### Wildcards and Anchors

**Wildcard**: `.` matches any single character
- `/beg.n/` matches "begin", "began", "begun"

**Anchors** (position markers):

| Symbol | Meaning | Example |
|--------|---------|---------|
| `^` | Start of line | `/^The/` matches lines starting with "The" |
| `$` | End of line | `/\.$/` matches lines ending with period |
| `\b` | Word boundary | `/\bthe\b/` matches "the" but not "there" |

### Grouping and Alternatives

**Disjunction** (`|`): Match either pattern
- `/cat|dog/` matches "cat" or "dog"

**Parentheses**: Group patterns
- `/gupp(y|ies)/` matches "guppy" or "guppies"

### Character Classes (Shortcuts)

| Shortcut | Meaning | Equivalent |
|----------|---------|------------|
| `\d` | Digit | `[0-9]` |
| `\D` | Non-digit | `[^0-9]` |
| `\w` | Word character | `[a-zA-Z0-9_]` |
| `\W` | Non-word | `[^a-zA-Z0-9_]` |
| `\s` | Whitespace | `[ \t\n\r]` |
| `\S` | Non-whitespace | `[^ \t\n\r]` |

### Putting It Together

**Find standalone "the" or "The"**:
```
/(^|[^a-zA-Z])[tT]he([^a-zA-Z]|$)/
```

Breaking it down:
- `(^|[^a-zA-Z])`: Start of line OR non-letter before
- `[tT]he`: "the" or "The"
- `([^a-zA-Z]|$)`: Non-letter after OR end of line

---

## Words and Tokens

### What Is a Word?

Seems simple, but it's surprisingly tricky!

**Challenges**:
- Contractions: Is "don't" one word or two?
- Hyphenation: Is "ice-cream" one word or two?
- Languages without spaces: Chinese, Japanese
- Multi-word expressions: "New York", "kick the bucket"

### Key Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Token** | An instance of a word/symbol | "the cat sat" has 3 tokens |
| **Type** | A unique word in vocabulary | "the cat sat on the mat" has 5 types |
| **Lemma** | Base/dictionary form | "runs", "ran", "running" → "run" |
| **Utterance** | Spoken equivalent of sentence | What you actually say |

### Heap's Law

Vocabulary size grows with corpus size, but sublinearly:
$$V = K \cdot N^\beta$$

Where:
- V = vocabulary size (types)
- N = corpus size (tokens)
- β ≈ 0.5–0.7, K ≈ 10–100

**Implication**: You'll always encounter new words!

---

## Text Normalization

Three main steps:
1. **Tokenization**: Break into tokens
2. **Normalization**: Standardize format
3. **Segmentation**: Find sentence boundaries

### Tokenization Approaches

**Rule-Based** (Penn Treebank):
- Standard for English NLP
- Specific rules for punctuation, contractions

**Regex-Based** (NLTK):
- Flexible, customizable
- Good for specific domains

**Subword** (BPE):
- Data-driven
- Handles unknown words gracefully

### Byte Pair Encoding (BPE)

**The Problem**: What about words we've never seen?
- Traditional tokenizers fail on "unigoogleable"
- OOV (out-of-vocabulary) tokens hurt performance

**The Solution**: Learn tokens from data!

**BPE Algorithm**:

1. **Start**: Vocabulary = individual characters
2. **Count**: Find most frequent adjacent pair
3. **Merge**: Add merged pair to vocabulary
4. **Repeat**: Until target vocabulary size reached

**Example**:
```
Corpus: "low lower lowest"
Initial vocab: {l, o, w, e, r, s, t, _}

Step 1: Most frequent pair = "lo" → add "lo"
Step 2: Most frequent pair = "low" → add "low"
...
```

**At test time**: Apply merges in the same order they were learned.

**Benefits**:
- No unknown words (can always fall back to characters)
- Common words stay whole
- Rare words split into meaningful pieces

---

## Word Normalization

### Case Folding

Convert to lowercase:
- "Apple" → "apple"
- "HELLO" → "hello"

**Trade-off**: Loses information!
- "US" (country) vs. "us" (pronoun)
- "Apple" (company) vs. "apple" (fruit)

### Lemmatization

Reduce to base form:
- "running", "ran", "runs" → "run"
- "better", "best" → "good"

**Requires**: Understanding of morphology and often part-of-speech.

### Stemming

Cruder approach — just chop suffixes:
- "running" → "run"
- "happily" → "happili" (imperfect!)

**Porter Stemmer**: Most common algorithm for English.

---

## Edit Distance

### The Problem

How similar are two strings?

**Applications**:
- Spell checking
- DNA sequence alignment
- Plagiarism detection

### Levenshtein Distance

Minimum number of **single-character edits**:
- **Insert**: cat → ca**t**s
- **Delete**: cats → cat
- **Substitute**: cat → c**o**t

### Dynamic Programming Solution

Build a matrix D where D[i,j] = distance between first i chars of string1 and first j chars of string2.

**Initialization**:
- D[i,0] = i (delete all characters)
- D[0,j] = j (insert all characters)

**Recurrence**:
```
D[i,j] = min(
    D[i-1,j] + 1,      # deletion
    D[i,j-1] + 1,      # insertion
    D[i-1,j-1] + cost  # substitution (cost=0 if match, 1 otherwise)
)
```

**Example**: Distance between "kitten" and "sitting"
- Answer: 3 (k→s, e→i, +g)

---

## Summary

| Concept | Purpose | Key Tool |
|---------|---------|----------|
| **Regex** | Find patterns | Pattern language |
| **Tokenization** | Split text | BPE, rules |
| **Normalization** | Standardize | Lemmatization, case folding |
| **Edit Distance** | Measure similarity | Dynamic programming |

### Practical Tips

1. **Always tokenize first** before any NLP task
2. **BPE** is the modern standard for neural models
3. **Be careful with normalization** — you might lose important information
4. **Regex takes practice** — use online testers to experiment!

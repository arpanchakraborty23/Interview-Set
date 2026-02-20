# Text Preprocessing Interview Questions - Complete Guide
## GenAI & NLP Roles

---

## SECTION 1: FUNDAMENTAL CONCEPTS (Beginner Level)

### Q1: What is text preprocessing and why is it important?

Text preprocessing is converting raw, unstructured text into clean, structured format suitable for ML models.

**Importance:**
- Raw text is noisy (typos, inconsistent formatting, irrelevant characters)
- Models only understand numbers, not text
- Better data = Better model accuracy and speed
- Reduces redundancy and noise

---

### Q2: List the main steps in a text preprocessing pipeline

1. Text Cleaning (remove HTML, URLs, special chars)
2. Tokenization (split into tokens)
3. Normalization (lowercase, remove punctuation)
4. Stopword Removal (optional, context-dependent)
5. Lemmatization/Stemming (optional)
6. Vectorization/Embedding (convert to numbers)

---

### Q3: What's the difference between raw text and preprocessed text?

**Raw:** "@user123 OMG!!! i LOVE this product!!! #awesome"
**Preprocessed:** "love product"

Raw has noise, inconsistency. Preprocessed is clean, standardized.

---

### Q4: Why can't we directly feed raw text to neural networks?

Neural networks require:
- Fixed-size numerical inputs
- Structured data format
- Consistent vocabulary
Raw text is unstructured, variable-length, symbolic

---

### Q5: What is tokenization at its core?

Breaking text into meaningful units (words, subwords, characters) that can be processed.

---

## SECTION 2: TOKENIZATION (Core Concept)

### Q6: What is tokenization? Why is it necessary?

**Definition:** Process of breaking text into meaningful units called tokens.

**Why necessary:**
- Bridge between text and numbers
- Vocabulary control
- Semantic preservation
- Computational efficiency

**Example:**
```
Text: "I love machine learning"
Word tokens: ["I", "love", "machine", "learning"]
Character tokens: ["I", " ", "l", "o", "v", "e", ...]
Subword tokens: ["I", "love", "mach", "##ine", "learning"]
```

---

### Q7: Compare word-level, character-level, and subword tokenization

| Type | Example | Pros | Cons |
|------|---------|------|------|
| **Word-level** | ["running", "quickly"] | Intuitive, preserves semantics | Large vocab, OOV problems |
| **Character-level** | ["r", "u", "n", "##n", "i", "n", "g"] | Small vocab, handles OOV | Loses word meaning, long sequences |
| **Subword** (BPE, WordPiece) | ["run", "##ning"] | Balances both | Slightly complex |

**Use cases:**
- Word: Traditional NLP (TF-IDF, old classifiers)
- Character: Handling typos, morphology
- Subword: Modern transformers (BERT, GPT)

---

### Q8: What is Byte-Pair Encoding (BPE)?

Algorithm that iteratively merges most frequent token pairs.

**Steps:**
1. Start with character-level tokens
2. Count all consecutive pairs
3. Merge most frequent pair
4. Repeat until vocabulary size reached

**Example:**
```
Initial: ["l", "o", "w", "l", "o", "w"]
Step 1: Merge "l"+"o" → ["lo", "w", "lo", "w"]
Step 2: Merge "lo"+"w" → ["low", "low"]
Step 3: Merge "low"+"low" → [vocabulary limit]
```

**Why used:** Language-agnostic, handles rare words, good vocabulary size

---

### Q9: What is WordPiece tokenization?

Similar to BPE but uses **likelihood-based merging** instead of frequency.

**Key difference:**
```
BPE: Merges most frequent pair
WordPiece: Merges pair maximizing P(X,Y)/(P(X)*P(Y))
           (prioritizes meaningful combinations)
```

**Used in:** BERT, RoBERTa
**Advantage:** More semantically meaningful subwords than BPE

---

### Q10: What are out-of-vocabulary (OOV) words? How are they handled?

**OOV words:** Tokens not in model's vocabulary

**Handling by tokenization type:**

**Word tokenization:**
```
Vocab: ["the", "cat", "sat"]
Text: "the dog sat"
"dog" is OOV → becomes [UNK] token
Problem: Information loss
```

**Subword tokenization:**
```
Vocab includes subword pieces
"unprecedented" → ["un", "preced", "ented"]
Handles: Rare words as combinations
```

**Best:** Subword tokenization balances OOV handling and sequence length

---

### Q11: What is the difference between SentencePiece and BPE?

**SentencePiece:**
- Works directly on raw text (no pre-tokenization)
- Language-agnostic (learns from data)
- Used in: T5, ALBERT, XLNet

**BPE:**
- Requires pre-tokenization (split by spaces/punctuation)
- Language-specific preprocessing needed
- Used in: GPT-2, GPT-3

**Key advantage of SentencePiece:** No language-specific preprocessing needed

---

## SECTION 3: TEXT NORMALIZATION

### Q12: What is text normalization?

Converting text into standard, consistent form:
- Lowercasing: "Hello" → "hello"
- Removing punctuation: "don't" → "dont"
- Removing accents: "café" → "cafe"
- Whitespace handling

**Benefits:**
- Reduces vocabulary
- Improves learning
- Better generalization

---

### Q13: When should you NOT normalize text aggressively?

**Cases where normalization hurts:**

1. **Acronyms/Proper nouns:**
```
"IBM", "USA" → lose meaning when lowercased
```

2. **Sentiment/Emphasis:**
```
"I LOVE this!" vs "I love this"
Uppercase conveys emphasis
```

3. **Code/Technical text:**
```
"Python", "True", "False" (case matters)
```

4. **Contractions:**
```
"don't" vs "dont" (different meaning)
```

**Modern approach:** Don't normalize aggressively for transformers. They're robust enough to ignore noise themselves.

---

### Q14: Expand contractions vs. remove them - which is better?

**Expand contractions:**
```
"don't" → "do not"
"can't" → "cannot"
Pros: Clearer meaning, better for grammar-sensitive tasks
Cons: Adds tokens, requires manual mapping
```

**Remove contractions:**
```
"don't" → "dont"
Pros: Simple, vocabulary reduction
Cons: Loses information, may confuse models
```

**Best practice:**
- Expand: Traditional NLP, rule-based
- Keep original: Deep learning (transformers learn)

---

### Q15: What is the impact of lowercasing on model performance?

**Impacts:**
- Reduces vocabulary (good)
- Loses information about proper nouns (bad)
- Makes "NLP" and "nlp" identical (mixed effect)

**Modern transformers:** Relatively robust to case differences. BERT learns to handle mixed case without preprocessing.

**When to lowercase:**
- Limited data
- Task doesn't depend on capitalization

**When to keep case:**
- Named entity recognition
- Question answering
- Tasks requiring NER

---

## SECTION 4: STOPWORDS & MORPHOLOGY

### Q16: What are stopwords? When to remove them?

Stopwords: Common words with little semantic value (the, is, a, and, or)

**REMOVE when:**
- Using traditional ML (TF-IDF, SVM, Naive Bayes)
- Small datasets
- Building search engines
- ~50% vocabulary reduction

**KEEP when:**
- Deep learning (CNN, RNN, Transformers)
- Syntax important: "not good" vs "good"
- Text generation, translation
- Modern LLMs (they learn what's important)

**Critical example:**
```
Sentiment: "This movie is NOT good"
Remove stopwords: ["movie", "good"]
Lost "NOT" → sentiment flipped!
```

---

### Q17: Stemming vs. Lemmatization - detailed comparison

**Stemming:**
- Rule-based, removes suffixes
- Fast
- Often produces non-words

```
running → runn (wrong)
jumped → jump (correct)
studies → studi (wrong)
```

**Lemmatization:**
- Dictionary-based, finds canonical form
- Slow
- Always produces real words

```
running → run (correct)
jumped → jump (correct)
studies → study (correct)
```

**Decision matrix:**
- Speed critical → Stemming
- Production system → Lemmatization
- Transformers → Neither (skip both)

---

### Q18: Why don't we use stemming/lemmatization with BERT/GPT?

1. **Subword tokenization handles it:**
```
"running" → ["run", "##ning"]
"runner" → ["run", "##ner"]
Model learns "run" is shared
```

2. **Context is key:**
```
"bank" (river) vs "bank" (financial)
Different lemmas, same word
Pre-lemmatization loses context
```

3. **Transformers learn morphology:**
- Self-attention captures relationships
- No need for explicit lemmatization

4. **Research shows:** Aggressive preprocessing actually *hurts* transformer performance

---

### Q19: Handling contractions - best approach for deep learning

**For transformers (BERT, GPT):**
Keep original contractions. Model handles them.

```
"don't" → tokenized as is
Model learns it means "do not"
More robust than manual expansion
```

**For classical ML:**
Expand for clarity and vocabulary reduction.

---

## SECTION 5: EMBEDDINGS & REPRESENTATIONS

### Q20: What are word embeddings?

Dense vectors representing words in continuous vector space.

**Why needed:**
- Neural networks require numbers
- Semantic relationships encoded
- Dimensionality reduction

**Example:**
```
One-hot: "king" = [0, 0, 1, 0, ..., 0] (50K dimensions)
Embedding: "king" = [0.2, -0.5, 0.8, ...] (300 dimensions)
                    (semantic meaning encoded)
```

**Magic property:**
```
king - man + woman ≈ queen
(Vector arithmetic reveals relationships)
```

---

### Q21: Compare TF-IDF, Word2Vec, and BERT embeddings

**TF-IDF:**
- Statistical, sparse vectors
- Bag-of-words, no context
- Best for: Search, traditional ML
```
Formula: TF(t) * log(N/df(t))
Output: High-dimensional, mostly zeros
```

**Word2Vec:**
- Dense vectors, static per word
- Captures some semantics
- One vector regardless of context
```
"bank" (financial) and "bank" (river) = same vector
Good for: Quick NLP, transfer learning
```

**BERT/Contextual:**
- Dense vectors, dynamic per context
- Polysemy handled
- Different vectors based on context
```
"bank" (financial) ≠ "bank" (river)
Best for: Deep learning, state-of-the-art
```

**Comparison table:**

| Feature | TF-IDF | Word2Vec | BERT |
|---------|--------|----------|------|
| Type | Statistical | Static | Contextual |
| Dimensions | 50K+ | 300 | 768 |
| Sparse/Dense | Sparse | Dense | Dense |
| Context-aware | No | No | Yes |
| Semantic | No | Yes | Yes |
| Speed | Fast | Medium | Slow (but pre-trained) |

---

### Q22: What is the difference between static and contextual embeddings?

**Static (Word2Vec, GloVe):**
```
Same word → always same vector
"bank" in "financial bank" = "bank" in "river bank"
Problem: Can't handle polysemy (multiple meanings)
```

**Contextual (BERT, ELMo, GPT):**
```
Same word → different vectors in different contexts
"bank" (financial) → [0.1, 0.2, 0.5, ...]
"bank" (river) → [0.5, 0.1, 0.2, ...]
Solution: Context-aware representations
```

**How BERT does it:**
- Processes entire sequence at once
- Self-attention for each word
- Each word influenced by all others
- Result: Context-dependent vectors

**Impact:**
Contextual embeddings are why transformers excel at NLP tasks.

---

### Q23: What is transfer learning with embeddings?

Using pre-trained embeddings instead of training from scratch.

**Traditional approach:**
```
1. Train embedding on huge corpus
2. Use those embeddings in your model
3. Fine-tune or freeze
Benefit: Much faster, less data needed
```

**Example:**
```
Pre-trained BERT embeddings:
- Trained on 3.3B words
- Wikipedia + BookCorpus

Your task:
- Sentiment analysis (small dataset)
- Use BERT embeddings directly
- Train only classifier on top
- Saves time & data
```

**Frozen vs. Fine-tuned:**
- **Frozen:** Use embeddings as-is (fast)
- **Fine-tuned:** Update embeddings for your task (slower but better)

---

## SECTION 6: DEEP LEARNING & TRANSFORMERS

### Q24: What is padding and truncation?

**Padding:**
Adding dummy tokens to make all sequences same length.

```
Seq 1: "I love cats" → [I, love, cats, <PAD>, <PAD>]
Seq 2: "I like dogs and birds" → [I, like, dogs, and, birds]
All sequences now length = 5
```

**Why necessary:** Neural networks need fixed input sizes for batching

**Truncation:**
Cutting sequences that exceed max length.

```
Seq: "I love this movie because it has great actors..."
Max length: 10
Result: "I love this movie because it has great"
Lost: "actors and..."
```

**Strategies:**
- Head truncation: Keep beginning (classification)
- Tail truncation: Keep end (better if important info at end)
- Both: Keep beginning and end (rare)

**Trade-offs:**
```
Longer max_length:
  ✓ More information
  ✗ Slower, more memory

Shorter max_length:
  ✓ Faster training
  ✗ Information loss
```

**Best practice:** Analyze distribution, set max_length to keep 95%+ of data

---

### Q25: What's the difference between encoding and decoding?

**Encoding:**
Text → Numerical representation

```
"I love machine learning"
→ Tokenization: ["I", "love", "machine", "learning"]
→ Embedding: [[0.2, -0.5, ...], [0.3, 0.1, ...], ...]
→ Encoding: [0.45, 0.12, ...] (compressed)
Result: Fixed-size vector
```

**Use cases:** Classification, semantic search, anomaly detection

**Decoding:**
Numerical representation → Text

```
Vector: [0.45, 0.12, ...]
→ Decoder
→ "I love machine learning"
```

**Use cases:** Generation, translation, summarization

---

### Q26: Explain encoder-only vs. decoder-only vs. encoder-decoder

**Encoder-only (BERT, RoBERTa):**
```
Input: "What is machine learning?"
Output: [semantic representation]
Used for: Classification, Q&A ranking, NER
Bidirectional context (can see all words)
Cannot generate text
```

**Decoder-only (GPT, LLaMA):**
```
Input: [representation]
Output: "Machine learning is..." (generated text)
Used for: Text generation, chatbots, code generation
Unidirectional context (only previous tokens)
Auto-regressive (one token at a time)
```

**Encoder-Decoder (T5, BART, Seq2Seq):**
```
Input: "Translate to French: Hello"
Encoder: Bidirectional understanding
Decoder: Generates "Bonjour"
Used for: Translation, summarization, QA generation
Both understanding and generation
```

**Comparison:**
| Aspect | Encoder | Decoder | Both |
|--------|---------|---------|------|
| Best for | Understanding | Generation | Translation |
| Example | BERT | GPT-3 | T5 |
| Speed | Fast | Slower | Medium |

---

### Q27: Why do transformers use subword tokenization?

**Problem with word tokenization:**
```
Vocabulary size: 50K-200K words
Every new word: adds to vocabulary
Rare words: "unprecedented" (OOV)
Misspellings: Becomes [UNK] (lost)
Result: Large model, information loss
```

**Benefits of subword:**

1. **Smaller vocabulary:**
```
Word: 50K tokens
Subword: 10K tokens
Savings: 80% smaller
```

2. **Handles rare words:**
```
"unprecedented" → ["un", "preced", "ented"]
All subwords in vocabulary → no [UNK]
```

3. **Language-agnostic:**
```
Doesn't need language-specific dictionary
Learned from data automatically
Works for any language
```

4. **Morphological:**
```
"running" → ["run", "##ning"]
"runner" → ["run", "##ner"]
Implicit stemming - shares "run"
```

5. **Best of both worlds:**
```
Character tokenization: Too many tokens
Word tokenization: Too large vocabulary
Subword: Sweet spot
```

---

### Q28: What are special tokens? ([CLS], [SEP], [UNK], [PAD])

**[CLS] - Classification token:**
```
Position: Beginning of sequence
Purpose: Aggregates sequence info
[CLS] This movie is great [SEP]
↓ (embedding contains full sequence meaning)
Used for classification
```

**[SEP] - Separator token:**
```
Purpose: Separates two sequences
"[CLS] Question: What is AI? [SEP] Answer: AI is..."
Tells model where one sequence ends, another begins
```

**[UNK] - Unknown token:**
```
Purpose: Represents out-of-vocabulary words
Text: "unprecedented" (not in vocab)
Becomes: [UNK]
Problem: Information loss
```

**[PAD] - Padding token:**
```
Purpose: Fills shorter sequences
Seq: ["the", "cat", "sat"]
Padded: ["the", "cat", "sat", "[PAD]", "[PAD]"]
Model learns to ignore via attention mask
```

**Attention mask (critical!):**
```
Tokens: [The, cat, sat, [PAD], [PAD]]
Mask:    [1,  1,   1,  0,      0]
         (1=attend, 0=ignore)
Without mask: Model attends to meaningless [PAD]
With mask: Model ignores [PAD]
```

---

## SECTION 7: PRACTICAL CONSIDERATIONS

### Q29: How do you choose max_length parameter?

**Analysis approach:**
```
Step 1: Get length distribution of all texts
Step 2: Calculate percentiles (50th, 95th, 99th)
Step 3: Choose max_length to capture 95-99%

Example:
50th percentile: 50 tokens (median)
95th percentile: 256 tokens
99th percentile: 512 tokens

Conservative choice: max_length=256 (95% coverage)
Aggressive choice: max_length=512 (99% coverage)
```

**Trade-offs:**
```
Longer max_length:
  ✓ Keep more info
  ✗ Slower training (quadratic with length)
  ✗ More memory

Shorter max_length:
  ✓ Faster training
  ✗ Truncation loss
```

**Practical:** Start with model default (512 for BERT). Increase if >90% truncated.

---

### Q30: How do you handle imbalanced text lengths?

**Problem:**
```
Some texts: 50 tokens
Other texts: 5000 tokens
Padding/truncation both suboptimal
```

**Solutions:**

1. **Bucketing:**
```
Group by length ranges
[0-100], [100-200], [200-500], [500+]
Set appropriate max_length per bucket
Reduces wasted padding
```

2. **Dynamic padding:**
```
Instead of fixed max_length
Pad each batch to max in that batch
Reduces padding overhead
(used in HuggingFace)
```

3. **Sampling strategy:**
```
Upsample short sequences
Downsample very long ones
Balance distribution
```

4. **Hierarchical approach:**
```
Very long documents: Split into chunks
Process chunks separately
Aggregate results
```

---

### Q31: How do you handle multiple languages in preprocessing?

**Challenges:**
- Different alphabets (Latin, Cyrillic, Arabic, CJK)
- Different tokenization needs
- Diacritics/accents

**Solutions:**

1. **Language-specific tokenizers:**
```
Spacy: Supports 20+ languages
NLTK: Language-specific rules
Jieba: Chinese segmentation
MeCab: Japanese morphology
```

2. **Universal approaches:**
```
SentencePiece: Language-agnostic
mBERT: Multilingual BERT
XLM-RoBERTa: 100+ languages
No language-specific preprocessing needed
```

3. **Preprocessing strategy:**
```
Use Unicode normalization (NFC/NFKC)
Don't lowercase for some languages (preserves meaning)
Handle diacritics carefully
Test language-specific edge cases
```

---

### Q32: How to handle domain-specific text? (Medical, Legal, Code)

**Challenges:**
- Domain-specific vocabulary
- Special characters/formatting
- Context-specific tokenization

**Solutions:**

1. **Custom tokenizers:**
```
Medical: Keep "mg", "ml" as single tokens
Legal: Preserve "§", "§§" formatting
Code: Handle operators, brackets specially
```

2. **Domain vocabularies:**
```
Include domain-specific terms
Medical: "carcinoma", "hypertension"
Legal: "plaintiff", "defendant"
```

3. **Pre-trained domain models:**
```
BioBERT: Biomedical texts
LegalBERT: Legal documents
CodeBERT: Source code
```

4. **Handling special chars:**
```
Code: Keep "{}", "()", "[]"
Chemical: Keep "H2O", "C6H12O6"
Math: Keep "∑", "∫", "√"
```

---

### Q33: How do you preprocess noisy social media text?

**Challenges:**
```
Typos: "u" instead of "you"
Slang: "lol", "omg", "fam"
Emojis: 😂, 💔
Hashtags: #awesome
Mentions: @user123
URLs: http://bit.ly/xyz
```

**Preprocessing strategy:**

```
1. Handle emojis:
   - Remove: 😂 → deleted
   - Convert: 😂 → happy_face
   - Keep: Original

2. Handle URLs:
   Remove or replace: http://... → [URL]

3. Handle mentions:
   - Remove: @user123 → deleted
   - Replace: @user123 → [USER]

4. Handle hashtags:
   - Remove #: #awesome → awesome
   - Keep: #awesome (for context)

5. Fix typos (optional):
   - "u" → "you"
   - "ur" → "your"
   - Careful: Can remove slang value

6. Normalize:
   - "lol" → keep (sentiment indicator)
   - "..." → ". "
   - Multiple spaces → single space
```

**Modern approach:**
Use transformers trained on Twitter/social data. They handle noise better than manual preprocessing.

---

## SECTION 8: ADVANCED & LLM-SPECIFIC

### Q34: Preprocessing for LLMs vs. Traditional NLP - differences

**Traditional NLP (TF-IDF, SVM):**
```
Needs aggressive preprocessing:
- Lowercase: YES
- Remove stopwords: YES
- Lemmatize: YES
- Remove punctuation: YES
Reason: Limited learning capacity, benefits from clean data
```

**Deep Learning (CNN/RNN):**
```
Moderate preprocessing:
- Lowercase: Optional
- Remove stopwords: NO (syntax matters)
- Lemmatize: NO
- Remove punctuation: Optional
Reason: Can learn from raw data, but some cleaning helps
```

**LLMs (GPT, BERT, LLaMA):**
```
Minimal preprocessing:
- Lowercase: NO (preserve case info)
- Remove stopwords: NO (they're important)
- Lemmatize: NO (lose variation)
- Remove punctuation: NO (syntax matters)
- Remove special chars: NO
Reason: Transformers are robust, aggressive preprocessing hurts
```

**Key insight:**
More powerful models need LESS preprocessing.
Research shows over-cleaning hurts LLM performance.

---

### Q35: Tokenization for LLMs - special considerations

**Standard approach fails:**
```
Text: "don't"
BERT tokenizes: ["don", "'", "t"]
GPT-2 might tokenize: ["don", "'", "t"] or differently
Issue: Different models tokenize differently
```

**LLM-specific tokenization:**

1. **Use model's official tokenizer:**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Always use the exact tokenizer the model was trained with
```

2. **Understand BPE variants:**
```
GPT-2: BPE with specific merges
GPT-3: Same as GPT-2
LLaMA: SentencePiece (different)
Mistral: LLaMA tokenizer

Different tokenizers = different tokens = different behavior
```

3. **Vocabulary size matters:**
```
GPT-2: 50K tokens
GPT-3: 50K tokens
LLaMA: 32K tokens
Mistral: 32K tokens (extended)

Smaller vocab → longer token sequences
Longer sequences → more computation
```

4. **Special tokens for prompting:**
```
Different models use different special tokens
GPT: No [CLS]/[SEP]
BERT: Uses [CLS]/[SEP]
T5: Uses [CLS], [SEP], etc.

For LLM prompts:
Use model's convention (end token, BOS, EOS)
```

---

### Q36: Handling long documents for LLMs

**Problem:**
```
Max context: 2K, 4K, 8K tokens (depends on model)
Document: 10K tokens
Solution: How to preprocess?
```

**Approaches:**

1. **Truncation (simple but loses info):**
```
Take first 2K tokens
Problem: End of document ignored
```

2. **Chunking + Sliding window:**
```
Split document into 1.5K chunks
Overlap: 500 tokens (context continuity)
Process each chunk separately
```

3. **Hierarchical summarization:**
```
Chunk 1 → Summarize → Summary 1
Chunk 2 → Summarize → Summary 2
...
Combine summaries
Then process combined
```

4. **Retrieval-augmented generation (RAG):**
```
Split into chunks (e.g., 500 tokens)
Embed all chunks
For query: Retrieve most relevant chunks
Only process relevant chunks
Avoids full document processing
```

5. **Sliding window with stride:**
```
Window: 2K tokens
Stride: 1K tokens
Chunk 1: [0:2K]
Chunk 2: [1K:3K]
Chunk 3: [2K:4K]
Overlap captures context
```

---

### Q37: Preprocessing for prompt engineering

**Key differences:**
```
Traditional task: "Classify sentiment of: 'Great movie!'"
Prompt engineering: "You are a movie critic. Review: 'Great movie!'"
```

**Preprocessing considerations:**

1. **Preserve formatting:**
```
Don't remove punctuation
Don't lowercase
Don't remove special chars
Format matters for instruction-following
```

2. **Structured prompts:**
```
Input:
---
Task: Sentiment analysis
Text: "Great movie"
---

Preprocessing: Preserve structure exactly
```

3. **Few-shot examples:**
```
Example 1: "Terrible movie" → Negative
Example 2: "Excellent film" → Positive

Preprocessing: Keep examples unchanged
Same style as test input
```

4. **System prompts:**
```
"You are a helpful assistant..."
Preprocessing: No cleaning
Exact wording matters
```

---

### Q38: Token efficiency and cost optimization

**Problem:**
```
LLM pricing: $X per 1K tokens
Long preprocessing → more tokens
More tokens → higher cost
```

**Optimization strategies:**

1. **Eliminate redundancy:**
```
Remove duplicate sentences
Remove repeated explanations
Combine related paragraphs
Effect: Fewer tokens, same info
```

2. **Smart truncation:**
```
Instead of: Cut at max_length
Smart: Cut at sentence boundary
Effect: Cleaner input, better results
```

3. **Compression techniques:**
```
Summarize long sections
Use bullet points instead of prose
Abbreviate common phrases
Effect: 30-50% token reduction
```

4. **Batching efficiency:**
```
Preprocess similar-length texts together
Reduces padding overhead
Effect: 10-20% savings
```

5. **Token counting before submission:**
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode(text)
num_tokens = len(tokens)
cost = (num_tokens / 1000) * price_per_1k
# Know cost before API call
```

---

### Q39: Handling code in preprocessing

**Challenges:**
```
Code != Natural language
Different syntax rules
Formatting matters
Indentation is semantic
```

**Preprocessing strategy:**

1. **Don't lowercase:**
```
"Python" != "python"
"True" != "true" (boolean)
Preserve case
```

2. **Don't remove special chars:**
```
Keep: "{}", "()", "[]", ";"
Remove: Harmful
```

3. **Handle comments:**
```
Keep: Code comments (explain intent)
Or: Remove (redundant with code)
Depends on task
```

4. **Tokenization:**
```
Use code-specific tokenizer:
- Pygments tokenizer
- Tree-sitter
- Language-specific parser

Better than generic tokenizer
```

5. **Preserve formatting:**
```
Keep indentation (semantic in Python)
Keep line breaks
Keep operator spacing
```

---

### Q40: Multilingual LLM preprocessing

**Challenge:**
```
Single model for 100+ languages
Different scripts, grammars, edge cases
One preprocessing strategy for all?
```

**Strategy:**

1. **Language detection first:**
```python
from langdetect import detect_langs
lang = detect_langs(text)[0].lang
# Process based on language
```

2. **Unified preprocessing:**
```
Use: SentencePiece tokenizer
Works: All languages equally
No: Language-specific preprocessing needed
```

3. **Minimal preprocessing:**
```
Don't: Assume English preprocessing
Do: Preserve original formatting
Let: Multilingual model handle diversity
```

4. **Script-specific handling:**
```
CJK (Chinese/Japanese/Korean):
  - No word boundaries
  - Different segmentation
  - Use language-specific tokenizers

Arabic:
  - Right-to-left
  - Diacritics matter (sometimes)
  - Special handling needed
```

---

## SECTION 9: COMMON MISTAKES & BEST PRACTICES

### Q41: What are common preprocessing mistakes?

**Mistake 1: Over-preprocessing**
```
Remove too much info
Lowercase + remove punctuation + lemmatize + remove stopwords
Result: Lost signal
Fix: Use minimal preprocessing for deep learning
```

**Mistake 2: Preprocessing mismatch**
```
Preprocess training data one way
Preprocess test data differently
Result: Distribution shift
Fix: Same preprocessing for train and test
```

**Mistake 3: Ignoring data distribution**
```
Set max_length = 512 for all data
90% of data is <100 tokens
Result: Wasted padding, slower training
Fix: Analyze distribution first
```

**Mistake 4: Using wrong tokenizer**
```
Train with BERT tokenizer
Use GPT tokenizer at inference
Result: Different tokens, wrong results
Fix: Always use model's exact tokenizer
```

**Mistake 5: Forgetting about test set**
```
Preprocess training data
Different preprocessing on test
Result: Train-test mismatch
Fix: Fit preprocessing on train, apply identically to test
```

**Mistake 6: Removing important tokens**
```
Remove stopwords: "not", "no"
Task: Sentiment analysis
Lose negation signal
Result: "not good" becomes "good" (opposite!)
Fix: Keep stopwords for tasks where syntax matters
```

---

### Q42: Best practices for text preprocessing

1. **Start minimal:**
```
Baseline: No preprocessing
Add: One step at a time
Measure: Impact on validation accuracy
Principle: Only add if it helps
```

2. **Know your task:**
```
Classification: Can afford more preprocessing
Generation: Need to preserve structure
NER: Must keep context
Adjust: Preprocessing accordingly
```

3. **Use standard tools:**
```
NLTK: Traditional NLP
Spacy: Modern, efficient
HuggingFace Tokenizers: Fast, compatible
Don't: Reinvent the wheel
```

4. **Validate impact:**
```
Preprocess version A: Accuracy 85%
Preprocess version B: Accuracy 87%
Use: Version B if other factors equal
```

5. **Document everything:**
```
What preprocessing: Applied
Why: Reasoning
Parameters: max_length, vocab_size
Results: Accuracy, speed
Code: Reproducibility
```

6. **Test edge cases:**
```
Emojis: How handled?
URLs: Removed or kept?
Multiple spaces: How normalized?
Special chars: What happens?
```

7. **Version control:**
```
Save: Tokenizer vocab
Save: Preprocessing config
Version: Track changes
Reason: Reproducibility
```

---

## SECTION 10: INTERVIEW TIPS

### Key Concepts to Master:
1. Tokenization (word, char, subword)
2. BPE and WordPiece
3. When to preprocess, when not to
4. Embeddings: TF-IDF vs Word2Vec vs BERT
5. Transformers: encoder, decoder, encoder-decoder
6. Special tokens and attention masks
7. Practical considerations: max_length, vocab_size
8. LLM-specific preprocessing
9. Domain-specific preprocessing

### Common Interview Questions:
- "Why does your preprocessing choice matter?" → Task-specific
- "Compare approaches A and B" → Trade-offs always exist
- "How would you handle [X]?" → Problem-solving approach
- "Why is [X] better than [Y]?" → Reasoning over memorization
- "What would you change?" → Critical thinking

### Pro Tips:
1. Ask clarifying questions: "For what task?"
2. Discuss trade-offs: "Benefits and drawbacks?"
3. Mention modern approaches: "Transformers need less preprocessing"
4. Show experience: "In project X, I..."
5. Admit uncertainty: "I'm not sure, let me think..."

---

## Quick Reference: Preprocessing Decision Tree

```
Task: ?
├── Traditional ML (TF-IDF, SVM)
│   └── Aggressive preprocessing (lowercase, lemmatize, remove stopwords)
├── Deep Learning (CNN/RNN)
│   └── Moderate preprocessing (keep stopwords, minimal normalization)
└── LLMs (BERT, GPT)
    └── Minimal preprocessing (preserve everything, let model learn)

Tokenization?
├── Word vocabulary < 10K → Use word tokenization
├── Large vocabulary / rare words → Use subword (BPE/WordPiece)
└── Any language, need simplicity → Use SentencePiece

Long documents?
├── < 512 tokens → Pad to max_length
├── 512-2K tokens → Truncate intelligently
└── > 2K tokens → Chunk or use retrieval-augmented approach

Multiple languages?
├── Single language → Language-specific tokenizer
└── Multiple languages → Multilingual BERT or SentencePiece

Noisy data (social media)?
├── Keep original structure
├── Handle URLs/mentions/hashtags contextually
└── Use transformer trained on similar data
```

---

End of Complete Guide

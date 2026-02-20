# Text Preprocessing Coding Interview Problems
## Solve These Yourself! 🎯

---

## SECTION 1: BASIC TEXT CLEANING (Easy)

### Problem 1: Remove HTML Tags

**Difficulty:** Easy
**Time Limit:** 5 minutes
**Concepts:** String manipulation, Regex

**Problem Statement:**
Remove all HTML tags from text while preserving the content inside.

**Input & Output Examples:**

```python
Test Case 1:
Input: "<p>Hello World</p>"
Output: "Hello World"

Test Case 2:
Input: "<div><span>Python</span> is <b>awesome</b></div>"
Output: "Python is awesome"

Test Case 3:
Input: "<h1>Title</h1><p>Content with <br/> line break</p>"
Output: "TitleContent with  line break"

Test Case 4:
Input: "<a href='https://example.com'>Click here</a>"
Output: "Click here"

Test Case 5:
Input: "No tags here"
Output: "No tags here"

Test Case 6:
Input: "<p>Text with <img src='pic.jpg'/> image</p>"
Output: "Text with  image"
```

**Constraints:**
- Handle self-closing tags (`<br/>`, `<img/>`)
- Remove tag attributes
- Don't worry about HTML entities (&nbsp;, &lt;, etc.)
- Preserve spaces between content

**Expected Function Signature:**
```python
def remove_html_tags(text: str) -> str:
    pass
```

---

### Problem 2: Remove URLs from Text

**Difficulty:** Easy
**Time Limit:** 5 minutes
**Concepts:** Regex, String patterns

**Problem Statement:**
Remove URLs (http, https, ftp, www) from text.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Check out https://example.com for more info"
Output: "Check out  for more info"

Test Case 2:
Input: "Visit http://www.google.com or www.github.com"
Output: "Visit  or "

Test Case 3:
Input: "Download from ftp://files.example.com/document.pdf"
Output: "Download from "

Test Case 4:
Input: "No URLs here, just text"
Output: "No URLs here, just text"

Test Case 5:
Input: "Multiple URLs: https://site1.com and https://site2.org and www.site3.net"
Output: "Multiple URLs:  and  and "

Test Case 6:
Input: "Email notaurl@example.com should not be removed"
Output: "Email notaurl@example.com should not be removed"

Test Case 7:
Input: "https://example.com/path?query=value&other=123"
Output: ""
```

**Constraints:**
- Match http://, https://, ftp://, www.
- Remove entire URL (including query parameters, paths)
- Don't remove email addresses
- Preserve surrounding text

**Expected Function Signature:**
```python
def remove_urls(text: str) -> str:
    pass
```

---

### Problem 3: Normalize Whitespace

**Difficulty:** Easy
**Time Limit:** 5 minutes
**Concepts:** String manipulation, Regex

**Problem Statement:**
Normalize whitespace by:
- Removing leading/trailing spaces
- Converting multiple spaces to single space
- Handling tabs and newlines

**Input & Output Examples:**

```python
Test Case 1:
Input: "   Hello   World   "
Output: "Hello World"

Test Case 2:
Input: "Python\n\nJavaScript"
Output: "Python JavaScript"

Test Case 3:
Input: "One\tTwo\tThree"
Output: "One Two Three"

Test Case 4:
Input: "Text\r\nwith\r\nwindows\r\nlinebreaks"
Output: "Text with windows linebreaks"

Test Case 5:
Input: "Mixed   spaces\t\ttabs\n\nnewlines"
Output: "Mixed spaces tabs newlines"

Test Case 6:
Input: "Normal text"
Output: "Normal text"

Test Case 7:
Input: "\n\n\t\t  \n"
Output: ""
```

**Constraints:**
- Replace tabs with spaces
- Replace newlines with spaces
- Remove leading/trailing whitespace
- Collapse multiple spaces to one

**Expected Function Signature:**
```python
def normalize_whitespace(text: str) -> str:
    pass
```

---

### Problem 4: Convert Text to Lowercase (with edge cases)

**Difficulty:** Easy
**Time Limit:** 3 minutes
**Concepts:** String methods, Unicode

**Problem Statement:**
Convert text to lowercase, handling special cases.

**Input & Output Examples:**

```python
Test Case 1:
Input: "HELLO WORLD"
Output: "hello world"

Test Case 2:
Input: "Python123ABC"
Output: "python123abc"

Test Case 3:
Input: "Café"
Output: "café"

Test Case 4:
Input: "I'm HAPPY!"
Output: "i'm happy!"

Test Case 5:
Input: "123 !@# Special"
Output: "123 !@# special"

Test Case 6:
Input: "ÉLÈVE NAÏVE"
Output: "élève naïve"

Test Case 7:
Input: ""
Output: ""
```

**Constraints:**
- Handle Unicode characters
- Numbers and special chars unchanged
- Preserve order and position

**Expected Function Signature:**
```python
def to_lowercase(text: str) -> str:
    pass
```

---

## SECTION 2: PUNCTUATION & SPECIAL CHARACTERS (Medium)

### Problem 5: Remove Punctuation (Keep Contractions)

**Difficulty:** Medium
**Time Limit:** 8 minutes
**Concepts:** Regex, String methods

**Problem Statement:**
Remove punctuation but preserve apostrophes in contractions.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Don't worry!"
Output: "Don't worry"

Test Case 2:
Input: "It's fine, isn't it?"
Output: "It's fine isn't it"

Test Case 3:
Input: "What's up? I'm good!"
Output: "What's up I'm good"

Test Case 4:
Input: "She said, 'Hello!'"
Output: "She said Hello"

Test Case 5:
Input: "Price: $99.99!!!"
Output: "Price 9999"

Test Case 6:
Input: "Email: test@example.com (contact us)"
Output: "Email testexamplecom contact us"

Test Case 7:
Input: "No punctuation here"
Output: "No punctuation here"

Test Case 8:
Input: "Can't, won't, shouldn't - we'll see!"
Output: "Can't won't shouldn't we'll see"

Test Case 9:
Input: "Multiple...periods and---dashes"
Output: "Multipleperiods anddashes"
```

**Constraints:**
- Keep apostrophes ONLY in contractions (don't, it's, etc.)
- Remove all other punctuation: .,!?;:'"()-
- Handle edge cases

**Expected Function Signature:**
```python
def remove_punctuation_keep_contractions(text: str) -> str:
    pass
```

---

### Problem 6: Remove Special Characters (Keep Alphanumeric)

**Difficulty:** Easy
**Time Limit:** 5 minutes
**Concepts:** Regex

**Problem Statement:**
Remove all special characters, keep only letters, numbers, and spaces.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Hello@World123"
Output: "HelloWorld123"

Test Case 2:
Input: "Price: $99.99"
Output: "Price 9999"

Test Case 3:
Input: "Email@test.com!!"
Output: "Emailtestcom"

Test Case 4:
Input: "123-456-7890"
Output: "1234567890"

Test Case 5:
Input: "No special chars"
Output: "No special chars"

Test Case 6:
Input: "!@#$%^&*()"
Output: ""

Test Case 7:
Input: "A-B_C*D&E"
Output: "ABCDE"

Test Case 8:
Input: "Text with émojis 😊 and symbols ™"
Output: "Text with mojis and symbols"
```

**Constraints:**
- Keep: a-z, A-Z, 0-9, spaces
- Remove: everything else
- Preserve word boundaries with spaces

**Expected Function Signature:**
```python
def remove_special_chars(text: str) -> str:
    pass
```

---

## SECTION 3: TOKENIZATION (Medium)

### Problem 7: Simple Word Tokenization

**Difficulty:** Medium
**Time Limit:** 8 minutes
**Concepts:** String manipulation, Regex

**Problem Statement:**
Split text into words (tokens). Handle punctuation appropriately.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Hello world"
Output: ["Hello", "world"]

Test Case 2:
Input: "Python,Java,C++"
Output: ["Python", "Java", "C"]

Test Case 3:
Input: "Don't worry, it's fine!"
Output: ["Don't", "worry", "it's", "fine"]

Test Case 4:
Input: "New York is great."
Output: ["New", "York", "is", "great"]

Test Case 5:
Input: "Price: $99.99"
Output: ["Price", "99", "99"]

Test Case 6:
Input: "Hello, world! How are you?"
Output: ["Hello", "world", "How", "are", "you"]

Test Case 7:
Input: "   Multiple   spaces   "
Output: ["Multiple", "spaces"]

Test Case 8:
Input: ""
Output: []

Test Case 9:
Input: "One-two-three"
Output: ["One", "two", "three"]

Test Case 10:
Input: "Mr. Smith went to Washington D.C."
Output: ["Mr", "Smith", "went", "to", "Washington", "D", "C"]
```

**Constraints:**
- Split by spaces and punctuation
- Keep contractions intact (don't, it's)
- Remove punctuation from tokens
- Handle multiple spaces
- Return list of tokens

**Expected Function Signature:**
```python
def tokenize(text: str) -> list[str]:
    pass
```

---

### Problem 8: Sentence Tokenization

**Difficulty:** Medium
**Time Limit:** 10 minutes
**Concepts:** Regex, Edge cases

**Problem Statement:**
Split text into sentences. Handle periods, question marks, and exclamation marks.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Hello world. How are you?"
Output: ["Hello world.", "How are you?"]

Test Case 2:
Input: "Dr. Smith went to the U.S.A. yesterday."
Output: ["Dr. Smith went to the U.S.A. yesterday."]

Test Case 3:
Input: "What?! Really?"
Output: ["What?!", "Really?"]

Test Case 4:
Input: "One. Two. Three."
Output: ["One.", "Two.", "Three."]

Test Case 5:
Input: "No punctuation"
Output: ["No punctuation"]

Test Case 6:
Input: "Mr. Johnson is here. He's happy!"
Output: ["Mr. Johnson is here.", "He's happy!"]

Test Case 7:
Input: "First sentence... Second sentence."
Output: ["First sentence...", "Second sentence."]

Test Case 8:
Input: "Hello! How are you? I'm fine."
Output: ["Hello!", "How are you?", "I'm fine."]

Test Case 9:
Input: "Question: What is this?"
Output: ["Question: What is this?"]

Test Case 10:
Input: ""
Output: []
```

**Constraints:**
- Split at sentence boundaries (. ? !)
- Handle abbreviations (Dr., U.S.A., etc.)
- Handle ellipsis (...)
- Keep punctuation with sentences
- Handle edge cases

**Expected Function Signature:**
```python
def sentence_tokenize(text: str) -> list[str]:
    pass
```

---

### Problem 9: N-gram Tokenization

**Difficulty:** Medium
**Time Limit:** 10 minutes
**Concepts:** Sliding window, List comprehension

**Problem Statement:**
Generate n-grams from text (contiguous sequences of n items).

**Input & Output Examples:**

```python
Test Case 1 (Bigrams - n=2):
Input: text = "The quick brown fox", n = 2
Output: [("The", "quick"), ("quick", "brown"), ("brown", "fox")]

Test Case 2 (Trigrams - n=3):
Input: text = "The quick brown", n = 3
Output: [("The", "quick", "brown")]

Test Case 3 (Unigrams - n=1):
Input: text = "Hello world", n = 1
Output: [("Hello",), ("world",)]

Test Case 4 (Bigrams with numbers):
Input: text = "1 2 3 4 5", n = 2
Output: [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")]

Test Case 5 (n > text length):
Input: text = "Short", n = 5
Output: []

Test Case 6 (Single word):
Input: text = "Hello", n = 2
Output: []

Test Case 7 (n=4):
Input: text = "A B C D E F", n = 4
Output: [("A", "B", "C", "D"), ("B", "C", "D", "E"), ("C", "D", "E", "F")]

Test Case 8 (Empty string):
Input: text = "", n = 2
Output: []
```

**Constraints:**
- n-grams should be tuples
- Handle edge cases (n > text length)
- Return in order
- Handle empty strings

**Expected Function Signature:**
```python
def get_ngrams(text: str, n: int) -> list[tuple]:
    pass
```

---

## SECTION 4: STOPWORDS (Medium)

### Problem 10: Remove Stopwords

**Difficulty:** Medium
**Time Limit:** 8 minutes
**Concepts:** List comprehension, Set operations

**Problem Statement:**
Remove common stopwords from text. Provided stopwords list.

**Input & Output Examples:**

```python
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "and", "or", 
             "in", "at", "to", "for", "of", "with", "by", "from", "on"}

Test Case 1:
Input: "The cat is on the table"
Output: "cat table"

Test Case 2:
Input: "I am going to the store for milk and eggs"
Output: "going store milk eggs"

Test Case 3:
Input: "The quick brown fox"
Output: "quick brown fox"

Test Case 4:
Input: "stopwords are important for NLP"
Output: "stopwords important NLP"

Test Case 5:
Input: "Remove the the the repeated stopwords"
Output: "Remove repeated stopwords"

Test Case 6:
Input: "No stopwords here"
Output: "stopwords"

Test Case 7:
Input: ""
Output: ""

Test Case 8:
Input: "a an the"
Output: ""

Test Case 9:
Input: "The cat and dog are friends"
Output: "cat dog friends"

Test Case 10:
Input: "Machine learning is awesome"
Output: "Machine learning awesome"
```

**Constraints:**
- Case-insensitive matching
- Preserve original case of remaining words
- Return as space-separated string
- Handle empty results
- Use provided stopwords set

**Expected Function Signature:**
```python
def remove_stopwords(text: str, stopwords: set) -> str:
    pass
```

---

## SECTION 5: STEMMING & LEMMATIZATION (Hard)

### Problem 11: Simple Stemming (Remove Common Suffixes)

**Difficulty:** Hard
**Time Limit:** 15 minutes
**Concepts:** String manipulation, Pattern matching

**Problem Statement:**
Implement simple stemming by removing common suffixes: -ing, -ed, -ly, -s, -es

**Input & Output Examples:**

```python
Test Case 1:
Input: "running"
Output: "runn"

Test Case 2:
Input: "jumped"
Output: "jump"

Test Case 3:
Input: "slowly"
Output: "slow"

Test Case 4:
Input: "cats"
Output: "cat"

Test Case 5:
Input: "boxes"
Output: "box"

Test Case 6:
Input: "walked"
Output: "walk"

Test Case 7:
Input: "happily"
Output: "happi"

Test Case 8:
Input: "computing"
Output: "comput"

Test Case 9:
Input: "word"
Output: "word"

Test Case 10:
Input: "flowers"
Output: "flower"

Test Case 11:
Input: ["running", "jumped", "slowly", "cats", "boxes"]
Output: ["runn", "jump", "slow", "cat", "box"]

Test Case 12:
Input: "The runner was running quickly"
Output: "the runn wa run quick"
```

**Suffix Priority (apply in this order):**
1. -ed → remove
2. -ing → remove  
3. -ly → remove
4. -es → remove
5. -s → remove (but not if only 1 char left)

**Constraints:**
- Apply suffixes in order
- Don't remove suffix if word becomes too short (< 3 chars)
- Handle full sentences
- Case-insensitive processing

**Expected Function Signature:**
```python
def simple_stem(text: str) -> str:
    pass
```

---

### Problem 12: Word Normalization with Dictionary

**Difficulty:** Hard
**Time Limit:** 12 minutes
**Concepts:** Dictionary, String replacement

**Problem Statement:**
Normalize text by expanding common contractions and abbreviations.

**Input & Output Examples:**

```python
EXPANSION_DICT = {
    "don't": "do not",
    "can't": "cannot",
    "it's": "it is",
    "i'm": "i am",
    "won't": "will not",
    "n't": "not",
    "'ve": "have",
    "'ll": "will",
    "mr.": "mister",
    "dr.": "doctor",
    "etc.": "et cetera"
}

Test Case 1:
Input: "Don't worry"
Output: "do not worry"

Test Case 2:
Input: "I'm happy"
Output: "i am happy"

Test Case 3:
Input: "Can't do it"
Output: "cannot do it"

Test Case 4:
Input: "He won't come"
Output: "he will not come"

Test Case 5:
Input: "She's smart"
Output: "she is smart"

Test Case 6:
Input: "Mr. Smith and Dr. Johnson"
Output: "mister smith and doctor johnson"

Test Case 7:
Input: "I've got things to do"
Output: "i have got things to do"

Test Case 8:
Input: "No contractions here"
Output: "no contractions here"

Test Case 9:
Input: "They'll be here"
Output: "they will be here"

Test Case 10:
Input: "I'm sorry, don't worry, it's fine!"
Output: "i am sorry do not worry it is fine"
```

**Constraints:**
- Case-insensitive matching
- Return lowercase
- Remove punctuation after expansion
- Handle overlapping patterns
- Apply in order of longest first

**Expected Function Signature:**
```python
def expand_contractions(text: str, expansion_dict: dict) -> str:
    pass
```

---

## SECTION 6: ADVANCED TEXT PREPROCESSING (Hard)

### Problem 13: Complete Text Cleaning Pipeline

**Difficulty:** Hard
**Time Limit:** 20 minutes
**Concepts:** Composing functions, Regex, String methods

**Problem Statement:**
Create a complete preprocessing pipeline with multiple steps:
1. Remove HTML tags
2. Remove URLs
3. Convert to lowercase
4. Remove special characters (keep alphanumeric + space)
5. Normalize whitespace
6. Remove stopwords
7. Tokenize

**Input & Output Examples:**

```python
STOPWORDS = {"the", "a", "an", "is", "are", "was", "and", "or", "in", "to"}

Test Case 1:
Input: "<p>Check out https://example.com for the BEST deals!</p>"
Output: ["check", "out", "best", "deals"]

Test Case 2:
Input: "Hello   WORLD! Visit www.google.com today."
Output: ["hello", "world", "visit", "today"]

Test Case 3:
Input: "<h1>Python & Java are GREAT</h1>"
Output: ["python", "java", "great"]

Test Case 4:
Input: "The cat is sleeping."
Output: ["cat", "sleeping"]

Test Case 5:
Input: "<div><span>Data</span> <b>Science</b></div>"
Output: ["data", "science"]

Test Case 6:
Input: "Email: contact@company.com (visit https://company.com)"
Output: ["email", "contact", "company", "com", "visit", "company", "com"]

Test Case 7:
Input: ""
Output: []

Test Case 8:
Input: "Price: $99.99 at www.store.com!!!"
Output: ["price", "99", "99"]

Test Case 9:
Input: "   <p>Multiple   spaces</p>   "
Output: ["multiple", "spaces"]

Test Case 10:
Input: "The quick brown fox jumps over the lazy dog"
Output: ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
```

**Order of operations (IMPORTANT):**
1. Remove HTML
2. Remove URLs
3. Lowercase
4. Remove special chars (keep alphanumeric + space)
5. Normalize whitespace
6. Tokenize
7. Remove stopwords

**Constraints:**
- Follow order exactly
- Return list of tokens (strings)
- Handle all edge cases
- Empty results should return empty list

**Expected Function Signature:**
```python
def preprocess_text(text: str, stopwords: set) -> list[str]:
    pass
```

---

### Problem 14: Remove Accents and Normalize Unicode

**Difficulty:** Hard
**Time Limit:** 10 minutes
**Concepts:** Unicode, Unicode normalization

**Problem Statement:**
Remove accents from characters (café → cafe) using Unicode normalization.

**Input & Output Examples:**

```python
Test Case 1:
Input: "café"
Output: "cafe"

Test Case 2:
Input: "naïve"
Output: "naive"

Test Case 3:
Input: "Élève"
Output: "Eleve"

Test Case 4:
Input: "España"
Output: "Espana"

Test Case 5:
Input: "Zürich"
Output: "Zurich"

Test Case 6:
Input: "résumé"
Output: "resume"

Test Case 7:
Input: "Bône"
Output: "Bone"

Test Case 8:
Input: "No accents"
Output: "No accents"

Test Case 9:
Input: "Crème brûlée"
Output: "Creme brulee"

Test Case 10:
Input: ""
Output: ""

Test Case 11:
Input: "François Müller"
Output: "Francois Muller"
```

**Constraints:**
- Use Unicode normalization (NFD mode)
- Remove combining characters (accents)
- Preserve non-accented characters
- Preserve spaces and non-letters

**Hint:** Use unicodedata module
```python
import unicodedata
```

**Expected Function Signature:**
```python
def remove_accents(text: str) -> str:
    pass
```

---

### Problem 15: Emoji & Special Unicode Handling

**Difficulty:** Hard
**Time Limit:** 12 minutes
**Concepts:** Unicode, Emoji detection, Character categories

**Problem Statement:**
Remove or handle emojis and other special Unicode characters.

**Input & Output Examples:**

```python
Test Case 1 (Remove emojis):
Input: "I love Python 😊"
Output: "I love Python"

Test Case 2:
Input: "Hello 👋 world 🌍"
Output: "Hello world"

Test Case 3:
Input: "Happy 😂😂😂"
Output: "Happy"

Test Case 4:
Input: "Price: $99 💰"
Output: "Price: $99"

Test Case 5:
Input: "No emojis here"
Output: "No emojis here"

Test Case 6 (Keep only ASCII):
Input: "Café ☕"
Output: "Caf"

Test Case 7:
Input: "😊"
Output: ""

Test Case 8:
Input: "Text with ™ and © symbols"
Output: "Text with and symbols"

Test Case 9:
Input: "Hearts ❤️💕💖"
Output: "Hearts"

Test Case 10:
Input: "Numbers ①②③ and letters ABC"
Output: "Numbers and letters ABC"
```

**Constraints:**
- Remove emoji characters
- Remove special symbols (™, ©, etc.)
- Keep alphanumeric and basic punctuation
- Preserve spaces between words

**Expected Function Signature:**
```python
def remove_emoji(text: str) -> str:
    pass
```

---

## SECTION 7: PRACTICAL CHALLENGES (Hard)

### Problem 16: Handle Multiple Languages

**Difficulty:** Hard
**Time Limit:** 15 minutes
**Concepts:** Unicode, Language-specific rules

**Problem Statement:**
Preprocess text that contains multiple languages (English, French, Spanish). Handle accents appropriately.

**Input & Output Examples:**

```python
Test Case 1 (Mixed English and French):
Input: "Hello! Bonjour! Ça va?"
Output: ["hello", "bonjour", "ca", "va"]

Test Case 2 (Spanish with accents):
Input: "¿Cómo estás? Muy bien."
Output: ["como", "estas", "muy", "bien"]

Test Case 3 (Multiple languages):
Input: "English and français with español"
Output: ["english", "and", "francais", "with", "espanol"]

Test Case 4 (Punctuation and special chars):
Input: "C'est magnifique! ¡Excelente!"
Output: ["c", "est", "magnifique", "excelente"]

Test Case 5 (Mixed case and accents):
Input: "FRANÇAIS español ENGLISH"
Output: ["francais", "espanol", "english"]

Test Case 6 (Quotes and accents):
Input: "'Élève' estudiant"
Output: ["eleve", "estudiant"]

Test Case 7 (Numbers preserved):
Input: "2023 es buen año"
Output: ["2023", "es", "buen", "ano"]

Test Case 8 (URLs in multiple languages):
Input: "Visit https://français.com for más info"
Output: ["visit", "for", "mas", "info"]

Test Case 9 (Empty):
Input: ""
Output: []

Test Case 10 (Special French accents):
Input: "Château créme brûlée"
Output: ["chateau", "creme", "brulee"]
```

**Constraints:**
- Remove accents
- Handle different punctuation rules
- Convert to lowercase
- Tokenize
- Remove URLs
- Handle Unicode characters

**Expected Function Signature:**
```python
def preprocess_multilingual(text: str) -> list[str]:
    pass
```

---

### Problem 17: Decontract Text (Expand Contractions)

**Difficulty:** Medium
**Time Limit:** 10 minutes
**Concepts:** Regex, Dictionary, String replacement

**Problem Statement:**
Expand all contractions in text while handling edge cases.

**Input & Output Examples:**

```python
Test Case 1:
Input: "Don't worry"
Output: "Do not worry"

Test Case 2:
Input: "I'm happy, you're sad"
Output: "I am happy, you are sad"

Test Case 3:
Input: "Can't won't shouldn't"
Output: "Cannot will not should not"

Test Case 4:
Input: "It's been great"
Output: "It has been great"

Test Case 5:
Input: "They'll be here soon"
Output: "They will be here soon"

Test Case 6:
Input: "We've got problems"
Output: "We have got problems"

Test Case 7:
Input: "No contractions here"
Output: "No contractions here"

Test Case 8:
Input: "I'd like that"
Output: "I would like that"

Test Case 9:
Input: "You'll see it's worth it"
Output: "You will see it is worth it"

Test Case 10:
Input: "They've couldn't wouldn't"
Output: "They have could not would not"

Test Case 11 (Mixed case):
Input: "DON'T WORRY, I'M FINE"
Output: "DO NOT WORRY, I AM FINE"

Test Case 12 (Multiple in sequence):
Input: "I'm sure you've noticed they're here"
Output: "I am sure you have noticed they are here"
```

**Common Contractions:**
- n't → not
- 'm → am
- 're → are
- 've → have
- 'll → will
- 'd → would/had
- 's → is/has

**Constraints:**
- Case-insensitive matching
- Preserve original case of non-apostrophe part
- Handle edge cases
- All contractions expanded

**Expected Function Signature:**
```python
def decontract_text(text: str) -> str:
    pass
```

---

### Problem 18: Handle Repeated Characters

**Difficulty:** Medium
**Time Limit:** 10 minutes
**Concepts:** Regex, String compression

**Problem Statement:**
Normalize repeated characters (e.g., "helllloooo" → "helo").

**Input & Output Examples:**

```python
Test Case 1:
Input: "Helloooo world"
Output: "Helo world"

Test Case 2:
Input: "Yessssss!!!"
Output: "Yes!"

Test Case 3:
Input: "nooooo wayyyy"
Output: "no way"

Test Case 4:
Input: "hahahaha"
Output: "haha"

Test Case 5:
Input: "Normal text"
Output: "Normal text"

Test Case 6:
Input: "aaaa"
Output: "a"

Test Case 7:
Input: "Good morning!!!"
Output: "God morning!"

Test Case 8:
Input: "Goooood afternoon"
Output: "God afternoon"

Test Case 9:
Input: "Multiple   spaces"
Output: "Multiple spaces"

Test Case 10 (Numbers):
Input: "Year 20000 or 2000"
Output: "Year 200 or 200"

Test Case 11 (Mix):
Input: "Heyyy guysss... Let's go!!!!"
Output: "Hey guys. Let's go!"

Test Case 12 (Edge case):
Input: ""
Output: ""
```

**Rules:**
- Keep max 2 consecutive identical characters
- Exception: Don't collapse numbers/special chars
- Apply to letters only

**Constraints:**
- Only compress 3+ consecutive chars to 2
- Case-insensitive
- Don't compress single or double chars

**Expected Function Signature:**
```python
def normalize_repeated_chars(text: str) -> str:
    pass
```

---

## SECTION 8: REAL-WORLD SCENARIOS (Hard)

### Problem 19: Process Social Media Text

**Difficulty:** Hard
**Time Limit:** 20 minutes
**Concepts:** Regex, Edge cases, Practical preprocessing

**Problem Statement:**
Clean and preprocess social media text (Twitter-like).
Handle: mentions, hashtags, URLs, emojis, excessive punctuation.

**Input & Output Examples:**

```python
Test Case 1:
Input: "@user Check out #Python at https://python.org 🐍"
Output: ["check", "out", "python"]

Test Case 2:
Input: "OMG!!! #AI is AMAZING 😍😍😍"
Output: ["omg", "ai", "amazing"]

Test Case 3:
Input: "@john @jane Let's meet up! #meetup"
Output: ["let", "meet", "up"]

Test Case 4:
Input: "Just posted on https://medium.com/my-blog"
Output: ["just", "posted"]

Test Case 5:
Input: "Love #MachineLearning & #AI #DeepLearning"
Output: ["love", "machinelearning", "ai", "deeplearning"]

Test Case 6:
Input: "No mentions, hashtags, or URLs"
Output: ["no", "mentions", "hashtags", "urls"]

Test Case 7:
Input: "@bot #hashtag repeated repeated"
Output: ["repeated"]

Test Case 8:
Input: "https://t.co/abc123 #promo @brand $$$"
Output: ["promo", "brand"]

Test Case 9 (Complex):
Input: "@everyone Check #AI & #ML here: https://bit.ly/xyz 🚀 #awesome"
Output: ["check", "ai", "ml", "awesome"]

Test Case 10 (Empty result):
Input: "@user @bot #tag https://example.com 😊"
Output: []
```

**Processing Steps:**
1. Remove URLs
2. Remove mentions (@user)
3. Keep hashtags but remove #
4. Remove emojis
5. Remove excessive punctuation
6. Lowercase
7. Remove extra whitespace
8. Tokenize
9. Remove stopwords (use: {is, at, out, on})

**Constraints:**
- Order matters!
- Hashtags become part of tokens (without #)
- Mentions completely removed
- Preserve natural word breaks

**Expected Function Signature:**
```python
def preprocess_social_media(text: str) -> list[str]:
    pass
```

---

### Problem 20: Deduplication and Normalization

**Difficulty:** Hard
**Time Limit:** 12 minutes
**Concepts:** List operations, Uniqueness, Ordering

**Problem Statement:**
Remove duplicate words while preserving order. Case-insensitive duplicates.

**Input & Output Examples:**

```python
Test Case 1:
Input: "hello world hello"
Output: ["hello", "world"]

Test Case 2:
Input: "Python python PYTHON"
Output: ["python"]

Test Case 3:
Input: ["the", "quick", "brown", "fox", "the", "quick"]
Output: ["the", "quick", "brown", "fox"]

Test Case 4:
Input: "a b c d e"
Output: ["a", "b", "c", "d", "e"]

Test Case 5:
Input: "machine learning machine"
Output: ["machine", "learning"]

Test Case 6:
Input: "a a a a"
Output: ["a"]

Test Case 7:
Input: []
Output: []

Test Case 8:
Input: ["Apple", "banana", "APPLE", "Cherry"]
Output: ["apple", "banana", "cherry"]

Test Case 9 (Mixed input):
Input: "The the quick Quick brown BROWN"
Output: ["the", "quick", "brown"]

Test Case 10 (Preserve order):
Input: "z a z b a c"
Output: ["z", "a", "b", "c"]
```

**Constraints:**
- Case-insensitive comparison
- Return lowercase tokens
- Preserve first occurrence order
- Return as list
- Handle empty input

**Expected Function Signature:**
```python
def deduplicate_tokens(tokens: list[str]) -> list[str]:
    pass
```

---

## ANSWER FORMAT

When you solve these, organize your solution like this:

```python
def function_name(input):
    """
    Solution approach:
    1. Step 1
    2. Step 2
    ...
    
    Time Complexity: O(n)
    Space Complexity: O(m)
    """
    # Your code here
    pass

# Test with provided test cases
if __name__ == "__main__":
    # Test Case 1
    assert function_name(input1) == expected1
    # Test Case 2
    assert function_name(input2) == expected2
    # ... etc
    print("All tests passed!")
```

---

## DIFFICULTY PROGRESSION

- **Easy (Problems 1-4):** Basic string operations, familiarize with problem format
- **Medium (Problems 5-12):** Regex, pattern matching, edge cases
- **Hard (Problems 13-20):** Real-world scenarios, composing solutions

**Recommended approach:**
1. Read problem carefully
2. Understand test cases
3. Plan your approach (pseudocode)
4. Write code
5. Test with all provided test cases
6. Optimize if needed

Good luck! 🎯

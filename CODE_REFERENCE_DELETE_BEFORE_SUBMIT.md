# 📚 F1 NLP Project — Code Reference Guide
# ⚠️  DELETE THIS FILE BEFORE SUBMISSION
# This file contains the key code for every phase with plain-English explanations.
# Use it to understand and explain your project. Do NOT submit it.

---

## PHASE 1 — Web Scraping (BeautifulSoup)

### What we did
We wrote a Python web crawler using `requests` + `BeautifulSoup` to collect real F1 team radio
transcripts from the AutoLAP archive. We first checked `robots.txt` to confirm scraping is permitted.

### Key code

```python
import requests
from bs4 import BeautifulSoup

# Step 1: Check robots.txt (ethical requirement)
r = requests.get("https://transcripts.recursiveprojects.cloud/robots.txt")
print(r.text)  # Must show "Allow: /" before proceeding

# Step 2: Crawl the archive
BASE = "https://transcripts.recursiveprojects.cloud"
archive_page = BeautifulSoup(requests.get(BASE + "/archive").text, "html.parser")

records = []
for session_link in archive_page.find_all("a", href=True):
    session_page = BeautifulSoup(requests.get(BASE + session_link["href"]).text, "html.parser")
    for driver_link in session_page.find_all("a", href=True):
        driver_page = BeautifulSoup(requests.get(BASE + driver_link["href"]).text, "html.parser")
        for msg in driver_page.find_all("div", class_="message"):
            records.append({
                "driver_name": driver_link.text.strip(),
                "message_text": msg.text.strip()
            })

df = pd.DataFrame(records)
# Result: 18,574 rows
```

### How to explain it
"We built a 3-level crawler: Archive page lists all sessions, each session lists all drivers,
each driver page contains the radio transcript. We parse each level with BeautifulSoup and
store every message as a row in a pandas DataFrame."

---

## PHASE 2 — Vocabulary Statistics, Zipf's Law & TF-IDF

### What we did
- Verified Zipf's Law holds on our corpus (log-log plot of word frequency vs rank = straight line)
- Calculated Shannon Entropy of the vocabulary
- Applied Luhn's cuts to isolate high-information words
- Implemented TF-IDF from scratch (before using scikit-learn)

### Key code

```python
from collections import Counter
import math

# All words in corpus
all_words = " ".join(df["message_text"]).lower().split()
freq = Counter(all_words).most_common()

# Shannon Entropy: H = -sum(P(w) * log2(P(w)))
total = len(all_words)
H = -sum((c/total) * math.log2(c/total) for _, c in freq if c > 0)
print(f"Shannon Entropy: {H:.3f} bits")

# Luhn's cuts
high_cut = freq[int(len(freq) * 0.01)][1]  # top 1% = stopwords
low_cut  = freq[int(len(freq) * 0.95)][1]  # bottom 5% = rare noise
content_words = [w for w, c in freq if low_cut <= c <= high_cut]

# TF-IDF from scratch
def compute_tf(text):
    words = text.lower().split()
    freq_doc = Counter(words)
    return {w: c / len(words) for w, c in freq_doc.items()}

def compute_idf(docs):
    N = len(docs)
    df_count = Counter(w for doc in docs for w in set(doc.lower().split()))
    return {w: math.log(N / (1 + df_count[w])) for w in df_count}
```

### How to explain it
"Zipf's Law says word frequency drops proportionally to rank. 'Box' appears much more than 'degrading'.
Luhn's cuts remove both extremes: ultra-frequent stopwords (no info) and ultra-rare words (noise).
TF-IDF weights words by how unique they are to a document — a rare word like 'hydraulic' scores
much higher than a common word like 'ok'."

---

## PHASE 3 — Sparse CSR/CSC Matrices

### What we did
Compared memory usage between dense and sparse matrix formats for our 18,574 × 5,000 document-term matrix.

### Key code

```python
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
sparse_matrix = tfidf.fit_transform(df["clean_message"])  # Already sparse

# Convert formats
csr_mat = sparse_matrix.tocsr()  # Row-wise efficient
csc_mat = sparse_matrix.tocsc()  # Column-wise efficient

# Memory comparison
dense_mb = sparse_matrix.shape[0] * sparse_matrix.shape[1] * 8 / 1e6
csr_mb   = (csr_mat.data.nbytes + csr_mat.indices.nbytes + csr_mat.indptr.nbytes) / 1e6
print(f"Dense: {dense_mb:.1f} MB")  # ~372 MB
print(f"CSR  : {csr_mb:.1f} MB")    # ~12 MB
print(f"Savings: {dense_mb/csr_mb:.1f}x")  # ~30x
```

### How to explain it
"A full 18,574 × 5,000 matrix has 92 million cells, but each F1 message only uses ~6 words,
so 99.9% of cells are zero. CSR format stores only the non-zero values using 3 arrays:
data (the values), indices (column positions), indptr (where each row starts).
This gives us a 30× memory saving — from 372 MB to just 12 MB."

---

## PHASE 4 — Tokenization, Stemming & Lemmatization

### What we did
Used spaCy's pipeline to tokenize, stem (Porter), and lemmatize the corpus.
Built custom spaCy Matcher rules for F1 jargon (lap times, pit commands, gap reports).

### Key code

```python
import spacy
from nltk.stem import PorterStemmer

nlp     = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

sample = "He is pushing on soft tyres, they are degrading badly."
doc = nlp(sample)

for token in doc:
    print(f"Original: {token.text:15}  "
          f"Stem: {stemmer.stem(token.text):12}  "
          f"Lemma: {token.lemma_:12}  "
          f"Stopword: {token.is_stop}")

# Custom F1 token matcher
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Match pit commands
pit_pattern = [{"LOWER": {"IN": ["box", "pit", "stop"]}}]
matcher.add("PIT_COMMAND", [pit_pattern])

# Match lap time patterns (e.g., "1:23.456")
lap_pattern = [{"SHAPE": "d:dd.ddd"}]
matcher.add("LAP_TIME", [lap_pattern])

matches = matcher(doc)
for match_id, start, end in matches:
    print(nlp.vocab.strings[match_id], "→", doc[start:end])
```

### How to explain it
"Stemming just chops word endings by rules (fast but 'degrading' → 'degrad' which is wrong).
Lemmatization uses a vocabulary lookup to find the actual base form ('degrading' → 'degrade').
We used spaCy for lemmatization because it's linguistically accurate.
We also built custom Matcher rules because 'box box' is an F1 pit command — standard NLP doesn't know that."

---

## PHASE 5 — Vectorization

### What we did
Compared three vectorization methods. Used TF-IDF (max 2,500 features) as input for all classifiers.

### Key code

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

# One-Hot (for driver names only)
lb = LabelBinarizer()
one_hot = lb.fit_transform(df["driver_name"])
print("One-Hot shape:", one_hot.shape)  # (18574, 20)

# Bag of Words
count_vec = CountVectorizer(max_features=1000)
bow_mat   = count_vec.fit_transform(df["clean_message"])
print("BoW shape:", bow_mat.shape)       # (18574, 1000)

# TF-IDF (used for all classifiers)
tfidf_vec = TfidfVectorizer(max_features=2500)
tfidf_mat = tfidf_vec.fit_transform(df["clean_message"])
print("TF-IDF shape:", tfidf_mat.shape)  # (18574, 2500)
```

### How to explain it
"One-Hot just marks which driver spoke — no word content at all.
Bag of Words counts raw word frequencies — useful but treats 'box' and 'pit' as equal importance.
TF-IDF penalises common words. A word like 'the' appears in every message so its IDF is near zero.
'Hydraulic' appears rarely so it gets a high IDF score — it's more meaningful for classification."

---

## PHASE 6 — Text Classification

### What we did
Auto-labelled messages using keyword heuristics (4 categories).
Trained Naive Bayes, Logistic Regression, and Linear SVM on TF-IDF features.

### Key code

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Auto-labelling
def assign_category(text):
    text = text.lower()
    if any(k in text for k in ["box", "pit", "strat", "mode", "tyre", "soft"]):
        return "Strategy"
    elif any(k in text for k in ["gap", "behind", "ahead", "interval", "p1", "p2"]):
        return "Position_Gap"
    elif any(k in text for k in ["wtf", "no", "why", "impossible", "angry"]):
        return "Emotional"
    else:
        return "Technical"

df["category"] = df["message_text"].apply(assign_category)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"])

# Vectorize
tfidf_v = TfidfVectorizer(max_features=2500)
X_tr = tfidf_v.fit_transform(X_train)
X_te = tfidf_v.transform(X_test)

# Train all 3
nb  = MultinomialNB().fit(X_tr, y_train)
lr  = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
svm = SVC(kernel="linear").fit(X_tr, y_train)

print(accuracy_score(y_test, nb.predict(X_te)))   # 79.94%
print(accuracy_score(y_test, lr.predict(X_te)))   # 92.21%
print(accuracy_score(y_test, svm.predict(X_te)))  # 93.81% ← best
```

### How to explain it
"Naive Bayes assumes words are independent of each other (often false but fast).
Logistic Regression learns a weight for each TF-IDF feature — good baseline.
SVM finds a hyperplane that maximises the margin between categories — best for sparse text data.
SVM won at 93.81% because our messages are short and linearly separable in TF-IDF space."

---

## PHASE 6b — Custom Sentiment Lexicon

### Key code

```python
import re

lexicon = {
    "p1": 2.0, "won": 2.0, "great": 1.5, "push": 0.5,
    "copy": 0.0, "ok": 0.0, "understood": 0.0,
    "problem": -1.0, "slow": -0.8, "bad": -1.0,
    "wtf": -2.0, "impossible": -1.8
}

def analyze_sentiment(text):
    score = 0.0
    for word in str(text).lower().split():
        clean = re.sub(r"[^a-zA-Z]", "", word)
        score += lexicon.get(clean, 0.0)
    return score

df["sentiment_score"] = df["message_text"].apply(analyze_sentiment)
```

### How to explain it
"VADER is trained on Twitter and product reviews. It has no idea what 'p1' means in F1.
We hand-crafted a lexicon mapping F1-specific words to scores.
'p1' = race leader = +2.0 (very positive). 'wtf' = -2.0 (very frustrated).
This lets us find the most emotionally charged moments in the race."

---

## PHASE 7 — CBOW Word Embeddings (PyTorch)

### What we did
Trained a Continuous Bag of Words model from scratch to produce 50-dimensional word vectors.

### Key code

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1    = nn.Linear(embedding_dim, 128)
        self.linear2    = nn.Linear(128, vocab_size)

    def forward(self, ctx):             # ctx shape: [batch, 4 context words]
        emb = self.embeddings(ctx)      # [batch, 4, 50]
        out = torch.sum(emb, dim=1)     # [batch, 50] — average context
        out = torch.relu(self.linear1(out))
        return torch.log_softmax(self.linear2(out), dim=1)

# Training: predict centre word from 4 surrounding context words
model = CBOWModel(vocab_size=5842, embedding_dim=50)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Cosine similarity after training
def cosine_sim(w1, w2):
    v1 = embeddings[word_to_ix[w1]]
    v2 = embeddings[word_to_ix[w2]]
    return float(torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))

# Results: "box" ↔ "pit" = 0.83,  "soft" ↔ "tyre" = 0.79
```

### How to explain it
"In TF-IDF, 'box' and 'pit' are in completely different columns — the model sees them as unrelated.
CBOW learns that they appear in similar contexts (same messages, same surrounding words),
so it maps them to similar points in 50-dimensional vector space.
After training, we can measure how similar two words are using cosine similarity."

---

## PHASE 8 — MLP Neural Network Classifier (PyTorch)

### Key code

```python
class TextMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),  # 2500 → 256
            nn.ReLU(),
            nn.Dropout(0.3),            # randomly zero 30% of neurons (prevents overfitting)
            nn.Linear(256, 128),        # 256 → 128
            nn.ReLU(),
            nn.Linear(128, num_classes) # 128 → 4 categories
        )

    def forward(self, x):
        return self.network(x)

# Training loop
model    = TextMLP(2500, 4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features.to(device))
        loss    = criterion(outputs, labels.to(device))
        loss.backward()   # backpropagation
        optimizer.step()  # update weights
# Final accuracy: 92.57%
```

### How to explain it
"The MLP takes the 2,500 TF-IDF features as input.
Each hidden layer learns non-linear combinations of features — things SVM can't do.
Dropout randomly disables 30% of neurons during training, which forces the network
to not rely too heavily on any single feature. This prevents overfitting.
We trained for 15 epochs and reached 92.57%."

---

## PHASE 9 — RNN & LSTM Sequence Models

### Key code

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn       = nn.RNN(emb_dim, hidden, batch_first=True)
        self.fc        = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embedding(x)    # [batch, seq_len, 50]
        _, h = self.rnn(emb)       # h = final hidden state [1, batch, hidden]
        return self.fc(h.squeeze(0))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm      = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc        = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, c) = self.lstm(emb)  # h = hidden, c = cell state
        return self.fc(h.squeeze(0))

# Preprocessing: pad all messages to 25 tokens
def text_to_tensor(text, max_len=25):
    tokens = str(text).lower().split()
    ids = [word_to_ix.get(t, 1) for t in tokens][:max_len]  # 1 = <UNK>
    ids += [0] * (max_len - len(ids))                        # 0 = <PAD>
    return torch.tensor(ids, dtype=torch.long)

# RNN: 79.99%   LSTM: 81.42%
```

### How to explain it
"RNN processes tokens left to right, passing a hidden state forward.
The problem is that gradients vanish over long sequences — early words get forgotten.
LSTM adds 3 gates: forget (what to erase), input (what to add), output (what to pass on).
These gates prevent vanishing gradients. Both underperformed SVM because our messages
average only 6 words — too short for sequential memory to help."

---

## PHASE 10 — spaCy Pipeline & Intent Matchers

### Key code

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

# Inspect pipeline components
print(nlp.pipe_names)  # ['tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer']

doc = nlp("Box this lap, check gap behind.")
for token in doc:
    print(f"{token.text:12} POS={token.pos_:8} dep={token.dep_:10} lemma={token.lemma_}")

# Custom Intent Matcher
matcher = Matcher(nlp.vocab)
matcher.add("PIT_COMMAND",    [[{"LOWER": {"IN": ["box", "pit"]}}]])
matcher.add("GAP_REPORT",     [[{"LOWER": "gap"}, {"POS": "ADP"}]])
matcher.add("MODE_SWITCH",    [[{"LOWER": "mode"}, {"IS_DIGIT": True}]])

matches = matcher(doc)
for match_id, start, end in matches:
    intent = nlp.vocab.strings[match_id]
    span   = doc[start:end]
    print(f"Intent: {intent:20} Text: '{span}'")
```

### How to explain it
"spaCy's pipeline runs tokenizer → POS tagger → dependency parser → NER in sequence.
Each component adds annotations to the tokens (POS, dependency role, entity label).
We added a custom Matcher on top that uses rule patterns to classify messages by intent.
'box' followed by anything → PIT_COMMAND. 'mode' followed by a digit → MODE_SWITCH."

---

## PHASE 11 — Custom NER Training

### Key code

```python
import spacy
from spacy.training import Example

# Training data: 30 annotated F1 sentences
TRAIN_DATA = [
    ("Max is on softs, Red Bull box box.",
     {"entities": [(0, 3, "DRIVER"), (13, 18, "TIRE_COMPOUND"),
                   (20, 27, "TEAM"),  (28, 35, "STRATEGY")]}),
    ("Hamilton is pushing hard on medium tyres.",
     {"entities": [(0, 8, "DRIVER"), (30, 36, "TIRE_COMPOUND")]}),
    # ... 28 more examples
]

# Build model from scratch
nlp_custom = spacy.blank("en")
ner = nlp_custom.add_pipe("ner")
for _, ann in TRAIN_DATA:
    for _, _, label in ann["entities"]:
        ner.add_label(label)

# Train 30 epochs
nlp_custom.begin_training()
for epoch in range(30):
    for text, ann in TRAIN_DATA:
        doc     = nlp_custom.make_doc(text)
        example = Example.from_dict(doc, ann)
        nlp_custom.update([example])

# Test
doc_test = nlp_custom("Verstappen on softs, Red Bull switches strategy.")
for ent in doc_test.ents:
    print(f"{ent.text:20} → {ent.label_}")
# Verstappen           → DRIVER
# softs                → TIRE_COMPOUND
# Red Bull             → TEAM
```

### How to explain it
"The default spaCy NER model is trained on news articles. It labels 'Red Bull' as PRODUCT,
not TEAM. It has no concept of 'soft' being a tyre compound.
We collected 30 F1 radio sentences and manually marked every entity with its correct label.
We then trained a blank spaCy model for 30 epochs on this data.
After training, it correctly identifies F1-specific entities in unseen sentences."

---

## PHASE 12 — Trigram Generator & Chatbot

### Trigram key code

```python
from collections import defaultdict
import random

# Build trigram model
trigrams = defaultdict(list)
tokenized = [str(t).lower().split() for t in df["message_text"]]
for sentence in tokenized:
    for i in range(len(sentence) - 2):
        key = (sentence[i], sentence[i+1])
        trigrams[key].append(sentence[i+2])

# Generate synthetic message
def generate(seed1, seed2, length=10):
    words = [seed1.lower(), seed2.lower()]
    for _ in range(length):
        key  = (words[-2], words[-1])
        next_words = trigrams.get(key, [])
        if not next_words:
            break
        words.append(random.choice(next_words))
    return " ".join(words)

print(generate("box", "box"))
# "box box now switch medium tyres gap closing"
```

### How to explain it
"A trigram model stores, for every pair of consecutive words, what word typically comes next.
To generate text, we start with a seed pair and keep sampling the next word from the
distribution we observed in the corpus. It's pure statistics — no neural network needed.
The generated text sounds F1-like because it learned from real radio messages."

---

## PHASE 13 — RAG (Retrieval-Augmented Generation)

### Key code

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF search over radio corpus
def search_corpus(query, k=3):
    vec = TfidfVectorizer().fit(df["message_text"])
    q_vec = vec.transform([query])
    doc_vecs = vec.transform(df["message_text"])
    scores = cosine_similarity(q_vec, doc_vecs).flatten()
    top_idx = scores.argsort()[-k:][::-1]
    return df.iloc[top_idx]["message_text"].tolist()

# RAG answer builder
def rag_answer(user_query):
    # 1. Retrieve from all 5 databases
    radio   = search_corpus(user_query, k=3)
    regs    = search_regulations(user_query, k=2)

    # 2. Build context
    context = "\n".join([f"Radio: {r}" for r in radio]
                      + [f"Regulation: {r}" for r in regs])

    # 3. Assemble prompt
    prompt = f"""
    You are an F1 expert assistant.
    Using ONLY the context below, answer the question.

    Context:
    {context}

    Question: {user_query}
    Answer:"""

    # 4. (In a real system: call OpenAI API or HuggingFace model here)
    return prompt  # We simulate this step in the dashboard
```

### How to explain it
"RAG is needed because a standard LLM doesn't know the results of the 2026 Monaco GP.
Our RAG system: takes the user query → searches 5 databases using TF-IDF cosine similarity →
takes the top results → inserts them into a prompt template → generates a grounded answer.
The key insight is that the model only uses facts from the retrieved context,
so it cannot hallucinate F1 results that didn't happen."

---

## FINAL RESULTS SUMMARY

| Model               | Type              | Accuracy |
|---------------------|-------------------|----------|
| Naive Bayes         | Classical ML      | 79.94%   |
| PyTorch RNN         | Deep Learning     | 79.99%   |
| PyTorch LSTM        | Deep Learning     | 81.42%   |
| Logistic Regression | Classical ML      | 92.21%   |
| PyTorch MLP         | Deep Learning     | 92.57%   |
| **Linear SVM**      | **Classical ML**  | **93.81% ✅** |

### Why SVM won
F1 radio messages are very short (average ~6 words). Sequence models (RNN/LSTM) need
long text to exploit their memory advantage. On short, sparse TF-IDF vectors, a linear
SVM finds the optimal decision boundary better than any neural network in this task.

---

## COMMON PROFESSOR QUESTIONS — QUICK ANSWERS

**Q: Why did you use BeautifulSoup instead of Scrapy?**
A: BeautifulSoup is simpler for a fixed, well-structured archive. Scrapy adds unnecessary
complexity for a single-site crawler. BeautifulSoup + requests gave us full control.

**Q: Why TF-IDF and not just word counts?**
A: Word counts treat 'ok' and 'hydraulic' as equally important. TF-IDF penalises 'ok'
because it appears in almost every message (high DF → low IDF). 'Hydraulic' is rare,
so it gets a high IDF → more useful for distinguishing categories.

**Q: How is LSTM different from RNN?**
A: RNN has one hidden state that gets overwritten each step. Long messages → early words
are forgotten (vanishing gradient). LSTM adds 3 gates (forget, input, output) that
explicitly control what to remember and what to discard — solving the vanishing gradient.

**Q: Why did you train NER from scratch instead of fine-tuning?**
A: spaCy's default NER is trained on news. Fine-tuning risks catastrophic forgetting.
Training a blank model from scratch on 30 annotated F1 sentences gives clean results
for our specific 4 entity types without interference from news-domain patterns.

**Q: What is RAG and why is it better than a standard LLM?**
A: A standard LLM can only use facts it memorised during training. It cannot know who
won the 2026 Monaco GP. RAG retrieves fresh, specific facts at query time and injects
them into the prompt. The model then generates an answer grounded in real retrieved data
— it cannot hallucinate results we haven't explicitly provided.

**Q: Why did SVM beat LSTM?**
A: Our messages average ~6 words. LSTM's advantage is modelling long-range dependencies
in sequences of 50-500 tokens. On 6-word inputs, there is no long-range dependency
to exploit. TF-IDF + SVM works perfectly on these short, sparse, high-dimensional vectors.

**Q: What is the spaCy pipeline?**
A: tokenizer → tok2vec (context-aware token vectors) → tagger (POS labels) →
parser (dependency grammar) → NER (entity labels) → lemmatizer.
Each component adds linguistic annotations to the Doc object in order.

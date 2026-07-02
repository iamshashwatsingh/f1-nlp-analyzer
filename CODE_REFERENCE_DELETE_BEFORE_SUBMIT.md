# 📚 F1 NLP Project — Code Reference Guide
# ⚠️  DELETE THIS FILE BEFORE SUBMISSION
# This file contains the key code for every phase with plain-English explanations.
# Use it to understand and explain your project. Do NOT submit it.

---

## PHASE 1 — Web Scraping (BeautifulSoup)

### What we did
We wrote a Python web crawler using `requests` + `BeautifulSoup` to collect F1 team radio transcripts from the AutoLAP archive (a static sandbox mirror transcripts.recursiveprojects.cloud). We first checked `robots.txt` to confirm scraping is permitted.

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
```

---

## PHASE 2 — Vocabulary Statistics & Zipf's Law

### What we did
- Verified Zipf's Law holds on our corpus (log-log plot of word frequency vs rank = straight line).

### Key code
```python
from collections import Counter
import matplotlib.pyplot as plt

# All words in corpus
all_words = " ".join(df["message_text"]).lower().split()
freq = Counter(all_words).most_common()

# Plot frequency vs rank on log-log scale
ranks = range(1, len(freq) + 1)
counts = [count for _, count in freq]

plt.loglog(ranks, counts)
plt.title("Zipf's Law Verification")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.show()
```

---

## PHASE 3 — TF-IDF Search Engine from Scratch

### What we did
Calculated Term Frequency (TF) and Inverse Document Frequency (IDF) manually using basic math logic, matching queries using cosine similarity.

### Key code
```python
import math

def compute_tf(text):
    words = text.lower().split()
    freq_doc = Counter(words)
    return {w: c / len(words) for w, c in freq_doc.items()}

def compute_idf(docs):
    N = len(docs)
    df_count = Counter(w for doc in docs for w in set(doc.lower().split()))
    return {w: math.log(N / (1 + df_count[w])) for w in df_count}
```

---

## PHASE 4 — Text Preprocessing (spaCy & NLTK)

### What we did
Used spaCy's pipeline to tokenize, stem (Porter), and lemmatize the corpus.

### Key code
```python
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

sample = "He is pushing on soft tyres, they are degrading badly."
doc = nlp(sample)

# Lemmatization & Stopword check
clean_tokens = [
    token.lemma_.lower()
    for token in doc
    if not token.is_stop and not token.is_punct
]
```

---

## PHASE 5 — Vectorization (Bag of Words & TF-IDF)

### What we did
Mapped raw text strings to numerical vectors using CountVectorizer and TfidfVectorizer.

### Key code
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of Words
count_vec = CountVectorizer(max_features=1000)
bow_mat = count_vec.fit_transform(df["clean_message"])

# TF-IDF (used for classifiers)
tfidf_vec = TfidfVectorizer(max_features=2500)
tfidf_mat = tfidf_vec.fit_transform(df["clean_message"])
```

---

## PHASE 6 — Sentiment Analysis & Category Classification

### What we did
Custom lexicon scoring for domain sentiment. Trained Naive Bayes, Logistic Regression, and SVM.

### Key code
```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Sentiment weights mapping
lexicon = {"p1": 2.0, "won": 2.0, "great": 1.5, "slow": -0.8, "wtf": -2.0}

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["category"], test_size=0.2, stratify=df["category"]
)

# Classifiers
nb = MultinomialNB().fit(X_train_tfidf, y_train)
lr = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
svm = SVC(kernel="linear").fit(X_train_tfidf, y_train) # 93.81% accuracy
```

---

## PHASE 7 — spaCy Pipeline & Intent Matchers

### What we did
Inspected the spaCy processing token attributes and created custom Matcher rules.

### Key code
```python
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
# Match 'box' or 'pit'
pit_pattern = [{"LOWER": {"IN": ["box", "pit"]}}]
matcher.add("PIT_COMMAND", [pit_pattern])
```

---

## PHASE 8 — Custom NER Model Training

### What we did
Trained a blank spaCy model on annotated sentences to find F1 entities.

### Key code
```python
import spacy
from spacy.training import Example

TRAIN_DATA = [
    ("Hamilton is on softs, Red Bull box box.",
     {"entities": [(0, 8, "DRIVER"), (14, 19, "TIRE_COMPOUND"), (21, 29, "TEAM")]})
]

nlp_blank = spacy.blank("en")
ner = nlp_blank.add_pipe("ner")
ner.add_label("DRIVER")
ner.add_label("TEAM")
ner.add_label("TIRE_COMPOUND")

nlp_blank.begin_training()
for epoch in range(30):
    for text, ann in TRAIN_DATA:
        example = Example.from_dict(nlp_blank.make_doc(text), ann)
        nlp_blank.update([example])
```

---

## PHASE 9 — Retrieval-Augmented Generation (RAG) System

### What we did
Set up a multi-source retrieval engine using TF-IDF cosine similarity search across 5 F1 databases (Radio Corpus, Teams/Drivers, Results, Tech Specs, Regulations) to assemble prompt context.

### Key code
```python
def rag_answer(user_query):
    # 1. Retrieve matching facts
    radio_context = search_corpus(user_query, k=2)
    regs_context = search_regulations(user_query, k=2)

    # 2. Compile Context Block
    context = "\n".join([f"Radio: {r}" for r in radio_context]
                        + [f"Regulation: {r}" for r in regs_context])

    # 3. Formulate strategical prompt
    prompt = f"""
    You are an expert F1 assistant. Using only the context below, answer the question.
    Context:
    {context}
    Question: {user_query}
    Answer:
    """
    return prompt
```

---

## 📊 Performance Benchmarks
* **Multinomial Naive Bayes:** 79.94%
* **Logistic Regression:** 92.21%
* **Linear SVM:** **93.81%**

### Q: Why did SVM beat deep learning or sequence models?
F1 radio messages average only 6 words. Short sequences do not have enough word dependency history for recurrent sequence architectures (like LSTMs/RNNs) to be useful. SVM works perfectly to find high-margin boundaries on sparse TF-IDF spaces.

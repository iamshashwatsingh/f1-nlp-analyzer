# 🏎️ F1 Team Radio NLP Analyzer

> **HS Aalen — NLP Course Project**
> A comprehensive Natural Language Processing pipeline applied to real Formula 1 team radio communications.

## Project Overview

This project scrapes F1 team radio transcripts from [AutoLAP](https://transcripts.recursiveprojects.cloud/archive) using BeautifulSoup, then applies NLP techniques covering all 13 course modules — from web crawling to Retrieval-Augmented Generation (RAG).

## Dataset

- **Source**: transcripts.recursiveprojects.cloud (AutoLAP)
- **Content**: AI-transcribed F1 team radio messages (Whisper-based)
- **Coverage**: 2025-2026 F1 seasons
- **Size**: ~5,000-15,000 individual radio messages
- **Fields**: Grand Prix, session type, driver name, timestamp, message text

## Course Modules Covered

| Module | Topic | Key Techniques |
|--------|-------|---------------|
| 1 | WWW & Search Engines | BeautifulSoup crawler, robots.txt, web graph |
| 2 | Information Retrieval | Zipf's Law, TF-IDF, search engine |
| 3 | Sparse Matrices | CSR/CSC, memory benchmarks |
| 4 | NLP Preprocessing | Tokenization, stopwords, stemming, lemmatization |
| 5 | Text as Numbers | One-Hot, CountVectorizer, TF-IDF Vectorizer |
| 6 | ML for NLP | Classification, sentiment analysis |
| 7 | Word Embeddings | CBOW, word similarity, t-SNE |
| 8 | Neural Networks | PyTorch feed-forward classifier |
| 9 | Sequence Models | POS tagging, RNN, LSTM |
| 10 | spaCy | Pipeline, matchers, intent detection |
| 11 | NER | Custom F1 entity recognition |
| 12 | Chatbots | Intent-based F1 radio chatbot |
| 13 | Modern LLMs | Semantic search, RAG pipeline |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Option A: Run the Jupyter Notebook
jupyter notebook f1_nlp_project.ipynb

# Option B: Run the Interactive Streamlit Dashboard
streamlit run dashboard.py

```

## Project Structure

```
nlp/
├── f1_nlp_project.ipynb          # Main Jupyter Notebook
├── data/
│   ├── raw/                      # Raw scraped HTML
│   └── processed/
│       └── f1_radio_corpus.csv   # Cleaned corpus
├── models/                       # Saved models
├── requirements.txt              # Python dependencies
├── scrape_data.py                # BeautifulSoup web crawler script
├── dashboard.py                  # Streamlit dashboard interface
└── README.md                     # Setup instructions
```

## Tech Stack

- **Scraping**: BeautifulSoup, requests
- **NLP**: scikit-learn, spaCy, NLTK
- **Deep Learning**: PyTorch
- **Visualization**: matplotlib, seaborn, wordcloud
- **Graph**: NetworkX
- **Embeddings**: sentence-transformers

## Author

- **University**: Hochschule Aalen
- **Team**: e-motion Formula Student Team

## License

Academic project — for educational purposes.

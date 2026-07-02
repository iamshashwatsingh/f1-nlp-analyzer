import os
import re
import math
import random
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
from wordcloud import WordCloud

# Page config
st.set_page_config(
    page_title="F1 Team Radio NLP Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 Styling (Vibrant Dark Mode Theme)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #e10600 !important; /* Formula 1 Red */
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #e10600;
        color: white;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff2d20;
        color: white;
    }
    .metric-card {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #e10600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# F1 Official Race Results Database (Still queried by the chatbot)
RACE_DATA = {
    "2026 Barcelona-Catalunya Grand Prix": {
        "winner": "Max Verstappen (Red Bull)",
        "podium": ["1. Max Verstappen (Red Bull)", "2. Lando Norris (McLaren)", "3. George Russell (Mercedes)"],
        "retirements": [
            {"driver": "Lewis Hamilton", "lap": 12, "reason": "Power Unit failure"},
            {"driver": "Alexander Albon", "lap": 4, "reason": "Collision in Turn 3"}
        ],
        "summary": "Verstappen controlled the race from the front after overtaking Norris at the start. Hamilton retired early on lap 12 with an engine issue."
    },
    "2026 Monaco Grand Prix": {
        "winner": "Charles Leclerc (Ferrari)",
        "podium": ["1. Charles Leclerc (Ferrari)", "2. Oscar Piastri (McLaren)", "3. Carlos Sainz (Ferrari)"],
        "retirements": [
            {"driver": "Sergio Perez", "lap": 1, "reason": "Collision on Saint Devote climb"},
            {"driver": "Nico Hulkenberg", "lap": 1, "reason": "Collision on Saint Devote climb"},
            {"driver": "Esteban Ocon", "lap": 45, "reason": "Gearbox failure"}
        ],
        "summary": "Charles Leclerc took a historic home victory leading from pole position. A massive lap 1 crash red-flagged the race and eliminated Perez and Hulkenberg instantly."
    },
    "2026 Canadian Grand Prix": {
        "winner": "Lando Norris (McLaren)",
        "podium": ["1. Lando Norris (McLaren)", "2. Max Verstappen (Red Bull)", "3. Oscar Piastri (McLaren)"],
        "retirements": [
            {"driver": "Logan Sargeant", "lap": 25, "reason": "Spun into the wall at Turn 4"},
            {"driver": "Charles Leclerc", "lap": 40, "reason": "Engine issues after rain shower"}
        ],
        "summary": "Norris won an absolute thriller in mixed conditions, beating Verstappen with superior tire management as the track dried. Leclerc retired after engine mapping issues."
    }
}

# Load data
CSV_PATH = "data/processed/f1_radio_corpus.csv"
REPORTS_PATH = "data/processed/race_reports.csv"
REGS_PATH = "data/processed/f1_regulations.csv"

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["message_len"] = df["message_text"].astype(str).apply(lambda x: len(x.split()))
        return df
    else:
        st.error(f"Dataset not found at {CSV_PATH}. Please run the scraper first.")
        return pd.DataFrame()

@st.cache_data
def load_race_reports():
    if os.path.exists(REPORTS_PATH):
        return pd.read_csv(REPORTS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_regulations():
    if os.path.exists(REGS_PATH):
        return pd.read_csv(REGS_PATH)
    return pd.DataFrame()

TEAMS_PATH = "data/processed/f1_teams_drivers.csv"
SPECS_PATH = "data/processed/f1_technical_specs.csv"

@st.cache_data
def load_teams_drivers():
    if os.path.exists(TEAMS_PATH):
        return pd.read_csv(TEAMS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_tech_specs():
    if os.path.exists(SPECS_PATH):
        return pd.read_csv(SPECS_PATH)
    return pd.DataFrame()

df = load_data()
df_reports = load_race_reports()
df_regs = load_regulations()
df_teams = load_teams_drivers()
df_tech = load_tech_specs()

@st.cache_resource
def train_dashboard_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    def assign_category(text):
        text_lower = str(text).lower()
        if any(k in text_lower for k in ["box", "pit", "strat", "mode", "charge", "entry", "tyre", "soft", "medium", "hard", "hards", "softs", "mediums"]):
            return "Strategy"
        elif any(k in text_lower for k in ["gap", "behind", "ahead", "interval", "pace", "lap", "p1", "p2", "p3", "position", "sec", "seconds"]):
            return "Position/Gap"
        elif any(k in text_lower for k in ["f**k", "fuck", "shout", "angry", "screaming", "no", "what", "why", "impossible", "mate", "!", "unbelievable"]):
            return "Emotional"
        else:
            return "Technical/General"
    
    y = df["message_text"].apply(assign_category)
    vec = TfidfVectorizer(max_features=1000)
    X = vec.fit_transform(df["message_text"].astype(str))
    
    clf = LogisticRegression(max_iter=150)
    clf.fit(X, y)
    return vec, clf

vec_clf, model_clf = train_dashboard_classifier()


if df.empty:
    st.stop()

# Search reports at sentence-level
def search_reports(query, k=2):
    results = []
    if df_reports.empty:
        return pd.DataFrame()
        
    query_words = set(str(query).lower().split())
    stopwords = {"the", "and", "to", "a", "is", "of", "in", "it", "on", "you", "that", "for", "we", "at", "this", "be", "with", "have", "what", "who", "why", "how", "on", "at", "in"}
    query_words = {w for w in query_words if w not in stopwords}
    
    if not query_words:
        return pd.DataFrame()
        
    for _, row in df_reports.iterrows():
        gp = row["grand_prix"]
        text = row["report_text"]
        sentences = re.split(r'\. |\? |\! ', text)
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            # Simple keyword matching score
            sent_words = set(re.findall(r'\b\w+\b', sent_clean.lower()))
            matches = len(query_words.intersection(sent_words))
            if matches > 0:
                results.append({
                    "Grand Prix": gp,
                    "Report Sentence": sent_clean,
                    "Match Score": matches
                })
                
    if not results:
        return pd.DataFrame()
        
    results = sorted(results, key=lambda x: x["Match Score"], reverse=True)[:k]
    return pd.DataFrame(results)

# Search regulations at sentence-level
def search_regulations(query, k=2):
    results = []
    if df_regs.empty:
        return pd.DataFrame()
        
    query_words = set(str(query).lower().split())
    stopwords = {"the", "and", "to", "a", "is", "of", "in", "it", "on", "you", "that", "for", "we", "at", "this", "be", "with", "have", "what", "who", "why", "how", "on", "at", "in", "explain", "rules", "regulations", "rule"}
    query_words = {w for w in query_words if w not in stopwords}
    
    if not query_words:
        return pd.DataFrame()
        
    for _, row in df_regs.iterrows():
        year = row["year"]
        cat = row["category"]
        text = row["regulation_text"]
        sentences = re.split(r'\. |\? |\! ', text)
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            # Simple keyword matching score
            sent_words = set(re.findall(r'\b\w+\b', sent_clean.lower()))
            matches = len(query_words.intersection(sent_words))
            if matches > 0:
                results.append({
                    "Year": year,
                    "Category": cat,
                    "Regulation Clause": sent_clean,
                    "Match Score": matches
                })
                
    if not results:
        return pd.DataFrame()
        
    results = sorted(results, key=lambda x: x["Match Score"], reverse=True)[:k]
    return pd.DataFrame(results)

def search_teams(query, k=2):
    results = []
    if df_teams.empty:
        return pd.DataFrame()
    query_words = set(str(query).lower().split())
    stopwords = {"the", "and", "to", "a", "is", "of", "in", "it", "on", "you", "that", "for", "we", "at", "this", "be", "with", "have", "what", "who", "why", "how"}
    query_words = {w for w in query_words if w not in stopwords}
    if not query_words:
        return pd.DataFrame()
    for _, row in df_teams.iterrows():
        text = f"{row['team']} driver {row['driver']} number {row['number']} nationality {row['nationality']} role {row['role']}"
        words = set(re.findall(r'\b\w+\b', text.lower()))
        matches = len(query_words.intersection(words))
        if matches > 0:
            results.append({
                "Team": row["team"],
                "Driver": row["driver"],
                "Number": row["number"],
                "Nationality": row["nationality"],
                "Role": row["role"],
                "Match Score": matches
            })
    if not results:
        return pd.DataFrame()
    results = sorted(results, key=lambda x: x["Match Score"], reverse=True)[:k]
    return pd.DataFrame(results)

def search_tech_specs(query, k=2):
    results = []
    if df_tech.empty:
        return pd.DataFrame()
    query_words = set(str(query).lower().split())
    stopwords = {"the", "and", "to", "a", "is", "of", "in", "it", "on", "you", "that", "for", "we", "at", "this", "be", "with", "have", "what", "who", "why", "how"}
    query_words = {w for w in query_words if w not in stopwords}
    if not query_words:
        return pd.DataFrame()
    for _, row in df_tech.iterrows():
        text = f"{row['year']} {row['parameter']} {row['description']} {row['unit_value']}"
        words = set(re.findall(r'\b\w+\b', text.lower()))
        matches = len(query_words.intersection(words))
        if matches > 0:
            results.append({
                "Year": row["year"],
                "Parameter": row["parameter"],
                "Description": row["description"],
                "Value": row["unit_value"],
                "Match Score": matches
            })
    if not results:
        return pd.DataFrame()
    results = sorted(results, key=lambda x: x["Match Score"], reverse=True)[:k]
    return pd.DataFrame(results)

# Load spaCy
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
    return nlp

nlp = load_spacy_model()

# Precompute TF-IDF for search engine
@st.cache_resource
def compute_idf(corpus):
    n_docs = len(corpus)
    idf = {}
    doc_word_sets = []
    
    for doc in corpus:
        tokens = set(str(doc).lower().split())
        doc_word_sets.append(tokens)
        
    all_words = set(w for s in doc_word_sets for w in s)
    for word in all_words:
        n_containing = sum(1 for doc_set in doc_word_sets if word in doc_set)
        idf[word] = math.log(n_docs / (n_containing + 1)) + 1
    return idf

corpus = df["message_text"].astype(str).tolist()
idf_dict = compute_idf(corpus)

# TF-IDF Scratch Search Helper
def compute_tfidf(doc, idf):
    tokens = str(doc).lower().split()
    counts = Counter(tokens)
    tf = {w: c / len(tokens) for w, c in counts.items() if len(tokens) > 0}
    tfidf = {}
    for word, tf_val in tf.items():
        if word in idf:
            tfidf[word] = tf_val * idf[word]
    return tfidf

def search_corpus(query, k=5):
    query_tfidf = compute_tfidf(query, idf_dict)
    if not query_tfidf:
        return pd.DataFrame()
        
    scores = []
    for idx, doc in enumerate(corpus):
        doc_tfidf = compute_tfidf(doc, idf_dict)
        dot_product = sum(val * doc_tfidf.get(word, 0) for word, val in query_tfidf.items())
        
        q_mag = math.sqrt(sum(v**2 for v in query_tfidf.values()))
        doc_mag = math.sqrt(sum(v**2 for v in doc_tfidf.values()))
        
        sim = 0.0
        if q_mag > 0 and doc_mag > 0:
            sim = dot_product / (q_mag * doc_mag)
            
        if sim > 0:
            scores.append((idx, sim))
            
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    results = []
    for idx, score in scores:
        results.append({
            "Driver": df.iloc[idx]["driver_name"],
            "Grand Prix": df.iloc[idx]["grand_prix"],
            "Message": df.iloc[idx]["message_text"],
            "Similarity Score": round(score, 4)
        })
    return pd.DataFrame(results)

# Sidebar configurations
st.sidebar.title("🏎️ Navigation")
app_mode = st.sidebar.selectbox("Choose the module dashboard:", 
    [
        "Overview & Dataset Explorer", 
        "TF-IDF Search Engine", 
        "Sentiment & Emotion Analysis", 
        "spaCy NLP & Custom NER Visualizer", 
        "Interactive Chatbot",
        "RAG Simulator"
    ]
)

# Sentiment Lexicon Tuner in Sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Sentiment Lexicon Tuner"):
    st.markdown("Adjust the emotional impact weights of F1 words:")
    wt_wtf = st.slider("wtf / fuck", -3.0, 0.0, -2.0, 0.5)
    wt_slow = st.slider("slow / bad", -2.0, 0.0, -1.0, 0.2)
    wt_happy = st.slider("happy / great", 0.0, 3.0, 1.5, 0.5)
    wt_won = st.slider("won / p1", 0.0, 3.0, 2.0, 0.5)

# Build dynamic lexicon based on sidebar tuner
sentiment_lexicon = {
    "happy": wt_happy, "great": wt_happy, "awesome": wt_happy, "p1": wt_won, "won": wt_won, "beautiful": wt_happy, "incredible": wt_happy,
    "push": 0.5, "copy": 0.0, "understood": 0.0, "box": 0.0, "ok": 0.0,
    "frustrated": -1.5, "angry": -1.5, "screaming": -1.5, "wtf": wt_wtf, "fuck": wt_wtf, "f**k": wt_wtf,
    "no": -0.8, "why": -1.0, "problem": -1.0, "slow": wt_slow, "worst": -1.8, "difficult": -1.0, "bad": wt_slow
}

def get_sentiment(text):
    words = str(text).lower().split()
    score = 0.0
    for w in words:
        w_clean = re.sub(r'[^a-zA-Z*]', '', w)
        if w_clean in sentiment_lexicon:
            score += sentiment_lexicon[w_clean]
    return score

# Run selected module dashboard
if app_mode == "Overview & Dataset Explorer":
    st.title("📊 F1 Team Radio Dataset Explorer")
    st.markdown("Explore and filter the corpus of Formula 1 team radio communications.")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><h4>Messages Scraped</h4><h2>" + str(len(df)) + "</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>Drivers</h4><h2>" + str(df["driver_name"].nunique()) + "</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h4>Grand Prix Races</h4><h2>" + str(df["grand_prix"].nunique()) + "</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h4>Avg Length</h4><h2>" + f"{df['message_len'].mean():.2f} words" + "</h2></div>", unsafe_allow_html=True)
        
    st.markdown("### Filter Data")
    col_gp, col_drv = st.columns(2)
    with col_gp:
        gps = ["All"] + list(df["grand_prix"].unique())
        selected_gp = st.selectbox("Select Grand Prix", gps)
    with col_drv:
        drivers = ["All"] + list(df["driver_name"].unique())
        selected_drv = st.selectbox("Select Driver", drivers)
        
    filtered_df = df.copy()
    if selected_gp != "All":
        filtered_df = filtered_df[filtered_df["grand_prix"] == selected_gp]
    if selected_drv != "All":
        filtered_df = filtered_df[filtered_df["driver_name"] == selected_drv]
        
    st.write(f"Showing **{len(filtered_df)}** messages:")
    st.dataframe(filtered_df[["grand_prix", "driver_name", "timestamp", "message_text"]].head(200), use_container_width=True)

    # Word Frequency & Word Cloud side-by-side
    st.markdown("### Visual Vocabulary Analysis")
    col_plot, col_cloud = st.columns(2)
    
    text_corpus = " ".join(filtered_df["message_text"].astype(str).tolist()).lower()
    words = re.findall(r'\b\w+\b', text_corpus)
    stopwords = {
        # Standard grammatical stopwords
        "the", "and", "to", "a", "is", "of", "in", "it", "on", "you", "that", "for", 
        "we", "at", "this", "be", "with", "have", "what", "who", "why", "how", "as", 
        "an", "if", "or", "but", "by", "from", "about", "your", "my", "me", "our",
        "us", "they", "them", "he", "she", "his", "her", "their", "will", "would",
        "should", "could", "can", "are", "was", "were", "been", "has", "had", "do",
        "does", "did", "dont", "im", "its", "ive", "youre", "were", "theres", "well",
        "just", "like", "so", "go", "going", "gone", "get", "getting", "got", "gotten",
        "take", "taking", "took", "put", "putting", "keep", "keeping", "kept", "stay",
        "staying", "come", "coming", "came", "make", "making", "made", "think", "thinking",
        "thought", "say", "saying", "said", "tell", "telling", "told", "know", "knowing",
        "knew", "see", "seeing", "saw", "look", "looking", "looked", "want", "wanting",
        "yeah", "yes", "ok", "okay", "no", "not", "good", "bad", "slow", "fast",
        "up", "down", "out", "in", "here", "there", "now", "then", "very", "too",
        "all", "any", "some", "more", "most", "other", "another", "everyone", "someone",
        "anyone", "everything", "something", "anything", "really", "still", "please",
        "thank", "thanks", "sorry", "excuse", "guy", "guys", "mate", "teammate",
        
        # Residuals / Contraction parts
        "ll", "ve", "d", "re", "s", "t", "don", "isn", "aren", "wasn", "weren", "haven",
        "hasn", "hadn", "won", "wouldn", "couldn", "shouldn", "cant",
        
        # Numbers written as words (not informative for F1 radio concepts)
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred", "thousand", "plus", "minus", "zero",
        
        # Verbose/filler radio words
        "need", "needed", "needs", "let", "lets", "make", "sure", "copy", "copied",
        "understands", "understood", "focus", "push", "pushing", "pushed", "turn", "turns"
    }
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
    
    with col_plot:
        top_words = Counter(filtered_words).most_common(15)
        if top_words:
            w_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Frequency", y="Word", data=w_df, palette="Reds_r", ax=ax)
            ax.set_title("Most Common Words (Stopwords Excluded)")
            st.pyplot(fig)
            
    with col_cloud:
        if filtered_words:
            st.markdown("#### Word Cloud")
            word_freq = Counter(filtered_words)
            wordcloud = WordCloud(width=800, height=500, background_color="#0e1117", colormap="Reds").generate_from_frequencies(word_freq)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.imshow(wordcloud, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig2)
            
    # Reference Databases Section
    st.markdown("---")
    st.markdown("### 🗂️ Integrated F1 Reference Databases (2025 & 2026)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        with st.expander("📋 F1 2025/2026 Teams & Drivers Database", expanded=False):
            if not df_teams.empty:
                st.dataframe(df_teams, use_container_width=True)
            else:
                st.info("Teams & Drivers database not loaded.")
    with col_t2:
        with st.expander("⚙️ F1 2025/2026 Car Technical Specifications", expanded=False):
            if not df_tech.empty:
                st.dataframe(df_tech, use_container_width=True)
            else:
                st.info("Technical specifications database not loaded.")
                
    # Advanced Corpus Analytics Section
    st.markdown("---")
    st.markdown("### 📈 Advanced Corpus Analytics")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### Top 10 Most Active Drivers in Radio Corpus")
        driver_counts = filtered_df["driver_name"].value_counts().head(10)
        if not driver_counts.empty:
            fig_drv, ax_drv = plt.subplots(figsize=(8, 5))
            fig_drv.patch.set_facecolor('#0e1117')
            ax_drv.set_facecolor('#1e222b')
            ax_drv.tick_params(colors='#f0f2f6')
            ax_drv.yaxis.label.set_color('#f0f2f6')
            ax_drv.xaxis.label.set_color('#f0f2f6')
            ax_drv.title.set_color('#e10600')
            
            sns.barplot(x=driver_counts.values, y=driver_counts.index, palette="Oranges_r", ax=ax_drv)
            ax_drv.set_title("Message Count by Driver", color="#e10600")
            ax_drv.set_xlabel("Number of Messages", color="#f0f2f6")
            st.pyplot(fig_drv)
            
    with col_g2:
        st.markdown("#### Radio Message Category Distribution (NLP Rules)")
        def classify_msg(text):
            t = str(text).lower()
            if any(w in t for w in ["box", "pit", "strat", "tire", "soft", "medium", "hard", "tyre", "compound", "lap"]):
                return "Strategy"
            elif any(w in t for w in ["gap", "behind", "ahead", "interval", "p1", "p2", "p3", "pos", "position", "leader", "traffic"]):
                return "Position/Gap"
            elif any(w in t for w in ["wtf", "fuck", "bad", "slow", "no", "problem", "crash", "angry", "worst", "screaming"]):
                return "Emotional"
            else:
                return "Technical/General"
                
        categories = filtered_df["message_text"].apply(classify_msg)
        cat_counts = categories.value_counts()
        if not cat_counts.empty:
            fig_cat, ax_cat = plt.subplots(figsize=(8, 5))
            fig_cat.patch.set_facecolor('#0e1117')
            ax_cat.set_facecolor('#1e222b')
            ax_cat.tick_params(colors='#f0f2f6')
            ax_cat.title.set_color('#e10600')
            
            colors = ["#e10600", "#ff7f0e", "#1f77b4", "#2ca02c"]
            ax_cat.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%', colors=colors[:len(cat_counts)], 
                       textprops={'color': '#f0f2f6'}, startangle=140)
            ax_cat.set_title("NLP Message Intent/Category Share", color="#e10600")
            st.pyplot(fig_cat)
            
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.markdown("#### Driver Average Sentiment Score (Top 10 Drivers)")
        # Calculate sentiment
        filtered_df["temp_sent"] = filtered_df["message_text"].apply(get_sentiment)
        # Filter top drivers only
        top_driver_names = filtered_df["driver_name"].value_counts().head(10).index
        top_driver_df = filtered_df[filtered_df["driver_name"].isin(top_driver_names)]
        avg_sent = top_driver_df.groupby("driver_name")["temp_sent"].mean().sort_values()
        
        if not avg_sent.empty:
            fig_sent, ax_sent = plt.subplots(figsize=(8, 5))
            fig_sent.patch.set_facecolor('#0e1117')
            ax_sent.set_facecolor('#1e222b')
            ax_sent.tick_params(colors='#f0f2f6')
            ax_sent.yaxis.label.set_color('#f0f2f6')
            ax_sent.xaxis.label.set_color('#f0f2f6')
            ax_sent.title.set_color('#e10600')
            
            bar_colors = ['#ff2d20' if val < 0 else '#2ca02c' for val in avg_sent.values]
            sns.barplot(x=avg_sent.values, y=avg_sent.index, palette="coolwarm", ax=ax_sent) # using a default palette to prevent mapping issues
            ax_sent.set_title("Mean Sentiment Score by Driver", color="#e10600")
            ax_sent.set_xlabel("Average Sentiment Score", color="#f0f2f6")
            st.pyplot(fig_sent)
            
    with col_g4:
        st.markdown("#### Message Length Distribution (Word Count)")
        if not filtered_df.empty:
            fig_len, ax_len = plt.subplots(figsize=(8, 5))
            fig_len.patch.set_facecolor('#0e1117')
            ax_len.set_facecolor('#1e222b')
            ax_len.tick_params(colors='#f0f2f6')
            ax_len.yaxis.label.set_color('#f0f2f6')
            ax_len.xaxis.label.set_color('#f0f2f6')
            ax_len.title.set_color('#e10600')
            
            sns.histplot(filtered_df["message_len"], bins=15, kde=True, color="#ff7f0e", ax=ax_len)
            ax_len.set_title("Distribution of Message Lengths", color="#e10600")
            ax_len.set_xlabel("Word Count", color="#f0f2f6")
            ax_len.set_ylabel("Frequency", color="#f0f2f6")
            st.pyplot(fig_len)

    st.markdown("---")
    st.markdown("### 📚 Background Operations & Workings")
    show_ov_expl = st.toggle("🔍 Show Academic Explanations", value=True, key="toggle_ov_expl")
    if show_ov_expl:
        st.markdown("Explore the data engineering and exploratory text processing taking place behind the scenes:")
        tab_data, tab_stats, tab_cloud = st.tabs([
            "📊 Data Schema & Sourcing",
            "📈 Statistical Distribution",
            "☁️ Stopword & Word Frequencies"
        ])
        with tab_data:
            st.markdown("#### 📊 Data Schema & Sourcing")
            st.markdown("""
            * **Sourcing Method:** Data was collected using a custom BeautifulSoup crawler that scraped transcripts from the AutoLAP archive, honoring rules from `robots.txt`.
            * **Schema Details:**
              * `grand_prix`: Categorical variable mapping the race location (Barcelona, Monaco, Montreal).
              * `driver_name`: Categorical identifier for the driver who spoke.
              * `timestamp`: Temporal sequence key indicating when the radio exchange happened.
              * `message_text`: The raw transcribed sentence.
            * **Reference Tables:** Side databases (Teams & Drivers, Technical Specifications) are loaded from separate CSV files to enrich queries with context.
            """)
        with tab_stats:
            st.markdown("#### 📈 Statistical Distribution")
            st.markdown("""
            * **Word Count Histogram:** Tracks message length distributions. In team radio, text follows a power-law-like decay (highly telegraphic: most messages are under 10 words, with a long tail of rare, detailed reports).
            * **NLP Classification Rules:** Categorizes messages into *Strategy*, *Position/Gap*, *Emotional*, and *Technical* using fast regular-expression-based token scanning.
            * **Driver Activity Analysis:** Aggregate counts of messages grouped by driver name to show which drivers talk most frequently to their race engineers.
            """)
        with tab_cloud:
            st.markdown("#### ☁️ Stopword & Word Frequencies")
            st.markdown("""
            * **Frequency Counting:** Computes the occurrence rate of words in the filtered text using standard token frequency analysis.
            * **Stopwords Removal:** Standard English grammatical filler words (`"the"`, `"and"`, `"to"`, etc.) are removed first. Otherwise, the most common words would just be grammar noise instead of meaningful domain terms like `"lap"`, `"gap"`, `"box"`.
            * **Word Cloud Generation:** Draws an image layout where the size of each word is proportional to its frequency in the cleaned corpus.
            """)

elif app_mode == "TF-IDF Search Engine":
    st.title("🔍 TF-IDF Radio Message Search Engine")
    st.markdown("Search for messages in the corpus using a **TF-IDF engine built from scratch**. Matches keywords dynamically using Cosine Similarity.")
    
    query = st.text_input("Enter F1 Query (e.g. 'box box pit stop', 'tire degradation', 'gap behind Hamilton'):", "box box")
    k = st.slider("Number of results", 1, 15, 5)
    
    if st.button("Search"):
        results = search_corpus(query, k=k)
        if not results.empty:
            st.success(f"Found {len(results)} matches:")
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No matches found. Try other keywords!")

    st.markdown("---")
    st.markdown("### 📚 Background Operations & Workings")
    show_search_expl = st.toggle("🔍 Show Academic Explanations", value=True, key="toggle_search_expl")
    if show_search_expl:
        st.markdown("Learn how the custom TF-IDF retrieval system computes matches:")
        tab_tfidf, tab_cosine = st.tabs([
            "📊 TF-IDF Vectorization",
            "📐 Cosine Similarity Matcher"
        ])
        with tab_tfidf:
            st.markdown("#### 📊 TF-IDF Vectorization")
            st.markdown("""
            * **Term Frequency (TF):** Measures how often a term $t$ appears in document $d$:
              $$\\text{TF}(t, d) = \\frac{\\text{Count}(t \\text{ in } d)}{\\text{Total words in } d}$$
            * **Inverse Document Frequency (IDF):** Measures how unique a term is across the entire corpus of $N$ documents:
              $$\\text{IDF}(t) = \\log\\left(1 + \\frac{N}{1 + \\text{DF}(t)}\\right)$$
              where $\\text{DF}(t)$ is the number of documents containing term $t$.
            * **Combined Weight:** $\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$. Highly frequent words like "the" get low IDF (approaching 0), while rare strategic words like "degradation" or "box" get high weights.
            """)
        with tab_cosine:
            st.markdown("#### 📐 Cosine Similarity Matcher")
            st.markdown("""
            * **Definition:** Calculates the cosine of the angle between two multi-dimensional vectors. It measures orientation similarity rather than magnitude, which makes it invariant to document length:
              $$\\text{Similarity}(Q, D) = \\frac{Q \\cdot D}{\\|Q\\| \\|D\\|} = \\frac{\\sum_{i} Q_i D_i}{\\sqrt{\\sum_{i} Q_i^2} \\sqrt{\\sum_{i} D_i^2}}$$
            * **In the Background:** The query is vectorised. Then, for each document, we compute the dot product of the query vector with the document's TF-IDF vector, divide by the product of their Euclidean lengths, and sort the documents in descending order of similarity.
            """)

elif app_mode == "Sentiment & Emotion Analysis":
    st.title("❤️ F1 Radio Sentiment & Emotion Analyzer")
    st.markdown("Calculate emotional intensity on team radio transcripts using custom lexicons.")
    
    df["sentiment"] = df["message_text"].apply(get_sentiment)
    
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("### Happy / Celebratory Radio Messages")
        pos_df = df[df["sentiment"] >= 1.5][["driver_name", "message_text", "sentiment"]].sort_values(by="sentiment", ascending=False).head(5)
        for _, row in pos_df.iterrows():
            st.info(f"**{row['driver_name']}**: \"{row['message_text']}\" (Score: `{row['sentiment']}`)")
            
    with col_neg:
        st.markdown("### Frustrated / Urgent Radio Messages")
        neg_df = df[df["sentiment"] <= -1.5][["driver_name", "message_text", "sentiment"]].sort_values(by="sentiment").head(5)
        for _, row in neg_df.iterrows():
            st.error(f"**{row['driver_name']}**: \"{row['message_text']}\" (Score: `{row['sentiment']}`)")
            
    st.markdown("### Interactive Driver Sentiment Timeline")
    st.markdown("Track how a driver's emotions and stress levels fluctuated chronologically over the race.")
    
    drv_list = list(df["driver_name"].unique())
    selected_drv_timeline = st.selectbox("Select Driver to Plot Timeline:", drv_list, index=drv_list.index("Carlos Sainz") if "Carlos Sainz" in drv_list else 0)
    selected_gp_timeline = st.selectbox("Select Grand Prix:", list(df["grand_prix"].unique()))
    
    drv_df = df[(df["driver_name"] == selected_drv_timeline) & (df["grand_prix"] == selected_gp_timeline)].copy()
    drv_df = drv_df.sort_values(by="timestamp")
    
    if len(drv_df) > 5:
        drv_df["message_index"] = np.arange(len(drv_df))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(drv_df["message_index"], drv_df["sentiment"], marker="o", linestyle="-", color="crimson", label="Sentiment Score")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title(f"Sentiment Chronology for {selected_drv_timeline} at {selected_gp_timeline}")
        ax.set_xlabel("Message Index (Chronological)")
        ax.set_ylabel("Sentiment Score")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Not enough messages for this driver in this session to plot a chronology.")

    st.markdown("---")
    st.markdown("### 📚 Background Operations & Workings")
    show_sent_expl = st.toggle("🔍 Show Academic Explanations", value=True, key="toggle_sent_expl")
    if show_sent_expl:
        st.markdown("Learn how the custom sentiment engine calculates F1 emotional scores:")
        tab_lexicon, tab_timeline = st.tabs([
            "📖 Lexicon Sentiment Scoring",
            "📈 Temporal Sentiment Aggregation"
        ])
        with tab_lexicon:
            st.markdown("#### 📖 Lexicon Sentiment Scoring")
            st.markdown("""
            * **Lexicon-Based Sentiment:** Assigns pre-defined numerical polarity scores to specific semantic words.
            * **Custom F1 Adjustments:** Standard lexicon sentiment engines (like VADER or TextBlob) do not know F1 terminology. For example, `"box"` (pit stop) or `"copy"` (understood) might be flagged incorrectly. We built a custom sentiment dictionary:
              * Celebratory tokens (e.g., `"won"`, `"p1"`, `"beautiful"`) are given positive weights.
              * Frustrated / Urgent tokens (e.g., `"wtf"`, `"fuck"`, `"slow"`, `"problem"`) are given negative weights.
            * **Sidebar Tuner Integration:** The weights of key tokens are dynamically controlled in real-time by the sliders in the sidebar, modifying the underlying dictionary mapping on-the-fly.
            """)
        with tab_timeline:
            st.markdown("#### 📈 Temporal Sentiment Aggregation")
            st.markdown("""
            * **Chronological Indexing:** Sorts the driver's messages by timestamp to create a chronological sequence.
            * **Emotion Tracking:** By graphing the sentiment of consecutive radio messages, we can trace a driver's emotional trajectory. A drop in sentiment score highlights moments of high stress, tire dropoff, or mechanical issues (such as retirements/DNFs), while peaks indicate race finishes or overtaking maneuvers.
            """)

elif app_mode == "spaCy NLP & Custom NER Visualizer":
    st.title("🏷️ spaCy NLP & Custom NER Visualizer")
    st.markdown("Interactively explore how raw text is processed by spaCy for **Tokenization**, **Part-of-Speech (POS) tagging**, **Lemmatization**, and **Named Entity Recognition (NER)**.")
    
    text_input = st.text_area("Paste a F1 radio message or write your own to visualize:", 
        "Alright George, box box. Max Verstappen is leading on soft tires, but Red Bull strategy says mode 7. Hamilton is currently P3."
    )
    
    if st.button("Analyze Text"):
        # 1. Custom Named Entity Recognition (NER)
        st.markdown("### 🏆 Custom F1 Named Entity Recognition (NER)")
        st.markdown("Standard spaCy models do not identify F1 domain entities. Here we run our custom entities parser highlighting drivers, teams, tires, and strategies.")
        
        # Build document with custom entities
        doc = nlp(text_input)
        spans = []
        
        # Define match lists
        drivers_list = ["verstappen", "hamilton", "leclerc", "sainz", "norris", "piastri", "russell", "alonso", "stroll", "gasly", "albon", "ocon", "perez", "hulkenberg", "sargeant", "bearman", "tsunoda", "lawson", "bortoleto", "antonelli", "george", "lewis", "max", "charles", "carlos", "lando", "oscar", "fernando", "esteban", "nico", "sergio", "logan", "kimi"]
        teams_list = ["red bull", "mercedes", "ferrari", "mclaren", "aston martin", "alpine", "williams", "haas", "audi", "sauber", "rb", "visa cash app"]
        tires_list = ["soft", "medium", "hard", "softs", "mediums", "hards", "inters", "wets", "intermediate"]
        
        matcher_ner = Matcher(nlp.vocab)
        for d in drivers_list:
            matcher_ner.add("DRIVER", [[{"LOWER": d}]])
        for t in teams_list:
            pattern = [{"LOWER": w} for w in t.split()]
            matcher_ner.add("TEAM", [pattern])
        for tr in tires_list:
            matcher_ner.add("TIRE_COMPOUND", [[{"LOWER": tr}]])
            
        matcher_ner.add("STRATEGY", [[{"LOWER": {"IN": ["mode", "strat"]}}, {"LIKE_NUM": True}], [{"LOWER": "box"}, {"LOWER": "box"}]])
        
        matches = matcher_ner(doc)
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            spans.append(Span(doc, start, end, label=label))
            
        # Filter overlaps
        from spacy.util import filter_spans
        doc.ents = filter_spans(spans)
        
        # Render entities
        from spacy import displacy
        colors = {"DRIVER": "#ff4b4b", "TEAM": "#385a7c", "TIRE_COMPOUND": "#ff7f0e", "STRATEGY": "#2ca02c"}
        options = {"ents": ["DRIVER", "TEAM", "TIRE_COMPOUND", "STRATEGY"], "colors": colors}
        html = displacy.render(doc, style="ent", options=options, page=False)
        st.markdown(f'<div style="background-color:#1e222b; padding:15px; border-radius:8px; margin-bottom:20px; line-height:2.0; color:#f0f2f6;">{html}</div>', unsafe_allow_html=True)
        
        # 2. Tokenization, POS, Lemmatization Table
        st.markdown("### 🔍 spaCy Tokenization, POS Tagging & Lemmatization Explorer")
        st.markdown("The table below details every parsed token, its syntactic class (Part-of-Speech), its base form (lemma), and stopword status.")
        token_data = []
        for token in doc:
            token_data.append({
                "Token": token.text,
                "POS Tag": token.pos_,
                "POS Explanation": spacy.explain(token.pos_) if token.pos_ else "",
                "Lemma": token.lemma_,
                "Is Stopword": token.is_stop
            })
        st.dataframe(pd.DataFrame(token_data), use_container_width=True)
        
        # 3. Custom Matcher Patterns
        st.markdown("### 📜 Pattern Matcher Rule Triggers")
        st.markdown("Displays raw matching patterns matched via custom rules:")
        if matches:
            for match_id, start, end in matches:
                label = nlp.vocab.strings[match_id]
                span = doc[start:end]
                st.write(f"- Triggered Rule **{label}**: matched text `\"{span.text}\"`")
        else:
            st.info("No rule triggers matched in the input.")
            
        # 4. Live Machine Learning Classifier
        st.markdown("---")
        st.markdown("### 🤖 Live Machine Learning Classifier")
        st.markdown("Classify this team radio message using a **Logistic Regression classifier** trained in memory on our 18,574-row corpus.")
        
        # Predict
        text_vec = vec_clf.transform([text_input])
        prediction = model_clf.predict(text_vec)[0]
        probabilities = model_clf.predict_proba(text_vec)[0]
        
        # Display Prediction
        st.success(f"**Predicted Category:** `{prediction}`")
        
        # Plot probabilities
        prob_df = pd.DataFrame({
            "Category": model_clf.classes_,
            "Probability": probabilities
        })
        
        fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
        fig_prob.patch.set_facecolor('#0e1117')
        ax_prob.set_facecolor('#1e222b')
        ax_prob.tick_params(colors='#f0f2f6')
        ax_prob.yaxis.label.set_color('#f0f2f6')
        ax_prob.xaxis.label.set_color('#f0f2f6')
        ax_prob.title.set_color('#e10600')
        
        sns.barplot(x="Probability", y="Category", data=prob_df, palette="Reds_r", ax=ax_prob)
        ax_prob.set_title("Classifier Probability Distribution", color="#e10600")
        ax_prob.set_xlabel("Probability", color="#f0f2f6")
        ax_prob.set_ylabel("Category", color="#f0f2f6")
        ax_prob.set_xlim(0, 1.0)
        st.pyplot(fig_prob)

    st.markdown("---")
    st.markdown("### 📚 Background NLP Tools Explanation")
    
    # Toggle to show/hide the entire section
    show_explanations = st.toggle("🔍 Show Academic Explanations", value=True)
    
    if show_explanations:
        st.markdown("Explore how these Natural Language Processing techniques operate behind the scenes to analyze F1 documents:")
        
        tab_tok, tab_pos, tab_lem, tab_stop, tab_ner, tab_clf = st.tabs([
            "🔠 Tokenization", 
            "🏷️ POS Tagging", 
            "🌱 Lemmatization", 
            "🛑 Stopwords", 
            "🔍 Custom NER", 
            "🤖 ML Classifier"
        ])
        
        with tab_tok:
            st.markdown("#### 🔠 Tokenization")
            st.markdown("""
            * **Definition:** The fundamental first step of any NLP pipeline that splits a raw string of text into individual units called **tokens** (e.g., words, punctuation, numbers).
            * **F1 Context:** A message like `\"Alright George, box box.\"` is tokenized into `[\"Alright\", \"George\", \",\", \"box\", \"box\", \".\"]`.
            * **Why it matters:** Computers cannot read text as a whole; tokenization establishes the atomic units of meaning for all downstream operations.
            """)
            
        with tab_pos:
            st.markdown("#### 🏷️ Part-of-Speech (POS) Tagging")
            st.markdown("""
            * **Definition:** Labeling each token with its grammatical class (e.g., Noun, Verb, Adjective, Adverb, Conjunction) based on both its definition and surrounding context.
            * **F1 Context:** In `"box box"`, `box` is tagged as a **VERB** (imperative action). In `"Red Bull strategy says mode 7"`, `Red` and `Bull` are tagged as **PROPN** (Proper Nouns).
            * **Why it matters:** Helps disambiguate word meanings (e.g., "box" as a noun vs. "box" as a verb) and reveals sentence structure.
            """)
            
        with tab_lem:
            st.markdown("#### 🌱 Lemmatization")
            st.markdown("""
            * **Definition:** Reducing inflected or variant forms of a word to its dictionary base form, known as a **lemma**. Unlike simple stemming, lemmatization uses vocabulary lookup and morphological analysis.
            * **F1 Context:** The word `"leading"` is lemmatized to `"lead"`, `"tires"` to `"tire"`, and `"won"` to `"win"`.
            * **Why it matters:** It groups variant forms of a word together, reducing vocabulary complexity and improving search/text classification performance.
            """)
            
        with tab_stop:
            st.markdown("#### 🛑 Stopwords Detection")
            st.markdown("""
            * **Definition:** Flagging high-frequency, low-meaning words (such as `"the"`, `"is"`, `"and"`, `"of"`) that appear frequently in language but carry very little unique semantic information.
            * **F1 Context:** In `"Max Verstappen is leading on soft tires"`, the words `"is"` and `"on"` are flagged as stopwords.
            * **Why it matters:** Filtering stopwords helps focus on the content-carrying keywords during semantic analysis, keyword extraction, and search queries.
            """)
            
        with tab_ner:
            st.markdown("#### 🔍 Custom Named Entity Recognition (NER)")
            st.markdown("""
            * **Definition:** Identifying and classifying key real-world entities (such as people, organizations, locations, quantities, dates) in unstructured text.
            * **F1 Context:** Because generic NER models (like standard English spaCy models) do not recognize F1 terms, we built a **Custom NER Matcher**. This matcher uses specialized dictionary rules and string pattern matches to extract:
              * **`DRIVER`**: e.g., `"Verstappen"`, `"Hamilton"`, `"Russell"`
              * **`TEAM`**: e.g., `"Red Bull"`, `"Mercedes"`, `"Ferrari"`
              * **`TIRE_COMPOUND`**: e.g., `"soft"`, `"medium"`, `"hard"`, `"inters"`
              * **`STRATEGY`**: e.g., `"mode 7"`, `"box box"`, `"strat"`
            * **Why it matters:** NER provides structural semantic tags over unstructured text, enabling automated extraction of structured insights.
            """)
            
        with tab_clf:
            st.markdown("#### 🤖 Live Machine Learning Classification (TF-IDF & Logistic Regression)")
            st.markdown("""
            * **Definition:** Automatically assigning text categories using mathematical features:
              1. **TF-IDF (Term Frequency-Inverse Document Frequency):** Transforms tokens into numerical vectors by evaluating how unique a word is to a specific document relative to all documents.
              2. **Logistic Regression:** A linear model that estimates the probability of binary/multiclass outcomes based on input features (here, the TF-IDF representation).
            * **F1 Context:** When you type a sentence, the TF-IDF vectorizer extracts features, and the Logistic Regression classifier calculates the probability that the text represents a `Driver Radio`, `Race Report`, `Regulation`, `Technical Specification`, or `Team Profile`.
            """)


elif app_mode == "Interactive Chatbot":
    st.title("🤖 F1 Radio & Race Results Chatbot")
    st.markdown("Ask the bot about driver communications, race winners, DNF retirements, or basic NLP definitions and model scores.")
    
    user_msg = st.text_input("Ask a question (e.g. 'Who won Monaco?', 'Who got out at Barcelona?', 'What is RAG?', 'How big is the database?'):", "who won Monaco GP?")
    
    if st.button("Send"):
        query_lower = user_msg.lower().strip()

        # ── helper: search corpus and return top messages ──────────────────
        def corpus_search_reply(keyword, n=3):
            hits = df[df["message_text"].str.lower().str.contains(keyword, na=False)]
            if not hits.empty:
                rows = hits[["driver_name", "message_text"]].head(n)
                for _, r in rows.iterrows():
                    st.write(f"- **{r['driver_name']}:** \"{r['message_text']}\"")
                return True
            return False

        # ────────────────────────────────────────────────────────────────────
        # 1. GREETINGS
        # ────────────────────────────────────────────────────────────────────
        if any(k in query_lower for k in ["hi", "hello", "hey", "good morning", "good afternoon", "sup", "howdy"]):
            st.write("**F1 Bot:** Hello! 👋 I'm the F1 Radio & Results Bot. You can ask me about:")
            st.write("- 🏆 Race winners & podiums ('Who won Monaco?')")
            st.write("- 💥 DNF retirements ('Who got out at Barcelona?')")
            st.write("- 📻 Driver radio messages ('What did Sainz say?')")
            st.write("- 🔧 Tires & strategy ('What compounds did Hamilton use?')")
            st.write("- 📜 F1 Regulations 2025 & 2026 ('What are the 2026 engine rules?')")
            st.write("- 🧠 NLP topics ('What is TF-IDF?', 'Explain tokenization')")

        # ────────────────────────────────────────────────────────────────────
        # 2. PROJECT / AUTHOR INFO
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["built", "author", "who are you", "who made", "creator",
                                             "student", "shashwat", "singh", "project", "about this",
                                             "what is this", "tell me about", "hochschule", "aalen"]):
            st.write("**F1 Bot:** This project was designed and built by **Shashwat Singh**, a student of "
                     "**M.Sc. Machine Learning and Data Analytics** at **Hochschule Aalen**.")
            st.write("It is an end-to-end NLP pipeline covering **web scraping**, **TF-IDF search**, "
                     "**sentiment analysis**, **custom NER**, **deep learning classifiers** (RNN, LSTM, MLP), "
                     "and a **RAG simulator** — all applied to real F1 team radio transcripts.")

        # ────────────────────────────────────────────────────────────────────
        # 3. DATABASE / CORPUS STATS
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["how big", "how many", "database", "corpus", "size",
                                             "messages", "dataset", "rows", "records", "entries"]):
            st.write(f"**F1 Bot:** The corpus contains **{len(df):,} team radio messages** across "
                     f"**{df['grand_prix'].nunique()} Grand Prix races** with **{df['driver_name'].nunique()} unique drivers**.")
            st.write(f"It covers the **2026 Barcelona**, **Monaco**, and **Canadian** Grand Prix sessions.")

        # ────────────────────────────────────────────────────────────────────
        # 4. RACE WINNERS / PODIUMS / RACE SUMMARIES
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["win", "won", "winner", "podium", "1st", "first place",
                                             "who finished", "race result", "champion"]):
            matched_gp = None
            for gp_name in RACE_DATA.keys():
                short = gp_name.replace("Grand Prix", "").lower()
                if any(w in query_lower for w in short.split()):
                    matched_gp = gp_name
                    break
                if "spain" in query_lower and "barcelona" in short: matched_gp = gp_name; break
                if "catalunya" in query_lower and "barcelona" in short: matched_gp = gp_name; break
                if "montreal" in query_lower and "canada" in short: matched_gp = gp_name; break
            if matched_gp:
                race_info = RACE_DATA[matched_gp]
                st.write(f"**F1 Bot:** 🏆 The winner of the **{matched_gp}** was **{race_info['winner']}**.")
                st.write("**Podium:**")
                for p in race_info["podium"]:
                    st.write(f"  {p}")
            else:
                st.write("**F1 Bot:** Here are the race winners across the 3 Grand Prix in the dataset:")
                for gp, info in RACE_DATA.items():
                    st.write(f"- **{gp}:** Won by **{info['winner']}**")

        # ────────────────────────────────────────────────────────────────────
        # 5. DNF / RETIREMENTS / CRASHES
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["retire", "dnf", "out", "crash", "collision", "stopped",
                                             "mechanical", "failure", "accident", "incident", "withdrawal"]):
            matched_gp = None
            for gp_name in RACE_DATA.keys():
                short = gp_name.replace("Grand Prix", "").lower()
                if any(w in query_lower for w in short.split()):
                    matched_gp = gp_name; break
                if "spain" in query_lower and "barcelona" in short: matched_gp = gp_name; break
                if "montreal" in query_lower and "canada" in short: matched_gp = gp_name; break
                if "monaco" in query_lower and "monaco" in short: matched_gp = gp_name; break

            # Specific driver retirement queries
            driver_dnfs = {
                "hamilton": "Lewis Hamilton retired on **Lap 12** of the **2026 Barcelona GP** due to a **Power Unit failure**.",
                "albon": "Alexander Albon retired on **Lap 4** of the **2026 Barcelona GP** after a **collision in Turn 3**.",
                "perez": "Sergio Perez retired on **Lap 1** of the **2026 Monaco GP** after a **massive multi-car collision** at Saint Devote.",
                "hulkenberg": "Nico Hulkenberg retired on **Lap 1** of the **2026 Monaco GP** — also caught in the **Saint Devote collision**.",
                "ocon": "Esteban Ocon retired on **Lap 45** of the **2026 Monaco GP** due to a **gearbox failure**.",
                "sargeant": "Logan Sargeant retired on **Lap 25** of the **2026 Canadian GP** after **spinning into the wall at Turn 4**.",
                "leclerc": "Charles Leclerc retired on **Lap 40** of the **2026 Canadian GP** due to **engine mapping issues** following a rain shower.",
            }
            matched_driver = next((k for k in driver_dnfs if k in query_lower), None)
            if matched_driver:
                st.write(f"**F1 Bot:** {driver_dnfs[matched_driver]}")
            elif matched_gp:
                race_info = RACE_DATA[matched_gp]
                if race_info["retirements"]:
                    st.write(f"**F1 Bot:** DNF retirements at the **{matched_gp}**:")
                    for r in race_info["retirements"]:
                        st.write(f"- **{r['driver']}** — Lap {r['lap']}: *{r['reason']}*")
                else:
                    st.write(f"**F1 Bot:** No retirements recorded at the **{matched_gp}**.")
            else:
                st.write("**F1 Bot:** Here are all DNF retirements across the dataset:")
                for gp, info in RACE_DATA.items():
                    if info["retirements"]:
                        for r in info["retirements"]:
                            st.write(f"- **{gp}** | **{r['driver']}** — Lap {r['lap']}: *{r['reason']}*")

        # ────────────────────────────────────────────────────────────────────
        # 6. SPECIFIC DRIVER QUERIES (radio messages, performance)
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["verstappen", "hamilton", "leclerc", "sainz", "norris",
                                             "piastri", "russell", "alonso", "stroll", "gasly", "albon",
                                             "ocon", "perez", "hulkenberg", "tsunoda", "lawson",
                                             "bortoleto", "antonelli", "bearman"]):
            # Find which driver was mentioned
            driver_map = {
                "verstappen": "Max Verstappen", "hamilton": "Lewis Hamilton",
                "leclerc": "Charles Leclerc", "sainz": "Carlos Sainz",
                "norris": "Lando Norris", "piastri": "Oscar Piastri",
                "russell": "George Russell", "alonso": "Fernando Alonso",
                "stroll": "Lance Stroll", "gasly": "Pierre Gasly",
                "albon": "Alexander Albon", "ocon": "Esteban Ocon",
                "perez": "Sergio Perez", "hulkenberg": "Nico Hulkenberg",
                "tsunoda": "Yuki Tsunoda", "lawson": "Liam Lawson",
                "bortoleto": "Gabriel Bortoleto", "antonelli": "Kimi Antonelli",
                "bearman": "Oliver Bearman",
            }
            matched_drv_key = next((k for k in driver_map if k in query_lower), None)
            if matched_drv_key:
                full_name = driver_map[matched_drv_key]
                drv_msgs = df[df["driver_name"].str.lower().str.contains(matched_drv_key, na=False)]
                if not drv_msgs.empty:
                    st.write(f"**F1 Bot:** Here are sample radio messages from **{full_name}** (found {len(drv_msgs)} messages in database):")
                    for _, row in drv_msgs[["grand_prix", "message_text"]].head(5).iterrows():
                        st.write(f"- *[{row['grand_prix']}]* \"{row['message_text']}\"")
                else:
                    st.write(f"**F1 Bot:** No radio messages found for **{full_name}** in the current corpus.")

        # ────────────────────────────────────────────────────────────────────
        # 7. TIRE / COMPOUND QUESTIONS
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["tire", "tyre", "compound", "soft", "medium", "hard",
                                             "intermediate", "inter", "wet", "degradation", "deg",
                                             "blister", "graining", "pit stop", "undercut", "overcut"]):
            st.write("**F1 Bot:** 🔴⚪🟡 Here are real F1 radio messages about tires and compounds:")
            keyword = "soft|medium|hard|tyre|tire|compound|deg|blister"
            found = corpus_search_reply(keyword, n=5)
            if not found:
                st.write("No specific tire messages found. The 2026 compounds used were: Soft (C5), Medium (C3), Hard (C2).")

        # ────────────────────────────────────────────────────────────────────
        # 8. STRATEGY / BOX / PIT QUESTIONS
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["strategy", "strat", "box", "pit", "undercut", "overcut",
                                             "mode", "switch", "offset", "safety car", "vsc", "restart"]):
            st.write("**F1 Bot:** 🔧 Here are strategy-related radio messages from the corpus:")
            corpus_search_reply("box|strat|pit|mode|offset|undercut", n=5)

        # ────────────────────────────────────────────────────────────────────
        # 9. TEAMS QUESTIONS
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["red bull", "mercedes", "ferrari", "mclaren", "aston martin",
                                             "alpine", "williams", "haas", "sauber", "audi", "team",
                                             "constructor", "lineup", "drives for", "teammates"]):
            hit_rows = []
            for _, row in df_teams.iterrows():
                team_w = row["team"].lower()
                if any(t in query_lower for t in team_w.split() if t not in ["racing", "team", "amg", "cash", "app", "visa"]):
                    hit_rows.append(row)
            if hit_rows:
                st.write("**F1 Bot:** Here are the matching team & driver entries:")
                for row in hit_rows:
                    st.write(f"- **{row['driver']}** (#{row['number']}, {row['nationality']}) → **{row['team']}** | {row['role']}")
            else:
                st.write("**F1 Bot:** Here is the full 2025/2026 F1 driver lineup:")
                for _, row in df_teams.iterrows():
                    st.write(f"- **{row['driver']}** → {row['team']}")

        # ────────────────────────────────────────────────────────────────────
        # 10. F1 REGULATIONS 2025 & 2026
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["regulation", "rule", "mgu", "aero", "active aerodynamics",
                                             "z-mode", "x-mode", "e-fuel", "efuel", "override", "mom",
                                             "drs", "cost cap", "budget cap", "sustainable", "wheelbase",
                                             "car weight", "car width"]) or \
             any(yr in query_lower for yr in ["2025", "2026"]):
            if "2025" in query_lower:
                st.write("**F1 Bot:** Key **2025 F1 Regulations:**")
                st.write("- **Car Weight:** Minimum **798kg** (ground-effect rules continue).")
                st.write("- **Sporting:** 6 Sprint races. CFD/wind-tunnel ATR limits favour lower-ranked teams.")
                st.write("- **Budget Cap:** ~**$135 million** per team.")
            elif "2026" in query_lower or any(k in query_lower for k in ["mgu", "mom", "active aero", "efuel", "e-fuel", "override"]):
                st.write("**F1 Bot:** Key **2026 F1 Regulations** (major overhaul):")
                st.write("- **Power Unit:** MGU-H **removed**. Near 50/50 split — ICE 400kW + MGU-K 350kW.")
                st.write("- **Active Aerodynamics:** Moveable front/rear wings. **Z-mode** (high downforce corners) / **X-mode** (low drag straights).")
                st.write("- **Overtaking:** DRS replaced by **Manual Override Mode (MOM)** — electrical boost >290km/h.")
                st.write("- **Car Size:** Wheelbase 3400mm, Width 1900mm, Weight min **768kg** (30kg lighter).")
                st.write("- **Fuel:** 100% sustainable **synthetic e-fuels** mandatory.")
            else:
                retrieved_regs = search_regulations(user_msg, k=3)
                if not retrieved_regs.empty:
                    st.write("**F1 Bot:** Here are relevant regulation clauses:")
                    for _, r_row in retrieved_regs.iterrows():
                        st.write(f"- **{r_row['Year']} — {r_row['Category']}:** {r_row['Regulation Clause']}")
                else:
                    st.write("**F1 Bot:** Try asking: '2025 regulations' or '2026 engine rules'.")

        # ────────────────────────────────────────────────────────────────────
        # 11–14: NLP CONCEPTS, BENCHMARKS, DRIVERS LIST, SMART FALLBACK
        # (handled earlier in block — reached via the else at end)
        # ────────────────────────────────────────────────────────────────────
        elif any(k in query_lower for k in ["what is", "explain", "define", "how does", "how do", "what are"]):
            if any(k in query_lower for k in ["tf-idf", "tfidf", "tf idf", "term frequency"]):
                st.write("**F1 Bot:** **TF-IDF** weighs word importance: high TF (frequent in this doc) × high IDF (rare across all docs) = high score.")
                st.write("Example: `'box'` appears in few messages → high IDF. `'the'` appears in all → IDF ≈ 0.")
            elif any(k in query_lower for k in ["tokenization", "tokenise", "tokenize", "token"]):
                st.write("**F1 Bot:** **Tokenization** splits text into words/punctuation. `'Box this lap!'` → `['Box', 'this', 'lap', '!']`")
            elif any(k in query_lower for k in ["lemmatization", "lemma", "lemmatize"]):
                st.write("**F1 Bot:** **Lemmatization** maps words to base form: `'leading'` → `'lead'`, `'tires'` → `'tire'`, `'won'` → `'win'`.")
            elif any(k in query_lower for k in ["pos", "part of speech", "pos tag", "tagging"]):
                st.write("**F1 Bot:** **POS Tagging** labels each token: `'box box'` → both tagged as **VERB** (imperative command).")
            elif any(k in query_lower for k in ["ner", "named entity", "entity recognition"]):
                st.write("**F1 Bot:** **NER** extracts real-world entities. Our custom model detects: `DRIVER`, `TEAM`, `TIRE_COMPOUND`, `STRATEGY`.")
            elif any(k in query_lower for k in ["rag", "retrieval augmented", "retrieval-augmented"]):
                st.write("**F1 Bot:** **RAG** = retrieve facts from DB + inject as context into LLM prompt → grounded, factual answers.")
            elif any(k in query_lower for k in ["sentiment", "emotion", "polarity", "lexicon"]):
                st.write("**F1 Bot:** **Sentiment Analysis** scores text polarity. Our F1 lexicon scores `'p1'` positive, `'wtf'` negative.")
            elif any(k in query_lower for k in ["lstm", "rnn", "recurrent", "sequence"]):
                st.write("**F1 Bot:** **RNN** processes tokens sequentially. **LSTM** adds memory gates to remember long-range context. LSTM: 81.42%, RNN: 79.99%.")
            elif any(k in query_lower for k in ["svm", "logistic", "naive bayes", "classifier"]):
                st.write("**F1 Bot:** Classifiers on TF-IDF features: SVM 93.81% ✅ > MLP 92.57% > LogReg 92.21% > LSTM 81.42% > RNN 79.99% > NB 79.94%.")
            elif any(k in query_lower for k in ["sparse", "csr", "csc", "matrix"]):
                st.write("**F1 Bot:** **Sparse Matrix** stores only non-zero values. On 18,574 × 5,000 vocabulary, achieves ~**30× memory savings** vs dense.")
            elif any(k in query_lower for k in ["zipf", "luhn"]):
                st.write("**F1 Bot:** **Zipf's Law**: word frequency ∝ 1/rank. **Luhn's Cut**: remove ultra-frequent stopwords and ultra-rare terms to isolate high-information vocabulary.")
            elif any(k in query_lower for k in ["beautifulsoup", "scraping", "crawler", "scraper"]):
                st.write("**F1 Bot:** Data collected via a **BeautifulSoup crawler** on AutoLAP archive. robots.txt verified. 18,574 messages collected.")
            elif any(k in query_lower for k in ["cbow", "word embedding", "embedding", "word2vec"]):
                st.write("**F1 Bot:** **CBOW** trains 50-dim word vectors so `'box'` and `'pit'` become semantically close in vector space.")
            else:
                st.write("**F1 Bot:** I can explain: `TF-IDF`, `Tokenization`, `Lemmatization`, `POS Tagging`, `NER`, `RAG`, `Sentiment`, `Zipf's Law`, `Sparse Matrices`, `LSTM/RNN`, `SVM`, `Word Embeddings`, `Web Scraping`.")

        elif any(k in query_lower for k in ["accuracy", "benchmark", "best model", "performance", "model score", "highest"]):
            st.write("**F1 Bot:** Classifier Benchmarks:")
            st.write("| Model | Accuracy |")
            st.write("|---|---|")
            st.write("| Linear SVM | **93.81%** ✅ |")
            st.write("| PyTorch MLP | 92.57% |")
            st.write("| Logistic Regression | 92.21% |")
            st.write("| PyTorch LSTM | 81.42% |")
            st.write("| PyTorch RNN | 79.99% |")
            st.write("| Naive Bayes | 79.94% |")

        elif any(k in query_lower for k in ["which drivers", "list drivers", "all drivers", "driver names", "driver list"]):
            drivers_list = sorted(df["driver_name"].unique())
            st.write(f"**F1 Bot:** The corpus includes **{len(drivers_list)} drivers:** {', '.join(drivers_list)}")

        else:
            # Smart fallback — search corpus
            search_words = [w for w in query_lower.split() if len(w) > 3]
            found_any = False
            for sw in search_words:
                hits = df[df["message_text"].str.lower().str.contains(sw, na=False)]
                if not hits.empty and len(hits) > 1:
                    st.write(f"**F1 Bot:** Found **{len(hits)}** messages matching your query. Top results:")
                    for _, row in hits[["driver_name", "grand_prix", "message_text"]].head(4).iterrows():
                        st.write(f"- **{row['driver_name']}** *[{row['grand_prix']}]*: \"{row['message_text']}\"")
                    found_any = True
                    break
            if not found_any:
                regs = search_regulations(user_msg, k=2)
                if not regs.empty:
                    st.write("**F1 Bot:** Closest matching regulation clauses:")
                    for _, r_row in regs.iterrows():
                        st.write(f"- **{r_row['Year']} — {r_row['Category']}:** {r_row['Regulation Clause']}")
                    found_any = True
            if not found_any:
                st.warning("**F1 Bot:** I couldn't find a match. Try:")
                st.write("- 🏆 *'Who won Monaco?'* | 💥 *'Who retired at Barcelona?'*")
                st.write("- 📻 *'What did Verstappen say?'* | 🔧 *'Show me tire messages'*")
                st.write("- 🧠 *'What is TF-IDF?'* | 📜 *'2026 engine rules'* | 📊 *'Best model accuracy'*")

    st.markdown("---")
    st.markdown("### Trigram Radio Text Generation")
    st.markdown("Generate synthetic F1 radio messages using n-gram word sequences.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        seed1 = st.text_input("Seed Word 1:", "box")
    with col_s2:
        seed2 = st.text_input("Seed Word 2:", "box")
        
    if st.button("Generate Synthetic Message"):
        tokenized_corpus = [str(text).lower().split() for text in df["message_text"]]
        trigrams = {}
        for sentence in tokenized_corpus:
            if len(sentence) < 3:
                continue
            for i in range(len(sentence) - 2):
                key = (sentence[i], sentence[i+1])
                if key not in trigrams:
                    trigrams[key] = []
                trigrams[key].append(sentence[i+2])
                
        words = [seed1.lower(), seed2.lower()]
        for _ in range(12):
            k_key = (words[-2], words[-1])
            if k_key in trigrams:
                next_w = random.choice(trigrams[k_key])
                words.append(next_w)
            else:
                break
        st.write(f"Generated text: **\"{' '.join(words)}\"**")

    st.markdown("---")
    st.markdown("### 📚 Background Operations & Workings")
    show_chat_expl = st.toggle("🔍 Show Academic Explanations", value=True, key="toggle_chat_expl")
    if show_chat_expl:
        st.markdown("Learn how the Chatbot and Trigram Generator perform language synthesis:")
        tab_qa, tab_trigram = st.tabs([
            "💬 Intent & Information Retrieval Q&A",
            "🔗 Trigram Language Model"
        ])
        with tab_qa:
            st.markdown("#### 💬 Intent & Information Retrieval Q&A")
            st.markdown("""
            * **Keyword Intent Matching:** The chatbot acts as a deterministic expert system. It scans queries for intent trigger keys (e.g., `"built"`, `"author"`, `"regulations"`, `"Hamilton"`, `"out"`, `"retire"`) and dispatches them to specialized retrieval algorithms.
            * **Reference Database Lookups:** For inquiries on lineups (e.g. teammate queries) and technical specifications (e.g. top speed queries), the chatbot runs matching filters against the `df_teams` and `df_tech` datasets in memory.
            """)
        with tab_trigram:
            st.markdown("#### 🔗 Trigram Language Model")
            st.markdown("""
            * **Trigram Concept:** An $n$-gram language model is a type of probabilistic model that predicts the next word in a sequence based on the history of the previous $n-1$ words. For a trigram ($n=3$), the probability of word $w_i$ is conditioned on $w_{i-2}$ and $w_{i-1}$:
              $$P(w_i \\mid w_{i-2}, w_{i-1}) = \\frac{\\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\\text{Count}(w_{i-2}, w_{i-1})}$$
            * **In the Background:** The generator scans the entire corpus of 18,574 radio messages to count occurrences of word triplets. When you provide two seed words, the model looks up all candidate third words that followed that pair in the training data, randomly samples one of those candidates (simulating a probability distribution), slides the window forward, and repeats.
            """)

elif app_mode == "RAG Simulator":
    st.title("🧠 RAG (Retrieval-Augmented Generation) Simulator")
    st.markdown("Simulate a modern LLM system. The RAG pipeline performs semantic search on the F1 corpus (team radios, race reports, and official regulations) to retrieve relevant context and constructs a strategist prompt template.")
    
    user_query = st.text_input("Ask a question about the race state or rules (e.g. 'Why did Hamilton retire?', 'What are the 2026 engine rules?', 'Explain 2026 active aerodynamics'):", 
        "What are the 2026 engine rules?"
    )
    
    if st.button("Retrieve and Generate Prompt"):
        # 1. Retrieve team radios
        retrieved_radio = search_corpus(user_query, k=2)
        
        # 2. Retrieve race reports
        retrieved_reports = search_reports(user_query, k=2)
        
        # 3. Retrieve regulations
        retrieved_regs = search_regulations(user_query, k=2)
        
        # 4. Retrieve teams & drivers
        retrieved_teams = search_teams(user_query, k=2)
        
        # 5. Retrieve tech specs
        retrieved_tech = search_tech_specs(user_query, k=2)
        
        if (not retrieved_radio.empty) or (not retrieved_reports.empty) or (not retrieved_regs.empty) or (not retrieved_teams.empty) or (not retrieved_tech.empty):
            st.success("Successfully retrieved context from F1 database!")
            
            context_str = ""
            
            if not retrieved_radio.empty:
                st.markdown("#### Retrieved Driver Team Radios:")
                st.dataframe(retrieved_radio, use_container_width=True)
                context_str += "\n--- TEAM RADIO TRANSCRIPTS ---\n"
                for idx, row in retrieved_radio.iterrows():
                    context_str += f"[{idx+1}] Driver: {row['Driver']} | GP: {row['Grand Prix']} | Radio: \"{row['Message']}\"\n"
                    
            if not retrieved_reports.empty:
                st.markdown("#### Retrieved Official Race Report Chunks:")
                st.dataframe(retrieved_reports, use_container_width=True)
                context_str += "\n--- OFFICIAL RACE REPORTS ---\n"
                for idx, row in retrieved_reports.iterrows():
                    context_str += f"[{idx+1}] Grand Prix: {row['Grand Prix']} | Summary: \"{row['Report Sentence']}\"\n"
                    
            if not retrieved_regs.empty:
                st.markdown("#### Retrieved F1 Official Regulations:")
                st.dataframe(retrieved_regs, use_container_width=True)
                context_str += "\n--- F1 OFFICIAL REGULATIONS ---\n"
                for idx, row in retrieved_regs.iterrows():
                    context_str += f"[{idx+1}] Year: {row['Year']} | Category: {row['Category']} | Regulation: \"{row['Regulation Clause']}\"\n"

            if not retrieved_teams.empty:
                st.markdown("#### Retrieved F1 Teams & Drivers Database:")
                st.dataframe(retrieved_teams, use_container_width=True)
                context_str += "\n--- F1 TEAMS & DRIVERS DATABASE ---\n"
                for idx, row in retrieved_teams.iterrows():
                    context_str += f"[{idx+1}] Team: {row['Team']} | Driver: {row['Driver']} | Number: {row['Number']} | Nationality: {row['Nationality']} | Role: {row['Role']}\n"
                    
            if not retrieved_tech.empty:
                st.markdown("#### Retrieved F1 Technical Specifications:")
                st.dataframe(retrieved_tech, use_container_width=True)
                context_str += "\n--- F1 TECHNICAL SPECIFICATIONS ---\n"
                for idx, row in retrieved_tech.iterrows():
                    context_str += f"[{idx+1}] Year: {row['Year']} | Parameter: {row['Parameter']} | Value: {row['Value']} | Description: {row['Description']}\n"
            
            prompt_template = f"""System: You are an expert F1 Race Strategist AI. Answer the user's question using the context of driver team radio messages, official race reports, and F1 technical/sporting regulations and database records provided below.

Context:
{context_str}

User Question: {user_query}
Race Strategist Answer:"""
            
            st.markdown("### Generated RAG Prompt Template:")
            st.code(prompt_template, language="markdown")
            
            st.markdown("### Race Strategist AI Answer (Simulated):")
            
            # Formulate smart answers based on reports, regulations, specs, teams, or query
            if "engine" in user_query.lower() and "2026" in user_query.lower():
                st.info("Based on the 2026 regulations, the MGU-H (Motor Generator Unit - Heat) is completely removed. Power units will rely on a near 50-50 power split between internal combustion (down to 400kW) and electrical power (up to 350kW via MGU-K).")
            elif "aerodynamics" in user_query.lower() and "2026" in user_query.lower():
                st.info("Based on the 2026 regulations, active aerodynamics are introduced. The cars will switch between Z-mode (high downforce for cornering) and X-mode (low drag configuration for straight-line speed).")
            elif "fuel" in user_query.lower() and "2026" in user_query.lower():
                st.info("Based on the 2026 regulations, 100% sustainable synthetic e-fuels are mandated for all teams, supporting F1's net-zero carbon goals.")
            elif "hamilton" in user_query.lower() and "retire" in user_query.lower():
                st.info("Based on the official race reports, Lewis Hamilton retired on Lap 12 of the Barcelona Grand Prix due to a terminal Power Unit (engine) failure.")
            elif "sainz" in user_query.lower() and "tire" in user_query.lower():
                st.info("Based on the context, Carlos Sainz was managing tire compounds during the race, experiencing degradation on his current set.")
            elif "leclerc" in user_query.lower() and "retire" in user_query.lower():
                st.info("Based on the Canadian GP report, Charles Leclerc retired on Lap 40 after experiencing engine mapping issues after a wet tire change.")
            elif "monaco" in user_query.lower() and "happen" in user_query.lower():
                st.info("Based on the reports, Charles Leclerc won his home race in Monaco. A major collision on Lap 1 involving Sergio Perez and Nico Hulkenberg red-flagged the race.")
            else:
                if not retrieved_tech.empty:
                    p = retrieved_tech.iloc[0]
                    st.info(f"Based on the official F1 technical specifications: \"In {p['Year']}, the parameter '{p['Parameter']}' has a value of {p['Value']} ({p['Description']}).\"")
                elif not retrieved_teams.empty:
                    t = retrieved_teams.iloc[0]
                    st.info(f"Based on the F1 teams & drivers database: \"{t['Driver']} (No. {t['Number']}, {t['Nationality']}) drives for {t['Team']} as a {t['Role']}.\"")
                elif not retrieved_regs.empty:
                    fact = retrieved_regs.iloc[0]["Regulation Clause"]
                    st.info(f"Based on the official F1 regulations: \"{fact}\"")
                elif not retrieved_reports.empty:
                    fact = retrieved_reports.iloc[0]["Report Sentence"]
                    st.info(f"Based on the official race reports: \"{fact}\"")
                else:
                    st.info("Based on the retrieved team radios, the drivers are communicating about strategy adjustments, traffic gaps, or pit calls.")
        else:
            st.warning("No relevant context found in the database. Try asking another question!")

    st.markdown("---")
    st.markdown("### 📚 Background Operations & Workings")
    show_rag_expl = st.toggle("🔍 Show Academic Explanations", value=True, key="toggle_rag_expl")
    if show_rag_expl:
        st.markdown("Learn how Retrieval-Augmented Generation bridges raw databases and AI prompts:")
        tab_rag_arch, tab_prompt, tab_gen = st.tabs([
            "🧠 RAG Architecture",
            "📝 Prompt Engineering",
            "⚙️ Context Slicing"
        ])
        with tab_rag_arch:
            st.markdown("#### 🧠 RAG Architecture")
            st.markdown("""
            * **Why RAG?** Large Language Models (LLMs) suffer from hallucinations and are limited by their training cutoff dates. **Retrieval-Augmented Generation (RAG)** solves this by retrieving factual, up-to-date information from external databases and injecting it into the model's context window.
            * **In the Background:** When a query is made, the simulator queries five distinct F1 datasets. This acts as the *Retrieval* step.
            """)
        with tab_prompt:
            st.markdown("#### 📝 Prompt Engineering")
            st.markdown("""
            * **Prompt Templates:** A standardized template wraps the retrieved facts alongside a custom system persona instruction.
            * **In the Background:** The simulator structures this template to force the model to behave like an *F1 Race Strategist*, citing the references explicitly and ignoring outside knowledge.
            """)
        with tab_gen:
            st.markdown("#### ⚙️ Context Slicing")
            st.markdown("""
            * **Retrieval Slicing:** To avoid context overflow and noise, we restrict the search to top-$k$ ($k=2$) matches from each database.
            * **Context Aggregation:** Data frames are dynamically parsed into natural language key-value formats (e.g. `[1] Driver: Hamilton | GP: Barcelona | Radio: "..."`), forming a cohesive context block.
            """)


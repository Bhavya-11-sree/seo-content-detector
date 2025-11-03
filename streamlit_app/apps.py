import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease
import numpy as np
import matplotlib.pyplot as plt

# ✅ Download NLTK resources
nltk.download('punkt', quiet=True)

# ✅ Load Pretrained Model & Vectorizer
import os

BASE_DIR = os.path.dirname(__file__)  # current folder (streamlit_app)

model = joblib.load(os.path.join(BASE_DIR, "model", "quality_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))
dataset = pd.read_csv(os.path.join(BASE_DIR, "featured.csv"))
  # Contains previous URLs & text data

# ✅ Function to extract webpage content
def extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else "No Title"
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        word_count = len(text.split())
        sentence_count = text.count(".")
        readability = flesch_reading_ease(text)

        return title, text, word_count, sentence_count, readability
    except:
        return None, None, 0, 0, 0

# ✅ Function to generate Summary (without gensim)
def generate_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    summary_sentences = [sentences[i] for i in sorted(top_indices)]
    return " ".join(summary_sentences)

# ✅ Keyword Extraction
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray()[0]
    indices = np.argsort(scores)[::-1][:top_n]
    return [vectorizer.get_feature_names_out()[i] for i in indices]

# ========================= Streamlit App UI =========================

st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
st.title("🔍 SEO Content Quality & Duplicate Content Detector")

url = st.text_input("🔗 Enter a website URL to analyze:")

if st.button("🚀 Analyze"):
    if url:
        title, text, wc, sc, readability = extract_content(url)

        if text:
            input_features = pd.DataFrame([{
                "word_count": wc,
                "sentence_count": sc,
                "readability": readability
            }])

            if wc > 1200 and readability >= 45:
                quality_label = "✅ High Quality"
            elif wc > 600 and readability >= 30:
                quality_label = "🟡 Medium Quality"
            else:
                quality_label = "⚠️ Low Quality"
            st.subheader(f"Content Quality: {quality_label}")

            col1, col2, col3 = st.columns(3)
            col1.metric("📄 Word Count", wc)
            col2.metric("✍️ Sentences", sc)
            col3.metric("📚 Readability", round(readability, 2))

            st.subheader(f"Content Quality: {quality_label}")

            # ✅ Summary
            st.subheader("🧾 Summary of Page:")
            summary = generate_summary(text)
            st.write(summary)

            # ✅ Keywords
            st.subheader("🔑 Top Keywords:")
            keywords = extract_keywords(text)
            st.write(", ".join(keywords))

            with st.expander("📝 View Full Extracted Text"):
                st.write(text[:2000] + "...")

            # ✅ Duplicate Detection
            st.subheader("🔁 Duplicate Content Detection")
            dataset_vectors = vectorizer.transform(dataset['clean_text'].fillna(""))
            input_vec = vectorizer.transform([text])
            sims = cosine_similarity(input_vec, dataset_vectors)[0]
            dataset['similarity'] = sims

            similar_pages = dataset[dataset["similarity"] > 0.5].sort_values(by="similarity", ascending=False)

            if not similar_pages.empty:
                st.warning("⚠ Similar pages found!")
                st.table(similar_pages[['url', 'similarity']].head(5))

                # ✅ Bar Chart for Similarity
                st.subheader("📊 Similar Pages Similarity Chart")
                fig, ax = plt.subplots()
                ax.bar(similar_pages['url'].head(5), similar_pages['similarity'].head(5))
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.success("✅ No similar content found.")

    else:
        st.warning("⚠ Enter a valid URL")





import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease

# ✅ Load Pretrained Model & Vectorizer
model = joblib.load("C:/Users/bhavy/Downloads/seo-content-detector/streamlit_app/model/quality_model.pkl")          # Your saved ML model
vectorizer = joblib.load("C:/Users/bhavy/Downloads/seo-content-detector/streamlit_app/model/vectorizer.pkl")  # TF-IDF vectorizer
dataset = pd.read_csv("C:/Users/bhavy/Downloads/seo-content-detector/streamlit_app/quality_scored.csv")  # Contains previous URLs & text data

# ✅ Page Setup

st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
st.title("🔍 SEO Content Quality & Duplicate Content Detector")
st.write("Analyze any webpage for **SEO quality**, **readability**, and **duplicate content risk**.")

# ✅ Sidebar - Project Info
with st.sidebar:
    st.header("📊 Project Overview")
    st.markdown("""
    **Objective:**  
    Build an AI system to:
    - ✅ Extract webpage content  
    - ✅ Predict SEO content quality  
    - ✅ Detect duplicate/similar pages  
    """)
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Developer:** *Your Name*")
    st.markdown("---")

# ✅ Function: Extract content from URL
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

# ✅ Main Input Section
url = st.text_input("🔗 Enter a website URL to analyze:")

if st.button("🚀 Analyze"):
    if url:
        title, text, word_count, sentence_count, readability = extract_content(url)

        if text:
            # ✅ Prepare Data
            input_features = pd.DataFrame([{
                "word_count": word_count,
                "sentence_count": sentence_count,
                "readability": readability
            }])

            # ✅ Predict Quality
            prediction = model.predict(input_features)[0]
            quality_label = "✅ High Quality" if prediction == 1 else "⚠️ Low Quality"

            # ✅ Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("📄 Word Count", word_count)
            col2.metric("✍️ Sentences", sentence_count)
            col3.metric("📚 Readability", round(readability, 2))

            st.subheader(f"Content Quality: {quality_label}")

            # ✅ Show text in expandable box
            with st.expander("📝 View Extracted Text"):
                st.write(text[:2000] + "...")

            # ✅ Duplicate / Similarity Check
            st.subheader("🔁 Checking for Duplicate or Similar Content...")
            if "text" in dataset.columns:
                dataset_vectors = vectorizer.transform(dataset['text'].fillna(""))
                input_vector = vectorizer.transform([text])
                
                similarity_scores = cosine_similarity(input_vector, dataset_vectors)[0]
                dataset["similarity"] = similarity_scores
                similar_pages = dataset[dataset["similarity"] > 0.5].sort_values(by="similarity", ascending=False)

                if not similar_pages.empty:
                    st.warning("⚠ Similar content found!")
                    st.table(similar_pages[['url', 'similarity']].head(5))
                else:
                    st.success("✅ No similar content found in the dataset.")

            # ✅ Download Report
            if st.button("📥 Download Analysis Report"):
                report = pd.DataFrame([{
                    "url": url,
                    "title": title,
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "readability": readability,
                    "quality": quality_label
                }])
                report.to_csv("seo_report.csv", index=False)
                st.success("✅ Report saved as 'seo_report.csv'")

        else:
            st.error("❌ Failed to extract content.")
    else:
        st.warning("⚠ Please enter a URL first.")

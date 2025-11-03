import streamlit as st
import pandas as pd
from utils.parser import extract_content
from utils.features import extract_features
from utils.scorer import load_model, predict_quality, find_similar

# Load dataset & model
DATA_PATH = "C:/Users/bhavy/Downloads/seo-content-detector/streamlit_app/featured.csv"
MODEL_PATH = "C:/Users/bhavy/Downloads/seo-content-detector/streamlit_app/model/quality_model.pkl"


df = pd.read_csv(DATA_PATH)
model = load_model(MODEL_PATH)

st.title("🔍 SEO Content Quality & Duplicate Detector")

url = st.text_input("Enter a URL to analyze ⬇️")

if st.button("Analyze"):
    if not url:
        st.warning("⚠️ Please enter a valid URL.")
    else:
        with st.spinner("Extracting and analyzing content..."):
            content = extract_content(url)
            features = extract_features(content)
            result = predict_quality(features, model)

            result["similar_to"] = find_similar(features, df)

            st.success("✅ Analysis Complete!")
            st.json(result)

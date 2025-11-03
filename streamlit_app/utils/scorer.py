import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(path):
    return joblib.load(path)

def predict_quality(features, model):
    X = [[features["word_count"], features["sentence_count"], features["readability"]]]
    label = model.predict(X)[0]
    features["quality_label"] = label
    return features

def find_similar(features, df, threshold=0.8):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"].fillna(""))
    new_vec = vectorizer.transform([features["text"]])

    similarity_scores = cosine_similarity(new_vec, tfidf_matrix)[0]
    similar_items = df[similarity_scores > threshold]["url"].tolist()

    return [{"url": url, "similarity": float(score)} 
            for url, score in zip(similar_items, similarity_scores) if score > threshold]

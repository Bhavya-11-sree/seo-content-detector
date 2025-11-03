import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('punkt')

def extract_features(content):
    text = content["body"]
    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    readability = flesch_reading_ease(text)

    return {
        "url": content["url"],
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": readability,
        "text": text
    }

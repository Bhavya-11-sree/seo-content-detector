

# 🚀 SEO Content Quality & Duplicate Detector

A lightweight yet powerful tool that analyzes website content for SEO quality, readability, word structure, and duplicate/similar content. It helps bloggers, marketers, and developers ensure their content is original, readable, and high-quality.

---

## ✅ Features

- 📥 **Scrape or Load Web Content** (URL or HTML input)
- ✨ **Clean & Extract**: Titles, body text, word/sentence count
- 📊 **Readability Score** (Flesch Reading Ease)
- 🧠 **Quality Labels**: High / Medium / Low using custom logic or ML
- 🔍 **Duplicate Content Detection** (TF-IDF + Cosine Similarity)
- 😊 **(Optional)** Sentiment / Emotion Analysis
- 📁 **CSV Output** with all metrics and quality labels

---

## 📂 Project Structure



seo-content-detector/
│── notebooks/
│ └── seo_pipeline.ipynb
│── app.py # Streamlit app (optional)
│── data/
│ ├── input_urls.csv
│ └── output_results.csv
│── models/
│ └── quality_model.pkl
│── requirements.txt
│── README.md


---

## ⚙️ Setup Instructions


git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb

🚀 Quick Start

Add URLs in data/input_urls.csv

Run seo_pipeline.ipynb or app.py

Output file output_results.csv will include:

url	word_count	readability	quality_label
https://example.com
	2011	53.27	✅ High
🌐 Streamlit Deployment (Optional)
streamlit run app.py


If deployed online, include URL here:
🔗 Live Demo: https://seo-content-detector-6b8mypqfzxfcmsrfkcugrw.streamlit.app/

💡 Key Decisions

BeautifulSoup + lxml → Clean HTML parsing

TF-IDF + Cosine Similarity → Accurate duplicate detection

Rule-Based Quality System → Transparent and explainable

Flesch Reading Ease → Standard readability metric

RandomForest Model (optional) → High accuracy with feature importance insights

📈 Results Summary

✅ Quality Labeling Results:

High Quality: Clear structure + high readability

Medium Quality: Informative but slightly complex

Low Quality: Too short / overly complex / low readability

✅ Model Metrics:

Accuracy: 0.96  
F1-Score: 0.97  
Baseline Accuracy (word count only): 0.49  
Top Features: readability, word_count, sentence_count


✅ Sample Confusion Matrix:

High     → 3/3 correct
Medium   → 8/9 correct
Low      → 13/13 correct

⚠️ Limitations

❌ JS-heavy websites not fully supported

❌ Doesn’t analyze keywords, backlinks, or E-E-A-T yet

❌ Rule-based system might miss nuanced human tone

🔮 Future Enhancements

Add sentiment/emotion scoring

Extract meta tags + keyword density

API endpoint for automation

Full Streamlit/Flask deployment

🤝 Contributing

Pull requests, ideas, or feature suggestions are always welcome!

📜 License

MIT License – Free to use, modify, and distribute.

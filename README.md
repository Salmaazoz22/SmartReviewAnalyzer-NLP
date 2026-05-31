<div align="center">

# 🎬 SmartReviewAnalyzer-NLP

### An End-to-End NLP Sentiment Analysis Pipeline

*Classifying movie reviews as Positive or Negative — from raw text to fine-tuned Transformers*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BERT-FFD21E?style=for-the-badge)](https://huggingface.co/transformers/)

[![GitHub Stars](https://img.shields.io/github/stars/Salmaazoz22/SmartReviewAnalyzer-NLP?style=social)](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Salmaazoz22/SmartReviewAnalyzer-NLP?style=social)](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP/network/members)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [NLP & ML Pipeline](#-nlp--ml-pipeline)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Screenshots & Visualizations](#-screenshots--visualizations)
- [Future Improvements](#-future-improvements)
- [Team & Contributions](#-team--contributions)
- [License](#-license)

---

## 🔍 Overview

**SmartReviewAnalyzer-NLP** is a complete, end-to-end Natural Language Processing project that classifies movie reviews from the **IMDB Dataset** (50,000 reviews) as **Positive** or **Negative**. The project follows a structured, notebook-driven pipeline that progresses from raw text cleaning through classical machine learning to state-of-the-art transformer fine-tuning — providing a rigorous benchmarking study at each stage.

> 🏆 **Champion Model: BERT** — **89.37% accuracy** | **F1-Score: 0.8947**

**Key highlights:**
- 🔡 Multi-stage text preprocessing with NLTK
- 📐 Dual feature extraction: TF-IDF and Word2Vec embeddings
- 🏅 Three-tier modeling: Logistic Regression → LSTM → BERT
- 📊 Rich visualizations: word clouds, confusion matrices, training curves, model comparison

---

## 🚩 Problem Statement

User-generated reviews contain valuable insights, but their volume and unstructured nature make manual analysis impractical at scale. Automated sentiment analysis enables businesses and researchers to extract actionable intelligence from thousands of reviews efficiently. This project builds and rigorously compares multiple approaches — from interpretable classical ML to deep bidirectional transformers — to accurately determine the sentiment polarity of any given review.

---

## ✨ Features

- ✅ Complete data cleaning pipeline (HTML removal, lowercasing, punctuation stripping, deduplication)
- ✅ Advanced text preprocessing (tokenization, stopword removal, lemmatization, normalization)
- ✅ TF-IDF vectorization (5,000 features, sparse matrix: 49,582 × 5,000)
- ✅ Word2Vec embeddings (100-dim, vocabulary: 64,530 tokens, L2-normalized sentence vectors)
- ✅ Baseline model: Logistic Regression — **88.00% accuracy**
- ✅ Deep learning model: Bidirectional LSTM (Keras/TensorFlow) — **85.41% accuracy**
- ✅ Transformer model: BERT fine-tuned (`bert-base-uncased`) — **89.37% accuracy**
- ✅ Comprehensive evaluation suite with confusion matrices and per-class metrics
- ✅ Modular, reproducible notebooks — one notebook per pipeline stage
- ✅ Saved model artifacts (`.pkl`, `.h5`) for downstream inference

---

## 🛠️ Tech Stack

| Layer | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Handling** | pandas, numpy |
| **NLP / Text Processing** | NLTK (punkt, stopwords, wordnet, WordNetLemmatizer), re |
| **Feature Extraction** | scikit-learn (TfidfVectorizer), Gensim (Word2Vec) |
| **Baseline ML** | scikit-learn (LogisticRegression, train_test_split, metrics) |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Transformers** | Hugging Face Transformers (BERT `bert-base-uncased`) |
| **Visualization** | matplotlib, seaborn, WordCloud |
| **Serialization** | pickle |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 🧠 NLP & ML Pipeline

The project is organized as a sequential, 6-stage notebook pipeline:

```
Raw IMDB Reviews (50,000 samples)
           │
           ▼
┌──────────────────────────────────┐
│  Stage 1 · Data Cleaning         │  01_data_cleaning.ipynb
│  ─ Lowercase & strip HTML tags   │
│  ─ Remove punctuation & numbers  │
│  ─ NLTK tokenization             │
│  ─ Stopword removal              │
│  ─ WordNet lemmatization         │
│  ─ Drop nulls & duplicates       │
│  → Output: cleaned_reviews.csv   │
└──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Stage 2 · Text Preprocessing    │  02_Text_Preprocessing.ipynb
│  ─ Re-tokenize cleaned text      │
│  ─ Normalize (keep alpha only)   │
│  ─ Filter short words (< 3 ch.)  │
│  ─ Custom extended stopwords     │
│  ─ Verb-form lemmatization       │
│  ─ Reconstruct processed_text    │
│  → Output: processed_text.csv    │
└──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Stage 3 · Feature Extraction    │  03_feature_extraction_tfidf_word2vec.ipynb
│  ─ TF-IDF: max_features=5,000    │
│    matrix shape: (49,582 × 5,000)│
│  ─ Word2Vec: vector_size=100,    │
│    window=5, min_count=2         │
│    vocab: 64,530 tokens          │
│    sentence embeddings: L2-norm  │
│  → tfidf_vectors.pkl             │
│  → word2vec_vectors.pkl          │
└──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Stage 4 · Baseline Model        │  04_Baseline_Model.ipynb
│  ─ Logistic Regression (TF-IDF)  │
│  ─ 80 / 20 stratified split      │
│  ─ Accuracy:  88.00%             │
│  ─ F1-Score:  0.8800             │
│  → Output: baseline_model.pkl    │
└──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Stage 5 · Advanced Models       │  05_Advanced_Model.ipynb
│  ─ LSTM (Keras / TensorFlow)     │
│    Accuracy: 85.41%  F1: 0.8502  │
│  ─ BERT fine-tuning              │
│    (bert-base-uncased)           │
│    Accuracy: 89.37%  F1: 0.8947  │
│  → advanced_model.h5             │
│  → lstm_training_history.pkl     │
│  → bert_training_history.pkl     │
└──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Stage 6 · Evaluation            │  06_Evaluation.ipynb
│  ─ Full metrics comparison       │
│  ─ Confusion matrices            │
│  ─ Training curves               │
│  ─ Side-by-side model comparison │
└──────────────────────────────────┘
```

---

## 🏗️ System Architecture

```
SmartReviewAnalyzer-NLP/
│
├── 📁 notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_Text _Preprocessing.ipynb
│   ├── 03_feature_extraction_tfidf_word2vec.ipynb
│   ├── 04_Baseline_Model.ipynb
│   ├── 05_Advanced_Model.ipynb
│   └── 06_Evaluation.ipynb
│
├── 📁 models/
│   ├── baseline_model.pkl            # Logistic Regression
│   ├── advanced_model.h5             # LSTM (Keras)
│   ├── lstm_training_history.pkl
│   └── bert_training_history.pkl
│
├── 📁 data/
│   └── sample_features.csv
│
├── 📁 outputs/
│   ├── Insights/
│   └── Tables/
│
├── 📁 Plots & Visualization/
│   ├── sentiment_distribution.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── The Confusion Matrix for The baseline Model.png
│   ├── lstm_confusion_matrix.png
│   ├── lstm_training_curves.png
│   ├── bert_confusion_matrix.png
│   ├── bert_training_curves.png
│   └── model_comparison.png
│
├── Smart Review Analyzer - NLP Presentation.pdf
└── README.md
```

---

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.8 or higher.

**1. Clone the repository**

```bash
git clone https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP.git
cd SmartReviewAnalyzer-NLP
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install all dependencies**

```bash
pip install pandas numpy nltk scikit-learn gensim tensorflow keras \
            transformers matplotlib seaborn wordcloud
```

**4. Download required NLTK resources**

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

**5. Download the dataset**

Download the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`IMDB Dataset.csv`) and place it in the project root directory.

---

## 🚀 How to Run

Run the notebooks **in order** for the full pipeline:

| Step | Notebook | Input | Output |
|:----:|----------|-------|--------|
| 1 | `01_data_cleaning.ipynb` | `IMDB Dataset.csv` | `cleaned_reviews.csv` |
| 2 | `02_Text _Preprocessing.ipynb` | `cleaned_reviews.csv` | `processed_text.csv` |
| 3 | `03_feature_extraction_tfidf_word2vec.ipynb` | `processed_text.csv` | `tfidf_vectors.pkl`, `word2vec_vectors.pkl` |
| 4 | `04_Baseline_Model.ipynb` | `tfidf_vectors.pkl` | `baseline_model.pkl` |
| 5 | `05_Advanced_Model.ipynb` | `processed_text.csv` | `advanced_model.h5`, training histories |
| 6 | `06_Evaluation.ipynb` | All models & data | Charts, metrics, comparison tables |

**Launch Jupyter:**

```bash
jupyter notebook notebooks/
```

> 💡 All notebooks were developed and tested on **Google Colab**. You can open them directly in Colab by uploading the notebook files and mounting your Drive for data access.

---

## 📊 Results

> Performance evaluated on a held-out test set (20% stratified split). **BERT is the champion model.**

<div align="center">

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression (TF-IDF) | 0.8800 | 0.88 | 0.88 | 0.8800 |
| LSTM | 0.8541 | 0.877 | 0.825 | 0.8502 |
| **BERT** 🏆 | **0.8937** | **0.8897** | **0.8997** | **0.8947** |

</div>

**Key Takeaways:**

- 🏆 **BERT** outperforms all models across every metric by leveraging deep bidirectional contextual understanding of the review text.
- ⚡ **Logistic Regression** delivers surprisingly strong, balanced results as a fast and interpretable baseline — a testament to the power of well-tuned TF-IDF features.
- 📉 **LSTM** performs competitively on precision but shows lower recall, suggesting difficulty generalizing edge-case negative patterns without full contextual attention.

---

## 🎨 Screenshots & Visualizations

### Sentiment Distribution
![Sentiment Distribution](Plots%20%26%20Visualization/sentiment_distribution.png)

### Word Clouds
| Positive Reviews | Negative Reviews |
|:-:|:-:|
| ![Positive WordCloud](Plots%20%26%20Visualization/wordcloud_positive.png) | ![Negative WordCloud](Plots%20%26%20Visualization/wordcloud_negative.png) |

### Confusion Matrices
| Logistic Regression | LSTM | BERT |
|:-:|:-:|:-:|
| ![Baseline CM](Plots%20%26%20Visualization/The%20Confusion%20Matrix%20for%20The%20baseline%20Model.png) | ![LSTM CM](Plots%20%26%20Visualization/lstm_confusion_matrix.png) | ![BERT CM](Plots%20%26%20Visualization/bert_confusion_matrix.png) |

### Training Curves
| LSTM | BERT |
|:-:|:-:|
| ![LSTM Curves](Plots%20%26%20Visualization/lstm_training_curves.png) | ![BERT Curves](Plots%20%26%20Visualization/bert_training_curves.png) |

### Model Comparison
![Model Comparison](Plots%20%26%20Visualization/model_comparison.png)

---

## 🔮 Future Improvements

- [ ] Deploy as a web application (FastAPI backend + Streamlit UI)
- [ ] Extend to multi-class sentiment (e.g., 1–5 star ratings)
- [ ] Add aspect-based sentiment analysis (ABSA)
- [ ] Experiment with RoBERTa and DistilBERT for improved efficiency
- [ ] Support for multilingual reviews using multilingual BERT
- [ ] Add model explainability using LIME or SHAP
- [ ] Build a real-time review analysis REST API

---

## 👥 Team & Contributions

This project was built collaboratively as part of an NLP course at the **Faculty of Computers and Information, Helwan University**.

<div align="center">

| Member | Role | Contributions |
|:-------|:----:|:-------------|
| 👑 **Salma Mohamed** | *Team Lead* | Project management, TF-IDF & Word2Vec feature extraction, README |
| **Malak Tarek** | *Data Engineer* | Dataset sourcing, curation & initial cleaning |
| **Sameh Naeem** | *NLP Engineer* | Advanced text preprocessing & normalization pipeline |
| **Mohammed Saied** | *ML Engineer* | Baseline model development (Logistic Regression) |
| **Rawan Essam** | *Deep Learning Engineer* | LSTM and BERT architecture, training & evaluation |
| **Ahmed Khaled** | *Analyst* | Visualization suite & metric analysis |

</div>

---

## 📄 License

This project is intended for academic and educational purposes.

---

<div align="center">

**Built with ❤️ by the Smart Review Analyzer Team**

*If you found this project helpful, please consider giving it a ⭐ on [GitHub](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP)*

> 📎 See `Smart Review Analyzer - NLP Presentation.pdf` for the full project presentation.

</div>

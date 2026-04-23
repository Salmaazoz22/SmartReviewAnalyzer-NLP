<div align="center">

# 🧠 Smart Review Analyzer

### A Complete NLP Sentiment Analysis Pipeline

*Classifying user reviews into Positive / Negative with deep visual insights*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BERT-FFD21E?style=for-the-badge)](https://huggingface.co/transformers/)

[![GitHub Stars](https://img.shields.io/github/stars/Salmaazoz22/SmartReviewAnalyzer-NLP?style=social)](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Salmaazoz22/SmartReviewAnalyzer-NLP?style=social)](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP/network/members)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Results](#-results)
- [Visual Insights](#-visual-insights)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Team & Contributions](#-team--contributions)

---

## 🔍 Overview

**Smart Review Analyzer** is an end-to-end Natural Language Processing pipeline that classifies user-generated reviews as **Positive** or **Negative**. The project progresses through three tiers of modeling complexity — from a classical Baseline to a State-of-the-Art Transformer — providing a thorough benchmarking study alongside rich visual analytics.

> **Champion Model: BERT** — achieving **89.37% accuracy** and an **F1-Score of 0.8947**

**Key highlights:**
- 🔡 Robust multi-stage text preprocessing
- 📐 Dual feature extraction strategies (TF-IDF & Word2Vec)
- 🏆 Three-tier modeling: Logistic Regression → LSTM → BERT
- 📊 Comprehensive visualizations including Word Clouds, Distribution Charts, and Confusion Matrices

---

## ⚙️ Pipeline Architecture

### 1 · Text Preprocessing

Raw review text is normalized through a standardized cleaning pipeline before any modeling takes place:

| Step | Description |
|------|-------------|
| **Lowercasing** | Uniform case normalization across all tokens |
| **Punctuation Removal** | Strip non-alphabetic characters and special symbols |
| **Stopword Removal** | Eliminate high-frequency, low-information words via NLTK |
| **Tokenization** | Split sentences into discrete word tokens |
| **Lemmatization** | Reduce tokens to their canonical dictionary form |

---

### 2 · Feature Extraction

Two complementary representation strategies are implemented:

- **TF-IDF** — Captures term importance relative to the corpus, used as input for the Baseline model.
- **Word2Vec** — Dense semantic embeddings that encode contextual word relationships, feeding into deep learning models.

---

### 3 · Modeling Approach

A three-tier progression from classical ML to cutting-edge NLP:

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1 — Baseline    │  Logistic Regression + TF-IDF   │
│  Tier 2 — Intermediate│  LSTM + Word2Vec Embeddings      │
│  Tier 3 — SOTA 🏆     │  BERT (bert-base-uncased)        │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results

> The table below summarizes performance metrics evaluated on the held-out test set. **BERT is our champion model.**

<div align="center">

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.8800 | 0.88 | 0.88 | 0.8800 |
| LSTM | 0.8541 | 0.877 | 0.825 | 0.8502 |
| **BERT** 🏆 | **0.8937** | **0.8897** | **0.8997** | **0.8947** |

</div>

**Key Takeaways:**

- ✅ **BERT** outperforms all models across every metric, leveraging deep bidirectional context understanding.
- 📉 **LSTM** performs competitively on precision but sacrifices recall, indicating difficulty capturing edge-case negative patterns.
- ⚡ **Logistic Regression** delivers surprisingly strong and balanced results as a fast, interpretable baseline.

---

## 🎨 Visual Insights

The pipeline generates the following diagnostic visualizations automatically:

| Visualization | Purpose |
|---------------|---------|
| ☁️ **Word Clouds** | Surface the most frequent and influential tokens per sentiment class |
| 📊 **Bar Charts** | Examine class distribution and label balance across the dataset |
| 🟦 **Confusion Matrices** | Per-model breakdown of True/False Positives and Negatives |

All plots are saved to the `outputs/visualizations/` directory upon pipeline execution.

---

## 🗂️ Repository Structure

```
SmartReviewAnalyzer-NLP/
│
├── 📁 data/
│   ├── raw/                    # Original, unprocessed review datasets
│   └── processed/              # Cleaned and tokenized data ready for modeling
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis & class distribution
│   ├── 02_preprocessing.ipynb  # Full preprocessing pipeline walkthrough
│   ├── 03_feature_extraction.ipynb  # TF-IDF & Word2Vec feature generation
│   ├── 04_baseline_model.ipynb # Logistic Regression training & evaluation
│   ├── 05_lstm_model.ipynb     # LSTM architecture, training & evaluation
│   └── 06_bert_model.ipynb     # BERT fine-tuning, evaluation & analysis
│
├── 📁 src/
│   ├── preprocessing.py        # Text cleaning and normalization functions
│   ├── feature_extraction.py   # TF-IDF and Word2Vec pipelines
│   ├── models/
│   │   ├── baseline.py         # Logistic Regression model
│   │   ├── lstm.py             # LSTM model definition and training loop
│   │   └── bert.py             # BERT fine-tuning wrapper
│   └── visualizations.py       # Word Cloud, bar chart, confusion matrix utils
│
├── 📁 outputs/
│   ├── models/                 # Saved model weights and checkpoints
│   └── visualizations/         # Generated plots and figures
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

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
pip install -r requirements.txt
```

**4. Download required NLTK resources**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 🚀 How to Run

### Option A — Run via Jupyter Notebooks *(Recommended)*

Execute the notebooks in order for a step-by-step walkthrough:

```bash
jupyter notebook notebooks/
```

Open and run notebooks `01` through `06` sequentially.

### Option B — Run the full pipeline via script

```bash
# Step 1: Preprocess raw data
python src/preprocessing.py --input data/raw/ --output data/processed/

# Step 2: Generate features
python src/feature_extraction.py --input data/processed/

# Step 3: Train and evaluate all models
python src/models/baseline.py
python src/models/lstm.py
python src/models/bert.py

# Step 4: Generate all visualizations
python src/visualizations.py
```

All results, metrics, and plots will be saved under `outputs/`.

---

## 👥 Team & Contributions

This project was built collaboratively as part of an NLP course. Each team member owned a critical segment of the pipeline:

<div align="center">

| Member | Role | Contributions |
|:-------|:----:|:-------------|
| 👑 **Salma** | *Team Lead* | Project management, roadmap planning, TF-IDF & Word2Vec feature extraction |
| **Malak Tarek** | *Data Engineer* | Data sourcing, dataset curation & initial cleaning |
| **Sameh Naeem** | *NLP Engineer* | Advanced text preprocessing, normalization pipeline |
| **Mohammed Saied** | *ML Engineer* | Baseline model development (Logistic Regression) |
| **Rawan Essam** | *Deep Learning Engineer* | LSTM architecture design & implementation |
| **Ahmed Khaled** | *BERT Engineer & Analyst* | BERT fine-tuning, visualization suite & metric analysis |

</div>

---

## 📄 License

This project is intended for academic and educational purposes.

---

<div align="center">

**Built with ❤️ by the Smart Review Analyzer Team**

*If you found this project helpful, please consider giving it a ⭐ on [GitHub](https://github.com/Salmaazoz22/SmartReviewAnalyzer-NLP)*

</div>

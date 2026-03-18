# 📚 Kindle Sentiment Analysis

A natural-language processing project focused on analysing user reviews of Kindle e-readers to determine sentiment (positive/negative) and extract key themes driving user satisfaction or dissatisfaction.

---

## 📌 Project Overview

This project builds a full NLP pipeline: loading Kindle review data (e.g., from Amazon or Kaggle), cleaning and preprocessing text, performing exploratory text analysis, modelling sentiment classification (using algorithms like Logistic Regression, SVM, or neural nets), and extracting insights about what matters most to Kindle users. The goal is to inform product teams or marketers about user sentiment trends and product improvement areas.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, NLTK/spacy for NLP
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection & Pre-processing

* Load review dataset with fields such as review text, rating, helpfulness, date, verified purchase flag.
* Clean text: remove HTML tags, punctuation, stop-words, lemmatisation/stemming as appropriate.
* Optionally derive target label by summarising review rating (e.g., rating ≥ 4 ⇒ positive, else negative).
* Handle missing values and outliers (e.g., reviews with very few words).

### 2. Exploratory Text Analysis

* Visualise distribution of sentiments (positive vs negative).
* Word-clouds or bar graphs of most common words by sentiment class.
* N-gram analysis (bigrams/trigrams) for positive vs negative reviews.
* Analyse rating vs review length, helpfulness count, time trends.

### 3. Feature Engineering

* Convert text to numeric features: TF-IDF vectors, word embeddings (Word2Vec/Glove), or averaged embeddings.
* Add additional features: review length, sentiment polarity scores (via TextBlob or VADER), rating, helpfulness count.
* Split data into training and test sets (e.g., 80/20).

### 4. Modelling

* Baseline model: Logistic Regression with TF-IDF features.
* More advanced models: SVM, Random Forest, or Neural Network (e.g., simple dense network or LSTM).
* Tune hyper-parameters (regularisation strength, ensemble size, learning rate) via cross-validation.

### 5. Evaluation

* Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
* Possibly ROC-AUC if treated as binary classification.
* Review model performance: where it misclassifies, key feature/corpus patterns.

### 6. Insights & Application

* Analyse top keywords and themes driving negative sentiment (e.g., battery, screen, software updates).
* Provide actionable recommendations for product improvement or marketing messaging.
* Optionally build a simple UI or dashboard where a user can input a review and get sentiment prediction.

---



---

## 📈 Key Findings

* Positive reviews were heavily characterised by words relating to “battery life”, “readable display”, “lightweight”, while negative reviews often mentioned “software glitch”, “battery drain”, “customer service”.
* Including review length and helpfulness-count improved model accuracy by adding meta-context beyond text-only features.
* Logistic Regression performed reasonably well; adding SVM or ensemble boosted performance further, especially in recall of negative class.
* Pre-processing (lemmatisation, n-grams) significantly improved classification performance compared to raw text.

---

## 🚀 Future Improvements

* Expand to multi-class sentiment (e.g., negative, neutral, positive) for finer granularity.
* Use deep-learning models (e.g., LSTM, BERT) for more nuanced text understanding and better classification accuracy.
* Deploy as a web app where users or product teams can input any Kindle review and get sentiment + theme extraction.
* Add explainability: show which words or phrases contributed most to a given sentiment classification (e.g., via LIME or SHAP).
* Monitor model drift as review language evolves (new features, updates) and retrain periodically.

---

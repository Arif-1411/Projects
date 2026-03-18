# ✉️ Spam vs Ham Classification using BOW & TF-IDF

A natural-language-processing project focused on classifying SMS/email messages into spam (unwanted) or ham (legitimate) using Bag-of-Words and TF-IDF feature techniques.

---

## 📌 Project Overview

This project implements a full pipeline: collecting a message dataset labelled spam/ham, preprocessing text, engineering features using Bag-of-Words (BOW) and TF-IDF vectorisation, training classification models (e.g., Logistic Regression, Naive Bayes), evaluating performance, and delivering insights about how text features differentiate spam from ham. The goal is to build a model that accurately flags unwanted messages and understand key linguistic indicators of spam.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, NLTK or spaCy for NLP preprocessing
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection & Pre-processing

* Load labelled dataset of messages (SMS or email) with target column “spam” or “ham”.
* Clean text: convert to lowercase, remove punctuation/stop-words, optionally lemmatise or stem.
* Split dataset into training and test sets (e.g., 80/20) with stratification on target.

### 2. Feature Engineering: BOW & TF-IDF

* Vectorise text using:

  * **Bag-of-Words (CountVectorizer)**
  * **TF-IDF (TfidfVectorizer)**
* Optionally limit vocabulary size (e.g., top 5000 words) and trim rare words
* Stack or compare both feature representations for modelling

### 3. Modelling

* Build baseline classification models:

  * **Logistic Regression**
  * **Naive Bayes (MultinomialNB)**
* Optionally explore tree-based methods (Random Forest) or ensemble
* Train models on BOW and TF-IDF features; compare performance

### 4. Evaluation

* Metrics used: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
* Pay special attention to recall on spam class (missing spam is costly)
* Compare how BOW vs TF-IDF features affect performance

### 5. Insights & Application

* Identify which words/features strongly signal spam (e.g., “free”, “win”, “click”) versus ham features.
* Provide recommendations: filtering rules, alerting system, UI for message classification.
* Demonstrate how feature engineering and representation techniques impact classification quality.

---



## 📈 Key Findings

* TF-IDF features delivered slightly stronger performance than raw BOW in capturing discriminative terms and reducing noise.
* Words such as **“free”**, **“win”**, **“offer”**, **“now”** were among top signals for spam class; while common ham messages contained more personal/context words.
* Logistic Regression and Naive Bayes performed well given the relatively straightforward feature space and task; tree-based models had limited incremental benefit.
* Proper preprocessing (removal of stop-words, lemmatisation) improved classification stability across BOW and TF-IDF feature sets.

---

## 🚀 Future Improvements

* Move from BOW/TF-IDF to more advanced text representations (e.g., word embeddings, transformer-based features) to capture context better.
* Expand to multi-language spam detection and more diverse message formats (email, social-media DMs, chat).
* Deploy model via web or mobile app: input message → classification result + confidence score.
* Integrate model into messaging system for real-time spam filtering and user-feedback loop.
* Add explainability (LIME/SHAP) to show users why a message was flagged as spam.

---

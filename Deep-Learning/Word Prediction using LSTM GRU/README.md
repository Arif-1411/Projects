# 📝 Word Prediction using LSTM & GRU

A deep-learning project focused on next-word and sequence prediction using recurrent neural networks (LSTM and GRU) built on text data.

---

## 📌 Project Overview

This project implements a full pipeline: text data ingestion and preprocessing, tokenization, sequence generation, model building with LSTM and GRU layers, training and evaluation, and then deployment or inference of predicted next words. The goal is to enable a model to continue text sequences and support applications like autocomplete, writing assistance or chatbot inputs.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, TensorFlow/Keras or PyTorch, nltk/spacy (for tokenization)
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection & Pre-processing

* Load a text dataset (e.g., public domain books, movie scripts, corpus of articles)
* Clean text: lowercasing, removing punctuation/noise, tokenization
* Convert text into sequences for sequence-to-one or sequence-to-sequence modelling
* Encode words (tokenizer or vocabulary mapping) and pad/truncate sequences

### 2. Sequence Generation & Feature Engineering

* From tokenized text create inputs like:

  * `X = [w1, w2, …, wn-1]`
  * `y = wn` (next word)
* Create sequences of fixed length (e.g., length = 10)
* Split into training and validation sets

### 3. Modeling

* Construct RNN models with layers like:

  * Embedding layer for input word indices
  * LSTM or GRU layer (or stacked) with dropout/regularization
  * Dense layer with softmax activation for vocabulary size prediction

  ```python
  model = Sequential()
  model.add(Embedding(vocab_size, embed_dim, input_length=seq_length))
  model.add(GRU(units=128, return_sequences=True))
  model.add(LSTM(units=128))
  model.add(Dense(vocab_size, activation='softmax'))
  ```
* Compile with `categorical_crossentropy` and optimiser like `Adam`
* Train with epochs, monitor validation loss

### 4. Evaluation & Inference

* Evaluate on validation set: accuracy of predicted next word, perplexity maybe
* Generate text: given seed text, predict next word, append, repeat to form sequence
* Review qualitative output for coherence and fluency

---

## 📁 Project Structure

```
Word-Prediction-LSTM-GRU/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* LSTM and GRU effectively learn sequence dependencies in text for next-word prediction
* Larger sequence lengths and richer embedding size improved model predictions, up to a point
* Overfitting was common on small corpora — dropout and regularisation helped
* Seed length and vocabulary size impacted prediction quality significantly

---

## 🚀 Future Improvements

* Move to transformer-based architectures (e.g., GPT-style) for higher prediction quality
* Expand dataset and domain (multilingual text, code) for broader model generalisation
* Deploy model as autocomplete web interface or API
* Monitor model drift and update with new text data periodically
* Introduce beam search or temperature sampling for more creative text generation

---

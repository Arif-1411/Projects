# 🎙️ Speech Emotion Recognition

A deep-learning project focused on classifying human emotions from spoken audio signals using audio feature extraction and neural network models.

---

## 📌 Project Overview

This project implements an end-to-end pipeline: loading audio data labelled with emotions, preprocessing and feature engineering (e.g., MFCCs, spectrograms), training and evaluating a neural network for emotion classification, and extracting insights that can be used in call-centres, human-computer interaction, or wellbeing monitoring systems.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** librosa (audio processing), numpy, pandas, matplotlib, seaborn, TensorFlow/Keras or PyTorch
* **Environment:** Jupyter Notebook / Google Colab
* **Techniques:** Audio feature extraction, deep neural network (CNN/RNN) architecture

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset comprising audio recordings of speech labeled with emotions such as neutral, calm, happy, sad, angry, fearful, disgusted, surprised.

### 2. Preprocessing & Feature Engineering

* Load audio clips and ensure consistent sampling rate.
* Extract features such as MFCCs (Mel-Frequency Cepstral Coefficients), chroma, mel-spectrograms, contrast, tonnetz.
* Optionally pad or truncate clips to fixed length, normalise.
* Combine features into structured dataframe: each example with feature vector + emotion label.
* Split dataset into training and validation/test sets.

### 3. Model Architecture & Training

* Baseline model: simple dense neural network on aggregated features.
* More advanced: CNN on spectrogram images or RNN on time-series features.
* Compile model (e.g., `categorical_crossentropy` loss, `accuracy` metric).
* Train model for multiple epochs with validation monitoring, early stopping as needed.

### 4. Evaluation & Inference

* Compute metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
* Visualise training/validation loss and accuracy curves.
* Test model on unseen audio samples and display predicted emotion.

---


## 📈 Key Findings

* Acoustic features like **MFCCs** and **mel-spectrograms** were strong predictors of emotional categories.
* Deep models (e.g., CNN/RNN) outperformed shallow models due to their ability to capture temporal and spectral patterns in audio.
* The model’s performance was better for clearly distinctive emotions (e.g., happy vs sad) and struggled more with subtle ones (e.g., calm vs neutral).
* Augmentation (noise, pitch shift) and balanced class sampling improved generalisation.

---

## 🚀 Future Improvements

* Expand dataset to multi-language or more natural conversational audio to improve real-world robustness.
* Use transformer-based audio models or multimodal fusion (speech + text transcript) for higher accuracy.
* Deploy the model as a web or mobile app where a user can record audio and get emotion feedback in real-time.
* Integrate interpretability (e.g., saliency maps over spectrograms) so users understand what parts of audio signal triggered the prediction.
* Monitor and mitigate bias: ensure model performs well across speaker demographics (age, gender, accent).

---

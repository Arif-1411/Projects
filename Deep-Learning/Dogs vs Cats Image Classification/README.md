# 🐶 Cats vs Dogs Image Classification

A deep-learning project focused on classifying images into cat or dog categories using convolutional neural networks.

---

## 📌 Project Overview

This project builds an end-to-end pipeline: loading image data, preprocessing (resizing, normalisation, augmentation), building and training a CNN model, and evaluating its performance. The goal is to accurately distinguish between cat and dog images and understand the key features learned by the model.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow/Keras or PyTorch, numpy, matplotlib, seaborn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection & Pre-processing

* Use the “Dogs vs Cats” image dataset (e.g., Kaggle’s dataset) containing labelled dog and cat images.
* Pre-process images: resize to a fixed size (e.g., 150 × 150 or 224 × 224), normalise pixel values (e.g., [0,1]).
* Perform data augmentation: random flips, rotations, zooms to improve generalisation.

### 2. Feature Engineering & Model Preparation

* Set up image data generators or custom PyTorch dataset with augmentation.
* Define CNN architecture: convolutional layers → pooling → dropout → flatten → dense → output layer with sigmoid activation for binary classification.
* Compile model with `binary_crossentropy`, optimizer like `Adam`, and metric `accuracy`.

### 3. Training & Validation

* Train model over multiple epochs, monitor training and validation accuracy and loss.
* Use callbacks (e.g., EarlyStopping or ModelCheckpoint) to avoid overfitting.
* Visualise training/validation loss and accuracy curves.

### 4. Evaluation & Prediction

* Evaluate model on reserved test set: accuracy, confusion matrix, ROC curve, precision/recall.
* Test on new images to predict ‘cat’ vs ‘dog’ and display image with predicted label.

---

## 📁 Project Structure

```
Dogs-vs-Cats-Image-Classification/
│── data/
│   ├── train/
│   ├── validation/
│   └── test/
│── notebooks/
│   └── image_classification.ipynb
│── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Data augmentation significantly reduced overfitting and improved validation accuracy.
* Transfer-learning using a pretrained backbone (e.g., MobileNet, ResNet) often boosted performance versus training from scratch.
* The model achieved high accuracy on the binary classification task, with most errors occurring on ambiguous images (blurry, small pets).
* Visualisation of activation maps revealed that the model focuses on pet fur texture, ear shape, and face orientation.

---

## 🚀 Future Improvements

* Expand to multi-class pet classification (e.g., cat, dog, rabbit, bird) to generalise further.
* Use higher resolution images (e.g., 224×224 or 299×299) and deeper architectures (e.g., EfficientNet, DenseNet) for improved accuracy.
* Deploy as a web app or mobile app where users upload a photo and receive pet-type prediction.
* Incorporate explainability tools (e.g., Grad-CAM) so users see which part of the image influenced the classification.
* Create a production-pipeline for inference (image upload endpoint, preprocessing, prediction, result API) and monitor model latency/accuracy in deployment.

---

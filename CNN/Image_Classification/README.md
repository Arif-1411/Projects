# Image Classification using CNN & Transfer Learning

A deep learning project that builds and compares two image classifiers on the CIFAR-10 dataset — a custom Convolutional Neural Network trained from scratch, and EfficientNetB0 fine-tuned using transfer learning.

---

## Overview

| | |
|---|---|
| Dataset | CIFAR-10 — 60,000 color images across 10 classes |
| Image size | 32 x 32 x 3 (RGB) |
| Train / Val / Test | 40,000 / 10,000 / 10,000 |
| Models | Custom CNN · EfficientNetB0 (Transfer Learning) |
| Framework | TensorFlow / Keras |

---

## Classes

```
Airplane   Automobile   Bird   Cat   Deer
Dog        Frog         Horse  Ship  Truck
```

---

## Project Structure

```
image_classification_cnn.py   main script
```

---

## Pipeline

```
Load Data
    |
Normalize + One-Hot Encode
    |
Train / Val Split  (80 / 20)
    |
Data Augmentation
    |
    +------------------+------------------+
    |                                     |
Custom CNN                     EfficientNetB0
(trained from scratch)         (Transfer Learning)
    |                                     |
    |                           Phase 1: Train head only
    |                                     |
    |                           Phase 2: Fine-tune top 20 layers
    |                                     |
    +------------------+------------------+
                       |
              Evaluate + Compare
```

---

## Models

### Custom CNN

Three convolutional blocks, each with two Conv2D layers, BatchNormalization, MaxPooling, and Dropout. Followed by two fully connected Dense layers with BatchNorm and Dropout before the softmax output.

```
Block 1  —  Conv2D(32)  x2  ->  MaxPool  ->  Dropout(0.25)
Block 2  —  Conv2D(64)  x2  ->  MaxPool  ->  Dropout(0.25)
Block 3  —  Conv2D(128) x2  ->  MaxPool  ->  Dropout(0.25)
FC       —  Dense(256)  ->  Dense(128)   ->  Softmax(10)
```

### EfficientNetB0 (Transfer Learning)

Pre-trained on ImageNet. A custom classification head is attached on top of the frozen base. Training happens in two phases:

- **Phase 1** — base model frozen, only the head is trained (15 epochs, lr=0.001)
- **Phase 2** — top 20 layers of base unfrozen for fine-tuning (10 epochs, lr=0.0001)

---

## Data Augmentation

Applied to the training set only using `ImageDataGenerator`:

```
rotation_range    = 15
width_shift       = 0.1
height_shift      = 0.1
horizontal_flip   = True
zoom_range        = 0.1
shear_range       = 0.1
fill_mode         = nearest
```

---

## Training Configuration

| Setting | CNN | Transfer Learning |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 / 0.0001 (fine-tune) |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Batch size | 64 | 32 |
| Max epochs | 50 | 15 + 10 |
| Early stopping | val_accuracy (patience=10) | val_accuracy (patience=8) |
| LR scheduler | ReduceLROnPlateau (patience=3) | ReduceLROnPlateau (patience=3) |

---

## Evaluation

Both models are evaluated on the held-out test set using:

- Accuracy and Loss
- Classification Report (precision, recall, F1 per class)
- Confusion Matrix

---

## Visualizations

The script produces the following plots in sequence:

```
1. Sample images grid (5x5)
2. Class distribution — train and test
3. Data augmentation examples — original vs augmented
4. CNN training curves — accuracy and loss
5. CNN confusion matrix
6. CNN prediction grid (15 images, color-coded correct/wrong)
7. Transfer learning training curves with fine-tune marker
8. Transfer learning confusion matrix
9. Final model comparison bar chart
```

---

## Requirements

```
tensorflow >= 2.10
scikit-learn >= 1.3
numpy
matplotlib
seaborn
```

Install:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
```

---

## Run

```bash
python image_classification_cnn.py
```

CIFAR-10 is downloaded automatically via `keras.datasets.cifar10.load_data()` on first run.

---

## Notes

- EfficientNetB0 expects 224x224 input — images are resized using `tf.image.resize` before training
- Validation split is stratified to preserve class balance
- Early stopping restores the best weights automatically — no manual model saving needed
- GPU is recommended for the transfer learning phase; the script prints GPU availability on startup

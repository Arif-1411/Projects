# 📄 README.md - Cats vs Dogs Classification

```markdown
# 🐱🐶 Cats vs Dogs Image Classification

A deep learning project using Transfer Learning (MobileNetV2) to classify images of cats and dogs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

This project implements a **binary image classification** system that can distinguish between cats and dogs. It uses **Transfer Learning** with a pre-trained **MobileNetV2** model, fine-tuned on a custom dataset.

### Key Features:
- ✅ Transfer Learning with MobileNetV2
- ✅ Data Augmentation for better generalization
- ✅ Fine-tuning of last 20 layers
- ✅ Early Stopping & Learning Rate Reduction
- ✅ Interactive image upload for testing

---

## 📦 Dataset

| Info | Details |
|------|---------|
| **Source** | [Kaggle - Cats and Dogs Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification) |
| **Total Images** | ~700 images |
| **Training Set** | 557 images (cats: 279, dogs: 278) |
| **Test Set** | 140 images (cats: 70, dogs: 70) |
| **Image Format** | JPEG |
| **Image Size** | Various (resized to 128x128) |

### Dataset Structure:
```
dataset/
├── train/
│   ├── cats/    # 279 images
│   └── dogs/    # 278 images
└── test/
    ├── cats/    # 70 images
    └── dogs/    # 70 images
```

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────┐
│ MobileNetV2 (Pre-trained ImageNet)  │
│ - Input: (128, 128, 3)              │
│ - Last 20 layers: Trainable         │
│ - Other layers: Frozen              │
├─────────────────────────────────────┤
│ GlobalAveragePooling2D              │
├─────────────────────────────────────┤
│ Dense (128, ReLU)                   │
├─────────────────────────────────────┤
│ Dropout (0.3)                       │
├─────────────────────────────────────┤
│ Dense (1, Sigmoid)                  │
└─────────────────────────────────────┘
```

### Model Summary:
| Parameter | Value |
|-----------|-------|
| Total Parameters | ~2.6M |
| Trainable Parameters | ~361K |
| Non-trainable Parameters | ~2.2M |

---

## ⚙️ Installation

### Prerequisites:
- Python 3.8+
- Google Colab (recommended) or local environment with GPU

### Steps:

1. **Clone the repository:**
```bash
git clone https://github.com/Arif-1411/Deep-Learning.git
cd cats-dogs-classification
```

2. **Install dependencies:**
```bash
pip install tensorflow numpy matplotlib kagglehub
```

3. **Run in Google Colab:**
   - Open `cats_dogs_classifier.ipynb` in Colab
   - Run all cells

---

## 🚀 Usage

### Training:
```python
# Download dataset
import kagglehub
path = kagglehub.dataset_download("samuelcortinhas/cats-and-dogs-image-classification")

# Train model
history = model.fit(
    train_iterator,
    epochs=20,
    validation_data=val_iterator,
    callbacks=[early_stop, reduce_lr]
)
```

### Prediction:
```python
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# Load and preprocess image
img = load_img('your_image.jpg', target_size=(128, 128))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 128, 128, 3)

# Predict
pred = model.predict(img_array)[0][0]
label = 'Dog' if pred > 0.5 else 'Cat'
confidence = pred if pred > 0.5 else 1 - pred

print(f"Prediction: {label} ({confidence*100:.1f}%)")
```

---

## 📊 Results

### Training Performance:

| Metric | Value |
|--------|-------|
| Best Training Accuracy | ~95% |
| Best Validation Accuracy | ~90% |
| Training Loss | ~0.15 |
| Validation Loss | ~0.30 |
| Epochs Trained | 15-20 |

### Accuracy & Loss Graphs:

![Training Results](training_results.png)

### Sample Predictions:

| Image | Prediction | Confidence |
|-------|------------|------------|
| 🖼️ cat1.jpg | Cat | 98.5% |
| 🖼️ dog1.jpg | Dog | 96.2% |
| 🖼️ cat2.jpg | Cat | 94.8% |

---

## 📁 Project Structure

```
cats-dogs-classification/
│
├── 📓 cats_dogs_classifier.ipynb   # Main notebook
├── 📄 README.md                     # Documentation
├── 📄 requirements.txt              # Dependencies
│
├── 📂 models/
│   └── cats_dogs_model.h5          # Saved model
│
├── 📂 results/
│   ├── training_results.png        # Accuracy/Loss graphs
│   └── predictions/                # Sample predictions
│
└── 📂 test_images/                 # Test images
    ├── cat_test.jpg
    └── dog_test.jpg
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3.8+-blue) | Programming Language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) | Deep Learning Framework |
| ![Keras](https://img.shields.io/badge/Keras-API-red) | High-level Neural Network API |
| ![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer_Learning-green) | Pre-trained Model |
| ![NumPy](https://img.shields.io/badge/NumPy-Array_Processing-blue) | Numerical Computing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue) | Data Visualization |
| ![Google Colab](https://img.shields.io/badge/Google_Colab-GPU-yellow) | Cloud Environment |

---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 128 x 128 |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 20 (with Early Stopping) |
| Dropout Rate | 0.3 |

### Data Augmentation:
- Rotation: 20°
- Width Shift: 10%
- Height Shift: 10%
- Zoom: 10%
- Horizontal Flip: Yes

---

## 🚀 Future Improvements

- [ ] Add more data (larger dataset)
- [ ] Try other architectures (ResNet, EfficientNet)
- [ ] Implement Grad-CAM for visualization
- [ ] Create web app using Streamlit/Flask
- [ ] Add multi-class classification (more animals)
- [ ] Mobile deployment using TensorFlow Lite

---


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Your Name**
- GitHub: [Arifudheen](https://github.com/Arif-1411)
- LinkedIn: [Arifudheen](https://www.linkedin.com/in/arifudheen-t-2199203a9?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)

---



---

## 📋 requirements.txt

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
kagglehub>=0.1.0
Pillow>=9.0.0
```

---



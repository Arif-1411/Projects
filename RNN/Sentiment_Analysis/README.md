# Sentiment Analysis using RNN, LSTM and Bidirectional LSTM

Binary sentiment classification on the IMDB Movie Reviews dataset using deep learning and NLP preprocessing.

---

## Project Overview

This project builds and compares three recurrent neural network models to classify movie reviews as positive or negative. The full pipeline covers text preprocessing, tokenization, model training, evaluation, and custom prediction.

---

## Dataset

- **Source:** IMDB Movie Reviews (via `keras.datasets.imdb`)
- **Size:** 50,000 reviews (25,000 train / 25,000 test)
- **Classes:** Positive, Negative (balanced)
- **Vocabulary Size:** 20,000 words

---

## Requirements

Install all dependencies before running the notebook:

```bash
pip install tensorflow nltk wordcloud scikit-learn matplotlib seaborn pandas numpy
```

After installation, restart the kernel before running any cells.

---

## How to Run

1. Install the requirements listed above.
2. Open `sentiment_analysis_rnn.ipynb` in Jupyter or VS Code.
3. Restart the kernel.
4. Run all cells from top to bottom in order.

> Note: Preprocessing all 50,000 reviews takes a few minutes. Training each model depends on your hardware. Using a GPU is recommended.

---

## Notebook Structure

| Cell | Section |
|------|---------|
| 1 | Import Libraries |
| 2 | Load Dataset |
| 3 | Data Exploration and Visualization |
| 4 | Text Preprocessing with NLP |
| 5 | Preprocess All Reviews |
| 6 | Word Cloud Visualization |
| 7 | Tokenization and Sequence Padding |
| 8 | Sequence Length Analysis |
| 9 | Model 1 - Simple RNN |
| 10 | Model 2 - LSTM |
| 11 | Model 3 - Bidirectional LSTM |
| 12 | Training Callbacks |
| 13 | Train All Models |
| 14 | Training History Visualization |
| 15 | Model Evaluation |
| 16 | Confusion Matrices |
| 17 | ROC Curves |
| 18 | Model Comparison |
| 19 | Predict on Custom Text |
| 20 | Word Embedding Visualization (PCA) |
| 21 | Save All Models |

---

## NLP Preprocessing Pipeline

Each review goes through the following steps before being fed into the model:

1. Lowercase conversion
2. HTML tag removal
3. URL removal
4. Special character and number removal
5. Tokenization using NLTK word_tokenize
6. Stopword removal (179 English stopwords)
7. Stemming using PorterStemmer
8. Lemmatization using WordNetLemmatizer
9. Word embedding via learned Embedding layer
10. Sequence padding to fixed length of 200

---

## Models

### Model 1 - Simple RNN

- Embedding layer (20000 x 128)
- SimpleRNN layer (64 units, dropout 0.3)
- Dense layers with Dropout
- Output: Sigmoid activation

### Model 2 - LSTM

- Embedding layer (20000 x 128)
- LSTM layer (128 units, dropout 0.3)
- Dense layers with BatchNormalization and Dropout
- Output: Sigmoid activation

### Model 3 - Bidirectional LSTM

- Embedding layer (20000 x 128)
- Bidirectional LSTM layer 1 (64 units, return_sequences=True)
- Bidirectional LSTM layer 2 (32 units)
- Dense layers with BatchNormalization and Dropout
- Output: Sigmoid activation

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Max Epochs | 20 (EarlyStopping applied) |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Validation Split | 20% of training data |

Callbacks used: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## Output Files

After running the full notebook, the following files are generated:

**Plots:**
- `data_exploration.png`
- `word_clouds.png`
- `sequence_analysis.png`
- `all_models_training_history.png`
- `confusion_matrices.png`
- `roc_curves.png`
- `model_comparison.png`
- `word_embeddings_pca.png`

**Saved Models:**
- `saved_models/simple_rnn_model.h5`
- `saved_models/lstm_model.h5`
- `saved_models/bidirectional_lstm_model.h5`
- `saved_models/tokenizer.pickle`

---

## Loading a Saved Model

```python
from tensorflow import keras
import pickle

model = keras.models.load_model('saved_models/lstm_model.h5')
tokenizer = pickle.load(open('saved_models/tokenizer.pickle', 'rb'))
```

---

## Project Structure

```
sentiment-analysis/
    sentiment_analysis_rnn.ipynb
    README.md
    saved_models/
        simple_rnn_model.h5
        lstm_model.h5
        bidirectional_lstm_model.h5
        tokenizer.pickle
    data_exploration.png
    word_clouds.png
    sequence_analysis.png
    all_models_training_history.png
    confusion_matrices.png
    roc_curves.png
    model_comparison.png
    word_embeddings_pca.png
```

---

## Author

Developed as part of an NLP and deep learning project using the IMDB dataset.

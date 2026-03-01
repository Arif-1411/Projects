# ====================
# Libraries
# ====================
# !pip install -q tensorflow tensorflow-hub tensorflow-text pandas scikit-learn matplotlib seaborn

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ====================
# Configuration
# ====================
CONFIG = {
    "NUM_CLASSES": 5,
    "TOKEN_EMBED_DIM": 128,
    "CHAR_EMBED_DIM": 64,
    "LSTM_UNITS": 24,
    "DENSE_UNITS": 256,
    "DROPOUT_RATE": 0.5,
    "BATCH_SIZE": 16,
    "EPOCHS": 2,
    "LABEL_SMOOTHING": 0.2,
    "MAX_CHAR_LENGTH": 200,
    "CHAR_VOCAB_SIZE": 1000,
    "RANDOM_SEED": 42
}

tf.random.set_seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

# ====================
# Simple Dataset
# ====================
data = {
    "abstract_id": [1]*5 + [2]*5,
    "sentence": [
        "Treatment options for humeral shaft fractures are limited.",
        "This study compares the safety and efficacy of treatments.",
        "Patients were randomized to receive either treatment A or B.",
        "There were no significant differences in outcomes.",
        "Use of treatment A is safe and effective.",
        "Osteoporosis is a major health problem.",
        "We analyzed the effects of calcium and vitamin D supplementation.",
        "Patients with low bone density received supplements.",
        "Bone density improved significantly after 6 months.",
        "Calcium and vitamin D supplementation is beneficial."
    ],
    "label": ["BACKGROUND","OBJECTIVE","METHODS","RESULTS","CONCLUSION"]*2
}

df = pd.DataFrame(data)

# ====================
# Feature Engineering
# ====================
# Line number & total lines
df["line_number"] = df.groupby("abstract_id").cumcount()
df["total_lines"] = df.groupby("abstract_id")["line_number"].transform("max") + 1

# One-hot encode labels
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(df["label"].values.reshape(-1,1))

# ====================
# Character Vectorization
# ====================
char_vectorizer = layers.TextVectorization(
    max_tokens=CONFIG["CHAR_VOCAB_SIZE"],
    output_mode="int",
    output_sequence_length=CONFIG["MAX_CHAR_LENGTH"]
)
char_vectorizer.adapt(df["sentence"])

# ====================
# Train-Test Split
# ====================
X = df[["sentence","line_number","total_lines"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, 
    random_state=CONFIG["RANDOM_SEED"], 
    stratify=y
)

# ====================
# Model Architecture
# ====================

# Token model (TF-Hub USE)
token_input = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_emb = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4", 
    trainable=False
)(token_input)
token_dense = layers.Dense(CONFIG["TOKEN_EMBED_DIM"], activation="relu")(token_emb)

# Char model
char_input = layers.Input(
    shape=(CONFIG["MAX_CHAR_LENGTH"],), 
    dtype=tf.int32, 
    name="char_input"
)
char_emb = layers.Embedding(
    CONFIG["CHAR_VOCAB_SIZE"], 
    CONFIG["CHAR_EMBED_DIM"], 
    mask_zero=True
)(char_input)
char_lstm = layers.Bidirectional(layers.LSTM(CONFIG["LSTM_UNITS"]))(char_emb)

# Positional features
line_input = layers.Input(shape=(1,), name="line_input")
line_dense = layers.Dense(32, activation="relu")(line_input)

total_input = layers.Input(shape=(1,), name="total_input")
total_dense = layers.Dense(32, activation="relu")(total_input)

# Combine all
combined = layers.Concatenate()([token_dense, char_lstm])
combined = layers.Dense(CONFIG["DENSE_UNITS"], activation="relu")(combined)
combined = layers.Dropout(CONFIG["DROPOUT_RATE"])(combined)

tribrid = layers.Concatenate()([combined, line_dense, total_dense])
output = layers.Dense(CONFIG["NUM_CLASSES"], activation="softmax")(tribrid)

# Build model
model = tf.keras.Model(
    [token_input, char_input, line_input, total_input], 
    output
)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=CONFIG["LABEL_SMOOTHING"]
    ),
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ====================
# Dataset Preparation
# ====================
def make_dataset(X, y):
    char_seq = char_vectorizer(X["sentence"].values)
    return tf.data.Dataset.from_tensor_slices((
        {
            "token_input": X["sentence"].values,
            "char_input": char_seq,
            "line_input": X["line_number"].values.reshape(-1,1),
            "total_input": X["total_lines"].values.reshape(-1,1)
        },
        y
    )).batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train)
test_ds = make_dataset(X_test, y_test)

# ====================
# Training
# ====================
print("\nTraining Started...")
history = model.fit(
    train_ds, 
    epochs=CONFIG["EPOCHS"], 
    validation_data=test_ds
)

# ====================
# Visualization
# ====================
print("\nPlotting Results...")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()

# ====================
# Prediction Demo
# ====================
print("\nPrediction Demo:")

sample_text = "Patients were randomized to receive treatment A."
sample_char = char_vectorizer([sample_text])
sample_line = np.array([[2]])
sample_total = np.array([[5]])

pred = model.predict([
    np.array([sample_text]), 
    sample_char, 
    sample_line, 
    sample_total
])

pred_class = ohe.inverse_transform(pred)[0][0]
confidence = np.max(pred)

print(f"\nSentence: {sample_text}")
print(f"Predicted class: {pred_class}")
print(f"Confidence: {confidence:.4f}")
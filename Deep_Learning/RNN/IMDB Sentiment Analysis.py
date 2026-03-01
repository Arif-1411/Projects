# ==========================================
# Step 1: Comments Header
# ==========================================
"""
IMDB Sentiment Analysis using RNN/LSTM
- Simple RNN
- LSTM
- Bidirectional LSTM
- Custom review prediction
"""

# ==========================================
# Step 2: Libraries Install
# ==========================================
# pip install tensorflow numpy pandas matplotlib scikit-learn

# ==========================================
# Step 3: Libraries Import
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# Step 4: Dataset Load & Explore
# ==========================================
vocab_size = 10000
maxlen = 200

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"\nDataset Info:")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Positive reviews in train: {sum(y_train)}")
print(f"Negative reviews in train: {len(y_train) - sum(y_train)}")

# Plot review lengths
review_lengths = [len(review) for review in x_train]
plt.figure(figsize=(10, 5))
plt.hist(review_lengths, bins=50, edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.axvline(x=maxlen, color='r', linestyle='--', label=f'Max Length = {maxlen}')
plt.legend()
plt.show()

# ==========================================
# Step 5: Data Preprocessing
# ==========================================
print("\nPadding sequences...")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Get word index for decoding
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])

# Show sample review
print("\nSample Review:")
print(decode_review(x_train[0]))
print(f"Label: {'Positive' if y_train[0] == 1 else 'Negative'}")

# ==========================================
# Step 6: Simple RNN Model
# ==========================================
print("\n" + "="*50)
print("MODEL 1: Simple RNN")
print("="*50)

model_rnn = models.Sequential([
    layers.Embedding(vocab_size, 32, input_length=maxlen),
    layers.SimpleRNN(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])

model_rnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_rnn.summary()

# ==========================================
# Step 7: LSTM Model
# ==========================================
print("\n" + "="*50)
print("MODEL 2: LSTM")
print("="*50)

model_lstm = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=maxlen),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_lstm.summary()

# ==========================================
# Step 8: Bidirectional LSTM Model
# ==========================================
print("\n" + "="*50)
print("MODEL 3: Bidirectional LSTM")
print("="*50)

model_bilstm = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=maxlen),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Dropout(0.5),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_bilstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_bilstm.summary()

# ==========================================
# Step 9: Model Training (LSTM)
# ==========================================
print("\n" + "="*50)
print("TRAINING LSTM MODEL")
print("="*50)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model_lstm.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# Step 10: Evaluate Model
# ==========================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

test_loss, test_acc = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# Step 11: Custom Review Prediction (CORRECTED)
# ==========================================
print("\n" + "="*50)
print("CUSTOM REVIEW PREDICTION")
print("="*50)

def predict_sentiment_correct(review_text, model=model_lstm):
    """
    Predict sentiment using IMDB word index
    """
    # Convert words to integers using IMDB word_index
    encoded = []
    for word in review_text.lower().split():
        index = word_index.get(word, 2)  # 2 = OOV (out of vocabulary)
        if index < vocab_size:
            encoded.append(index)
        else:
            encoded.append(2)
    
    # Pad sequence
    padded = sequence.pad_sequences([encoded], maxlen=maxlen)
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nReview: {review_text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Score: {prediction:.4f}")
    
    return sentiment, confidence

# ==========================================
# Step 12: Test Custom Reviews
# ==========================================
print("\nTesting custom reviews:\n")

test_reviews = [
    "This movie was amazing and fun!",
    "Terrible movie, waste of time",
    "The best film I have ever seen",
    "Boring and poorly acted",
    "Happy Journey",
    "Absolutely brilliant masterpiece",
    "Worst movie ever made"
]

for review in test_reviews:
    predict_sentiment_correct(review)
    print("-" * 50)

# ==========================================
# Step 13: Advanced Prediction Function
# ==========================================
def predict_with_explanation(review_text, model=model_lstm):
    """
    Predict with detailed explanation
    """
    # Encode
    encoded = []
    found_words = []
    unknown_words = []
    
    for word in review_text.lower().split():
        index = word_index.get(word, 2)
        if index < vocab_size and index != 2:
            encoded.append(index)
            found_words.append(word)
        else:
            encoded.append(2)
            unknown_words.append(word)
    
    # Pad and predict
    padded = sequence.pad_sequences([encoded], maxlen=maxlen)
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nReview: {review_text}")
    print(f"Found words: {', '.join(found_words)}")
    print(f"Unknown words: {', '.join(unknown_words) if unknown_words else 'None'}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Prediction Score: {prediction:.4f}")
    
    return sentiment, confidence, found_words, unknown_words

# Test advanced prediction
print("\n" + "="*50)
print("ADVANCED PREDICTION WITH EXPLANATION")
print("="*50)

predict_with_explanation("This movie was absolutely fantastic and amazing!")
predict_with_explanation("Horrible acting and terrible plot")
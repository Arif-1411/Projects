"""
Movie Review Sentiment Analysis using RNN (LSTM)
IMDb Dataset - 50,000 Reviews
Binary Classification: Positive (1) or Negative (0)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Step 1: Load IMDb Dataset
# ==========================================
print("="*60)
print("MOVIE SENTIMENT ANALYSIS USING LSTM")
print("="*60)

vocab_size = 10000
print(f"\nLoading IMDb dataset...")
print(f"Vocabulary size: {vocab_size} words")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print("\nDataset Info:")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Positive reviews in train: {sum(y_train)}")
print(f"Negative reviews in train: {len(y_train) - sum(y_train)}")

# Show sample review
print("\nSample review (encoded):")
print(f"Length: {len(x_train[0])} words")
print(f"First 20 word indices: {x_train[0][:20]}")
print(f"Label: {'Positive' if y_train[0] == 1 else 'Negative'}")

# ==========================================
# Step 2: Decode a Sample Review
# ==========================================
# Get word index dictionary
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    """Decode integer-encoded review back to text"""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])

sample_review = decode_review(x_train[0])
print("\nDecoded sample review:")
print(sample_review[:300] + "...")
print(f"Sentiment: {'Positive' if y_train[0] == 1 else 'Negative'}")

# ==========================================
# Step 3: Pad Sequences
# ==========================================
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

max_len = 200
print(f"\nPadding sequences to max length: {max_len}")

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# ==========================================
# Step 4: Build RNN Model
# ==========================================
print("\n" + "="*60)
print("BUILDING LSTM MODEL")
print("="*60)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ==========================================
# Step 5: Train Model
# ==========================================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Complete!")

# ==========================================
# Step 6: Evaluate Model
# ==========================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ==========================================
# Step 7: Visualizations
# ==========================================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# Plot 1: Accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# Step 8: Custom Review Prediction
# ==========================================
print("\n" + "="*60)
print("CUSTOM REVIEW PREDICTION")
print("="*60)

def predict_sentiment(review_text, model, word_index, max_len=200):
    """
    Predict sentiment of custom review text
    
    Parameters:
    - review_text: Raw text review
    - model: Trained model
    - word_index: Word to index mapping
    - max_len: Maximum sequence length
    
    Returns:
    - sentiment: 'Positive' or 'Negative'
    - confidence: Probability score
    """
    # Convert text to sequence
    words = review_text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # +3 offset for IMDb
    
    # Pad sequence
    padded = pad_sequences([encoded], maxlen=max_len)
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence, prediction

# Test with custom reviews
test_reviews = [
    "This movie was absolutely amazing! Best film I've ever seen.",
    "Terrible movie. Complete waste of time and money.",
    "The acting was brilliant and the story was captivating.",
    "Boring and predictable. I fell asleep halfway through.",
    "A masterpiece of cinema with outstanding performances."
]

print("\nTesting custom reviews:\n")

for i, review in enumerate(test_reviews, 1):
    sentiment, confidence, raw_score = predict_sentiment(review, model, word_index, max_len)
    
    print(f"{i}. Review: {review}")
    print(f"   Sentiment: {sentiment}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Raw Score: {raw_score:.4f}")
    print("-"*60)

# ==========================================
# Step 9: Interactive Prediction
# ==========================================
def interactive_sentiment_analysis():
    """
    Interactive mode for user input
    """
    print("\n" + "="*60)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*60)
    print("\nType your movie review (or 'quit' to exit):\n")
    
    while True:
        user_review = input("Enter review: ").strip()
        
        if user_review.lower() == 'quit':
            print("Exiting... Thank you!")
            break
        
        if not user_review:
            print("Please enter a valid review.\n")
            continue
        
        sentiment, confidence, raw_score = predict_sentiment(user_review, model, word_index, max_len)
        
        print(f"\nSentiment: {sentiment}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Raw Score: {raw_score:.4f}\n")
        print("-"*60 + "\n")

# ==========================================
# Step 10: Confusion Matrix & Detailed Metrics
# ==========================================
print("\n" + "="*60)
print("DETAILED PERFORMANCE METRICS")
print("="*60)

# Get predictions for test set
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate metrics
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], fontsize=12)
plt.yticks(tick_marks, ['Negative', 'Positive'], fontsize=12)

# Add text annotations
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16)

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# Step 11: Save Model
# ==========================================
model_filename = 'movie_sentiment_lstm_model.h5'
model.save(model_filename)
print("\n" + "="*60)
print(f"Model saved: {model_filename}")
print("="*60)

# ==========================================
# Step 12: Load and Test Saved Model
# ==========================================
print("\nLoading saved model...")
loaded_model = load_model(model_filename)
print("Model loaded successfully!")

# Test loaded model
test_review = "This movie is fantastic and entertaining!"
sentiment, confidence, raw_score = predict_sentiment(test_review, loaded_model, word_index, max_len)

print(f"\nTest with loaded model:")
print(f"Review: {test_review}")
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence*100:.2f}%")

# ==========================================
# Final Summary
# ==========================================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"Dataset: IMDb 50,000 movie reviews")
print(f"Vocabulary Size: {vocab_size} words")
print(f"Max Sequence Length: {max_len} words")
print(f"Model: LSTM with Embedding layer")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Model saved: {model_filename}")
print("="*60)

# Ask user if they want interactive mode
print("\nWould you like to try interactive mode? (yes/no)")
choice = input().strip().lower()

if choice == 'yes':
    interactive_sentiment_analysis()
else:
    print("\nProject complete. Thank you!")
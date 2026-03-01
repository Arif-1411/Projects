import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------
# Sample Training Text
# (You can replace with large text later)
# --------------------------
text = """
deep learning is a subset of machine learning
machine learning is a subset of artificial intelligence
artificial intelligence is the future
deep learning models learn patterns from data
"""

# --------------------------
# Tokenization
# --------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

print(f" Total Words in Vocabulary: {total_words}")

# --------------------------
# Create Input Sequences
# --------------------------
input_sequences = []
for line in text.split("\n"):
    if line.strip():  # Empty lines skip ചെയ്യാൻ
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i+1])

print(f" Total Input Sequences: {len(input_sequences)}")

# --------------------------
# Padding
# --------------------------
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f" Input Shape (X): {X.shape}")
print(f" Output Shape (y): {y.shape}")

# --------------------------
# Build LSTM Model
# --------------------------
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --------------------------
# Train Model
# --------------------------
print("\n Training Started...\n")
history = model.fit(X, y, epochs=300, verbose=0)

print(f" Training Complete! Final Accuracy: {history.history['accuracy'][-1]:.4f}")

# --------------------------
# Predict Next Word Function
# --------------------------
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return None

# --------------------------
# Demo - Multiple Tests
# --------------------------

test_sentences = [
    "deep learning",
    "machine learning",
    "artificial intelligence",
    "deep learning models",
    "patterns from"
]

for input_text in test_sentences:
    next_word = predict_next_word(input_text)
    print(f" Input: '{input_text}'")
    print(f" Predicted Next Word: '{next_word}'")
    print(f" Complete Sentence: '{input_text} {next_word}'")

# --------------------------
# Interactive Prediction
# --------------------------
print("\n Try Your Own Sentence:")
print("(Type words from the training text)\n")

user_input = input("Enter text: ").strip()
if user_input:
    predicted = predict_next_word(user_input)
    if predicted:
        print(f"\n Predicted Next Word: {predicted}")
        print(f" Complete: {user_input} {predicted}")
    else:
        print(" Could not predict. Try words from training text.")
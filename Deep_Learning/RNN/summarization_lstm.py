"""
Abstractive Text Summarization using Seq2Seq LSTM
Generates new summary text
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("TensorFlow Version:", tf.__version__)

# ==========================================
# Sample Dataset (Text, Summary pairs)
# ==========================================
texts = [
    "artificial intelligence is transforming the world and many industries",
    "machine learning allows systems to learn from data automatically",
    "deep learning uses neural networks to solve complex problems",
    "natural language processing helps computers understand human language"
]

summaries = [
    "ai transforms industries",
    "machine learning learns from data",
    "deep learning solves problems",
    "nlp understands language"
]

print(f"Training Samples: {len(texts)}")
print("\nSample Data:")
for i in range(2):
    print(f"\nText: {texts[i]}")
    print(f"Summary: {summaries[i]}")

# ==========================================
# Tokenization
# ==========================================
print("\n" + "="*60)
print("TOKENIZATION")
print("="*60)

# Add start and end tokens to summaries
summaries_with_tokens = ['<start> ' + s + ' <end>' for s in summaries]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts + summaries_with_tokens)
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocabulary Size: {vocab_size}")

# Tokenize sequences
text_seq = tokenizer.texts_to_sequences(texts)
summary_seq = tokenizer.texts_to_sequences(summaries_with_tokens)

# Get max lengths
max_text_len = max(len(seq) for seq in text_seq)
max_summary_len = max(len(seq) for seq in summary_seq)

print(f"Max Text Length: {max_text_len}")
print(f"Max Summary Length: {max_summary_len}")

# Pad sequences
encoder_input = pad_sequences(text_seq, maxlen=max_text_len, padding='post')
decoder_input = pad_sequences(summary_seq, maxlen=max_summary_len, padding='post')

# Decoder target (shifted by one position)
decoder_target_data = np.zeros((len(texts), max_summary_len, vocab_size))

for i, seq in enumerate(summary_seq):
    for t, word_index in enumerate(seq):
        if t > 0:  # Skip the <start> token
            decoder_target_data[i, t-1, word_index] = 1

print(f"\nEncoder Input Shape: {encoder_input.shape}")
print(f"Decoder Input Shape: {decoder_input.shape}")
print(f"Decoder Target Shape: {decoder_target_data.shape}")

# ==========================================
# Build Seq2Seq Model
# ==========================================
print("\n" + "="*60)
print("BUILDING MODEL")
print("="*60)

# Encoder
encoder_inputs = Input(shape=(max_text_len,), name='encoder_input')
enc_emb = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(128, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_summary_len,), name='decoder_input')
dec_emb = Embedding(vocab_size, 64, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# Train Model
# ==========================================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

history = model.fit(
    [encoder_input, decoder_input],
    decoder_target_data,
    epochs=300,
    batch_size=2,
    verbose=0
)

print(f"\nTraining Complete!")
print(f"Final Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")

# ==========================================
# Inference Models
# ==========================================
print("\n" + "="*60)
print("BUILDING INFERENCE MODELS")
print("="*60)

# Encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = dec_emb
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

print("Inference models created!")

# ==========================================
# Prediction Function
# ==========================================
def decode_sequence(input_seq):
    # Encode the input
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Generate empty target sequence
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get('<start>', 1)
    
    # Reverse word index
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Decode
    decoded_sentence = []
    stop_condition = False
    max_length = 20
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Sample token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')
        
        if sampled_word == '<end>' or len(decoded_sentence) > max_length:
            stop_condition = True
        elif sampled_word and sampled_word != '<start>':
            decoded_sentence.append(sampled_word)
        
        # Update target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return ' '.join(decoded_sentence)

# ==========================================
# Test Predictions
# ==========================================
print("\n" + "="*60)
print("PREDICTIONS")
print("="*60)

for i in range(len(texts)):
    input_seq = encoder_input[i:i+1]
    decoded_summary = decode_sequence(input_seq)
    
    print(f"\n{i+1}.")
    print(f"Original: {texts[i]}")
    print(f"Expected: {summaries[i]}")
    print(f"Predicted: {decoded_summary}")
    print("-"*60)
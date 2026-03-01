"""
Text Encoder - Transformer Based
Uses Sentence Transformers for text embeddings
"""

import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

class TextEncoder(tf.keras.Model):
    """Text Encoder using Transformer models"""
    
    def __init__(self, model_name: str, embed_dim: int, max_length: int = 512):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # HuggingFace Transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = TFAutoModel.from_pretrained(model_name)
        
        # Projection layer
        self.projection = layers.Dense(
            embed_dim, 
            activation='relu', 
            name='text_projection'
        )
        
        # Dropout
        self.dropout = layers.Dropout(0.3)
    
    def call(self, texts, training=False):
        """
        Forward pass
        texts: List of strings or tf.string tensor
        """
        # Tokenization
        if isinstance(texts, tf.Tensor):
            if texts.ndim == 0:
                texts = [texts.numpy().decode('utf-8')]
            else:
                texts = [t.numpy().decode('utf-8') for t in texts]
        
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Transformer inference
        outputs = self.transformer(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )
        
        # Extract [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Project
        projected = self.projection(pooled_output)
        projected = self.dropout(projected, training=training)
        
        return projected
    
    def get_embedding_dim(self):
        return self.embed_dim

# Test
if __name__ == "__main__":
    encoder = TextEncoder("sentence-transformers/all-MiniLM-L6-v2", 384)
    test_texts = ["This is a test sentence", "Another test text"]
    embeddings = encoder(test_texts)
    print(f"Text Embedding Shape: {embeddings.shape}")
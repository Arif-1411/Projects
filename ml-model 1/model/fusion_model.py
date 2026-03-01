"""
Multimodal Fusion Model
Combines text and image embeddings for classification
"""

import tensorflow as tf
from tensorflow.keras import layers
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder

class MultimodalFusionModel(tf.keras.Model):
    """Multimodal Fusion Model - Text + Image"""
    
    def __init__(self, config):
        super(MultimodalFusionModel, self).__init__()
        
        # Encoders
        self.text_encoder = TextEncoder(
            config["model"]["text_model"],
            config["model"]["projection_dim"]
        )
        
        self.image_encoder = ImageEncoder(
            config["model"]["image_model"],
            config["model"]["projection_dim"]
        )
        
        # Fusion method
        self.fusion_method = config["model"]["fusion_method"]
        
        if self.fusion_method == "attention":
            # Multimodal attention
            self.attention_layer = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=config["model"]["projection_dim"]
            )
        
        elif self.fusion_method == "weighted":
            # Trainable weights
            self.fusion_weight = tf.Variable(
                initial_value=0.5,
                trainable=True,
                name="fusion_weight"
            )
        
        # Classification head
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation='relu', name='fc1'),
            layers.Dropout(config["model"]["dropout_rate"]),
            layers.Dense(64, activation='relu', name='fc2'),
            layers.Dropout(config["model"]["dropout_rate"]),
            layers.Dense(config["data"]["num_classes"], activation='softmax', name='output')
        ])
    
    def call(self, inputs, training=False):
        """
        Forward pass
        inputs: {
            'texts': batch of text strings,
            'images': batch of image tensors/paths
        }
        """
        texts = inputs['texts']
        images = inputs['images']
        
        # Get embeddings
        text_embeds = self.text_encoder(texts, training=training)
        image_embeds = self.image_encoder(images, training=training)
        
        # Fusion
        if self.fusion_method == "concatenate":
            fused = layers.Concatenate()([text_embeds, image_embeds])
        
        elif self.fusion_method == "attention":
            combined = tf.stack([text_embeds, image_embeds], axis=1)
            attended = self.attention_layer(
                query=combined,
                key=combined,
                value=combined,
                training=training
            )
            fused = tf.reshape(attended, [tf.shape(attended)[0], -1])
        
        elif self.fusion_method == "weighted":
            weight = tf.sigmoid(self.fusion_weight)
            fused = weight * text_embeds + (1 - weight) * image_embeds
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Classification
        output = self.classifier(fused, training=training)
        return output
    
    def get_embeddings(self, inputs):
        """Extract embeddings only"""
        texts = inputs['texts']
        images = inputs['images']
        
        text_embeds = self.text_encoder(texts, training=False)
        image_embeds = self.image_encoder(images, training=False)
        
        return {
            'text_embeddings': text_embeds,
            'image_embeddings': image_embeds
        }

# Test
if __name__ == "__main__":
    import yaml
    
    # Sample config
    config = {
        "model": {
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "image_model": "resnet50",
            "fusion_method": "concatenate",
            "projection_dim": 256,
            "dropout_rate": 0.3
        },
        "data": {
            "num_classes": 2
        }
    }
    
    model = MultimodalFusionModel(config)
    
    # Test inputs
    test_inputs = {
        'texts': tf.constant(["Fake news text", "Real news text"]),
        'images': tf.random.uniform((2, 224, 224, 3), 0, 255, dtype=tf.float32)
    }
    
    predictions = model(test_inputs)
    print(f"Prediction Shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")